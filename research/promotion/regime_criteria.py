"""Regime-aware qualification gates for T24R."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence

from research.experiments.comparator import ComparisonResult
from research.hierarchy.modes import MODE_DEFENSIVE

_TOLERANCE = 1e-9
_REGIME_BUCKETS = ("high_vol", "mid_vol", "low_vol")


@dataclass(frozen=True)
class RegimeQualificationEvaluation:
    """Structured response for regime-aware gating."""

    hard_failures: Sequence[str]
    soft_failures: Sequence[str]
    report: Dict[str, Dict[str, Any]]


@dataclass(frozen=True)
class RegimeQualificationCriteria:
    """Configuration for regime-aware qualification requirements."""

    drawdown_regression_pct: float = 10.0
    high_vol_exposure_cap: float = 0.40
    sharpe_degradation_pct: float = 15.0
    turnover_increase_pct: float = 20.0
    mode_transition_fraction: float = 0.10
    execution_drawdown_regression_pct: float = 2.0

    def evaluate(
        self,
        comparison: ComparisonResult,
        candidate_metrics: Mapping[str, Any],
        baseline_metrics: Mapping[str, Any],
        *,
        hierarchy_enabled: bool,
        execution_metrics: Mapping[str, Any] | None = None,
    ) -> RegimeQualificationEvaluation:
        """Evaluate regime-specific gates and summarize failures."""

        regime_deltas = _copy_regime_deltas(comparison.regime_deltas)
        report: Dict[str, Dict[str, Any]] = {
            bucket: {
                "passed": True,
                "checks": {},
                "deltas": regime_deltas.get(bucket, {}),
                "execution_metrics": _execution_bucket_metrics(execution_metrics, bucket),
            }
            for bucket in _REGIME_BUCKETS
        }
        hard_failures: list[str] = []
        soft_failures: list[str] = []

        self._evaluate_high_vol_drawdown(regime_deltas, report, hard_failures)
        self._evaluate_execution_drawdown(regime_deltas, report, hard_failures)
        if hierarchy_enabled:
            self._evaluate_defensive_exposure(regime_deltas, report, hard_failures)
        self._evaluate_global_sharpe(candidate_metrics, baseline_metrics, report, soft_failures)
        self._evaluate_high_vol_turnover(regime_deltas, report, soft_failures)
        if hierarchy_enabled:
            self._evaluate_hierarchy_sanity(candidate_metrics, report, soft_failures)

        return RegimeQualificationEvaluation(
            hard_failures=hard_failures,
            soft_failures=soft_failures,
            report=report,
        )

    def _evaluate_high_vol_drawdown(
        self,
        regime_deltas: Mapping[str, Mapping[str, Mapping[str, float | None]]],
        report: Dict[str, Dict[str, Any]],
        failures: list[str],
    ) -> None:
        candidate, baseline, entry = _regime_metric(regime_deltas, "high_vol", "max_drawdown")
        limit = None
        status = "fail"
        reason = "regime_high_vol_drawdown_missing"
        if candidate is not None and baseline is not None:
            limit = baseline * (1.0 + self.drawdown_regression_pct / 100.0)
            if candidate <= (limit + _TOLERANCE):
                status = "pass"
                reason = None
            else:
                reason = "regime_high_vol_drawdown_regressed"
        if status == "fail" and reason:
            failures.append(reason)
        _record_check(
            report,
            "high_vol",
            "drawdown_protection",
            status,
            {
                "observed": candidate,
                "baseline": baseline,
                "limit": limit,
                "delta": entry.get("delta") if entry else None,
            },
        )

    def _evaluate_execution_drawdown(
        self,
        regime_deltas: Mapping[str, Mapping[str, Mapping[str, float | None]]],
        report: Dict[str, Dict[str, Any]],
        failures: list[str],
    ) -> None:
        entry = regime_deltas.get("high_vol", {}).get("max_drawdown")
        candidate = entry.get("candidate") if isinstance(entry, Mapping) else None
        baseline = entry.get("baseline") if isinstance(entry, Mapping) else None
        status = "pass"
        reason = None
        limit = None
        if candidate is not None and baseline is not None:
            limit = baseline + (self.execution_drawdown_regression_pct / 100.0)
            if candidate > limit + _TOLERANCE:
                status = "fail"
                reason = "execution_high_vol_drawdown_regressed"
                failures.append(reason)
        _record_check(
            report,
            "high_vol",
            "execution_drawdown_guard",
            status if candidate is not None and baseline is not None else "skip",
            {
                "observed": candidate,
                "baseline": baseline,
                "limit": limit,
                "delta": entry.get("delta") if isinstance(entry, Mapping) else None,
                "tolerance_pct": self.execution_drawdown_regression_pct,
            },
        )

    def _evaluate_defensive_exposure(
        self,
        regime_deltas: Mapping[str, Mapping[str, Mapping[str, float | None]]],
        report: Dict[str, Dict[str, Any]],
        failures: list[str],
    ) -> None:
        candidate, baseline, entry = _regime_metric(regime_deltas, "high_vol", "avg_exposure")
        status = "fail"
        limit = self.high_vol_exposure_cap
        if baseline is not None:
            limit = min(limit, baseline)
        reason = "regime_high_vol_exposure_missing"
        if candidate is not None:
            if candidate <= (limit + _TOLERANCE):
                status = "pass"
                reason = None
            else:
                reason = "regime_high_vol_exposure_exceeded"
        if status == "fail" and reason:
            failures.append(reason)
        _record_check(
            report,
            "high_vol",
            "defensive_exposure_guard",
            status,
            {
                "observed": candidate,
                "baseline": baseline,
                "limit": limit,
                "delta": entry.get("delta") if entry else None,
            },
        )

    def _evaluate_global_sharpe(
        self,
        candidate_metrics: Mapping[str, Any],
        baseline_metrics: Mapping[str, Any],
        report: Dict[str, Dict[str, Any]],
        warnings: list[str],
    ) -> None:
        candidate = _extract_metric(candidate_metrics, ("performance", "sharpe"))
        baseline = _extract_metric(baseline_metrics, ("performance", "sharpe"))
        status = "pass"
        degradation = None
        reason = None
        if candidate is None or baseline is None or baseline <= _TOLERANCE:
            status = "warn"
            reason = "regime_sharpe_missing"
        else:
            change = baseline - candidate
            if change > _TOLERANCE:
                degradation = (change / baseline) * 100.0
                if degradation > self.sharpe_degradation_pct + _TOLERANCE:
                    status = "warn"
                    reason = "regime_sharpe_degradation"
        if status == "warn" and reason:
            message = reason
            if degradation is not None:
                message = f"{reason}:{round(degradation, 6)}>{self.sharpe_degradation_pct}"
            warnings.append(message)
        _record_check(
            report,
            "high_vol",
            "global_sharpe_guard",
            status,
            {
                "observed": candidate,
                "baseline": baseline,
                "limit_pct": self.sharpe_degradation_pct,
                "degradation_pct": None if degradation is None else round(degradation, 10),
            },
        )

    def _evaluate_high_vol_turnover(
        self,
        regime_deltas: Mapping[str, Mapping[str, Mapping[str, float | None]]],
        report: Dict[str, Dict[str, Any]],
        warnings: list[str],
    ) -> None:
        candidate, baseline, entry = _regime_metric(regime_deltas, "high_vol", "avg_turnover")
        status = "pass"
        reason = None
        limit = None
        increase_pct = None
        if candidate is None or baseline is None or baseline <= _TOLERANCE:
            status = "warn"
            reason = "regime_high_vol_turnover_missing"
        else:
            limit = baseline * (1.0 + self.turnover_increase_pct / 100.0)
            delta = candidate - baseline
            if delta > _TOLERANCE:
                increase_pct = (delta / baseline) * 100.0
                if increase_pct > self.turnover_increase_pct + _TOLERANCE:
                    status = "warn"
                    reason = "regime_high_vol_turnover_spike"
        if status == "warn" and reason:
            if increase_pct is not None:
                warnings.append(f"{reason}:{round(increase_pct, 6)}>{self.turnover_increase_pct}")
            else:
                warnings.append(reason)
        _record_check(
            report,
            "high_vol",
            "turnover_stability",
            status,
            {
                "observed": candidate,
                "baseline": baseline,
                "limit": limit,
                "delta": entry.get("delta") if entry else None,
                "increase_pct": None if increase_pct is None else round(increase_pct, 10),
            },
        )

    def _evaluate_hierarchy_sanity(
        self,
        candidate_metrics: Mapping[str, Any],
        report: Dict[str, Dict[str, Any]],
        warnings: list[str],
    ) -> None:
        trading = candidate_metrics.get("trading")
        counts = trading.get("mode_counts") if isinstance(trading, Mapping) else None
        transitions = trading.get("mode_transitions") if isinstance(trading, Mapping) else None
        normalized_counts = _normalize_mapping(counts)
        total_steps = sum(normalized_counts.values())
        transition_total = sum(_normalize_mapping(transitions).values())
        status = "pass"
        limit = None
        reason = None
        if total_steps <= 0:
            status = "warn"
            reason = "hierarchy_mode_counts_missing"
        else:
            limit = math.ceil(total_steps * self.mode_transition_fraction)
            if transition_total > limit + _TOLERANCE:
                status = "warn"
                reason = "hierarchy_mode_transitions_exceeded"
        if status == "warn" and reason:
            warnings.append(reason)
        defensive_warn = None
        defensive_fraction = 0.0
        if total_steps > 0:
            defensive_fraction = normalized_counts.get(MODE_DEFENSIVE, 0.0) / total_steps
            if defensive_fraction <= _TOLERANCE:
                defensive_warn = "hierarchy_never_defensive"
            elif defensive_fraction >= 1.0 - _TOLERANCE:
                defensive_warn = "hierarchy_always_defensive"
        if defensive_warn:
            warnings.append(defensive_warn)
        _record_check(
            report,
            "high_vol",
            "hierarchy_sanity",
            status if status == "fail" else "warn" if reason or defensive_warn else "pass",
            {
                "total_steps": total_steps,
                "transition_total": transition_total,
                "transition_limit": limit,
                "defensive_fraction": round(defensive_fraction, 10) if total_steps > 0 else None,
            },
        )


def _record_check(
    report: Dict[str, Dict[str, Any]],
    bucket: str,
    check_id: str,
    status: str,
    payload: Mapping[str, Any],
) -> None:
    entry = report[bucket]
    entry["checks"][check_id] = {"status": status, **payload}
    if status == "fail":
        entry["passed"] = False


def _regime_metric(
    regime_deltas: Mapping[str, Mapping[str, Mapping[str, float | None]]],
    bucket: str,
    metric: str,
) -> tuple[float | None, float | None, Mapping[str, float | None] | None]:
    bucket_entry = regime_deltas.get(bucket)
    if not isinstance(bucket_entry, Mapping):
        return None, None, None
    metric_entry = bucket_entry.get(metric)
    if not isinstance(metric_entry, Mapping):
        return None, None, None
    candidate = metric_entry.get("candidate")
    baseline = metric_entry.get("baseline")
    return candidate, baseline, metric_entry


def _extract_metric(payload: Mapping[str, Any], path: Sequence[str]) -> float | None:
    node: Any = payload
    for key in path:
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return _as_float(node)


def _copy_regime_deltas(
    regime_deltas: Mapping[str, Mapping[str, Mapping[str, float | None]]],
) -> Dict[str, Dict[str, Dict[str, float | None]]]:
    copied: Dict[str, Dict[str, Dict[str, float | None]]] = {}
    for bucket, metrics in regime_deltas.items():
        copied[bucket] = {}
        for metric, values in metrics.items():
            copied[bucket][metric] = dict(values)
    return copied


def _execution_bucket_metrics(
    execution_metrics: Mapping[str, Any] | None,
    bucket: str,
) -> Dict[str, float | None]:
    if not isinstance(execution_metrics, Mapping):
        return {}
    regime = execution_metrics.get("regime")
    if not isinstance(regime, Mapping):
        return {}
    bucket_entry = regime.get(bucket)
    if not isinstance(bucket_entry, Mapping):
        return {}
    result: Dict[str, float | None] = {}
    for key in ("avg_slippage_bps", "p95_slippage_bps", "reject_rate", "fill_rate"):
        result[key] = _as_float(bucket_entry.get(key))
    return result


def _normalize_mapping(values: Mapping[str, Any] | None) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    if not isinstance(values, Mapping):
        return normalized
    for key, value in values.items():
        number = _as_float(value)
        if number is None:
            continue
        normalized[str(key)] = float(number)
    return normalized


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return float(number)


__all__ = ["RegimeQualificationCriteria", "RegimeQualificationEvaluation"]
