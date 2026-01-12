"""Execution qualification gates for T26R."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

from research.experiments.comparator import ComparisonResult, MetricComparison

_TOLERANCE = 1e-9
_SUMMARY_KEYS = (
    "avg_slippage_bps",
    "execution_halts",
    "fill_rate",
    "p95_slippage_bps",
    "partial_fill_rate",
    "reject_rate",
    "total_fees",
    "turnover_realized",
)
_SUMMARY_KEYS = (
    "avg_slippage_bps",
    "execution_halts",
    "fill_rate",
    "p95_slippage_bps",
    "partial_fill_rate",
    "reject_rate",
    "total_fees",
    "turnover_realized",
)


@dataclass(frozen=True)
class ExecutionGateEvaluation:
    """Structured output for execution gate checks."""

    hard_failures: List[str]
    soft_failures: List[str]
    report: Dict[str, Any]


@dataclass(frozen=True)
class ExecutionQualificationCriteria:
    """Configuration for execution gate enforcement."""

    min_fill_rate: float = 0.99
    max_reject_rate: float = 0.01
    max_p95_slippage_bps: float = 25.0
    soft_max_avg_slippage_delta_bps: float = 5.0
    soft_max_fees_delta_pct: float = 10.0
    soft_max_turnover_delta_pct: float = 10.0

    def evaluate(
        self,
        comparison: ComparisonResult,
        candidate_metrics: Mapping[str, Any],
        baseline_metrics: Mapping[str, Any],
        *,
        skip_delta_checks: bool = False,
    ) -> ExecutionGateEvaluation:
        candidate_execution = _execution_section(candidate_metrics)
        baseline_execution = _execution_section(baseline_metrics)
        if candidate_execution is None:
            report = {
                "candidate": {},
                "baseline": baseline_execution or {},
                "comparison": comparison.execution_metrics,
                "gates": [],
            }
            return ExecutionGateEvaluation(
                hard_failures=["execution_metrics_missing"],
                soft_failures=[],
                report=report,
            )

        summary_candidate = candidate_execution.get("summary")
        summary_baseline = baseline_execution.get("summary") if isinstance(baseline_execution, Mapping) else None
        gates: List[Dict[str, Any]] = []
        hard_failures: List[str] = []
        soft_failures: List[str] = []

        self._enforce_hard_gate(
            summary=summary_candidate,
            baseline=summary_baseline,
            gate_id="execution_halts_zero",
            metric="execution.summary.execution_halts",
            predicate=lambda value: value <= _TOLERANCE,
            threshold=0.0,
            severity_failures=hard_failures,
            gates=gates,
            failure_reason="execution_halts_detected",
        )
        self._enforce_hard_gate(
            summary=summary_candidate,
            baseline=summary_baseline,
            gate_id="execution_reject_rate_cap",
            metric="execution.summary.reject_rate",
            predicate=lambda value: value <= self.max_reject_rate + _TOLERANCE,
            threshold=self.max_reject_rate,
            severity_failures=hard_failures,
            gates=gates,
            failure_reason="execution_reject_rate_exceeded",
        )
        self._enforce_hard_gate(
            summary=summary_candidate,
            baseline=summary_baseline,
            gate_id="execution_fill_rate_floor",
            metric="execution.summary.fill_rate",
            predicate=lambda value: value >= self.min_fill_rate - _TOLERANCE,
            threshold=self.min_fill_rate,
            severity_failures=hard_failures,
            gates=gates,
            failure_reason="execution_fill_rate_below_minimum",
        )
        self._enforce_hard_gate(
            summary=summary_candidate,
            baseline=summary_baseline,
            gate_id="execution_p95_slippage_cap",
            metric="execution.summary.p95_slippage_bps",
            predicate=lambda value: abs(value) <= self.max_p95_slippage_bps + _TOLERANCE,
            threshold=self.max_p95_slippage_bps,
            severity_failures=hard_failures,
            gates=gates,
            failure_reason="execution_slippage_exceeded",
        )

        summary_block = _build_execution_summary_block(summary_candidate, summary_baseline)
        if skip_delta_checks:
            _record_delta_skip(gates, "execution_avg_slippage_delta", "execution.summary.avg_slippage_bps")
            _record_delta_skip(gates, "execution_fee_drift", "execution.summary.total_fees")
            _record_delta_skip(gates, "execution_turnover_drift", "execution.summary.turnover_realized")
        else:
            self._enforce_soft_delta_gate(
                comparison_summary=summary_block,
                gate_id="execution_avg_slippage_delta",
                metric="avg_slippage_bps",
                threshold=self.soft_max_avg_slippage_delta_bps,
                severity_failures=soft_failures,
                gates=gates,
                direction="absolute",
            )
            self._enforce_soft_delta_gate(
                comparison_summary=summary_block,
                gate_id="execution_fee_drift",
                metric="total_fees",
                threshold=self.soft_max_fees_delta_pct,
                severity_failures=soft_failures,
                gates=gates,
                direction="percent",
            )
            self._enforce_soft_delta_gate(
                comparison_summary=summary_block,
                gate_id="execution_turnover_drift",
                metric="turnover_realized",
                threshold=self.soft_max_turnover_delta_pct,
                severity_failures=soft_failures,
                gates=gates,
                direction="percent",
            )
            self._evaluate_sharpe_correlation(
                comparison=comparison,
                severity_failures=soft_failures,
                gates=gates,
            )

        comparison_report = dict(comparison.execution_metrics or {})
        comparison_report["summary"] = summary_block
        comparison_report["regime"] = _build_execution_regime_block(
            candidate_execution.get("regime") if isinstance(candidate_execution, Mapping) else None,
            baseline_execution.get("regime") if isinstance(baseline_execution, Mapping) else None,
        )
        report = {
            "candidate": candidate_execution,
            "baseline": baseline_execution or {},
            "comparison": comparison_report,
            "gates": gates,
        }
        return ExecutionGateEvaluation(
            hard_failures=hard_failures,
            soft_failures=soft_failures,
            report=report,
        )

    def _enforce_hard_gate(
        self,
        *,
        summary: Mapping[str, Any] | None,
        baseline: Mapping[str, Any] | None,
        gate_id: str,
        metric: str,
        predicate,
        threshold: float,
        severity_failures: List[str],
        gates: List[Dict[str, Any]],
        failure_reason: str,
    ) -> None:
        observed = _as_float(summary.get(metric.split(".")[-1])) if isinstance(summary, Mapping) else None
        baseline_value = _as_float(baseline.get(metric.split(".")[-1])) if isinstance(baseline, Mapping) else None
        status = "pass"
        message = "within threshold"
        if observed is None:
            status = "fail"
            message = "candidate_metric_missing"
            severity_failures.append(f"missing_execution_metric:{metric}")
        elif predicate(observed):
            status = "pass"
        else:
            status = "fail"
            message = failure_reason
            severity_failures.append(failure_reason)
        gates.append(
            {
                "gate_id": gate_id,
                "severity": "hard",
                "status": status,
                "metric": metric,
                "observed": observed,
                "baseline": baseline_value,
                "threshold": threshold,
                "message": message,
            }
        )

    def _enforce_soft_delta_gate(
        self,
        *,
        comparison_summary: Mapping[str, Any] | None,
        gate_id: str,
        metric: str,
        threshold: float,
        severity_failures: List[str],
        gates: List[Dict[str, Any]],
        direction: str,
        skip: bool = False,
    ) -> None:
        entry = comparison_summary.get(metric) if isinstance(comparison_summary, Mapping) else None
        candidate_value = _as_float(entry.get("candidate")) if isinstance(entry, Mapping) else None
        baseline_value = _as_float(entry.get("baseline")) if isinstance(entry, Mapping) else None
        delta = _as_float(entry.get("delta")) if isinstance(entry, Mapping) else None
        delta_pct = _as_float(entry.get("delta_pct")) if isinstance(entry, Mapping) else None
        observed = candidate_value
        status = "pass"
        message = "within threshold"
        metric_id = f"execution.summary.{metric}"

        if skip:
            status = "skip"
            message = "phase1_delta_skip"
        elif entry is None:
            status = "skip"
            message = "comparison_missing"
        elif baseline_value is None:
            status = "skip"
            message = "baseline_missing"
        elif candidate_value is None:
            status = "skip"
            message = "candidate_missing"
        else:
            value = delta if direction == "absolute" else delta_pct
            if value is None:
                status = "skip"
                message = "insufficient_baseline"
            else:
                if value > threshold + _TOLERANCE:
                    status = "warn"
                    message = f"{metric}_regressed"
                    severity_failures.append(f"{metric_id}_regressed")
        gates.append(
            {
                "gate_id": gate_id,
                "severity": "soft",
                "status": status,
                "metric": metric_id,
                "observed": observed,
                "baseline": baseline_value,
                "threshold": threshold,
                "delta": delta,
                "delta_pct": delta_pct,
                "message": message,
            }
        )

    def _evaluate_sharpe_correlation(
        self,
        *,
        comparison: ComparisonResult,
        severity_failures: List[str],
        gates: List[Dict[str, Any]],
    ) -> None:
        sharpe_entry = _metric_entry(comparison.metrics, "performance.sharpe")
        slippage_entry = None
        if comparison.execution_metrics:
            summary = comparison.execution_metrics.get("summary")
            if isinstance(summary, Mapping):
                slippage_entry = summary.get("avg_slippage_bps")
        slippage_delta = _as_float(slippage_entry.get("delta")) if isinstance(slippage_entry, Mapping) else None
        status = "pass"
        message = "within threshold"
        observed = None
        baseline_value = None
        delta = None
        if sharpe_entry and not sharpe_entry.reason and sharpe_entry.delta is not None:
            observed = sharpe_entry.candidate_value
            baseline_value = sharpe_entry.baseline_value
            delta = sharpe_entry.delta
            if sharpe_entry.delta < -_TOLERANCE:
                if slippage_delta is not None and slippage_delta > _TOLERANCE:
                    status = "warn"
                    message = "execution_sharpe_correlated_with_slippage"
                    severity_failures.append(message)
        gates.append(
            {
                "gate_id": "execution_sharpe_correlation",
                "severity": "soft",
                "status": status,
                "metric": "performance.sharpe",
                "observed": observed,
                "baseline": baseline_value,
                "delta": delta,
                "message": message,
            }
        )


def _execution_section(metrics_payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    section = metrics_payload.get("execution")
    if isinstance(section, Mapping):
        return section
    return None


def _build_execution_summary_block(
    candidate_summary: Mapping[str, Any] | None,
    baseline_summary: Mapping[str, Any] | None,
) -> Dict[str, Dict[str, float | None]]:
    block: Dict[str, Dict[str, float | None]] = {}
    for key in _SUMMARY_KEYS:
        candidate_value = _as_float(candidate_summary.get(key)) if isinstance(candidate_summary, Mapping) else None
        baseline_value = _as_float(baseline_summary.get(key)) if isinstance(baseline_summary, Mapping) else None
        delta = None
        delta_pct = None
        if candidate_value is not None and baseline_value is not None:
            delta = candidate_value - baseline_value
            if abs(baseline_value) > _TOLERANCE:
                delta_pct = (delta / baseline_value) * 100.0
        block[key] = {
            "candidate": candidate_value,
            "baseline": baseline_value,
            "delta": delta,
            "delta_pct": delta_pct,
        }
    return block


def _build_execution_regime_block(
    candidate_regime: Mapping[str, Any] | None,
    baseline_regime: Mapping[str, Any] | None,
) -> Dict[str, Dict[str, Dict[str, float | None]]]:
    block: Dict[str, Dict[str, Dict[str, float | None]]] = {}
    buckets = set()
    if isinstance(candidate_regime, Mapping):
        buckets.update(candidate_regime.keys())
    if isinstance(baseline_regime, Mapping):
        buckets.update(baseline_regime.keys())
    for bucket in sorted(buckets):
        cand_entry = candidate_regime.get(bucket) if isinstance(candidate_regime, Mapping) else None
        base_entry = baseline_regime.get(bucket) if isinstance(baseline_regime, Mapping) else None
        metrics = set()
        if isinstance(cand_entry, Mapping):
            metrics.update(cand_entry.keys())
        if isinstance(base_entry, Mapping):
            metrics.update(base_entry.keys())
        bucket_block: Dict[str, Dict[str, float | None]] = {}
        for metric in sorted(metrics):
            cand_value = _as_float(cand_entry.get(metric)) if isinstance(cand_entry, Mapping) else None
            base_value = _as_float(base_entry.get(metric)) if isinstance(base_entry, Mapping) else None
            delta = None
            delta_pct = None
            if cand_value is not None and base_value is not None:
                delta = cand_value - base_value
                if abs(base_value) > _TOLERANCE:
                    delta_pct = (delta / base_value) * 100.0
            bucket_block[metric] = {
                "candidate": cand_value,
                "baseline": base_value,
                "delta": delta,
                "delta_pct": delta_pct,
            }
        block[bucket] = bucket_block
    return block


def _record_delta_skip(gates: List[Dict[str, Any]], gate_id: str, metric: str) -> None:
    gates.append(
        {
            "gate_id": gate_id,
            "severity": "soft",
            "status": "skip",
            "metric": metric,
            "observed": None,
            "baseline": None,
            "threshold": None,
            "delta": None,
            "delta_pct": None,
            "message": "phase1_delta_skip",
        }
    )


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


def _metric_entry(entries: List[MetricComparison], metric_id: str) -> MetricComparison | None:
    for entry in entries:
        if entry.metric_id == metric_id:
            return entry
    return None


__all__ = ["ExecutionQualificationCriteria", "ExecutionGateEvaluation"]
