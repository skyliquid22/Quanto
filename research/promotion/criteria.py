"""Deterministic qualification criteria for experiment promotion."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from research.experiments.ablation import DEFAULT_SWEEP_ROOT
from research.experiments.comparator import compare_experiments
from research.experiments.registry import ExperimentRecord, ExperimentRegistry
from research.experiments.regression import RegressionGateRule, default_gate_rules, evaluate_gates

_SANITY_TURNOVER_METRIC = "trading.turnover_1d_mean"


@dataclass(frozen=True)
class SweepRobustnessCriteria:
    """Optional sweep-level quality thresholds."""

    min_completed: int | None = None
    max_failures: int | None = None
    severity: str = "soft"

    def __post_init__(self) -> None:
        normalized = self.severity.lower()
        if normalized not in {"hard", "soft"}:
            raise ValueError("severity must be 'hard' or 'soft'.")
        object.__setattr__(self, "severity", normalized)
        if self.min_completed is not None and self.min_completed < 0:
            raise ValueError("min_completed must be non-negative when provided.")
        if self.max_failures is not None and self.max_failures < 0:
            raise ValueError("max_failures must be non-negative when provided.")

    def evaluate(self, summary: Mapping[str, Any]) -> str | None:
        metrics_data = summary.get("metrics") or {}
        regression_data = summary.get("regression") or {}
        if self.min_completed is not None:
            completed = _coerce_int(metrics_data.get("completed"))
            if completed < self.min_completed:
                return f"sweep_completed_below_threshold:{completed}<{self.min_completed}"
        if self.max_failures is not None:
            failures = _coerce_int(regression_data.get("fails"))
            if failures > self.max_failures:
                return f"sweep_failures_exceeded:{failures}>{self.max_failures}"
        return None


@dataclass(frozen=True)
class RegimeDiagnosticsCriteria:
    """Optional regime/hierarchy diagnostics thresholds."""

    min_modes: int | None = None
    min_mode_fraction: float | None = None
    severity: str = "soft"

    def __post_init__(self) -> None:
        normalized = self.severity.lower()
        if normalized not in {"hard", "soft"}:
            raise ValueError("severity must be 'hard' or 'soft'.")
        object.__setattr__(self, "severity", normalized)
        if self.min_modes is not None and self.min_modes < 0:
            raise ValueError("min_modes must be non-negative when provided.")
        if self.min_mode_fraction is not None:
            if not (0.0 <= self.min_mode_fraction <= 1.0):
                raise ValueError("min_mode_fraction must be within [0, 1].")

    def evaluate(self, metrics_payload: Mapping[str, Any]) -> str | None:
        trading = metrics_payload.get("trading")
        if not isinstance(trading, Mapping):
            if self.min_modes or self.min_mode_fraction is not None:
                return "regime_trading_metrics_missing"
            return None
        mode_counts = trading.get("mode_counts")
        if not isinstance(mode_counts, Mapping) or not mode_counts:
            if self.min_modes or self.min_mode_fraction is not None:
                return "regime_mode_counts_missing"
            return None
        normalized_counts: Dict[str, float] = {}
        for mode, raw_value in mode_counts.items():
            value = _as_float(raw_value)
            if value is None:
                continue
            normalized_counts[str(mode)] = max(0.0, value)
        total = sum(normalized_counts.values())
        active_modes = sum(1 for value in normalized_counts.values() if value > 0)
        if self.min_modes is not None and active_modes < self.min_modes:
            return f"regime_active_modes_below_threshold:{active_modes}<{self.min_modes}"
        if self.min_mode_fraction is not None:
            if total <= 0:
                return "regime_mode_counts_empty"
            for mode, count in sorted(normalized_counts.items()):
                fraction = count / total if total else 0.0
                if fraction < self.min_mode_fraction:
                    return (
                        f"regime_mode_fraction_below_threshold:{mode}:{_round_float(fraction)}"
                        f"<{self.min_mode_fraction}"
                    )
        return None


@dataclass(frozen=True)
class QualificationEvaluation:
    """Final decision bundle returned by QualificationCriteria."""

    passed: bool
    failed_hard: List[str]
    failed_soft: List[str]
    metrics_snapshot: Mapping[str, Any]
    gate_summary: Dict[str, Any]
    sweep_summary: Mapping[str, Any] | None


@dataclass(frozen=True)
class QualificationCriteria:
    """Deterministic qualification requirements."""

    max_drawdown: float = 1.0
    max_turnover: float = 1.0
    turnover_metric: str = _SANITY_TURNOVER_METRIC
    required_artifacts: Sequence[str] = (
        "spec/experiment_spec.json",
        "evaluation/metrics.json",
        "logs/run_summary.json",
    )
    sweep_root: Path | None = None
    sweep_robustness: SweepRobustnessCriteria | None = None
    regime_diagnostics: RegimeDiagnosticsCriteria | None = None

    def evaluate(
        self,
        candidate_record: ExperimentRecord,
        baseline_record: ExperimentRecord,
        *,
        registry: ExperimentRegistry | None = None,
        gate_rules: Sequence[RegressionGateRule] | None = None,
        sweep_name: str | None = None,
    ) -> QualificationEvaluation:
        registry = registry or ExperimentRegistry()
        comparison = compare_experiments(
            candidate_record.experiment_id,
            baseline_record.experiment_id,
            registry=registry,
        )
        resolved_rules = list(gate_rules) if gate_rules else default_gate_rules()
        gate_report = evaluate_gates(comparison, resolved_rules)
        metrics_payload = _load_metrics(candidate_record.metrics_path)
        failed_hard: List[str] = []
        failed_soft: List[str] = []

        if gate_report.overall_status == "fail":
            failed_hard.append("regression_gates_failed")
        elif gate_report.soft_warnings:
            failed_soft.append("regression_gates_soft_warnings")

        constraint_reason = self._constraint_reason(metrics_payload)
        if constraint_reason:
            failed_hard.append(constraint_reason)

        for reason in self._sanity_reasons(metrics_payload):
            failed_hard.append(reason)

        missing_artifacts = self._missing_artifacts(candidate_record)
        if missing_artifacts:
            missing = ",".join(sorted(missing_artifacts))
            failed_hard.append(f"missing_artifacts:{missing}")

        sweep_summary: Mapping[str, Any] | None = None
        sweep_name = sweep_name.strip() if sweep_name else None
        if sweep_name:
            sweep_summary = self._load_sweep_summary(sweep_name)
            if sweep_summary is None:
                failed_soft.append(f"sweep_summary_missing:{sweep_name}")
            elif self.sweep_robustness:
                sweep_reason = self.sweep_robustness.evaluate(sweep_summary)
                if sweep_reason:
                    self._route_failure(sweep_reason, self.sweep_robustness.severity, failed_hard, failed_soft)
        elif self.sweep_robustness:
            failed_soft.append("sweep_summary_missing")

        if self.regime_diagnostics:
            regime_reason = self.regime_diagnostics.evaluate(metrics_payload)
            if regime_reason:
                self._route_failure(regime_reason, self.regime_diagnostics.severity, failed_hard, failed_soft)

        passed = not failed_hard
        return QualificationEvaluation(
            passed=passed,
            failed_hard=failed_hard,
            failed_soft=failed_soft,
            metrics_snapshot=metrics_payload,
            gate_summary=gate_report.to_dict(),
            sweep_summary=sweep_summary,
        )

    def _constraint_reason(self, metrics_payload: Mapping[str, Any]) -> str | None:
        safety = metrics_payload.get("safety")
        if not isinstance(safety, Mapping):
            return "safety_metrics_missing"
        keys = (
            "constraint_violations_count",
            "max_weight_violation_count",
            "exposure_violation_count",
            "turnover_violation_count",
        )
        for key in keys:
            value = _as_float(safety.get(key))
            if value is None:
                return f"missing_metric:safety.{key}"
            if value > 0.0:
                return f"constraint_violation:{key}={_round_float(value)}"
        return None

    def _sanity_reasons(self, metrics_payload: Mapping[str, Any]) -> List[str]:
        reasons: List[str] = []
        safety = metrics_payload.get("safety")
        performance = metrics_payload.get("performance")
        nan_violations = _as_float(safety.get("nan_inf_violations")) if isinstance(safety, Mapping) else None
        if nan_violations is None:
            reasons.append("missing_metric:safety.nan_inf_violations")
        elif nan_violations > 0:
            reasons.append("sanity:nan_inf_detected")
        drawdown = _as_float(performance.get("max_drawdown")) if isinstance(performance, Mapping) else None
        if drawdown is None:
            reasons.append("missing_metric:performance.max_drawdown")
        elif drawdown > self.max_drawdown + 1e-9:
            reasons.append(f"sanity:max_drawdown_exceeded:{_round_float(drawdown)}")
        turnover = _extract_metric(metrics_payload, self.turnover_metric)
        if turnover is None:
            reasons.append(f"missing_metric:{self.turnover_metric}")
        elif turnover > self.max_turnover + 1e-9:
            reasons.append(f"sanity:turnover_exceeded:{_round_float(turnover)}")
        return reasons

    def _missing_artifacts(self, record: ExperimentRecord) -> List[str]:
        missing: List[str] = []
        for rel_path in self.required_artifacts:
            candidate = record.root / rel_path
            if not candidate.exists():
                missing.append(rel_path)
        return missing

    def _route_failure(
        self,
        reason: str,
        severity: str,
        failed_hard: List[str],
        failed_soft: List[str],
    ) -> None:
        if severity == "hard":
            failed_hard.append(reason)
        else:
            failed_soft.append(reason)

    def _load_sweep_summary(self, sweep_name: str) -> Mapping[str, Any] | None:
        root = Path(self.sweep_root) if self.sweep_root else DEFAULT_SWEEP_ROOT
        artifact_dir = root / sweep_name
        metrics_path = artifact_dir / "aggregate_metrics.json"
        regression_path = artifact_dir / "regression_summary.json"
        if not metrics_path.exists() and not regression_path.exists():
            return None
        summary: Dict[str, Any] = {
            "sweep_name": sweep_name,
            "artifact_dir": str(artifact_dir),
        }
        if metrics_path.exists():
            summary["metrics_path"] = str(metrics_path)
            summary["metrics"] = _read_optional_json(metrics_path)
        if regression_path.exists():
            summary["regression_path"] = str(regression_path)
            summary["regression"] = _read_optional_json(regression_path)
        return summary


def _load_metrics(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unable to parse metrics.json at {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"metrics.json at {path} must contain a mapping.")
    return payload


def _extract_metric(payload: Mapping[str, Any], metric_id: str) -> float | None:
    parts = metric_id.split(".")
    node: Any = payload
    for part in parts:
        if not isinstance(node, Mapping):
            return None
        node = node.get(part)
    return _as_float(node)


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


def _coerce_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _round_float(value: float) -> float:
    rounded = round(float(value), 10)
    if rounded == -0.0:
        return 0.0
    return rounded


def _read_optional_json(path: Path) -> Mapping[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, Mapping):
        return {}
    return data


__all__ = [
    "QualificationCriteria",
    "QualificationEvaluation",
    "RegimeDiagnosticsCriteria",
    "SweepRobustnessCriteria",
]
