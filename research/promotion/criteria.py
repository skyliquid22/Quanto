"""Deterministic qualification criteria for experiment promotion."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import subprocess
import sys
import hashlib

from infra.paths import get_data_root
from research.experiments.ablation import DEFAULT_SWEEP_ROOT
from research.experiments.comparator import compare_experiments
from research.experiments.registry import ExperimentRecord, ExperimentRegistry
from research.experiments.regression import RegressionGateRule, default_gate_rules, evaluate_gates
from research.experiments.spec import ExperimentSpec
from research.promotion.execution_criteria import ExecutionQualificationCriteria
from research.promotion.execution_metrics_locator import (
    ExecutionMetricsLocatorResult,
    locate_execution_metrics,
)
from research.promotion.regime_criteria import RegimeQualificationCriteria

_SANITY_TURNOVER_METRIC = "trading.turnover_1d_mean"
_PHASE1_WARNING = "Baseline equals candidate; delta-based gates skipped (Phase 1 mode)."
_REGIME_TOLERANCE = 1e-9


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
    regime_report: Mapping[str, Any] | None = None
    execution_report: Mapping[str, Any] | None = None
    execution_resolution: Mapping[str, Any] | None = None


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
    regime_qualification: RegimeQualificationCriteria | None = None
    execution_qualification: ExecutionQualificationCriteria | None = field(default_factory=ExecutionQualificationCriteria)

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
        same_identifier = candidate_record.experiment_id == baseline_record.experiment_id
        resolved_rules = list(gate_rules) if gate_rules else default_gate_rules()
        gate_report = evaluate_gates(comparison, resolved_rules)
        registry_root = registry.root
        shadow_root = registry_root.parent / "shadow"
        metrics_payload = _load_metrics(candidate_record.metrics_path)
        baseline_metrics = _load_metrics(baseline_record.metrics_path)
        failed_hard: List[str] = []
        failed_soft: List[str] = []
        execution_resolution: Dict[str, Any] = {}
        deterministic_present = self._has_deterministic_replay(metrics_payload)

        if gate_report.overall_status == "fail":
            failed_hard.append("regression_gates_failed")
        elif gate_report.soft_warnings:
            failed_soft.append("regression_gates_soft_warnings")

        constraint_reason = self._constraint_reason(metrics_payload)
        if constraint_reason:
            failed_hard.append(constraint_reason)

        for reason in self._sanity_reasons(metrics_payload):
            failed_hard.append(reason)

        for reason in self._regime_high_vol_hard_failures(metrics_payload, baseline_metrics):
            failed_hard.append(reason)

        sharpe_reason = self._sharpe_improvement_reason(metrics_payload, baseline_metrics)
        if sharpe_reason:
            failed_soft.append(sharpe_reason)

        for reason in self._stability_soft_failures(metrics_payload, baseline_metrics):
            failed_soft.append(reason)

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

        candidate_resolution = locate_execution_metrics(
            candidate_record.experiment_id,
            registry_root=registry_root,
            shadow_root=shadow_root,
        )
        candidate_info = candidate_resolution.to_dict()
        if not candidate_resolution.found:
            replay_info = self._attempt_shadow_replay(
                candidate_record,
                registry_root=registry_root,
                label="candidate",
            )
            candidate_info["qualification_replay"] = replay_info
            if replay_info.get("returncode") == 0:
                candidate_resolution = locate_execution_metrics(
                    candidate_record.experiment_id,
                    registry_root=registry_root,
                    shadow_root=shadow_root,
                )
                candidate_info = candidate_resolution.to_dict()
                candidate_info["qualification_replay"] = replay_info
        execution_resolution["candidate"] = candidate_info
        if candidate_info.get("found"):
            metrics_payload = _attach_execution_section(
                metrics_payload,
                candidate_resolution,
                persist=True,
            )
        else:
            failed_hard.append("execution_metrics_missing")

        if (
            not deterministic_present
            and candidate_info.get("found")
            and _has_shadow_deterministic_artifacts(candidate_resolution)
        ):
            deterministic_present = True

        baseline_resolution = locate_execution_metrics(
            baseline_record.experiment_id,
            registry_root=registry_root,
            shadow_root=shadow_root,
        )
        baseline_info = baseline_resolution.to_dict()
        if not baseline_resolution.found:
            replay_info = self._attempt_shadow_replay(
                baseline_record,
                registry_root=registry_root,
                label="baseline",
            )
            baseline_info["qualification_replay"] = replay_info
            if replay_info.get("returncode") == 0:
                baseline_resolution = locate_execution_metrics(
                    baseline_record.experiment_id,
                    registry_root=registry_root,
                    shadow_root=shadow_root,
                )
                baseline_info = baseline_resolution.to_dict()
                baseline_info["qualification_replay"] = replay_info
        execution_resolution["baseline"] = baseline_info
        if baseline_info.get("found"):
            baseline_metrics = _attach_execution_section(
                baseline_metrics,
                baseline_resolution,
                persist=True,
            )
        else:
            failed_soft.append("execution_baseline_metrics_missing")

        if not deterministic_present:
            failed_hard.append("deterministic_replay_missing")

        comparison = compare_experiments(
            candidate_record.experiment_id,
            baseline_record.experiment_id,
            registry=registry,
        )

        regime_report: Mapping[str, Any] | None = None
        if self.regime_qualification is not None:
            hierarchy_enabled = self._hierarchy_enabled(candidate_record)
            regime_eval = self.regime_qualification.evaluate(
                comparison,
                metrics_payload,
                baseline_metrics,
                hierarchy_enabled=hierarchy_enabled,
                execution_metrics=metrics_payload.get("execution"),
            )
            regime_report = regime_eval.report
            failed_hard.extend(regime_eval.hard_failures)
            failed_soft.extend(regime_eval.soft_failures)

        execution_report: Mapping[str, Any] | None = None
        if self.execution_qualification is not None:
            skip_delta_checks = same_identifier or not baseline_resolution.found
            execution_eval = self.execution_qualification.evaluate(
                comparison,
                metrics_payload,
                baseline_metrics,
                skip_delta_checks=skip_delta_checks,
            )
            execution_report = execution_eval.report
            failed_hard.extend(execution_eval.hard_failures)
            failed_soft.extend(execution_eval.soft_failures)
            if same_identifier:
                failed_soft.append(_PHASE1_WARNING)
        else:
            execution_report = None

        passed = not failed_hard
        return QualificationEvaluation(
            passed=passed,
            failed_hard=failed_hard,
            failed_soft=failed_soft,
            metrics_snapshot=metrics_payload,
            gate_summary=gate_report.to_dict(),
            sweep_summary=sweep_summary,
            regime_report=regime_report,
            execution_report=execution_report,
            execution_resolution=execution_resolution,
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

    def _has_deterministic_replay(self, metrics_payload: Mapping[str, Any]) -> bool:
        returns = metrics_payload.get("returns")
        if not isinstance(returns, Sequence):
            return False
        normalized = []
        for value in returns:
            number = _as_float(value)
            if number is None:
                return False
            normalized.append(number)
        return bool(normalized)

    def _sharpe_improvement_reason(
        self,
        candidate_metrics: Mapping[str, Any],
        baseline_metrics: Mapping[str, Any],
    ) -> str | None:
        candidate = _extract_metric(candidate_metrics, "performance.sharpe")
        baseline = _extract_metric(baseline_metrics, "performance.sharpe")
        if candidate is None or baseline is None:
            return "soft_gate:sharpe_missing"
        if candidate + 1e-9 < baseline:
            delta = _round_float(candidate - baseline)
            return f"soft_gate:sharpe_not_improved:{delta}"
        return None

    def _stability_soft_failures(
        self,
        candidate_metrics: Mapping[str, Any],
        baseline_metrics: Mapping[str, Any],
    ) -> List[str]:
        reasons: List[str] = []
        candidate_stability = candidate_metrics.get("stability")
        baseline_stability = baseline_metrics.get("stability")
        if not isinstance(candidate_stability, Mapping) or not isinstance(baseline_stability, Mapping):
            return reasons
        keys = ("turnover_std", "mode_churn_rate", "cost_curve_span")
        for key in keys:
            candidate_value = _as_float(candidate_stability.get(key))
            baseline_value = _as_float(baseline_stability.get(key))
            if candidate_value is None or baseline_value is None:
                continue
            if candidate_value > baseline_value + 1e-9:
                reasons.append(f"stability_regressed:{key}")
        return reasons

    def _regime_high_vol_hard_failures(
        self,
        candidate_metrics: Mapping[str, Any],
        baseline_metrics: Mapping[str, Any],
    ) -> List[str]:
        reasons: List[str] = []
        candidate_entry = _regime_entry(candidate_metrics, "high_vol")
        baseline_entry = _regime_entry(baseline_metrics, "high_vol")
        if candidate_entry is None or baseline_entry is None:
            reasons.append("regime_high_vol_metrics_missing")
            return reasons
        cand_draw = _as_float(candidate_entry.get("max_drawdown"))
        base_draw = _as_float(baseline_entry.get("max_drawdown"))
        if cand_draw is None or base_draw is None:
            reasons.append("regime_high_vol_metrics_missing")
        elif cand_draw > (base_draw + _REGIME_TOLERANCE):
            reasons.append("regime_high_vol_drawdown_regressed")
        cand_exposure = _as_float(candidate_entry.get("avg_exposure"))
        base_exposure = _as_float(baseline_entry.get("avg_exposure"))
        if cand_exposure is None or base_exposure is None:
            if "regime_high_vol_metrics_missing" not in reasons:
                reasons.append("regime_high_vol_metrics_missing")
        elif cand_exposure > (base_exposure + _REGIME_TOLERANCE):
            reasons.append("regime_high_vol_exposure_increased")
        return reasons

    def _attempt_shadow_replay(
        self,
        record: ExperimentRecord,
        *,
        registry_root: Path,
        label: str,
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {"invoked": False}
        try:
            spec = ExperimentSpec.from_file(record.spec_path)
        except Exception as exc:  # pragma: no cover - defensive
            info["reason"] = f"spec_error:{exc}"
            return info
        start = getattr(spec, "start_date", None)
        end = getattr(spec, "end_date", None)
        if start is None or end is None:
            info["reason"] = "spec_missing_dates"
            return info
        script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_shadow.py"
        window_start = start.isoformat()
        window_end = end.isoformat()
        cmd = [
            sys.executable,
            str(script_path),
            "--experiment-id",
            record.experiment_id,
            "--replay",
            "--start-date",
            window_start,
            "--end-date",
            window_end,
            "--registry-root",
            str(registry_root),
            "--qualification-replay",
            "--qualification-reason",
            f"qualification_auto_{label}",
            "--execution-mode",
            "sim",
        ]
        run_dir = _shadow_run_dir(record.experiment_id, window_start, window_end)
        if run_dir.exists():
            cmd.append("--resume")
        result = subprocess.run(cmd, capture_output=True, text=True)
        info.update(
            {
                "invoked": True,
                "command": cmd,
                "returncode": int(result.returncode),
                "stdout_tail": result.stdout[-2000:],
                "stderr_tail": result.stderr[-2000:],
            }
        )
        return info

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

    def _hierarchy_enabled(self, record: ExperimentRecord) -> bool:
        try:
            payload = json.loads(record.spec_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        if not isinstance(payload, Mapping):
            return False
        return bool(payload.get("hierarchy_enabled"))


def _load_metrics(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unable to parse metrics.json at {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"metrics.json at {path} must contain a mapping.")
    return payload


def _shadow_run_dir(experiment_id: str, start: str, end: str) -> Path:
    run_id = _derive_shadow_run_id(experiment_id, start, end)
    return get_data_root() / "shadow" / experiment_id / run_id


def _derive_shadow_run_id(experiment_id: str, start: str, end: str) -> str:
    payload = {
        "experiment_id": experiment_id,
        "window_start": start,
        "window_end": end,
        "mode": "replay",
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"replay_{digest[:12]}"


def _has_shadow_deterministic_artifacts(locator_result: ExecutionMetricsLocatorResult) -> bool:
    if locator_result.source != "shadow":
        return False
    exec_path = locator_result.execution_metrics_path
    if exec_path is None:
        return False
    run_dir = exec_path.parent
    state_path = run_dir / "state.json"
    summary_path = run_dir / "summary.json"
    return state_path.exists() and summary_path.exists()


def _regime_entry(metrics_payload: Mapping[str, Any], bucket: str) -> Mapping[str, Any] | None:
    section = metrics_payload.get("performance_by_regime")
    if not isinstance(section, Mapping):
        return None
    entry = section.get(bucket)
    if not isinstance(entry, Mapping):
        return None
    return entry


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


def _attach_execution_section(
    metrics_payload: Mapping[str, Any],
    locator_result: ExecutionMetricsLocatorResult,
    *,
    persist: bool = False,
) -> Mapping[str, Any]:
    payload = dict(metrics_payload)
    exec_path = locator_result.execution_metrics_path
    metrics_path = locator_result.metrics_path
    execution = None
    if exec_path and exec_path.exists() and exec_path != metrics_path:
        try:
            execution = json.loads(exec_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            execution = None
    if execution is None and metrics_path and metrics_path.exists():
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, Mapping):
            execution = data.get("execution")
    if execution is not None:
        payload["execution"] = execution
        if persist:
            _persist_execution_section(metrics_path, execution)
    return payload


def _persist_execution_section(metrics_path: Path | None, execution: Mapping[str, Any]) -> None:
    if not metrics_path:
        return
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    payload = dict(payload)
    payload["execution"] = execution
    metrics_path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")


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
