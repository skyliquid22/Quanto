"""Execution qualification gates for T26R."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

from research.experiments.comparator import ComparisonResult, MetricComparison

_TOLERANCE = 1e-9


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

        self._enforce_soft_delta_gate(
            comparison_summary=comparison.execution_metrics.get("summary") if comparison.execution_metrics else {},
            gate_id="execution_avg_slippage_delta",
            metric="avg_slippage_bps",
            threshold=self.soft_max_avg_slippage_delta_bps,
            severity_failures=soft_failures,
            gates=gates,
            direction="absolute",
        )
        self._enforce_soft_delta_gate(
            comparison_summary=comparison.execution_metrics.get("summary") if comparison.execution_metrics else {},
            gate_id="execution_fee_drift",
            metric="total_fees",
            threshold=self.soft_max_fees_delta_pct,
            severity_failures=soft_failures,
            gates=gates,
            direction="percent",
        )
        self._enforce_soft_delta_gate(
            comparison_summary=comparison.execution_metrics.get("summary") if comparison.execution_metrics else {},
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

        report = {
            "candidate": candidate_execution,
            "baseline": baseline_execution or {},
            "comparison": comparison.execution_metrics,
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
    ) -> None:
        entry = comparison_summary.get(metric) if isinstance(comparison_summary, Mapping) else None
        observed = entry.get("candidate") if isinstance(entry, Mapping) else None
        baseline_value = entry.get("baseline") if isinstance(entry, Mapping) else None
        delta = entry.get("delta") if isinstance(entry, Mapping) else None
        delta_pct = entry.get("delta_pct") if isinstance(entry, Mapping) else None
        status = "pass"
        message = "within threshold"
        metric_id = f"execution.summary.{metric}"

        if entry is None or (delta is None and delta_pct is None):
            status = "skip"
            message = "baseline_missing"
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
        slippage_entry = comparison.execution_metrics.get("summary", {}).get("avg_slippage_bps") if comparison.execution_metrics else None
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
                if isinstance(slippage_entry, Mapping) and slippage_entry.get("delta", 0.0) > _TOLERANCE:
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
