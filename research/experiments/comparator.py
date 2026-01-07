"""Deterministic experiment comparison utilities."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping

from research.eval.metrics import MetricSchemaEntry, metric_schema_entries
from research.experiments.registry import ExperimentRecord, ExperimentRegistry

_TOLERANCE = 1e-9
_PERCENT_PRECISION = 6
_DELTA_PRECISION = 10


@dataclass(frozen=True)
class MetricComparison:
    """Per-metric comparison output."""

    metric_id: str
    category: str
    name: str
    directionality: str
    candidate_value: float | None
    baseline_value: float | None
    delta: float | None
    delta_pct: float | None
    status: str
    reason: str | None = None


@dataclass(frozen=True)
class ComparisonSummary:
    total_metrics: int
    compared_metrics: int
    num_improved: int
    num_regressed: int
    num_unchanged: int
    num_missing: int


@dataclass(frozen=True)
class ComparisonResult:
    candidate_experiment_id: str
    baseline_experiment_id: str
    metadata: Dict[str, Any]
    metrics: List[MetricComparison]
    summary: ComparisonSummary
    missing_metrics: List[Dict[str, str]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_experiment_id": self.candidate_experiment_id,
            "baseline_experiment_id": self.baseline_experiment_id,
            "metadata": self.metadata,
            "summary": asdict(self.summary),
            "metrics": [asdict(entry) for entry in self.metrics],
            "missing_metrics": list(self.missing_metrics),
        }


def compare_experiments(
    candidate_experiment_id: str,
    baseline_experiment_id: str,
    *,
    registry: ExperimentRegistry | None = None,
) -> ComparisonResult:
    """Compare two experiments using aligned metrics."""

    registry = registry or ExperimentRegistry()
    candidate_payload, candidate_record = _load_payload(candidate_experiment_id, registry)
    baseline_payload, baseline_record = _load_payload(baseline_experiment_id, registry)

    entries = metric_schema_entries()
    comparisons: List[MetricComparison] = []
    missing_details: List[Dict[str, str]] = []
    improved = regressed = unchanged = compared = 0

    for entry in entries:
        comparison = _compare_metric(entry, candidate_payload, baseline_payload)
        comparisons.append(comparison)
        if comparison.reason:
            missing_details.append({"metric_id": comparison.metric_id, "reason": comparison.reason})
            continue
        compared += 1
        if comparison.status == "improved":
            improved += 1
        elif comparison.status == "regressed":
            regressed += 1
        else:
            unchanged += 1

    summary = ComparisonSummary(
        total_metrics=len(entries),
        compared_metrics=compared,
        num_improved=improved,
        num_regressed=regressed,
        num_unchanged=unchanged,
        num_missing=len(missing_details),
    )
    metadata = {
        "candidate": _experiment_metadata(candidate_payload, candidate_record),
        "baseline": _experiment_metadata(baseline_payload, baseline_record),
    }
    return ComparisonResult(
        candidate_experiment_id=candidate_record.experiment_id,
        baseline_experiment_id=baseline_record.experiment_id,
        metadata=metadata,
        metrics=comparisons,
        summary=summary,
        missing_metrics=missing_details,
    )


def _load_payload(
    experiment_id: str,
    registry: ExperimentRegistry,
) -> tuple[Dict[str, Any], ExperimentRecord]:
    record = registry.get(experiment_id)
    payload = json.loads(record.metrics_path.read_text(encoding="utf-8"))
    return payload, record


def _compare_metric(
    entry: MetricSchemaEntry,
    candidate_payload: Mapping[str, Any],
    baseline_payload: Mapping[str, Any],
) -> MetricComparison:
    candidate_value = _extract_metric(candidate_payload, entry)
    baseline_value = _extract_metric(baseline_payload, entry)
    reason = None
    status = "unchanged"
    delta = None
    delta_pct = None

    if candidate_value is None and baseline_value is None:
        reason = "candidate_and_baseline_missing"
    elif candidate_value is None:
        reason = "candidate_missing"
    elif baseline_value is None:
        reason = "baseline_missing"
    else:
        raw_delta = candidate_value - baseline_value
        delta = _round(raw_delta, _DELTA_PRECISION)
        delta_pct = _percent_change(raw_delta, baseline_value)
        status = _determine_status(entry.direction, raw_delta)

    return MetricComparison(
        metric_id=entry.metric_id,
        category=entry.category,
        name=entry.key,
        directionality=entry.direction,
        candidate_value=candidate_value,
        baseline_value=baseline_value,
        delta=delta,
        delta_pct=delta_pct,
        status=status,
        reason=reason,
    )


def _experiment_metadata(payload: Mapping[str, Any], record: ExperimentRecord) -> Dict[str, Any]:
    metadata = payload.get("metadata") if isinstance(payload, Mapping) else {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    return {
        "experiment_id": record.experiment_id,
        "symbols": list(metadata.get("symbols", [])),
        "feature_set": metadata.get("feature_set"),
        "policy_id": metadata.get("policy_id"),
        "start_date": metadata.get("start_date"),
        "end_date": metadata.get("end_date"),
        "interval": metadata.get("interval"),
        "run_id": metadata.get("run_id"),
    }


def _extract_metric(payload: Mapping[str, Any], entry: MetricSchemaEntry) -> float | None:
    section = payload.get(entry.category)
    if not isinstance(section, Mapping):
        return None
    value = section.get(entry.key)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return float(number)


def _determine_status(direction: str, delta: float) -> str:
    if abs(delta) <= _TOLERANCE:
        return "unchanged"
    if direction == "lower_is_better":
        return "improved" if delta < 0 else "regressed"
    return "improved" if delta > 0 else "regressed"


def _percent_change(delta: float, baseline: float) -> float | None:
    denominator = abs(baseline)
    if denominator <= _TOLERANCE:
        return None
    value = (delta / baseline) * 100.0
    return _round(value, _PERCENT_PRECISION)


def _round(value: float, precision: int) -> float:
    rounded = round(float(value), precision)
    if rounded == -0.0:
        return 0.0
    return rounded


__all__ = [
    "ComparisonResult",
    "ComparisonSummary",
    "MetricComparison",
    "compare_experiments",
]
