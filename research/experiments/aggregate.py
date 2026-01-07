"""Aggregation utilities for experiment sweeps."""

from __future__ import annotations

import json
import math
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median, pstdev
from typing import Any, Dict, List, Mapping, Sequence

from research.experiments.ablation import SweepResult
from research.experiments.registry import ExperimentRecord, ExperimentRegistry
from research.experiments.sweep import SweepSpec

_METRIC_TARGETS = (
    ("performance.sharpe", "higher_is_better"),
    ("performance.total_return", "higher_is_better"),
    ("performance.max_drawdown", "lower_is_better"),
    ("trading.turnover_1d_mean", "lower_is_better"),
)
_PRECISION = 10


@dataclass(frozen=True)
class AggregationResult:
    """Bundle containing metric and regression aggregation outputs."""

    metrics_summary: Dict[str, Any]
    regression_summary: Dict[str, Any]


def aggregate_sweep(result: SweepResult, *, registry: ExperimentRegistry | None = None) -> AggregationResult:
    registry = registry or ExperimentRegistry()
    metric_entries, missing = _load_metrics(result, registry)
    metrics_summary = _build_metric_summary(result, metric_entries, missing)
    regression_summary = _build_regression_summary(result, metric_entries)
    return AggregationResult(metrics_summary=metrics_summary, regression_summary=regression_summary)


@dataclass(frozen=True)
class _MetricEntry:
    experiment_id: str
    dimensions: Dict[str, Any]
    payload: Mapping[str, Any]
    record: ExperimentRecord


def _load_metrics(result: SweepResult, registry: ExperimentRegistry) -> tuple[List[_MetricEntry], List[Dict[str, Any]]]:
    entries: List[_MetricEntry] = []
    missing_details: List[Dict[str, Any]] = []
    for experiment in result.experiments:
        try:
            record = registry.get(experiment.experiment_id)
        except FileNotFoundError:
            missing_details.append({"experiment_id": experiment.experiment_id, "reason": "not_found"})
            continue
        payload = json.loads(record.metrics_path.read_text(encoding="utf-8"))
        entries.append(
            _MetricEntry(
                experiment_id=experiment.experiment_id,
                dimensions=dict(experiment.dimensions),
                payload=payload,
                record=record,
            )
        )
    missing_details.sort(key=lambda entry: entry["experiment_id"])
    return entries, missing_details


def _build_metric_summary(
    result: SweepResult,
    metric_entries: Sequence[_MetricEntry],
    missing: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    metrics_payload: Dict[str, Any] = OrderedDict()
    extrema_payload: Dict[str, Any] = OrderedDict()
    missing_map: Dict[str, List[str]] = defaultdict(list)

    for metric_id, direction in _METRIC_TARGETS:
        values: List[float] = []
        best: tuple[float, _MetricEntry] | None = None
        worst: tuple[float, _MetricEntry] | None = None
        for entry in metric_entries:
            value = _extract_metric(entry.payload, metric_id)
            if value is None:
                missing_map[metric_id].append(entry.experiment_id)
                continue
            values.append(value)
            if best is None or _is_better(direction, value, best[0]):
                best = (value, entry)
            if worst is None or _is_worse(direction, value, worst[0]):
                worst = (value, entry)
        summary = {
            "samples": len(values),
            "mean": _round_float(fmean(values)) if values else None,
            "median": _round_float(median(values)) if values else None,
            "std": _round_float(pstdev(values)) if len(values) > 1 else (0.0 if values else None),
            "missing_experiments": sorted(missing_map.get(metric_id, [])),
        }
        metrics_payload[metric_id] = summary
        extrema_payload[metric_id] = {
            "best": _extreme_payload(best),
            "worst": _extreme_payload(worst),
        }

    configuration_counts = _configuration_counts(metric_entries, result.dimension_names)
    payload = OrderedDict(
        (
            ("sweep_name", result.sweep_name),
            ("dimension_names", list(result.dimension_names)),
            ("total_experiments", result.total_requested),
            ("completed", result.completed),
            ("skipped", result.skipped),
            ("metrics", metrics_payload),
            ("extrema", extrema_payload),
            ("configuration_counts", configuration_counts),
            ("missing_metrics", list(missing)),
        )
    )
    return payload


def _build_regression_summary(
    result: SweepResult,
    metric_entries: Sequence[_MetricEntry],
) -> Dict[str, Any]:
    gate_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"pass": 0, "fail": 0, "warn": 0})
    missing_reports: List[str] = []
    overall_pass = overall_fail = experiments_with_warnings = 0

    for entry in metric_entries:
        gate_path = entry.record.root / "comparison" / "gate_report.json"
        if not gate_path.exists():
            missing_reports.append(entry.experiment_id)
            continue
        gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
        overall_status = str(gate_payload.get("overall_status", "")).lower()
        if overall_status == "fail":
            overall_fail += 1
        else:
            overall_pass += 1
        soft_warnings = int(gate_payload.get("soft_warnings", 0))
        if soft_warnings:
            experiments_with_warnings += 1
        for evaluation in gate_payload.get("evaluations", []):
            gate_id = str(evaluation.get("gate_id") or evaluation.get("metric_id") or "unknown")
            status = str(evaluation.get("status", "pass")).lower()
            if status not in {"pass", "fail", "warn"}:
                status = "pass"
            gate_counts[gate_id][status] += 1

    ordered_gate_counts = OrderedDict(sorted(gate_counts.items(), key=lambda item: item[0]))
    return OrderedDict(
        (
            ("sweep_name", result.sweep_name),
            ("total_experiments", result.total_requested),
            ("reports_evaluated", overall_pass + overall_fail),
            ("passes", overall_pass),
            ("fails", overall_fail),
            ("experiments_with_warnings", experiments_with_warnings),
            ("missing_reports", sorted(missing_reports)),
            ("gate_counts", ordered_gate_counts),
        )
    )


def _extract_metric(payload: Mapping[str, Any], metric_id: str) -> float | None:
    parts = metric_id.split(".")
    node: Any = payload
    for part in parts:
        if not isinstance(node, Mapping):
            return None
        node = node.get(part)
    if node is None:
        return None
    try:
        value = float(node)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def _is_better(direction: str, value: float, current: float) -> bool:
    if direction == "lower_is_better":
        return value < current
    return value > current


def _is_worse(direction: str, value: float, current: float) -> bool:
    if direction == "lower_is_better":
        return value > current
    return value < current


def _extreme_payload(entry: tuple[float, _MetricEntry] | None) -> Dict[str, Any] | None:
    if entry is None:
        return None
    value, record = entry
    return {
        "experiment_id": record.experiment_id,
        "value": _round_float(value),
        "dimensions": record.dimensions,
    }


def _round_float(value: float | None) -> float | None:
    if value is None:
        return None
    return float(round(value, _PRECISION))


def _configuration_counts(metric_entries: Sequence[_MetricEntry], dimension_names: Sequence[str]) -> List[Dict[str, Any]]:
    counts: Dict[str, Dict[str, Any]] = {}
    for entry in metric_entries:
        key = "|".join(f"{name}={entry.dimensions.get(name)}" for name in dimension_names)
        if key not in counts:
            counts[key] = {
                "dimensions": OrderedDict((name, entry.dimensions.get(name)) for name in dimension_names),
                "sample_count": 0,
            }
        counts[key]["sample_count"] += 1
    ordered = sorted(counts.values(), key=lambda entry: tuple(str(v) for v in entry["dimensions"].values()))
    return ordered


__all__ = ["AggregationResult", "aggregate_sweep"]
