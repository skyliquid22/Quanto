"""Utilities to derive evaluation timeseries artifacts from rollout outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


def load_rollout_json(experiment_dir: Path) -> Dict[str, Any]:
    """Load runs/rollout.json for an experiment."""

    rollout_path = experiment_dir / "runs" / "rollout.json"
    if not rollout_path.exists():
        raise FileNotFoundError(f"rollout artifact not found: {rollout_path}")
    return json.loads(rollout_path.read_text(encoding="utf-8"))


def extract_timeseries_from_rollout(
    rollout_payload: Mapping[str, Any],
    *,
    float_precision: int = 10,
) -> Dict[str, Any]:
    """Derive per-step metrics needed for regime slicing from a rollout payload."""

    series = rollout_payload.get("series") or {}
    returns = list(series.get("returns") or [])
    timestamps = list(series.get("timestamps") or [])
    if not returns or len(timestamps) < len(returns) + 1:
        raise ValueError("rollout payload missing aligned timestamps and returns")
    step_timestamps = timestamps[1 : len(returns) + 1]
    metadata = rollout_payload.get("metadata") or {}
    symbol_order = _resolve_symbol_order(metadata, series)
    weight_entries = _expand_weight_entries(series.get("weights"), symbol_order)
    if len(weight_entries) < len(returns) + 1:
        raise ValueError("rollout weights must include T+1 entries")

    exposures = [
        _round(sum(max(weight_entries[idx].get(symbol, 0.0), 0.0) for symbol in symbol_order), float_precision)
        for idx in range(1, len(returns) + 1)
    ]
    turnover_by_step = [
        _round(
            sum(abs(weight_entries[idx].get(symbol, 0.0) - weight_entries[idx - 1].get(symbol, 0.0)) for symbol in symbol_order),
            float_precision,
        )
        for idx in range(1, len(returns) + 1)
    ]
    regime_payload = _extract_regime_series(series)
    return {
        "timestamps": step_timestamps,
        "returns": [float(value) for value in returns],
        "exposures": exposures,
        "turnover_by_step": turnover_by_step,
        "regime": regime_payload,
    }


def write_timeseries_json(experiment_dir: Path, timeseries: Mapping[str, Any]) -> Path:
    """Persist evaluation/timeseries.json for downstream utilities."""

    evaluation_dir = experiment_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    path = evaluation_dir / "timeseries.json"
    path.write_text(json.dumps(timeseries, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _resolve_symbol_order(metadata: Mapping[str, Any], series: Mapping[str, Any]) -> List[str]:
    rollout_meta = metadata.get("rollout") if isinstance(metadata, Mapping) else None
    symbols = []
    if isinstance(rollout_meta, Mapping):
        symbols = list(rollout_meta.get("symbols") or [])
    if not symbols and isinstance(metadata.get("symbols"), list):
        symbols = list(metadata["symbols"])
    if not symbols:
        weights = series.get("weights")
        if isinstance(weights, Mapping):
            symbols = sorted(weights.keys())
        elif isinstance(weights, Sequence) and weights:
            symbols = ["asset"]
    if not symbols:
        raise ValueError("Unable to determine symbol order for rollout weights")
    return [str(symbol) for symbol in symbols]


def _expand_weight_entries(weights: Any, symbols: Sequence[str]) -> List[Dict[str, float]]:
    if isinstance(weights, list):
        if len(symbols) != 1:
            raise ValueError("List weights only supported for single-symbol rollouts")
        return [{symbols[0]: float(value)} for value in weights]
    if isinstance(weights, Mapping):
        entries: List[Dict[str, float]] = []
        length = max(len(series) for series in weights.values()) if weights else 0
        for idx in range(length):
            snapshot: Dict[str, float] = {}
            for symbol in symbols:
                series = weights.get(symbol) or []
                value = series[idx] if idx < len(series) else 0.0
                snapshot[symbol] = float(value)
            entries.append(snapshot)
        return entries
    raise ValueError("rollout weights must be a mapping or list")


def _extract_regime_series(series: Mapping[str, Any]) -> Dict[str, Any] | None:
    raw_regime = series.get("regime")
    if not isinstance(raw_regime, Mapping):
        return None
    feature_names = raw_regime.get("feature_names") or []
    values = raw_regime.get("series") or raw_regime.get("values") or []
    if not feature_names or not values:
        return None
    feature_names = [str(name) for name in feature_names]
    normalized: List[List[float]] = []
    expected = len(feature_names)
    for snapshot in values:
        row = list(snapshot) if isinstance(snapshot, Sequence) else []
        if len(row) < expected:
            row.extend([0.0] * (expected - len(row)))
        normalized.append([float(row[idx]) for idx in range(expected)])
    if not normalized:
        return None
    return {"feature_names": feature_names, "series": normalized}


def _round(value: float, precision: int) -> float:
    rounded = round(float(value), precision)
    if rounded == -0.0:
        return 0.0
    return rounded


__all__ = ["extract_timeseries_from_rollout", "load_rollout_json", "write_timeseries_json"]
