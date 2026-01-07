"""Deterministic regime slicing helpers used for metrics and qualification."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

_REGIME_SIGNAL = "market_vol_20d"
_REGIME_BUCKETS = ("high_vol", "mid_vol", "low_vol")


@dataclass(frozen=True)
class RegimeSliceResult:
    """Structured output capturing regime slicing metadata and metrics."""

    metadata: Dict[str, object]
    performance_by_regime: Dict[str, Dict[str, float | None]]


def compute_regime_slices(
    regime_series: Sequence[Sequence[float]] | None,
    feature_names: Sequence[str] | None,
    *,
    returns: Sequence[float],
    exposures: Sequence[float],
    turnover_by_step: Sequence[float],
    annualization_days: int,
    float_precision: int,
) -> RegimeSliceResult | None:
    """Compute regime quantiles and per-bucket metrics when data is available."""

    if not regime_series or not feature_names:
        return None
    signal_index = _resolve_signal_index(feature_names)
    if signal_index is None:
        return None
    available = min(len(regime_series), len(exposures), len(turnover_by_step))
    if available <= 0:
        return None
    signal_values = [
        _coerce_float(regime_series[idx], signal_index) for idx in range(available)
    ]
    if not signal_values:
        return None
    q33 = _quantile(signal_values, 0.33)
    q66 = _quantile(signal_values, 0.66)
    bucket_labels = [_bucket_label(value, q33, q66) for value in signal_values]
    exposure_map: Dict[str, List[float]] = {bucket: [] for bucket in _REGIME_BUCKETS}
    turnover_map: Dict[str, List[float]] = {bucket: [] for bucket in _REGIME_BUCKETS}
    for idx, bucket in enumerate(bucket_labels):
        exposure_map[bucket].append(float(exposures[idx]))
        turnover_map[bucket].append(float(turnover_by_step[idx]))
    returns_by_bucket: Dict[str, List[float]] = {bucket: [] for bucket in _REGIME_BUCKETS}
    return_steps = min(len(returns), max(available - 1, 0))
    for idx in range(return_steps):
        bucket = bucket_labels[idx]
        returns_by_bucket[bucket].append(float(returns[idx]))
    performance: Dict[str, Dict[str, float | None]] = {}
    for bucket in _REGIME_BUCKETS:
        performance[bucket] = _bucket_metrics(
            returns_by_bucket[bucket],
            exposure_map[bucket],
            turnover_map[bucket],
            annualization_days,
            float_precision,
        )
    metadata = {
        "signal": _REGIME_SIGNAL,
        "quantiles": {
            "q33": _round(q33, float_precision),
            "q66": _round(q66, float_precision),
        },
    }
    return RegimeSliceResult(metadata=metadata, performance_by_regime=performance)


def _resolve_signal_index(feature_names: Sequence[str]) -> int | None:
    for idx, name in enumerate(feature_names):
        if str(name).strip() == _REGIME_SIGNAL:
            return idx
    return None


def _coerce_float(row: Sequence[float], index: int) -> float:
    if index < len(row):
        value = row[index]
    else:
        value = 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return number


def _bucket_label(value: float, q33: float, q66: float) -> str:
    if value <= q33:
        return "low_vol"
    if value <= q66:
        return "mid_vol"
    return "high_vol"


def _bucket_metrics(
    returns: Sequence[float],
    exposures: Sequence[float],
    turnovers: Sequence[float],
    annualization_days: int,
    precision: int,
) -> Dict[str, float | None]:
    metrics: Dict[str, float | None] = {
        "total_return": None,
        "max_drawdown": None,
        "volatility_ann": None,
        "avg_exposure": None,
        "avg_turnover": None,
        "sharpe": None,
    }
    if exposures:
        metrics["avg_exposure"] = _round(sum(exposures) / len(exposures), precision)
    if turnovers:
        metrics["avg_turnover"] = _round(sum(turnovers) / len(turnovers), precision)
    if not returns:
        return metrics
    growth = 1.0
    account_path = [1.0]
    for value in returns:
        growth *= 1.0 + value
        account_path.append(growth)
    metrics["total_return"] = _round(growth - 1.0, precision)
    volatility = _volatility(returns, annualization_days)
    metrics["volatility_ann"] = _round(volatility, precision)
    mean_return = sum(returns) / len(returns)
    sharpe = None
    if volatility > 0 and annualization_days > 0:
        sharpe = (mean_return * annualization_days) / volatility
    metrics["sharpe"] = None if sharpe is None else _round(sharpe, precision)
    metrics["max_drawdown"] = _round(_max_drawdown(account_path), precision)
    return metrics


def _volatility(returns: Sequence[float], annualization_days: int) -> float:
    if not returns:
        return 0.0
    mean_value = sum(returns) / len(returns)
    variance = sum((value - mean_value) ** 2 for value in returns) / len(returns)
    daily_vol = math.sqrt(max(variance, 0.0))
    return daily_vol * math.sqrt(max(annualization_days, 0))


def _max_drawdown(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    peak = float(values[0])
    worst = 0.0
    for value in values:
        val = float(value)
        if val > peak:
            peak = val
        drawdown = (val - peak) / peak if peak else 0.0
        worst = min(worst, drawdown)
    return abs(worst)


def _quantile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = percentile * (len(sorted_values) - 1)
    low_index = int(math.floor(rank))
    high_index = int(math.ceil(rank))
    low_value = sorted_values[low_index]
    high_value = sorted_values[high_index]
    if low_index == high_index:
        return low_value
    weight = rank - low_index
    return low_value + weight * (high_value - low_value)


def _round(value: float, precision: int) -> float:
    rounded = round(float(value), precision)
    if rounded == -0.0:
        return 0.0
    return rounded


__all__ = ["RegimeSliceResult", "compute_regime_slices"]
