"""Deterministic regime slicing helpers used for metrics and qualification."""

from __future__ import annotations

import math
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

try:  # pragma: no cover - PyYAML optional
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

_REGIME_SIGNAL = "market_vol_20d"
_REGIME_BUCKETS = ("high_vol", "mid_vol", "low_vol")
_REGIME_V2_BUCKETS = (
    "high_vol_trend_up",
    "high_vol_trend_down",
    "high_vol_flat",
    "low_vol_trend_up",
    "low_vol_trend_down",
    "low_vol_flat",
)
_DEFAULT_THRESHOLD_PATH = Path(__file__).resolve().parents[2] / "configs" / "regime_thresholds.yml"


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
    labeling_version: str | None = None,
    thresholds_path: str | Path | None = None,
) -> RegimeSliceResult | None:
    """Compute regime quantiles and per-bucket metrics when data is available."""

    version = (labeling_version or "v1").strip().lower()
    if version in {"v1", "default"}:
        return _compute_regime_slices_v1(
            regime_series,
            feature_names,
            returns=returns,
            exposures=exposures,
            turnover_by_step=turnover_by_step,
            annualization_days=annualization_days,
            float_precision=float_precision,
        )
    if version == "v2":
        return _compute_regime_slices_v2(
            regime_series,
            feature_names,
            returns=returns,
            exposures=exposures,
            turnover_by_step=turnover_by_step,
            annualization_days=annualization_days,
            float_precision=float_precision,
            thresholds_path=thresholds_path,
        )
    raise ValueError(f"Unknown regime labeling version '{labeling_version}'")


def _compute_regime_slices_v1(
    regime_series: Sequence[Sequence[float]] | None,
    feature_names: Sequence[str] | None,
    *,
    returns: Sequence[float],
    exposures: Sequence[float],
    turnover_by_step: Sequence[float],
    annualization_days: int,
    float_precision: int,
) -> RegimeSliceResult | None:
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


def _compute_regime_slices_v2(
    regime_series: Sequence[Sequence[float]] | None,
    feature_names: Sequence[str] | None,
    *,
    returns: Sequence[float],
    exposures: Sequence[float],
    turnover_by_step: Sequence[float],
    annualization_days: int,
    float_precision: int,
    thresholds_path: str | Path | None,
) -> RegimeSliceResult | None:
    if not regime_series or not feature_names:
        return None
    signal_index = _resolve_signal_index(feature_names)
    trend_index = _resolve_trend_index(feature_names)
    if signal_index is None or trend_index is None:
        return None
    available = min(len(regime_series), len(exposures), len(turnover_by_step))
    if available <= 0:
        return None
    thresholds = _load_regime_thresholds(thresholds_path)
    vol_high = thresholds["thresholds_used"]["market_vol_20d_high"]
    deadzone = thresholds["thresholds_used"]["market_trend_deadzone"]
    bucket_labels = []
    for idx in range(available):
        vol_value = _coerce_float(regime_series[idx], signal_index)
        trend_value = _coerce_float(regime_series[idx], trend_index)
        bucket_labels.append(_bucket_label_v2(vol_value, trend_value, vol_high, deadzone))
    exposure_map: Dict[str, List[float]] = {bucket: [] for bucket in _REGIME_V2_BUCKETS}
    turnover_map: Dict[str, List[float]] = {bucket: [] for bucket in _REGIME_V2_BUCKETS}
    for idx, bucket in enumerate(bucket_labels):
        exposure_map[bucket].append(float(exposures[idx]))
        turnover_map[bucket].append(float(turnover_by_step[idx]))
    returns_by_bucket: Dict[str, List[float]] = {bucket: [] for bucket in _REGIME_V2_BUCKETS}
    return_steps = min(len(returns), max(available - 1, 0))
    for idx in range(return_steps):
        bucket = bucket_labels[idx]
        returns_by_bucket[bucket].append(float(returns[idx]))
    performance: Dict[str, Dict[str, float | None]] = {}
    for bucket in _REGIME_V2_BUCKETS:
        performance[bucket] = _bucket_metrics(
            returns_by_bucket[bucket],
            exposure_map[bucket],
            turnover_map[bucket],
            annualization_days,
            float_precision,
        )
    metadata = {
        "version": "v2",
        "signal": _REGIME_SIGNAL,
        "thresholds_file": thresholds["thresholds_file"],
        "thresholds_used": dict(thresholds["thresholds_used"]),
        "bucket_distribution": dict(thresholds["bucket_distribution"]),
        "universe": list(thresholds.get("universe", [])),
    }
    reference_window = thresholds.get("reference_window")
    if reference_window:
        metadata["reference_window"] = dict(reference_window)
    return RegimeSliceResult(metadata=metadata, performance_by_regime=performance)


def _resolve_signal_index(feature_names: Sequence[str]) -> int | None:
    for idx, name in enumerate(feature_names):
        if str(name).strip() == _REGIME_SIGNAL:
            return idx
    return None


def _resolve_trend_index(feature_names: Sequence[str]) -> int | None:
    for idx, name in enumerate(feature_names):
        if str(name).strip() == "market_trend_20d":
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


def _bucket_label_v2(value: float, trend: float, vol_high: float, deadzone: float) -> str:
    vol_bucket = "high_vol" if value >= vol_high else "low_vol"
    if trend >= deadzone:
        trend_bucket = "trend_up"
    elif trend <= -deadzone:
        trend_bucket = "trend_down"
    else:
        trend_bucket = "flat"
    return f"{vol_bucket}_{trend_bucket}"


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


def _load_regime_thresholds(path: str | Path | None) -> Dict[str, object]:
    threshold_path = Path(path) if path else _DEFAULT_THRESHOLD_PATH
    if not threshold_path.exists():
        raise FileNotFoundError(
            f"Regime thresholds file not found: {threshold_path}. "
            "Run scripts/build_regime_thresholds.py to generate it."
        )
    payload = _read_threshold_payload(threshold_path)
    if not isinstance(payload, Mapping):
        raise ValueError("Regime thresholds file must contain a mapping")
    if str(payload.get("version", "")).lower() != "v2":
        raise ValueError("Regime thresholds version must be v2")
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("Regime thresholds missing 'thresholds' mapping")
    vol_entry = thresholds.get("market_vol_20d")
    trend_entry = thresholds.get("market_trend_20d")
    if not isinstance(vol_entry, Mapping) or not isinstance(trend_entry, Mapping):
        raise ValueError("Regime thresholds missing market_vol_20d/market_trend_20d entries")
    vol_high = _coerce_threshold(vol_entry.get("high"), "market_vol_20d.high")
    deadzone = _coerce_threshold(trend_entry.get("deadzone"), "market_trend_20d.deadzone")
    bucket_distribution = payload.get("bucket_distribution")
    if not isinstance(bucket_distribution, Mapping) or not bucket_distribution:
        raise ValueError("Regime thresholds missing bucket_distribution")
    bucket_distribution = _normalize_bucket_distribution(bucket_distribution)
    missing = [bucket for bucket in _REGIME_V2_BUCKETS if bucket not in bucket_distribution]
    if missing:
        raise ValueError(f"Regime thresholds missing buckets: {missing}")
    return {
        "thresholds_file": str(threshold_path),
        "thresholds_used": {
            "market_vol_20d_high": vol_high,
            "market_trend_deadzone": deadzone,
        },
        "bucket_distribution": bucket_distribution,
        "universe": payload.get("universe") or [],
        "reference_window": payload.get("reference_window") or None,
    }


def _read_threshold_payload(path: Path) -> Mapping[str, object]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML must be installed to read regime thresholds")
        loaded = yaml.safe_load(text)
        if loaded is None:
            raise ValueError("Regime thresholds file is empty")
        if not isinstance(loaded, Mapping):
            raise ValueError("Regime thresholds file must be a mapping")
        return loaded
    return json.loads(text)


def _coerce_threshold(value: object, label: str) -> float:
    if value is None:
        raise ValueError(f"Regime threshold missing {label}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Regime threshold {label} must be numeric") from None
    if not math.isfinite(number):
        raise ValueError(f"Regime threshold {label} must be finite")
    return number


def _normalize_bucket_distribution(values: Mapping[str, object]) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    for key, value in values.items():
        try:
            normalized[str(key)] = float(value)
        except (TypeError, ValueError):
            raise ValueError("Bucket distribution values must be numeric") from None
    return normalized


__all__ = ["RegimeSliceResult", "compute_regime_slices"]
