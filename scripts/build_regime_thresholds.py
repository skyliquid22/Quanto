#!/usr/bin/env python3
"""Compute fixed regime thresholds from a reference window."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

try:  # pragma: no cover - PyYAML optional
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

import numpy as np

from infra.paths import get_data_root
from research.datasets.canonical_equity_loader import load_primary_regime_universe
from research.features.regime_features_v1 import compute_primary_regime_features
from research.regime.universe import PRIMARY_REGIME_UNIVERSE

UTC = timezone.utc
DEFAULT_START = "2005-01-01"
DEFAULT_END = "2025-12-31"
DEFAULT_P_HIGH = 0.90
DEFAULT_DEADZONE = 0.002
DEFAULT_OUTPUT = "configs/regime_thresholds.yml"

_REGIME_V2_BUCKETS = (
    "high_vol_trend_up",
    "high_vol_trend_down",
    "high_vol_flat",
    "low_vol_trend_up",
    "low_vol_trend_down",
    "low_vol_flat",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build regime v2 threshold configuration.")
    parser.add_argument("--start-date", default=DEFAULT_START, help="Reference start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=DEFAULT_END, help="Reference end date (YYYY-MM-DD).")
    parser.add_argument(
        "--p-high",
        type=float,
        default=DEFAULT_P_HIGH,
        help="High-volatility percentile (0-1).",
    )
    parser.add_argument(
        "--deadzone",
        type=float,
        default=DEFAULT_DEADZONE,
        help="Trend dead-zone for flat classification.",
    )
    parser.add_argument(
        "--min-bucket-share",
        type=float,
        default=0.02,
        help="Minimum share required for each bucket (default 2%).",
    )
    parser.add_argument("--data-root", help="Override QUANTO data root.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output thresholds file path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    if end < start:
        raise SystemExit("end-date cannot be earlier than start-date")
    if not (0.0 < args.p_high < 1.0):
        raise SystemExit("p-high must be between 0 and 1")
    if args.deadzone < 0:
        raise SystemExit("deadzone must be non-negative")

    data_root = Path(args.data_root).expanduser() if args.data_root else get_data_root()
    slices, _ = load_primary_regime_universe(start, end, data_root=data_root, interval="daily")
    close_panel = _build_close_panel(slices)
    features = compute_primary_regime_features(close_panel)
    vol_series = features["market_vol_20d"].to_numpy(dtype="float64")
    trend_series = features["market_trend_20d"].to_numpy(dtype="float64")
    vol_high = float(np.quantile(vol_series, args.p_high))

    distribution = compute_bucket_distribution(
        vol_series,
        trend_series,
        vol_high=vol_high,
        deadzone=args.deadzone,
    )
    validate_bucket_distribution(distribution, min_share=args.min_bucket_share)

    payload = {
        "version": "v2",
        "universe": list(PRIMARY_REGIME_UNIVERSE),
        "reference_window": {
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        },
        "features": {
            "market_vol_20d": {
                "method": "percentile",
                "p_high": float(args.p_high),
            },
            "market_trend_20d": {
                "method": "deadzone",
                "deadzone": float(args.deadzone),
            },
        },
        "thresholds": {
            "market_vol_20d": {"high": vol_high},
            "market_trend_20d": {"deadzone": float(args.deadzone)},
        },
        "bucket_distribution": distribution,
        "recompute_policy": {
            "mode": "manual",
            "annual_month": 1,
            "drift_check": {"window_days": 252, "ks_threshold": 0.10},
        },
        "metadata": {
            "computed_at": datetime.now(tz=UTC).date().isoformat(),
            "computed_by": "scripts/build_regime_thresholds.py",
        },
    }

    output_path = Path(args.output).expanduser()
    _write_payload(output_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _build_close_panel(slices: Mapping[str, object]) -> "pd.DataFrame":
    import pandas as pd  # local import to avoid import cost for callers

    timestamps = pd.DatetimeIndex([], tz="UTC")
    per_symbol = {}
    for symbol, slice_data in slices.items():
        frame = slice_data.frame  # type: ignore[attr-defined]
        series = frame["close"].astype(float)
        series.index = pd.to_datetime(series.index, utc=True)
        per_symbol[symbol] = series
        timestamps = timestamps.union(series.index)
    timestamps = pd.DatetimeIndex(timestamps.sort_values(), tz="UTC")
    close_panel = pd.DataFrame(index=timestamps)
    for symbol, series in per_symbol.items():
        close_panel[symbol] = series.reindex(timestamps).ffill()
    if close_panel.isna().all(axis=None):
        raise SystemExit("No valid close values found for primary regime universe.")
    return close_panel


def compute_bucket_distribution(
    vol_values: Iterable[float],
    trend_values: Iterable[float],
    *,
    vol_high: float,
    deadzone: float,
) -> Dict[str, float]:
    counts = {bucket: 0 for bucket in _REGIME_V2_BUCKETS}
    total = 0
    for vol, trend in zip(vol_values, trend_values):
        bucket = _bucket_label_v2(float(vol), float(trend), vol_high, deadzone)
        counts[bucket] += 1
        total += 1
    if total == 0:
        raise SystemExit("Unable to compute bucket distribution (no rows).")
    return {bucket: counts[bucket] / total for bucket in _REGIME_V2_BUCKETS}


def validate_bucket_distribution(distribution: Mapping[str, float], *, min_share: float) -> None:
    total = sum(float(value) for value in distribution.values())
    if not np.isfinite(total) or abs(total - 1.0) > 1e-6:
        raise SystemExit(f"Bucket distribution must sum to 1.0 (got {total:.6f}).")
    sparse = {
        bucket: value
        for bucket, value in distribution.items()
        if float(value) < min_share
    }
    if sparse:
        details = ", ".join(f"{bucket}={value:.4f}" for bucket, value in sparse.items())
        raise SystemExit(f"Buckets below minimum share ({min_share:.2%}): {details}")


def _bucket_label_v2(value: float, trend: float, vol_high: float, deadzone: float) -> str:
    vol_bucket = "high_vol" if value >= vol_high else "low_vol"
    if trend >= deadzone:
        trend_bucket = "trend_up"
    elif trend <= -deadzone:
        trend_bucket = "trend_down"
    else:
        trend_bucket = "flat"
    return f"{vol_bucket}_{trend_bucket}"


def _write_payload(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".yml", ".yaml"}:
        if yaml is None:
            raise SystemExit("PyYAML is required to write YAML thresholds.")
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
