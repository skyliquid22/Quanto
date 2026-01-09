# research/features/core_features_v1.py
"""core_v1: deterministic per-symbol OHLCV primitive features.

Aligned with FEATURE_SPEC.md OHLCV context:
- ret_1, ret_5, ret_21
- rv_21, rv_63
- dist_from_20d_high, dist_from_20d_low
Plus a stable volume feature:
- log_vol_z20
"""

from __future__ import annotations

from typing import Sequence

try:  # pragma: no cover
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None


CORE_V1_OBSERVATION_COLUMNS: tuple[str, ...] = (
    "close",
    "ret_1",
    "ret_5",
    "ret_21",
    "rv_21",
    "rv_63",
    "dist_from_20d_high",
    "dist_from_20d_low",
    "log_vol_z20",
)


def _ensure_pandas_available() -> None:
    if pd is None:
        raise RuntimeError("pandas is required to compute core_v1 features")


def compute_core_features_v1(equity_df: "pd.DataFrame") -> "pd.DataFrame":
    """Compute core_v1 features from a single-symbol equity frame.

    Required columns: timestamp, close, volume
    Output includes timestamp + CORE_V1_OBSERVATION_COLUMNS (and may include extras).
    """
    _ensure_pandas_available()

    required = {"timestamp", "close", "volume"}
    missing = required.difference(set(equity_df.columns))
    if missing:
        raise ValueError(f"core_v1 requires columns {sorted(required)}; missing {sorted(missing)}")

    df = equity_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df.sort_values("timestamp", inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)

    # 1) Returns (log)
    close = df["close"]
    df["ret_1"] = (close / close.shift(1)).apply(lambda x: float("nan") if x <= 0 else 0.0)  # placeholder
    # Use vectorized log safely
    ratio_1 = close / close.shift(1)
    ratio_5 = close / close.shift(5)
    ratio_21 = close / close.shift(21)
    df["ret_1"] = (ratio_1).apply(lambda x: float("nan") if x <= 0 else x)
    df["ret_5"] = (ratio_5).apply(lambda x: float("nan") if x <= 0 else x)
    df["ret_21"] = (ratio_21).apply(lambda x: float("nan") if x <= 0 else x)

    # now log (still deterministic)
    import numpy as np  # local import to keep module light

    df["ret_1"] = np.log(df["ret_1"])
    df["ret_5"] = np.log(df["ret_5"])
    df["ret_21"] = np.log(df["ret_21"])

    # 2) Realized vol (daily-scale, not annualized)
    df["rv_21"] = df["ret_1"].rolling(window=21, min_periods=21).std(ddof=0)
    df["rv_63"] = df["ret_1"].rolling(window=63, min_periods=63).std(ddof=0)

    # 3) Distance from rolling high/low
    roll_high_20 = close.rolling(window=20, min_periods=20).max()
    roll_low_20 = close.rolling(window=20, min_periods=20).min()
    df["dist_from_20d_high"] = (close / roll_high_20) - 1.0
    df["dist_from_20d_low"] = (close / roll_low_20) - 1.0

    # 4) Volume z-score (stable)
    vol_log = np.log1p(df["volume"])
    vol_mu = vol_log.rolling(window=20, min_periods=20).mean()
    vol_sd = vol_log.rolling(window=20, min_periods=20).std(ddof=0)
    df["log_vol_z20"] = (vol_log - vol_mu) / (vol_sd.replace(0.0, np.nan))

    # Deterministic NaN policy
    df[[
        "ret_1", "ret_5", "ret_21",
        "rv_21", "rv_63",
        "dist_from_20d_high", "dist_from_20d_low",
        "log_vol_z20"
    ]] = df[[
        "ret_1", "ret_5", "ret_21",
        "rv_21", "rv_63",
        "dist_from_20d_high", "dist_from_20d_low",
        "log_vol_z20"
    ]].fillna(0.0)

    return df
