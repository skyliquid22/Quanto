"""Deterministic regime feature engineering utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np

try:  # pragma: no cover - pandas optional in some environments
    import pandas as pd  # type: ignore
except Exception as exc:  # pragma: no cover
    pd = None
    _PANDAS_ERROR: Exception | None = exc
else:  # pragma: no cover
    _PANDAS_ERROR = None

from research.regime.universe import PRIMARY_REGIME_UNIVERSE

EPS = 1e-12
REGIME_FEATURE_COLUMNS: tuple[str, ...] = (
    "market_vol_20d",
    "market_trend_20d",
    "dispersion_20d",
    "corr_mean_20d",
)


def compute_regime_features(
    close_panel: "pd.DataFrame",
    *,
    window: int = 20,
) -> "pd.DataFrame":
    """Compute slow-moving market regime features from aligned close prices."""

    _ensure_pandas_available()
    if window <= 0:
        raise ValueError("window must be positive")
    if close_panel.empty:
        raise ValueError("close_panel cannot be empty")

    panel = close_panel.copy()
    panel = panel.sort_index()
    panel = panel.apply(pd.to_numeric, errors="coerce")
    if panel.isna().all(axis=None):
        raise ValueError("close_panel must include at least one finite value")

    returns = panel.pct_change().fillna(0.0)
    market_returns = returns.mean(axis=1).fillna(0.0)
    market_vol = market_returns.rolling(window=window, min_periods=2).std(ddof=0).fillna(0.0)
    market_trend = market_returns.rolling(window=window, min_periods=1).mean().fillna(0.0)
    dispersion_daily = returns.std(axis=1, ddof=0).fillna(0.0)
    dispersion = dispersion_daily.rolling(window=window, min_periods=1).mean().fillna(0.0)
    corr_mean = _rolling_corr_mean(returns, window=window)

    features = pd.DataFrame(
        {
            "market_vol_20d": market_vol,
            "market_trend_20d": market_trend,
            "dispersion_20d": dispersion,
            "corr_mean_20d": corr_mean,
        },
        index=panel.index,
    )
    features = features.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return features


def compute_primary_regime_features(
    close_panel: "pd.DataFrame",
    *,
    window: int = 20,
    universe: Sequence[str] = PRIMARY_REGIME_UNIVERSE,
) -> "pd.DataFrame":
    """Compute regime features using a fixed primary market universe."""

    _ensure_pandas_available()
    required = tuple(str(symbol).strip().upper() for symbol in universe if str(symbol).strip())
    if not required:
        raise ValueError("primary regime universe must include at least one symbol")
    if close_panel.empty:
        raise ValueError("close_panel cannot be empty")
    missing = [symbol for symbol in required if symbol not in close_panel.columns]
    if missing:
        raise ValueError(f"close_panel missing primary regime symbols: {missing}")
    panel = close_panel.loc[:, required]
    return compute_regime_features(panel, window=window)


def _rolling_corr_mean(returns: "pd.DataFrame", *, window: int) -> "pd.Series":
    if returns.shape[1] < 2:
        return pd.Series(0.0, index=returns.index)
    values: list[float] = []
    for idx in range(len(returns.index)):
        start = max(0, idx - window + 1)
        window_frame = returns.iloc[start : idx + 1]
        if len(window_frame) < 2:
            values.append(0.0)
            continue
        corr = window_frame.corr()
        if corr.shape[0] < 2:
            values.append(0.0)
            continue
        matrix = corr.to_numpy()
        upper = np.triu_indices_from(matrix, k=1)
        upper_values = matrix[upper]
        finite = upper_values[np.isfinite(upper_values)]
        values.append(float(finite.mean()) if finite.size else 0.0)
    return pd.Series(values, index=returns.index).fillna(0.0)


def _ensure_pandas_available() -> None:
    if pd is None:
        raise RuntimeError("pandas is required for regime feature computation") from _PANDAS_ERROR


__all__ = [
    "REGIME_FEATURE_COLUMNS",
    "compute_regime_features",
    "compute_primary_regime_features",
]
