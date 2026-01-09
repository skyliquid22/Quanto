"""Label engineering utilities for supervised proxies used by alpha/RL training."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

try:  # pragma: no cover - pandas optional
    import pandas as pd  # type: ignore
except Exception as exc:  # pragma: no cover
    pd = None
    _PANDAS_ERROR: Exception | None = exc
else:  # pragma: no cover
    _PANDAS_ERROR = None


def compute_forward_return_labels(
    frame: "pd.DataFrame",
    horizons: Sequence[int] = (1, 5, 20),
    *,
    price_column: str = "close",
) -> "pd.DataFrame":
    """Append forward return labels for the requested horizons."""

    _ensure_pandas_available()
    if price_column not in frame:
        raise ValueError(f"{price_column} column missing from frame")
    if not horizons:
        return frame
    df = frame.copy()
    prices = pd.to_numeric(df[price_column], errors="coerce")
    for horizon in horizons:
        if horizon <= 0:
            raise ValueError("horizons must be positive integers")
        future = prices.shift(-horizon)
        label = (future - prices) / prices.replace(0.0, np.nan)
        label = label.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        df[f"label_fwd_return_{horizon}"] = label
    return df


def compute_forward_drawdown_labels(
    frame: "pd.DataFrame",
    horizons: Sequence[int] = (5, 20),
    *,
    price_column: str = "close",
) -> "pd.DataFrame":
    """Append maximum forward drawdown labels over the supplied horizons."""

    _ensure_pandas_available()
    if price_column not in frame:
        raise ValueError(f"{price_column} column missing from frame")
    if not horizons:
        return frame
    df = frame.copy()
    prices = pd.to_numeric(df[price_column], errors="coerce").to_numpy(dtype=float)
    n = len(prices)
    for horizon in horizons:
        if horizon <= 0:
            raise ValueError("horizons must be positive integers")
        labels = np.zeros(n, dtype=float)
        for idx in range(n):
            start = idx + 1
            stop = min(n, idx + 1 + horizon)
            if start >= stop or not np.isfinite(prices[idx]) or prices[idx] == 0.0:
                labels[idx] = 0.0
                continue
            window = prices[start:stop]
            window = window[np.isfinite(window)]
            if window.size == 0:
                labels[idx] = 0.0
                continue
            forward_min = float(window.min())
            drawdown = (forward_min - prices[idx]) / prices[idx]
            labels[idx] = drawdown if drawdown < 0 else 0.0
        df[f"label_fwd_drawdown_{horizon}"] = labels
    return df


def compute_regime_transition_flags(
    frame: "pd.DataFrame",
    *,
    column: str = "regime_state",
    horizons: Sequence[int] = (1,),
) -> "pd.DataFrame":
    """Append binary regime transition flags for the provided horizons."""

    _ensure_pandas_available()
    if column not in frame:
        raise ValueError(f"{column} column missing from frame")
    if not horizons:
        return frame
    df = frame.copy()
    regime_series = df[column].astype(str)
    for horizon in horizons:
        if horizon <= 0:
            raise ValueError("horizons must be positive integers")
        shifted = regime_series.shift(-horizon)
        label = (shifted.notna() & (shifted != regime_series)).astype(int)
        df[f"label_regime_transition_{horizon}"] = label
    return df


def _ensure_pandas_available() -> None:
    if pd is None:
        raise RuntimeError("pandas is required for label engineering") from _PANDAS_ERROR


__all__ = [
    "compute_forward_drawdown_labels",
    "compute_forward_return_labels",
    "compute_regime_transition_flags",
]
