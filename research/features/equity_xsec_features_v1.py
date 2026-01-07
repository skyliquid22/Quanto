"""Cross-sectional equity feature engineering for universe mode."""

from __future__ import annotations

from datetime import date, timedelta, timezone
from typing import Dict, Mapping, Sequence

import numpy as np

try:  # pragma: no cover - pandas optional in some environments
    import pandas as pd  # type: ignore
except Exception as exc:  # pragma: no cover
    pd = None
    _PANDAS_ERROR: Exception | None = exc
else:  # pragma: no cover
    _PANDAS_ERROR = None

from research.datasets.canonical_equity_loader import CanonicalEquitySlice, build_union_calendar

EPS = 1e-9
UTC = timezone.utc
BASE_FEATURE_COLUMNS = ("close", "ret_1d", "ret_5d", "vol_10d", "sma_20_dist")
XSEC_FEATURE_COLUMNS = ("ret_1d_z", "ret_5d_z", "vol_10d_z", "ret_1d_rank", "ret_5d_rank", "rel_strength_20d")
MARKET_FEATURE_COLUMNS = ("market_ret_1d", "market_vol_20d", "dispersion_1d", "corr_mean_20d")
EQUITY_XSEC_OBSERVATION_COLUMNS = BASE_FEATURE_COLUMNS + XSEC_FEATURE_COLUMNS + MARKET_FEATURE_COLUMNS


def build_equity_xsec_feature_frames(
    slices: Mapping[str, CanonicalEquitySlice],
    *,
    start_date: date,
    end_date: date,
    symbol_order: Sequence[str] | None = None,
    forward_fill_limit: int = 3,
) -> Dict[str, "pd.DataFrame"]:
    """Compute deterministic cross-sectional features for every symbol."""

    _ensure_pandas_available()
    if not slices:
        raise ValueError("slices cannot be empty")
    order = tuple(dict.fromkeys(symbol_order or sorted(slices.keys())))
    if len(order) < 2:
        raise ValueError("Cross-sectional features require at least two symbols")
    calendar = build_union_calendar(slices)
    if start_date:
        start_ts = pd.Timestamp(start_date, tz=UTC)
        calendar = calendar[calendar >= start_ts]
    if end_date:
        end_exclusive = pd.Timestamp(end_date, tz=UTC) + pd.Timedelta(days=1)
        calendar = calendar[calendar < end_exclusive]
    if calendar.empty:
        raise ValueError("No overlapping timestamps available for cross-sectional features")
    close_panel = _align_close_panel(slices, calendar, order, forward_fill_limit)
    if close_panel.empty:
        raise ValueError("Unable to align closing prices across the requested universe")

    ret_1d = close_panel.pct_change().fillna(0.0)
    ret_5d = close_panel.pct_change(5).fillna(0.0)
    ret_20d = close_panel.pct_change(20).fillna(0.0)
    vol_10d = ret_1d.rolling(window=10, min_periods=2).std(ddof=0).fillna(0.0)
    sma_20 = close_panel.rolling(window=20, min_periods=1).mean()
    sma_20_dist = ((close_panel - sma_20) / sma_20.replace(0.0, np.nan)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    rel_strength_20d = ret_20d.sub(ret_20d.mean(axis=1), axis=0).fillna(0.0)

    ret_1d_z = _compute_zscores(ret_1d)
    ret_5d_z = _compute_zscores(ret_5d)
    vol_10d_z = _compute_zscores(vol_10d)
    ret_1d_rank = _compute_ranks(ret_1d, order)
    ret_5d_rank = _compute_ranks(ret_5d, order)

    market_ret_1d = ret_1d.mean(axis=1).fillna(0.0)
    dispersion_1d = ret_1d.std(axis=1, ddof=0).fillna(0.0)
    market_vol_20d = market_ret_1d.rolling(window=20, min_periods=2).std(ddof=0).fillna(0.0)
    corr_mean_20d = _rolling_corr_mean(ret_1d, window=20)

    frames: Dict[str, "pd.DataFrame"] = {}
    for symbol in order:
        combined = pd.DataFrame(
            {
                "close": close_panel[symbol],
                "ret_1d": ret_1d[symbol],
                "ret_5d": ret_5d[symbol],
                "vol_10d": vol_10d[symbol],
                "sma_20_dist": sma_20_dist[symbol],
                "ret_1d_z": ret_1d_z[symbol],
                "ret_5d_z": ret_5d_z[symbol],
                "vol_10d_z": vol_10d_z[symbol],
                "ret_1d_rank": ret_1d_rank[symbol],
                "ret_5d_rank": ret_5d_rank[symbol],
                "rel_strength_20d": rel_strength_20d[symbol],
                "market_ret_1d": market_ret_1d,
                "market_vol_20d": market_vol_20d,
                "dispersion_1d": dispersion_1d,
                "corr_mean_20d": corr_mean_20d,
            }
        )
        combined = combined.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        combined.reset_index(inplace=True)
        combined.rename(columns={"index": "timestamp"}, inplace=True)
        combined = combined[["timestamp", *EQUITY_XSEC_OBSERVATION_COLUMNS]]
        frames[symbol] = combined
    return frames


def _align_close_panel(
    slices: Mapping[str, CanonicalEquitySlice],
    calendar: "pd.DatetimeIndex",
    order: Sequence[str],
    forward_fill_limit: int,
) -> "pd.DataFrame":
    limit = max(0, int(forward_fill_limit))
    panel = pd.DataFrame(index=calendar)
    for symbol in order:
        slice_data = slices[symbol]
        if slice_data.frame.empty or "close" not in slice_data.frame.columns:
            raise ValueError(f"slice for {symbol} is missing close data")
        closes = slice_data.frame["close"].astype(float)
        aligned = closes.reindex(calendar)
        if limit > 0:
            aligned = aligned.ffill(limit=limit)
        panel[symbol] = aligned
    panel = panel.dropna(how="any")
    return panel


def _compute_zscores(values: "pd.DataFrame") -> "pd.DataFrame":
    mean = values.mean(axis=1)
    std = values.std(axis=1, ddof=0)
    centered = values.sub(mean, axis=0)
    denom = std.replace(0.0, np.nan) + EPS
    zscores = centered.div(denom, axis=0)
    zscores = zscores.where(std > 0, 0.0)
    return zscores.fillna(0.0)


def _compute_ranks(values: "pd.DataFrame", order: Sequence[str]) -> "pd.DataFrame":
    ranks = pd.DataFrame(index=values.index, columns=order, dtype="float64")
    denom = max(len(order) - 1, 1)
    for timestamp in values.index:
        row = values.loc[timestamp]
        items = []
        for symbol in order:
            val = row.get(symbol, 0.0)
            if pd.isna(val):
                val = 0.0
            items.append((float(val), symbol))
        items.sort(key=lambda pair: (pair[0], pair[1]))
        for idx, (_, symbol) in enumerate(items):
            ranks.at[timestamp, symbol] = 0.0 if denom == 0 else idx / denom
    return ranks.fillna(0.0)


def _rolling_corr_mean(returns: "pd.DataFrame", window: int) -> "pd.Series":
    if returns.shape[1] < 2:
        return pd.Series(0.0, index=returns.index)
    corr_values: list[float] = []
    for idx in range(len(returns)):
        start = max(0, idx - window + 1)
        window_frame = returns.iloc[start : idx + 1]
        if len(window_frame) < 2:
            corr_values.append(0.0)
            continue
        corr = window_frame.corr()
        if corr.shape[0] < 2:
            corr_values.append(0.0)
            continue
        matrix = corr.to_numpy()
        upper_indices = np.triu_indices_from(matrix, k=1)
        upper_values = matrix[upper_indices]
        finite = upper_values[np.isfinite(upper_values)]
        corr_values.append(float(finite.mean()) if finite.size else 0.0)
    return pd.Series(corr_values, index=returns.index).fillna(0.0)


def _ensure_pandas_available() -> None:
    if pd is None:
        raise RuntimeError("pandas is required for cross-sectional features") from _PANDAS_ERROR


__all__ = [
    "EQUITY_XSEC_OBSERVATION_COLUMNS",
    "build_equity_xsec_feature_frames",
]
