"""Historical regime window finder for stress test scenarios.

Scans a replay data source for contiguous date windows that match a requested
regime filter (e.g. "crash", "high_vol") and returns qualifying windows sorted
longest-first.

Regime labels are derived from the vol signal using the same quantile logic as
the shadow engine's _compute_regime_buckets. Filter aliases map user-facing
names (like "crash") to the underlying bucket labels.
"""

from __future__ import annotations

from datetime import date
from typing import Sequence

import pandas as pd

from research.stress.config import HistoricalRegimeSpec

_FILTER_ALIASES: dict[str, list[str]] = {
    "crash": ["high_vol", "high_vol_trend_down"],
    "high_vol": ["high_vol", "high_vol_trend_up", "high_vol_trend_down", "high_vol_flat"],
    "low_vol": ["low_vol", "low_vol_trend_up", "low_vol_trend_down", "low_vol_flat"],
    "mid_vol": ["mid_vol"],
    "trending": ["low_vol_trend_up", "high_vol_trend_up"],
    "trend_up": ["low_vol_trend_up", "high_vol_trend_up"],
    "trend_down": ["low_vol_trend_down", "high_vol_trend_down"],
}

_VOL_SIGNAL = "market_vol_20d"


def _quantile(values: list[float], percentile: float) -> float:
    import math
    if not values:
        return 0.0
    sv = sorted(values)
    if len(sv) == 1:
        return sv[0]
    rank = percentile * (len(sv) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return sv[lo]
    return sv[lo] + (rank - lo) * (sv[hi] - sv[lo])


def _bucket_label(value: float, q33: float, q66: float) -> str:
    if value <= q33:
        return "low_vol"
    if value <= q66:
        return "mid_vol"
    return "high_vol"


def _resolve_target_labels(regime_filter: str | None) -> set[str]:
    """Map a filter name to the set of bucket labels it covers."""
    if regime_filter is None:
        return set()
    key = regime_filter.lower().strip()
    if key in _FILTER_ALIASES:
        return set(_FILTER_ALIASES[key])
    # Treat unknown filters as a direct label match.
    return {key}


def _label_calendar(
    calendar: list[pd.Timestamp],
    regime_series: Sequence[Sequence[float]],
    feature_names: Sequence[str],
) -> list[str]:
    """Assign a regime bucket label to each trading day."""
    try:
        vol_idx = list(feature_names).index(_VOL_SIGNAL)
    except ValueError:
        # No vol signal available — label every day as mid_vol.
        return ["mid_vol"] * len(calendar)

    n = min(len(calendar), len(regime_series))
    signal_values = [float(regime_series[i][vol_idx]) for i in range(n)]
    if not signal_values:
        return ["mid_vol"] * len(calendar)

    q33 = _quantile(signal_values, 0.33)
    q66 = _quantile(signal_values, 0.66)
    labels = [_bucket_label(v, q33, q66) for v in signal_values]
    # Pad any remaining days if calendar is longer than regime_series.
    labels += ["mid_vol"] * (len(calendar) - len(labels))
    return labels


def _find_contiguous_windows(
    calendar: list[pd.Timestamp],
    labels: list[str],
    target_labels: set[str],
    min_period_days: int,
) -> list[tuple[date, date]]:
    """Find contiguous runs of days whose label is in target_labels."""
    windows: list[tuple[date, date]] = []
    run_start: int | None = None

    for i, label in enumerate(labels):
        if label in target_labels:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                length = i - run_start
                if length >= min_period_days:
                    windows.append((
                        calendar[run_start].date(),
                        calendar[i - 1].date(),
                    ))
                run_start = None

    # Close any open run at the end of the calendar.
    if run_start is not None:
        length = len(calendar) - run_start
        if length >= min_period_days:
            windows.append((
                calendar[run_start].date(),
                calendar[-1].date(),
            ))

    # Sort longest window first.
    windows.sort(key=lambda w: (w[1] - w[0]).days, reverse=True)
    return windows


def find_regime_windows(
    calendar: list[pd.Timestamp],
    regime_series: Sequence[Sequence[float]],
    feature_names: Sequence[str],
    hist_spec: HistoricalRegimeSpec,
) -> list[tuple[date, date]]:
    """Return qualifying date windows matching hist_spec.regime_filter.

    Args:
        calendar: Ordered list of trading timestamps from ReplayMarketDataSource.
        regime_series: Per-step regime feature vectors, same length as calendar.
        feature_names: Names corresponding to each position in regime_series vectors.
        hist_spec: HistoricalRegimeSpec defining the filter and min_period_days.

    Returns:
        List of (start_date, end_date) tuples, sorted longest-first.

    Raises:
        ValueError: If no qualifying windows are found.
    """
    target_labels = _resolve_target_labels(hist_spec.regime_filter)
    if not target_labels:
        # No filter — treat the entire calendar as one window if it qualifies.
        if len(calendar) >= hist_spec.min_period_days:
            return [(calendar[0].date(), calendar[-1].date())]
        raise ValueError(
            f"Full calendar ({len(calendar)} days) shorter than min_period_days "
            f"({hist_spec.min_period_days})."
        )

    labels = _label_calendar(calendar, regime_series, feature_names)
    windows = _find_contiguous_windows(calendar, labels, target_labels, hist_spec.min_period_days)

    if not windows:
        raise ValueError(
            f"No regime windows found matching filter '{hist_spec.regime_filter}' "
            f"with min_period_days={hist_spec.min_period_days}. "
            f"Available labels: {sorted(set(labels))}."
        )

    return windows


__all__ = ["find_regime_windows"]
