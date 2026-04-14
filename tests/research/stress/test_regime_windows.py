"""Unit tests for regime window finder."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from research.stress.config import HistoricalRegimeSpec
from research.stress.regime_windows import (
    find_regime_windows,
    _label_calendar,
    _find_contiguous_windows,
    _resolve_target_labels,
    _quantile,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cal(start: str, days: int) -> list[pd.Timestamp]:
    base = pd.Timestamp(start, tz="UTC")
    return [base + pd.Timedelta(days=i) for i in range(days)]


def _regime_series(vol_values: list[float]) -> list[tuple[float, ...]]:
    return [tuple([v]) for v in vol_values]


# ---------------------------------------------------------------------------
# _resolve_target_labels
# ---------------------------------------------------------------------------

def test_resolve_crash_alias():
    labels = _resolve_target_labels("crash")
    assert "high_vol" in labels


def test_resolve_high_vol_alias():
    labels = _resolve_target_labels("high_vol")
    assert "high_vol" in labels
    assert "high_vol_trend_up" in labels


def test_resolve_unknown_treated_as_direct():
    labels = _resolve_target_labels("custom_bucket")
    assert labels == {"custom_bucket"}


def test_resolve_none_returns_empty():
    assert _resolve_target_labels(None) == set()


# ---------------------------------------------------------------------------
# _label_calendar
# ---------------------------------------------------------------------------

def test_label_calendar_three_buckets():
    cal = _cal("2024-01-01", 6)
    # low=1,2  mid=5,6  high=9,10
    vols = [1.0, 2.0, 5.0, 6.0, 9.0, 10.0]
    series = _regime_series(vols)
    labels = _label_calendar(cal, series, ["market_vol_20d"])
    assert labels[0] == "low_vol"
    assert labels[1] == "low_vol"
    assert labels[4] == "high_vol"
    assert labels[5] == "high_vol"


def test_label_calendar_no_vol_feature():
    cal = _cal("2024-01-01", 4)
    series = _regime_series([0.1, 0.2, 0.3, 0.4])
    labels = _label_calendar(cal, series, ["some_other_feature"])
    assert all(l == "mid_vol" for l in labels)


def test_label_calendar_pads_short_series():
    cal = _cal("2024-01-01", 5)
    series = _regime_series([1.0, 2.0])  # shorter than calendar
    labels = _label_calendar(cal, series, ["market_vol_20d"])
    assert len(labels) == 5
    # Last 3 padded as mid_vol
    assert labels[2] == "mid_vol"


# ---------------------------------------------------------------------------
# _find_contiguous_windows
# ---------------------------------------------------------------------------

def test_contiguous_single_window():
    cal = _cal("2024-01-01", 5)
    labels = ["high_vol"] * 5
    windows = _find_contiguous_windows(cal, labels, {"high_vol"}, min_period_days=3)
    assert len(windows) == 1
    assert windows[0] == (date(2024, 1, 1), date(2024, 1, 5))


def test_contiguous_two_windows():
    cal = _cal("2024-01-01", 10)
    labels = ["high_vol"] * 4 + ["low_vol"] * 2 + ["high_vol"] * 4
    windows = _find_contiguous_windows(cal, labels, {"high_vol"}, min_period_days=3)
    assert len(windows) == 2


def test_contiguous_min_period_filters_short_windows():
    cal = _cal("2024-01-01", 10)
    labels = ["high_vol"] * 2 + ["low_vol"] * 3 + ["high_vol"] * 5
    windows = _find_contiguous_windows(cal, labels, {"high_vol"}, min_period_days=4)
    # Only the 5-day window qualifies
    assert len(windows) == 1
    assert (windows[0][1] - windows[0][0]).days == 4  # 5 days = 4-day span


def test_contiguous_sorted_longest_first():
    cal = _cal("2024-01-01", 12)
    # 3-day window then 7-day window
    labels = ["high_vol"] * 3 + ["low_vol"] * 2 + ["high_vol"] * 7
    windows = _find_contiguous_windows(cal, labels, {"high_vol"}, min_period_days=1)
    assert (windows[0][1] - windows[0][0]).days > (windows[1][1] - windows[1][0]).days


def test_contiguous_open_run_at_end():
    # Window starts and runs to end of calendar
    cal = _cal("2024-01-01", 8)
    labels = ["low_vol"] * 2 + ["high_vol"] * 6
    windows = _find_contiguous_windows(cal, labels, {"high_vol"}, min_period_days=5)
    assert len(windows) == 1
    assert windows[0][1] == cal[-1].date()


# ---------------------------------------------------------------------------
# find_regime_windows integration
# ---------------------------------------------------------------------------

def test_find_regime_windows_returns_matching():
    # Use a clear spread so the top third lands strictly above q66.
    cal = _cal("2024-01-01", 20)
    vols = list(range(1, 21))  # 1..20 — top 6-7 values land in high_vol bucket
    series = _regime_series(vols)
    spec = HistoricalRegimeSpec(regime_version="v1", regime_filter="high_vol", min_period_days=3)
    windows = find_regime_windows(cal, series, ["market_vol_20d"], spec)
    assert len(windows) >= 1
    start, end = windows[0]
    assert start < end


def test_find_regime_windows_raises_when_none_found():
    cal = _cal("2024-01-01", 10)
    # All identical vol — q33=q66=1.0, everything is "mid_vol", no "high_vol" window
    vols = [1.0] * 10
    series = _regime_series(vols)
    spec = HistoricalRegimeSpec(regime_version="v1", regime_filter="high_vol", min_period_days=5)
    with pytest.raises(ValueError, match="No regime windows found"):
        find_regime_windows(cal, series, ["market_vol_20d"], spec)


def test_find_regime_windows_no_filter_uses_full_calendar():
    cal = _cal("2024-01-01", 10)
    vols = [1.0] * 10
    series = _regime_series(vols)
    spec = HistoricalRegimeSpec(regime_version="v1", regime_filter=None, min_period_days=5)
    windows = find_regime_windows(cal, series, ["market_vol_20d"], spec)
    assert len(windows) == 1
    assert windows[0] == (date(2024, 1, 1), date(2024, 1, 10))


def test_find_regime_windows_crash_alias_finds_high_vol():
    cal = _cal("2024-01-01", 30)
    # Use a clear spread — values 21..30 land above q66
    vols = list(range(1, 31))
    series = _regime_series(vols)
    spec = HistoricalRegimeSpec(regime_version="v1", regime_filter="crash", min_period_days=5)
    windows = find_regime_windows(cal, series, ["market_vol_20d"], spec)
    assert len(windows) >= 1


def test_find_regime_windows_min_period_too_large_raises():
    cal = _cal("2024-01-01", 5)
    vols = [10.0] * 5
    series = _regime_series(vols)
    spec = HistoricalRegimeSpec(regime_version="v1", regime_filter="high_vol", min_period_days=100)
    with pytest.raises(ValueError):
        find_regime_windows(cal, series, ["market_vol_20d"], spec)
