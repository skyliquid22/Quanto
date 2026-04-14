"""Unit tests for StressDataSource synthetic mutation wrapper."""

from __future__ import annotations

from datetime import timezone
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from research.stress.config import SyntheticSpec
from research.stress.data_source import StressDataSource, _parse_inject_at, _day_offset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(date_str: str) -> pd.Timestamp:
    return pd.Timestamp(date_str, tz="UTC")


def _make_calendar(start: str, days: int) -> list[pd.Timestamp]:
    base = pd.Timestamp(start, tz="UTC")
    return [base + pd.Timedelta(days=i) for i in range(days)]


def _make_snapshot(
    as_of: pd.Timestamp,
    symbols: tuple[str, ...] = ("AAA",),
    feature_val: float = 10.0,
    regime_features: tuple[float, ...] = (0.02,),
    regime_feature_names: tuple[str, ...] = ("market_vol_20d",),
) -> dict[str, Any]:
    return {
        "as_of": as_of,
        "symbols": symbols,
        "panel": {sym: {"close_norm": feature_val, "sma_fast": feature_val} for sym in symbols},
        "regime_features": regime_features,
        "regime_feature_names": regime_feature_names,
        "observation_columns": ("close_norm", "sma_fast"),
    }


def _make_base(calendar: list[pd.Timestamp], feature_val: float = 10.0) -> MagicMock:
    base = MagicMock()
    base.calendar.return_value = calendar
    base.snapshot.side_effect = lambda as_of: _make_snapshot(as_of, feature_val=feature_val)
    return base


# ---------------------------------------------------------------------------
# _parse_inject_at
# ---------------------------------------------------------------------------

def test_parse_inject_at_iso_date():
    cal = _make_calendar("2024-01-01", 10)
    ts = _parse_inject_at("2024-01-05", cal)
    assert ts is not None
    assert ts.date().isoformat() == "2024-01-05"


def test_parse_inject_at_none_returns_none():
    cal = _make_calendar("2024-01-01", 10)
    assert _parse_inject_at(None, cal) is None


def test_parse_inject_at_event_marker_returns_none():
    cal = _make_calendar("2024-01-01", 10)
    assert _parse_inject_at("portfolio_rebalance", cal) is None


# ---------------------------------------------------------------------------
# Shock mutation
# ---------------------------------------------------------------------------

def test_shock_scales_panel_features():
    cal = _make_calendar("2024-01-01", 5)
    spec = SyntheticSpec(mutation="shock", magnitude=-0.10, duration_days=5)
    source = StressDataSource(_make_base(cal, feature_val=10.0), spec)
    snap = source.snapshot(cal[0])
    for symbol, features in snap["panel"].items():
        for feat, val in features.items():
            assert val == pytest.approx(10.0 * 0.9)


def test_shock_does_not_mutate_outside_window():
    cal = _make_calendar("2024-01-01", 10)
    # Inject on day 5 for 1 day only — days 0-4 and 6+ should be unmutated.
    spec = SyntheticSpec(mutation="shock", magnitude=-0.50, duration_days=1, inject_at="2024-01-06")
    source = StressDataSource(_make_base(cal, feature_val=10.0), spec)
    snap_day0 = source.snapshot(cal[0])
    assert snap_day0["panel"]["AAA"]["close_norm"] == pytest.approx(10.0)


def test_shock_mutates_inside_window():
    cal = _make_calendar("2024-01-01", 10)
    spec = SyntheticSpec(mutation="shock", magnitude=-0.50, duration_days=3, inject_at="2024-01-03")
    source = StressDataSource(_make_base(cal, feature_val=10.0), spec)
    snap = source.snapshot(_ts("2024-01-03"))
    assert snap["panel"]["AAA"]["close_norm"] == pytest.approx(5.0)


def test_shock_does_not_change_regime_features():
    cal = _make_calendar("2024-01-01", 5)
    spec = SyntheticSpec(mutation="shock", magnitude=-0.50, duration_days=5)
    base = MagicMock()
    base.calendar.return_value = cal
    base.snapshot.side_effect = lambda as_of: _make_snapshot(
        as_of, regime_features=(0.05,), regime_feature_names=("market_vol_20d",)
    )
    source = StressDataSource(base, spec)
    snap = source.snapshot(cal[0])
    # Shock only touches panel, not regime features
    assert snap["regime_features"] == (0.05,)


# ---------------------------------------------------------------------------
# Drift mutation
# ---------------------------------------------------------------------------

def test_drift_compounds_daily():
    cal = _make_calendar("2024-01-01", 5)
    spec = SyntheticSpec(mutation="drift", magnitude=0.10, duration_days=5)
    source = StressDataSource(_make_base(cal, feature_val=10.0), spec)
    # Day 0: (1.1)^0 = 1.0 → 10.0
    snap0 = source.snapshot(cal[0])
    assert snap0["panel"]["AAA"]["close_norm"] == pytest.approx(10.0)
    # Day 1: (1.1)^1 = 1.1 → 11.0
    snap1 = source.snapshot(cal[1])
    assert snap1["panel"]["AAA"]["close_norm"] == pytest.approx(11.0)
    # Day 3: (1.1)^3 → 10 * 1.331
    snap3 = source.snapshot(cal[3])
    assert snap3["panel"]["AAA"]["close_norm"] == pytest.approx(10.0 * 1.1**3)


# ---------------------------------------------------------------------------
# Volatility spike mutation
# ---------------------------------------------------------------------------

def test_vol_spike_scales_vol_feature():
    cal = _make_calendar("2024-01-01", 5)
    spec = SyntheticSpec(mutation="volatility_spike", factor=2.0, duration_days=5)
    base = MagicMock()
    base.calendar.return_value = cal
    base.snapshot.side_effect = lambda as_of: _make_snapshot(
        as_of, feature_val=10.0, regime_features=(0.02,), regime_feature_names=("market_vol_20d",)
    )
    source = StressDataSource(base, spec)
    snap = source.snapshot(cal[0])
    assert snap["regime_features"] == pytest.approx((0.04,))


def test_vol_spike_leaves_panel_unchanged():
    cal = _make_calendar("2024-01-01", 5)
    spec = SyntheticSpec(mutation="volatility_spike", factor=3.0, duration_days=5)
    source = StressDataSource(_make_base(cal, feature_val=10.0), spec)
    snap = source.snapshot(cal[0])
    assert snap["panel"]["AAA"]["close_norm"] == pytest.approx(10.0)


def test_vol_spike_non_vol_regime_features_unchanged():
    cal = _make_calendar("2024-01-01", 5)
    spec = SyntheticSpec(mutation="volatility_spike", factor=2.0, duration_days=5)
    base = MagicMock()
    base.calendar.return_value = cal
    base.snapshot.side_effect = lambda as_of: _make_snapshot(
        as_of,
        regime_features=(0.02, 0.5),
        regime_feature_names=("market_vol_20d", "market_trend_20d"),
    )
    source = StressDataSource(base, spec)
    snap = source.snapshot(cal[0])
    # vol scaled, trend unchanged
    assert snap["regime_features"][0] == pytest.approx(0.04)
    assert snap["regime_features"][1] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Calendar passthrough
# ---------------------------------------------------------------------------

def test_calendar_passthrough():
    cal = _make_calendar("2024-01-01", 7)
    spec = SyntheticSpec(mutation="shock", magnitude=-0.1, duration_days=3)
    source = StressDataSource(_make_base(cal), spec)
    assert source.calendar() == cal


# ---------------------------------------------------------------------------
# Unknown mutation falls through safely
# ---------------------------------------------------------------------------

def test_unknown_mutation_returns_unmodified_snapshot():
    cal = _make_calendar("2024-01-01", 3)
    # Bypass the Literal type by constructing directly via object.__setattr__
    spec = SyntheticSpec.__new__(SyntheticSpec)
    object.__setattr__(spec, "mutation", "unknown_future_mutation")
    object.__setattr__(spec, "magnitude", None)
    object.__setattr__(spec, "factor", None)
    object.__setattr__(spec, "duration_days", 3)
    object.__setattr__(spec, "inject_at", None)
    source = StressDataSource(_make_base(cal, feature_val=10.0), spec)
    snap = source.snapshot(cal[0])
    assert snap["panel"]["AAA"]["close_norm"] == pytest.approx(10.0)
