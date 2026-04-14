"""Synthetic mutation wrapper for stress test scenarios.

StressDataSource wraps a ReplayMarketDataSource and intercepts snapshot()
calls to apply feature-level mutations during a defined window. Mutations
operate on the pre-materialized feature panel, which is an approximation —
SMA values are not re-derived from mutated prices. This is acceptable for
stress testing purposes (probing robustness, not simulating exact prices).

Supported mutations:
  shock             — multiply all panel features by (1 + magnitude) for each
                      day in the mutation window.
  drift             — compound magnitude daily: day k applies ×(1+magnitude)^k
                      relative to the injection start.
  volatility_spike  — multiply regime features (market_vol_20d) by factor;
                      per-symbol panel features are unchanged.

inject_at behaviour:
  ISO date string   — mutation starts on that calendar date.
  Event marker      — any non-date string (e.g. "portfolio_rebalance") is
                      treated as day 0 (start of the replay window). This is
                      a deliberate simplification; lifecycle-hook injection is
                      left for a future iteration.
  None              — injection starts at day 0.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import pandas as pd

from research.shadow.data_source import MarketDataSource, ReplayMarketDataSource
from research.stress.config import SyntheticSpec

_VOL_REGIME_FEATURES = {"market_vol_20d", "market_vol_5d", "market_vol_60d"}


def _parse_inject_at(inject_at: str | None, calendar: list[pd.Timestamp]) -> pd.Timestamp | None:
    """Resolve inject_at to a calendar timestamp, or None for day-0 injection."""
    if inject_at is None:
        return None
    try:
        dt = datetime.fromisoformat(inject_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return pd.Timestamp(dt)
    except ValueError:
        # Non-date event marker — inject at start of calendar.
        return None


def _day_offset(as_of: pd.Timestamp, inject_start: pd.Timestamp) -> int:
    """Number of calendar days since injection start (0-based)."""
    delta = (as_of.normalize() - inject_start.normalize()).days
    return max(0, delta)


class StressDataSource(MarketDataSource):
    """MarketDataSource that applies a synthetic mutation during a date window.

    Args:
        base: The underlying ReplayMarketDataSource supplying real market data.
        spec: SyntheticSpec describing the mutation type, parameters, and window.
    """

    def __init__(self, base: ReplayMarketDataSource, spec: SyntheticSpec) -> None:
        self._base = base
        self._spec = spec
        cal = base.calendar()
        self._inject_start: pd.Timestamp = _parse_inject_at(spec.inject_at, cal) or (
            cal[0] if cal else pd.Timestamp("1970-01-01", tz="UTC")
        )
        # Normalise to UTC midnight for day-boundary comparisons.
        if self._inject_start.tzinfo is None:
            self._inject_start = self._inject_start.tz_localize("UTC")
        else:
            self._inject_start = self._inject_start.tz_convert("UTC")
        self._inject_end: pd.Timestamp = self._inject_start + pd.Timedelta(days=spec.duration_days - 1)

    def calendar(self) -> list[pd.Timestamp]:
        return self._base.calendar()

    def snapshot(self, as_of: pd.Timestamp) -> dict[str, Any]:
        snap = self._base.snapshot(as_of)
        if self._in_window(as_of):
            snap = self._apply_mutation(snap, as_of)
        return snap

    def _in_window(self, as_of: pd.Timestamp) -> bool:
        ts = as_of
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return self._inject_start.normalize() <= ts.normalize() <= self._inject_end.normalize()

    def _apply_mutation(self, snap: dict[str, Any], as_of: pd.Timestamp) -> dict[str, Any]:
        mutation = self._spec.mutation
        if mutation == "shock":
            return self._apply_shock(snap)
        if mutation == "drift":
            return self._apply_drift(snap, as_of)
        if mutation == "volatility_spike":
            return self._apply_vol_spike(snap)
        return snap

    def _apply_shock(self, snap: dict[str, Any]) -> dict[str, Any]:
        """Scale all per-symbol panel features by (1 + magnitude)."""
        magnitude = float(self._spec.magnitude or 0.0)
        factor = 1.0 + magnitude
        mutated_panel = {
            symbol: {feat: val * factor for feat, val in features.items()}
            for symbol, features in snap["panel"].items()
        }
        return {**snap, "panel": mutated_panel}

    def _apply_drift(self, snap: dict[str, Any], as_of: pd.Timestamp) -> dict[str, Any]:
        """Compound daily drift from injection start: factor = (1+magnitude)^day_offset."""
        magnitude = float(self._spec.magnitude or 0.0)
        offset = _day_offset(as_of, self._inject_start)
        factor = (1.0 + magnitude) ** offset
        mutated_panel = {
            symbol: {feat: val * factor for feat, val in features.items()}
            for symbol, features in snap["panel"].items()
        }
        return {**snap, "panel": mutated_panel}

    def _apply_vol_spike(self, snap: dict[str, Any]) -> dict[str, Any]:
        """Multiply regime vol features by factor; leave per-symbol panel unchanged."""
        factor = float(self._spec.factor or 1.0)
        feature_names = snap.get("regime_feature_names", ())
        regime_features = snap.get("regime_features", ())
        if not regime_features or not feature_names:
            return snap
        mutated_regime = tuple(
            val * factor if name in _VOL_REGIME_FEATURES else val
            for name, val in zip(feature_names, regime_features)
        )
        return {**snap, "regime_features": mutated_regime}


__all__ = ["StressDataSource"]
