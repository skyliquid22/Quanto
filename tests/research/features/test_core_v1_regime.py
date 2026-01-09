from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from research.datasets.canonical_equity_loader import CanonicalEquitySlice
from research.features.core_features_v1 import CORE_V1_OBSERVATION_COLUMNS
from research.features.feature_eng import build_universe_feature_results
from research.features.feature_registry import build_universe_feature_panel, normalize_feature_set_name
from research.features.regime_features_v1 import REGIME_FEATURE_COLUMNS
from research.regime import RegimeState


UTC = timezone.utc


def _make_slice(symbol: str, base_close: float) -> CanonicalEquitySlice:
    timestamps = pd.date_range(datetime(2023, 1, 1, tzinfo=UTC), periods=80, freq="D", tz=UTC)
    closes = [base_close + idx * 0.5 for idx in range(len(timestamps))]
    volumes = [1_000_000 + idx * 100 for idx in range(len(timestamps))]
    frame = pd.DataFrame(
        {
            "symbol": symbol,
            "open": closes,
            "high": [value * 1.01 for value in closes],
            "low": [value * 0.99 for value in closes],
            "close": closes,
            "volume": volumes,
        },
        index=timestamps,
    )
    frame.index.name = "timestamp"
    return CanonicalEquitySlice(symbol=symbol, frame=frame, file_paths=[])


def test_core_v1_regime_panel_includes_row_level_regime_state():
    symbols = ("AAA", "BBB", "CCC")
    slices = {
        "AAA": _make_slice("AAA", 100.0),
        "BBB": _make_slice("BBB", 50.0),
        "CCC": _make_slice("CCC", 75.0),
    }
    start = slices["AAA"].frame.index[0].date()
    end = slices["AAA"].frame.index[-1].date()
    feature_map = build_universe_feature_results(
        "core_v1_regime",
        slices,
        symbol_order=symbols,
        start_date=start,
        end_date=end,
    )
    for symbol in symbols:
        assert feature_map[symbol].feature_set == "core_v1_regime"
        assert feature_map[symbol].observation_columns == CORE_V1_OBSERVATION_COLUMNS

    calendar = slices["AAA"].frame.index
    panel = build_universe_feature_panel(
        feature_map,
        symbol_order=symbols,
        calendar=calendar,
        regime_feature_set="regime_v1",
    )
    assert panel.symbol_order == symbols
    assert panel.observation_columns == CORE_V1_OBSERVATION_COLUMNS + REGIME_FEATURE_COLUMNS
    row = panel.rows[10]
    assert "regime_state" in row
    assert isinstance(row["regime_state"], RegimeState)
    for symbol in symbols:
        per_symbol = row["panel"][symbol]
        assert all(column in per_symbol for column in CORE_V1_OBSERVATION_COLUMNS)
        for regime_column in REGIME_FEATURE_COLUMNS:
            assert regime_column not in per_symbol


def test_core_v1_regime_regime_state_deterministic():
    symbols = ("AAA", "BBB")
    slices = {
        "AAA": _make_slice("AAA", 120.0),
        "BBB": _make_slice("BBB", 80.0),
    }
    start = slices["AAA"].frame.index[0].date()
    end = slices["AAA"].frame.index[-1].date()
    feature_map = build_universe_feature_results(
        "core_v1_regime",
        slices,
        symbol_order=symbols,
        start_date=start,
        end_date=end,
    )
    calendar = slices["AAA"].frame.index
    panel_one = build_universe_feature_panel(
        feature_map,
        symbol_order=symbols,
        calendar=calendar,
        regime_feature_set="regime_v1",
    )
    panel_two = build_universe_feature_panel(
        feature_map,
        symbol_order=symbols,
        calendar=calendar,
        regime_feature_set="regime_v1",
    )

    def _extract_regime_series(panel_rows):
        series = []
        for row in panel_rows:
            state = row.get("regime_state")
            assert isinstance(state, RegimeState)
            series.append(tuple(float(value) for value in state.features))
        return series

    assert _extract_regime_series(panel_one.rows) == _extract_regime_series(panel_two.rows)
