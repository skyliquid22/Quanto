from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from research.datasets.canonical_equity_loader import CanonicalEquitySlice
from research.features.feature_eng import build_universe_feature_results
from research.features.feature_registry import (
    CORE_V1_XSEC_OBSERVATION_COLUMNS,
    build_universe_feature_panel,
)
from research.features.regime_features_v1 import REGIME_FEATURE_COLUMNS
from research.regime import RegimeState


UTC = timezone.utc


def _make_slice(symbol: str, base_close: float, trend: float) -> CanonicalEquitySlice:
    timestamps = pd.date_range(datetime(2023, 1, 1, tzinfo=UTC), periods=90, freq="D", tz=UTC)
    closes = [base_close + idx * trend for idx in range(len(timestamps))]
    volume = [1_000_000 + (idx * 500) + int(base_close * 10) for idx in range(len(timestamps))]
    frame = pd.DataFrame(
        {
            "symbol": symbol,
            "open": [price * 0.995 for price in closes],
            "high": [price * 1.01 for price in closes],
            "low": [price * 0.99 for price in closes],
            "close": closes,
            "volume": volume,
        },
        index=timestamps,
    )
    frame.index.name = "timestamp"
    return CanonicalEquitySlice(symbol=symbol, frame=frame, file_paths=[])


def _serialize_panel_rows(panel):
    payload = []
    for row in panel.rows:
        state = row.get("regime_state")
        assert isinstance(state, RegimeState)
        payload.append(
            {
                "timestamp": row["timestamp"],
                "panel": {
                    symbol: {column: row["panel"][symbol][column] for column in CORE_V1_XSEC_OBSERVATION_COLUMNS}
                    for symbol in panel.symbol_order
                },
                "regime": tuple(float(value) for value in state.features),
            }
        )
    return payload


def test_core_v1_xsec_regime_panel_contract():
    symbols = ("AAA", "BBB", "CCC")
    slices = {
        "AAA": _make_slice("AAA", 100.0, 0.8),
        "BBB": _make_slice("BBB", 60.0, 0.5),
        "CCC": _make_slice("CCC", 140.0, 1.1),
    }
    start = slices["AAA"].frame.index[0].date()
    end = slices["AAA"].frame.index[-1].date()

    feature_map = build_universe_feature_results(
        "core_v1_xsec_regime",
        slices,
        symbol_order=symbols,
        start_date=start,
        end_date=end,
    )
    for symbol in symbols:
        assert feature_map[symbol].feature_set == "core_v1_xsec_regime"
        assert feature_map[symbol].observation_columns == CORE_V1_XSEC_OBSERVATION_COLUMNS

    calendar = slices["AAA"].frame.index
    panel = build_universe_feature_panel(
        feature_map,
        symbol_order=symbols,
        calendar=calendar,
        regime_feature_set="regime_v1",
    )
    assert panel.symbol_order == symbols
    assert panel.observation_columns == CORE_V1_XSEC_OBSERVATION_COLUMNS + REGIME_FEATURE_COLUMNS
    row = panel.rows[40]
    assert "regime_state" in row
    assert isinstance(row["regime_state"], RegimeState)
    assert row["regime_state"].feature_names == REGIME_FEATURE_COLUMNS
    for symbol in symbols:
        per_symbol = row["panel"][symbol]
        assert all(column in per_symbol for column in CORE_V1_XSEC_OBSERVATION_COLUMNS)
    # Cross-sectional metrics should differ across symbols after warmup
    metric_name = "ret_1d_rank"
    values = [row["panel"][symbol][metric_name] for symbol in symbols]
    assert len(set(round(value, 6) for value in values)) > 1


def test_core_v1_xsec_regime_determinism_and_no_lookahead():
    symbols = ("AAA", "BBB", "CCC")
    slices = {
        "AAA": _make_slice("AAA", 90.0, 0.7),
        "BBB": _make_slice("BBB", 70.0, 0.4),
        "CCC": _make_slice("CCC", 110.0, 1.2),
    }
    start = slices["AAA"].frame.index[0].date()
    end = slices["AAA"].frame.index[-1].date()
    feature_map = build_universe_feature_results(
        "core_v1_xsec_regime",
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
    assert _serialize_panel_rows(panel_one) == _serialize_panel_rows(panel_two)

    mutated_slices = {
        symbol: CanonicalEquitySlice(symbol=symbol, frame=slice_data.frame.copy(deep=True), file_paths=[])
        for symbol, slice_data in slices.items()
    }
    last_idx = mutated_slices["AAA"].frame.index[-1]
    mutated_slices["AAA"].frame.loc[last_idx, "close"] *= 1.05
    mutated_slices["AAA"].frame.loc[last_idx, "open"] *= 1.05
    mutated_slices["AAA"].frame.loc[last_idx, "high"] *= 1.05
    mutated_slices["AAA"].frame.loc[last_idx, "low"] *= 1.05

    mutated_features = build_universe_feature_results(
        "core_v1_xsec_regime",
        mutated_slices,
        symbol_order=symbols,
        start_date=start,
        end_date=end,
    )
    mutated_panel = build_universe_feature_panel(
        mutated_features,
        symbol_order=symbols,
        calendar=calendar,
        regime_feature_set="regime_v1",
    )

    base_regime_series = [tuple(row["regime_state"].features.tolist()) for row in panel_one.rows[:-1]]
    mutated_series = [tuple(row["regime_state"].features.tolist()) for row in mutated_panel.rows[:-1]]
    assert base_regime_series == mutated_series
