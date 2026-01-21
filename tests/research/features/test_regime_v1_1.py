from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from research.datasets.canonical_equity_loader import CanonicalEquitySlice
from research.features import feature_registry
from research.features.feature_registry import FeatureSetResult, SMA_OBSERVATION_COLUMNS, build_universe_feature_panel
from research.features.regime_features_v1 import (
    PRIMARY_REGIME_UNIVERSE,
    REGIME_FEATURE_COLUMNS,
    compute_primary_regime_features,
    compute_regime_features,
)
from research.regime import RegimeState


UTC = timezone.utc


def _make_feature_result(symbol: str, closes: list[float]) -> FeatureSetResult:
    timestamps = pd.date_range(datetime(2023, 1, 1, tzinfo=UTC), periods=len(closes), freq="D", tz=UTC)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": closes,
            "sma_fast": closes,
            "sma_slow": closes,
            "sma_diff": [0.0 for _ in closes],
            "sma_signal": [0.0 for _ in closes],
        }
    )
    return FeatureSetResult(
        frame=frame,
        observation_columns=SMA_OBSERVATION_COLUMNS,
        feature_set="sma_universe_v1",
        inputs_used={},
    )


def _make_primary_slice(symbol: str, closes: list[float]) -> CanonicalEquitySlice:
    timestamps = pd.date_range(datetime(2023, 1, 1, tzinfo=UTC), periods=len(closes), freq="D", tz=UTC)
    frame = pd.DataFrame(
        {
            "symbol": symbol,
            "open": closes,
            "high": [value * 1.01 for value in closes],
            "low": [value * 0.99 for value in closes],
            "close": closes,
            "volume": [1_000_000 for _ in closes],
        },
        index=timestamps,
    )
    frame.index.name = "timestamp"
    return CanonicalEquitySlice(symbol=symbol, frame=frame, file_paths=[])


def test_regime_v1_1_uses_primary_universe(monkeypatch):
    primary_closes = {
        "SPY": [100.0, 101.0, 102.0, 103.0, 104.0],
        "QQQ": [200.0, 198.0, 202.0, 205.0, 207.0],
        "IWM": [50.0, 51.5, 52.0, 51.0, 53.0],
    }
    primary_slices = {
        symbol: _make_primary_slice(symbol, closes) for symbol, closes in primary_closes.items()
    }

    def fake_load_primary(start_date, end_date, *, data_root=None, interval="daily"):
        assert interval == "daily"
        assert start_date <= end_date
        return dict(primary_slices), {}

    monkeypatch.setattr(feature_registry, "load_primary_regime_universe", fake_load_primary)

    feature_map = {
        "AAA": _make_feature_result("AAA", [10.0, 10.5, 11.0, 11.5, 12.0]),
        "BBB": _make_feature_result("BBB", [20.0, 19.5, 19.0, 18.5, 18.0]),
    }
    calendar = feature_map["AAA"].frame["timestamp"]
    panel = build_universe_feature_panel(
        feature_map,
        symbol_order=("AAA", "BBB"),
        calendar=calendar,
        regime_feature_set="regime_v1_1",
    )

    expected_close_panel = pd.DataFrame(index=pd.DatetimeIndex(calendar))
    for symbol in PRIMARY_REGIME_UNIVERSE:
        expected_close_panel[symbol] = primary_slices[symbol].frame["close"].astype(float)
    expected = compute_primary_regime_features(expected_close_panel)
    experiment_close_panel = pd.DataFrame(
        {
            "AAA": feature_map["AAA"].frame["close"].astype(float).to_numpy(),
            "BBB": feature_map["BBB"].frame["close"].astype(float).to_numpy(),
        },
        index=pd.DatetimeIndex(calendar),
    )
    experiment_expected = compute_regime_features(experiment_close_panel)

    assert panel.observation_columns == SMA_OBSERVATION_COLUMNS + REGIME_FEATURE_COLUMNS
    for idx, row in enumerate(panel.rows):
        state = row.get("regime_state")
        assert isinstance(state, RegimeState)
        assert state.feature_names == REGIME_FEATURE_COLUMNS
        values = [float(value) for value in state.features]
        expected_values = [float(expected.iloc[idx][column]) for column in REGIME_FEATURE_COLUMNS]
        assert values == expected_values
        if idx == len(panel.rows) - 1:
            experiment_values = [
                float(experiment_expected.iloc[idx][column]) for column in REGIME_FEATURE_COLUMNS
            ]
            assert values != experiment_values


def test_regime_v1_1_missing_symbol_raises(monkeypatch):
    primary_slices = {
        "SPY": _make_primary_slice("SPY", [100.0, 101.0, 102.0]),
        "QQQ": _make_primary_slice("QQQ", [200.0, 201.0, 202.0]),
    }

    def fake_load_primary(start_date, end_date, *, data_root=None, interval="daily"):
        return dict(primary_slices), {}

    monkeypatch.setattr(feature_registry, "load_primary_regime_universe", fake_load_primary)

    feature_map = {
        "AAA": _make_feature_result("AAA", [10.0, 10.5, 11.0]),
        "BBB": _make_feature_result("BBB", [20.0, 19.5, 19.0]),
    }
    calendar = feature_map["AAA"].frame["timestamp"]

    with pytest.raises(ValueError, match="Primary regime universe missing canonical data"):
        build_universe_feature_panel(
            feature_map,
            symbol_order=("AAA", "BBB"),
            calendar=calendar,
            regime_feature_set="regime_v1_1",
        )
