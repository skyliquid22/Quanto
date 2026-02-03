import pandas as pd

from research.datasets.canonical_fundamentals_loader import CanonicalFundamentalsSlice
from research.features.feature_registry import (
    FUNDAMENTALS_V1_COLUMNS,
    build_features,
    observation_columns_for_feature_set,
)


def _equity_frame():
    timestamps = pd.to_datetime(["2025-02-14", "2025-02-16"], utc=True)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": [100.0, 101.0],
            "volume": [1000.0, 1100.0],
        }
    )


def test_core_v1_fundamentals_v1_columns(monkeypatch):
    def fake_load(symbols, start, end, *, data_root=None, lookback_days=400):
        empty = CanonicalFundamentalsSlice(symbol="AAA", frame=pd.DataFrame(), file_paths=[])
        return {"AAA": empty}, {}

    monkeypatch.setattr(
        "research.features.sets.fundamentals_v1.load_canonical_fundamentals",
        fake_load,
    )

    result = build_features(
        "core_v1_fundamentals_v1",
        _equity_frame(),
        underlying_symbol="AAA",
        start_date="2025-02-14",
        end_date="2025-02-16",
    )

    for column in FUNDAMENTALS_V1_COLUMNS:
        assert column in result.observation_columns
    expected_prefix = ["timestamp", *result.observation_columns]
    assert list(result.frame.columns[: len(expected_prefix)]) == expected_prefix


def test_observation_columns_for_new_feature_sets():
    for name in (
        "core_v1_fundamentals_v1",
        "core_v1_fundamentals_regime_v1",
        "core_v1_fundamentals_regime_xsec_v1",
        "core_v1_fundamentals_regime_opts_v1",
    ):
        columns = observation_columns_for_feature_set(name)
        for column in FUNDAMENTALS_V1_COLUMNS:
            assert column in columns
