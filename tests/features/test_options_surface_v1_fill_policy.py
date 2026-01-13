import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_bool_dtype

from research.datasets.options_surface_loader import OptionsSurfaceSlice
from research.features.feature_registry import build_features
from research.features.sets.options_surface_v1 import OPTIONS_SURFACE_V1_COLUMNS


@pytest.fixture()
def equity_frame():
    timestamps = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": np.linspace(100.0, 104.0, num=len(timestamps)),
        }
    )


@pytest.fixture()
def surface_slice():
    frame = pd.DataFrame(
        {
            "symbol": ["AAA"] * 3,
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04"], utc=True),
            "OPT:OI:CALL": [10.0, 6.0, np.nan],
            "OPT:OI:PUT": [0.1, 3.0, np.nan],
            "OPT:OI:TOTAL": [11.0, np.nan, 25.0],
            "OPT:OI:CALL_PUT_RATIO": [25.0, np.nan, np.nan],
            "OPT:VOL:CALL": [120.0, 80.0, np.nan],
            "OPT:VOL:PUT": [10.0, 5.0, np.nan],
            "OPT:VOL:TOTAL": [np.nan, 60.0, np.nan],
            "OPT:VOL:CALL_PUT_RATIO": [np.nan, np.nan, np.nan],
            "OPT:IVX:30": [np.nan, 22.0, 25.0],
            "OPT:IVX:90": [np.nan, 35.0, np.nan],
            "OPT:IVX:180": [np.nan, np.nan, 40.0],
            "OPT:IVR:30": [np.nan, 65.0, np.nan],
            "OPT:IVR:90": [np.nan, np.nan, 70.0],
            "OPT:COVERAGE:ROW_VALID": [True, True, True],
            "OPT:COVERAGE:HAS_OI": [True, True, True],
            "OPT:COVERAGE:HAS_OPT_VOLUME": [True, True, True],
            "OPT:COVERAGE:HAS_IVX": [True, True, True],
            "OPT:COVERAGE:HAS_IVR": [True, True, True],
        }
    )
    return OptionsSurfaceSlice(symbol="AAA", frame=frame, file_paths=[])


def test_options_surface_v1_fill_policy(monkeypatch, equity_frame, surface_slice):
    def fake_load(symbols, start, end, *, data_root=None):
        assert symbols == ["AAA"]
        return {"AAA": OptionsSurfaceSlice(symbol="AAA", frame=surface_slice.frame.copy(), file_paths=[])}, {}

    monkeypatch.setattr(
        "research.features.sets.options_surface_v1.load_options_surface",
        fake_load,
    )

    result = build_features(
        "options_surface_v1",
        equity_frame,
        underlying_symbol="AAA",
        start_date="2024-01-01",
        end_date="2024-01-05",
    )

    assert result.observation_columns == OPTIONS_SURFACE_V1_COLUMNS
    expected_columns = ["timestamp", *OPTIONS_SURFACE_V1_COLUMNS]
    assert list(result.frame.columns) == expected_columns
    coverage_columns = [
        "OPT:COVERAGE:ROW_VALID",
        "OPT:COVERAGE:HAS_OI",
        "OPT:COVERAGE:HAS_OPT_VOLUME",
        "OPT:COVERAGE:HAS_IVX",
        "OPT:COVERAGE:HAS_IVR",
    ]
    for column in coverage_columns:
        assert is_bool_dtype(result.frame[column])

    valid_mask = result.frame["OPT:COVERAGE:ROW_VALID"]
    numeric_columns = OPTIONS_SURFACE_V1_COLUMNS[:9]
    for column in numeric_columns:
        assert result.frame.loc[valid_mask, column].isna().sum() == 0

    row = result.frame.iloc[0]
    assert row["OPT:OI:CALL_PUT_RATIO"] == pytest.approx(20.0)
    assert row["OPT:VOL:TOTAL"] == 0.0

    # Last row (no surface data) should mark coverage flags False.
    assert not result.frame.loc[result.frame.index[-1], "OPT:COVERAGE:ROW_VALID"]
    assert result.frame["OPT:VOL:CALL_PUT_RATIO"].isna().iloc[-1]
