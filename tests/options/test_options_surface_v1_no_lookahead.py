from __future__ import annotations

import pandas as pd

from research.datasets.options_surface_loader import OptionsSurfaceSlice
from research.features.sets.opts_surface_v1 import attach_surface_columns


def test_options_surface_v1_no_lookahead_behavior():
    timestamps = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    base_frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": [100.0, 101.0, 102.0],
            "ret_1": [0.0, 0.01, 0.011],
        }
    )
    surface_frame = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "date": ["2024-01-02", "2024-01-05"],
            "OPT:OI:CALL": [1000.0, 1500.0],
            "OPT:COVERAGE:HAS_OI": [True, True],
            "OPT:COVERAGE:ROW_VALID": [True, True],
        }
    )
    slice_data = OptionsSurfaceSlice(symbol="AAA", frame=surface_frame, file_paths=[])
    enriched = attach_surface_columns(base_frame, slice_data)
    assert "OPT:OI:CALL" in enriched.columns
    assert "OPT:COVERAGE:HAS_OI" in enriched.columns

    row_jan2 = enriched.loc[enriched["timestamp"] == timestamps[1]]
    row_jan3 = enriched.loc[enriched["timestamp"] == timestamps[2]]
    row_jan1 = enriched.loc[enriched["timestamp"] == timestamps[0]]

    assert row_jan2["OPT:OI:CALL"].iloc[0] == 1000.0
    assert pd.isna(row_jan3["OPT:OI:CALL"].iloc[0])
    assert pd.isna(row_jan1["OPT:OI:CALL"].iloc[0])

    assert bool(row_jan3["OPT:COVERAGE:HAS_OI"].iloc[0]) is False
    assert bool(row_jan1["OPT:COVERAGE:HAS_OI"].iloc[0]) is False
