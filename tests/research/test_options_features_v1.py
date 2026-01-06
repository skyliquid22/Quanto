from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd

from research.features.options_features_v1 import OPTION_FEATURE_COLUMNS, compute_options_features


UTC = timezone.utc


def test_options_features_are_deterministic_and_fill_missing_days():
    equity_rows = [
        {"timestamp": datetime(2023, 1, 2, tzinfo=UTC), "close": 150.0, "sma_fast": 149.0, "sma_slow": 145.0},
        {"timestamp": datetime(2023, 1, 3, tzinfo=UTC), "close": 152.0, "sma_fast": 150.0, "sma_slow": 146.0},
        {"timestamp": datetime(2023, 1, 4, tzinfo=UTC), "close": 151.0, "sma_fast": 148.5, "sma_slow": 146.5},
    ]
    equity_df = pd.DataFrame(equity_rows)

    reference_df = pd.DataFrame(
        [
            {"option_symbol": "AAPL1", "underlying_symbol": "AAPL", "option_type": "call", "strike": 150},
            {"option_symbol": "AAPL2", "underlying_symbol": "AAPL", "option_type": "call", "strike": 155},
            {"option_symbol": "AAPL3", "underlying_symbol": "AAPL", "option_type": "put", "strike": 150},
            {"option_symbol": "AAPL4", "underlying_symbol": "AAPL", "option_type": "put", "strike": 145},
        ]
    )

    open_interest_df = pd.DataFrame(
        [
            {"timestamp": datetime(2023, 1, 2, tzinfo=UTC), "option_symbol": "AAPL1", "open_interest": 100},
            {"timestamp": datetime(2023, 1, 2, tzinfo=UTC), "option_symbol": "AAPL2", "open_interest": 80},
            {"timestamp": datetime(2023, 1, 2, tzinfo=UTC), "option_symbol": "AAPL3", "open_interest": 120},
            {"timestamp": datetime(2023, 1, 2, tzinfo=UTC), "option_symbol": "AAPL4", "open_interest": 60},
            {"timestamp": datetime(2023, 1, 3, tzinfo=UTC), "option_symbol": "AAPL1", "open_interest": 90},
            {"timestamp": datetime(2023, 1, 3, tzinfo=UTC), "option_symbol": "AAPL2", "open_interest": 70},
            {"timestamp": datetime(2023, 1, 3, tzinfo=UTC), "option_symbol": "AAPL3", "open_interest": 130},
            {"timestamp": datetime(2023, 1, 3, tzinfo=UTC), "option_symbol": "AAPL4", "open_interest": 50},
        ]
    )

    ohlcv_df = pd.DataFrame(
        [
            {"timestamp": datetime(2023, 1, 2, tzinfo=UTC), "option_symbol": "AAPL1", "volume": 10},
            {"timestamp": datetime(2023, 1, 2, tzinfo=UTC), "option_symbol": "AAPL3", "volume": 20},
            {"timestamp": datetime(2023, 1, 3, tzinfo=UTC), "option_symbol": "AAPL2", "volume": 30},
            {"timestamp": datetime(2023, 1, 3, tzinfo=UTC), "option_symbol": "AAPL4", "volume": 40},
        ]
    )

    features = compute_options_features(equity_df, reference_df, open_interest_df, ohlcv_df)

    assert list(features.columns[-len(OPTION_FEATURE_COLUMNS) :]) == list(OPTION_FEATURE_COLUMNS)

    day1 = features.iloc[0]
    assert math.isclose(day1["oi_total"], 360.0, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(day1["oi_call_total"], 180.0, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(day1["oi_put_total"], 180.0, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(day1["oi_put_call_ratio"], 1.0, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(day1["options_volume_total"], 30.0, rel_tol=0, abs_tol=1e-12)

    day2 = features.iloc[1]
    assert math.isclose(day2["oi_total"], 340.0, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(day2["oi_change_1d"], -20.0, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(day2["oi_change_pct_1d"], -20.0 / (360.0 + 1e-9), rel_tol=0, abs_tol=1e-12)
    assert math.isclose(day2["options_volume_total"], 70.0, rel_tol=0, abs_tol=1e-12)

    day3 = features.iloc[2]
    assert math.isclose(day3["oi_total"], day2["oi_total"], rel_tol=0, abs_tol=1e-12)
    assert math.isclose(day3["oi_change_1d"], 0.0, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(day3["options_volume_total"], 0.0, rel_tol=0, abs_tol=1e-12)
