import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_bool_dtype

from research.datasets.canonical_fundamentals_loader import CanonicalFundamentalsSlice
from research.features.feature_registry import build_features


def _fundamentals_frame():
    return pd.DataFrame(
        {
            "report_date": [
                "2023-12-31",
                "2024-03-31",
                "2024-06-30",
                "2024-09-30",
                "2024-12-31",
            ],
            "filing_date": [
                "2024-02-15",
                "2024-05-10",
                "2024-08-10",
                "2024-11-05",
                "2025-02-15",
            ],
            "period": ["quarterly"] * 5,
            "fiscal_period": ["Q4", "Q1", "Q2", "Q3", "Q4"],
            "revenue": [10.0, 11.0, 12.0, 13.0, 14.0],
            "net_income": [1.0, 1.1, 1.2, 1.3, 1.4],
            "operating_income": [2.0, 2.1, 2.2, 2.3, 2.4],
            "free_cash_flow": [3.0, 3.1, 3.2, 3.3, 3.4],
            "eps": [0.1, 0.11, 0.12, 0.13, 0.14],
            "total_assets": [100.0, 101.0, 102.0, 103.0, 104.0],
            "total_liabilities": [40.0, 41.0, 42.0, 43.0, 44.0],
            "shareholder_equity": [60.0, 60.0, 60.0, 60.0, 60.0],
            "shares_outstanding": [1000.0] * 5,
        }
    )


def _equity_frame():
    timestamps = pd.to_datetime(
        ["2025-02-14", "2025-02-16", "2025-09-01"],
        utc=True,
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": [100.0, 101.0, 102.0],
            "volume": [1000.0, 1100.0, 1200.0],
        }
    )


def test_fundamentals_asof_and_staleness(monkeypatch):
    def fake_load(symbols, start, end, *, data_root=None, lookback_days=400):
        assert symbols == ["AAA"]
        slice_data = CanonicalFundamentalsSlice(symbol="AAA", frame=_fundamentals_frame(), file_paths=[])
        return {"AAA": slice_data}, {}

    monkeypatch.setattr(
        "research.features.sets.fundamentals_v1.load_canonical_fundamentals",
        fake_load,
    )

    result = build_features(
        "core_v1_fundamentals_v1",
        _equity_frame(),
        underlying_symbol="AAA",
        start_date="2025-02-14",
        end_date="2025-09-01",
    )

    frame = result.frame
    assert is_bool_dtype(frame["fund_row_valid"])
    assert is_bool_dtype(frame["fund_stale"])

    row_pre_filing = frame.loc[frame["timestamp"] == pd.Timestamp("2025-02-14", tz="UTC")].iloc[0]
    assert row_pre_filing["fund_ttm_revenue"] == pytest.approx(46.0)

    row_post_filing = frame.loc[frame["timestamp"] == pd.Timestamp("2025-02-16", tz="UTC")].iloc[0]
    assert row_post_filing["fund_ttm_revenue"] == pytest.approx(50.0)
    assert row_post_filing["fund_row_valid"]

    stale_row = frame.loc[frame["timestamp"] == pd.Timestamp("2025-09-01", tz="UTC")].iloc[0]
    assert stale_row["fund_stale"]
    assert not stale_row["fund_row_valid"]
    assert np.isnan(stale_row["fund_ttm_revenue"])
