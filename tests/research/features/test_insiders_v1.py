from __future__ import annotations

from datetime import timezone

import numpy as np
import pandas as pd

from research.datasets.canonical_insiders_loader import CanonicalInsidersSlice
from research.features.feature_registry import FeatureBuildContext
from research.features.sets import insiders_v1

UTC = timezone.utc


def _equity_frame(periods: int) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=periods, freq="B", tz=UTC)
    close = np.full(len(dates), 100.0)
    volume = np.full(len(dates), 1_000_000.0)
    return pd.DataFrame({"timestamp": dates, "close": close, "volume": volume})


def test_insiders_bsr_cluster_and_ownership(monkeypatch):
    equity = _equity_frame(40)
    dates = equity["timestamp"].dt.normalize()

    insiders = pd.DataFrame(
        [
            {
                "filing_date": dates.iloc[25],
                "transaction_date": dates.iloc[25],
                "transaction_value": 200_000.0,
                "transaction_shares": 100.0,
                "shares_owned_before_transaction": 1000.0,
                "shares_owned_after_transaction": 1100.0,
                "security_title": "Common Stock",
                "title": "CEO",
                "is_board_director": False,
                "name": "Alice",
            },
            {
                "filing_date": dates.iloc[27],
                "transaction_date": dates.iloc[27],
                "transaction_value": 100_000.0,
                "transaction_shares": -50.0,
                "shares_owned_before_transaction": 1000.0,
                "shares_owned_after_transaction": 900.0,
                "security_title": "Common Stock",
                "title": "CFO",
                "is_board_director": False,
                "name": "Bob",
            },
            {
                "filing_date": dates.iloc[27],
                "transaction_date": dates.iloc[27],
                "transaction_value": 50_000.0,
                "transaction_shares": 10.0,
                "security_title": "Common Stock",
                "title": "CEO",
                "is_board_director": False,
                "name": "Noise",
            },
        ]
    )

    def _fake_loader(symbols, start_date, end_date, data_root=None, lookback_days=820):
        return {"AAPL": CanonicalInsidersSlice(symbol="AAPL", frame=insiders, file_paths=[])}, {}

    monkeypatch.setattr(insiders_v1, "load_canonical_insiders", _fake_loader)

    context = FeatureBuildContext(
        symbol="AAPL",
        start_date=equity["timestamp"].iloc[0].date(),
        end_date=equity["timestamp"].iloc[-1].date(),
        data_root=None,
        inputs_used={},
    )

    result = insiders_v1.build_insiders_v1_features(equity, context)
    target = result.iloc[30]

    expected_bsr = (200_000.0 - 100_000.0) / (200_000.0 + 100_000.0 + 1.0)
    assert np.isclose(target["insider_filtered_bsr_30d"], expected_bsr, atol=1e-6)

    expected_cluster = 3.0 * (200_000.0 / (100.0 * 1_000_000.0))
    assert np.isclose(target["insider_cluster_score_10d"], expected_cluster, atol=1e-6)

    assert np.isclose(target["insider_ownership_delta_30d"], 0.0, atol=1e-6)
    assert np.isnan(target["insider_intensity_pctl_30d"])

    early = result.iloc[5]
    assert early["insider_filtered_bsr_30d"] == 0.0
    assert early["insider_cluster_score_10d"] == 0.0


def test_insider_intensity_percentile(monkeypatch):
    periods = 805
    equity = _equity_frame(periods)
    dates = equity["timestamp"].dt.normalize()

    records = []
    # build 25 historical 30-session blocks with value=1000
    for block in range(25):
        idx = 19 + block * 30
        records.append(
            {
                "filing_date": dates.iloc[idx],
                "transaction_date": dates.iloc[idx],
                "transaction_value": 1000.0,
                "transaction_shares": 10.0,
                "shares_owned_before_transaction": 1000.0,
                "shares_owned_after_transaction": 1010.0,
                "security_title": "Common Stock",
                "title": "CEO",
                "is_board_director": False,
                "name": f"Insider{block}",
            }
        )
    # current window buy value larger
    records.append(
        {
            "filing_date": dates.iloc[775],
            "transaction_date": dates.iloc[775],
            "transaction_value": 2000.0,
            "transaction_shares": 10.0,
            "shares_owned_before_transaction": 1000.0,
            "shares_owned_after_transaction": 1010.0,
            "security_title": "Common Stock",
            "title": "CEO",
            "is_board_director": False,
            "name": "Current",
        }
    )
    insiders = pd.DataFrame(records)

    def _fake_loader(symbols, start_date, end_date, data_root=None, lookback_days=820):
        return {"AAPL": CanonicalInsidersSlice(symbol="AAPL", frame=insiders, file_paths=[])}, {}

    monkeypatch.setattr(insiders_v1, "load_canonical_insiders", _fake_loader)

    context = FeatureBuildContext(
        symbol="AAPL",
        start_date=equity["timestamp"].iloc[0].date(),
        end_date=equity["timestamp"].iloc[-1].date(),
        data_root=None,
        inputs_used={},
    )

    result = insiders_v1.build_insiders_v1_features(equity, context)
    last_row = result.iloc[-1]
    assert np.isclose(last_row["insider_intensity_pctl_30d"], 1.0, atol=1e-6)
