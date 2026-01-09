from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.features.core_features_v1 import (
    CORE_V1_OBSERVATION_COLUMNS,
    compute_core_features_v1,
)


def _make_equity_df(rows: int = 50) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC")
    closes = np.linspace(100.0, 150.0, rows)
    vols = np.linspace(1_000_000, 2_000_000, rows)
    return pd.DataFrame({"timestamp": dates, "close": closes, "volume": vols})


def test_core_features_column_contract_and_order():
    raw = _make_equity_df()
    result = compute_core_features_v1(raw)
    expected = ["timestamp", *CORE_V1_OBSERVATION_COLUMNS]
    assert list(result[expected].columns) == expected


def test_core_features_returns_match_log_ratio_no_lookahead():
    raw = _make_equity_df()
    result = compute_core_features_v1(raw)
    closes = result["close"].to_numpy()
    log_ratio = np.log(closes[1:] / closes[:-1])
    np.testing.assert_allclose(result["ret_1"].to_numpy()[1:], log_ratio, rtol=1e-10, atol=1e-10)
    # first value should be zero because of deterministic NaN fill
    assert result["ret_1"].iloc[0] == 0.0


def test_core_features_rolling_windows_zero_until_ready():
    raw = _make_equity_df(rows=80)
    result = compute_core_features_v1(raw)
    assert (result["rv_21"].iloc[:21] == 0.0).all()
    assert (result["rv_63"].iloc[:63] == 0.0).all()
    for column in ("dist_from_20d_high", "dist_from_20d_low", "log_vol_z20"):
        assert (result[column].iloc[:19] == 0.0).all()


def test_core_features_basic_ranges_and_finite_values():
    raw = _make_equity_df(rows=80)
    result = compute_core_features_v1(raw)
    for column in CORE_V1_OBSERVATION_COLUMNS[1:]:
        values = result[column].to_numpy()
        assert np.isfinite(values).all()
        assert not np.isnan(values).any()


def test_core_features_missing_column_raises():
    raw = _make_equity_df().drop(columns=["volume"])
    with pytest.raises(ValueError):
        compute_core_features_v1(raw)
