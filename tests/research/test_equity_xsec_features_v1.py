from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Dict, List

import math

import numpy as np
import pandas as pd
import pytest

from research.datasets.canonical_equity_loader import CanonicalEquitySlice
from research.features.equity_xsec_features_v1 import EQUITY_XSEC_OBSERVATION_COLUMNS, build_equity_xsec_feature_frames

UTC = timezone.utc


def _make_slice(symbol: str, closes: List[float]) -> CanonicalEquitySlice:
    timestamps = [datetime(2023, 1, idx + 2, 16, tzinfo=UTC) for idx in range(len(closes))]
    frame = pd.DataFrame(
        {
            "symbol": [symbol] * len(closes),
            "open": closes,
            "high": [value * 1.01 for value in closes],
            "low": [value * 0.99 for value in closes],
            "close": closes,
            "volume": [1_000_000 + idx for idx in range(len(closes))],
        },
        index=pd.DatetimeIndex(timestamps, tz=UTC, name="timestamp"),
    )
    return CanonicalEquitySlice(symbol=symbol, frame=frame, file_paths=[])


def test_equity_xsec_features_match_formulas():
    slices: Dict[str, CanonicalEquitySlice] = {
        "AAA": _make_slice("AAA", [100.0, 110.0, 115.5]),
        "BBB": _make_slice("BBB", [50.0, 45.0, 47.7]),
        "CCC": _make_slice("CCC", [80.0, 84.0, 79.8]),
    }
    frames = build_equity_xsec_feature_frames(
        slices,
        start_date=date(2023, 1, 2),
        end_date=date(2023, 1, 4),
        symbol_order=["AAA", "BBB", "CCC"],
    )
    assert sorted(frames.keys()) == ["AAA", "BBB", "CCC"]
    sample = frames["AAA"]
    assert tuple(sample.columns) == ("timestamp", *EQUITY_XSEC_OBSERVATION_COLUMNS)

    last_rows = {symbol: df.iloc[-1] for symbol, df in frames.items()}
    ret_values = [last_rows[symbol]["ret_1d"] for symbol in ("AAA", "BBB", "CCC")]
    ret_mean = sum(ret_values) / len(ret_values)
    ret_std = math.sqrt(sum((value - ret_mean) ** 2 for value in ret_values) / len(ret_values))

    # z-score should be (value - mean) / std when std > 0
    assert last_rows["AAA"]["ret_1d_z"] == pytest.approx(
        (last_rows["AAA"]["ret_1d"] - ret_mean) / ret_std, rel=1e-6
    )
    assert last_rows["BBB"]["ret_5d_z"] == pytest.approx(0.0, abs=1e-12)  # not enough lookback => zeroed
    assert last_rows["CCC"]["ret_1d_rank"] == 0.0  # lowest return in ordered symbol list
    assert last_rows["AAA"]["ret_1d_rank"] == pytest.approx(0.5)
    assert last_rows["BBB"]["ret_1d_rank"] == pytest.approx(1.0)
    assert last_rows["BBB"]["rel_strength_20d"] == pytest.approx(0.0)

    vol_values = [last_rows[symbol]["vol_10d"] for symbol in ("AAA", "BBB", "CCC")]
    vol_mean = sum(vol_values) / len(vol_values)
    vol_std = math.sqrt(sum((value - vol_mean) ** 2 for value in vol_values) / len(vol_values))
    assert last_rows["BBB"]["vol_10d_z"] == pytest.approx(
        (last_rows["BBB"]["vol_10d"] - vol_mean) / vol_std, rel=1e-6
    )

    # Market aggregates
    assert last_rows["AAA"]["market_ret_1d"] == pytest.approx(ret_mean)
    assert last_rows["BBB"]["dispersion_1d"] == pytest.approx(ret_std)

    market_series = frames["AAA"]["market_ret_1d"]
    expected_market_vol = (
        pd.Series(market_series).rolling(window=20, min_periods=2).std(ddof=0).iloc[-1]
    )
    assert last_rows["CCC"]["market_vol_20d"] == pytest.approx(expected_market_vol)

    ret_history = {symbol: df.set_index("timestamp")["ret_1d"] for symbol, df in frames.items()}
    ret_panel = pd.DataFrame(ret_history)

    def _expected_corr_mean(panel: pd.DataFrame, window: int) -> float:
        corr_values: list[float] = []
        for idx in range(len(panel)):
            start = max(0, idx - window + 1)
            window_frame = panel.iloc[start : idx + 1]
            if len(window_frame) < 2:
                corr_values.append(0.0)
                continue
            corr = window_frame.corr()
            if corr.shape[0] < 2:
                corr_values.append(0.0)
                continue
            upper = corr.to_numpy()[np.triu_indices_from(corr, k=1)]
            finite = upper[~pd.isna(upper)]
            corr_values.append(float(finite.mean()) if finite.size else 0.0)
        return corr_values[-1]

    expected_corr_mean = _expected_corr_mean(ret_panel, 20)
    assert last_rows["AAA"]["corr_mean_20d"] == pytest.approx(expected_corr_mean)
