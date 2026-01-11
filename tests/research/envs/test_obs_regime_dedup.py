from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from research.envs.signal_weight_env import SignalWeightEnvConfig, SignalWeightTradingEnv
from research.regime import RegimeState


UTC = timezone.utc


def _build_rows():
    timestamps = [datetime(2023, 1, 1, tzinfo=UTC) + timedelta(days=idx) for idx in range(3)]
    symbols = ("AAA", "BBB", "CCC")
    rows = []
    for idx, ts in enumerate(timestamps):
        panel = {
            symbol: {
                "close": 100.0 + idx + offset,
                "alpha": float(idx + offset),
            }
            for offset, symbol in enumerate(symbols)
        }
        regime_state = RegimeState(
            features=np.array([0.1 + idx, 0.2 + idx], dtype="float64"),
            feature_names=("regime_vol", "regime_trend"),
        )
        rows.append({"timestamp": ts, "panel": panel, "regime_state": regime_state})
    return rows


def test_regime_observation_columns_are_global():
    rows = _build_rows()
    env = SignalWeightTradingEnv(
        rows,
        config=SignalWeightEnvConfig(transaction_cost_bp=0.0),
        observation_columns=("close", "alpha", "regime_vol", "regime_trend"),
    )
    headers = env.observation_columns
    base_cols = ("close", "alpha")
    regime_cols = ("regime_vol", "regime_trend")
    symbols = env.symbols

    expected_length = len(symbols) * len(base_cols) + len(regime_cols) + len(symbols)
    assert len(headers) == expected_length
    assert headers.count("REGIME:regime_vol") == 1
    assert headers.count("REGIME:regime_trend") == 1
    for symbol in symbols:
        assert not any(name.startswith(f"{symbol}:regime_") for name in headers)
    obs = env.reset()
    assert len(obs) == expected_length
    base_end = len(symbols) * len(base_cols)
    assert tuple(obs[base_end : base_end + len(regime_cols)]) == rows[0]["regime_state"].as_tuple()
