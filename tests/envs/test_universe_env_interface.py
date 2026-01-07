from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Tuple

import numpy as np
import pytest

from research.envs.gym_weight_env import GymWeightTradingEnv
from research.envs.signal_weight_env import SignalWeightEnvConfig

UTC = timezone.utc


def _reset(env: GymWeightTradingEnv) -> Tuple[np.ndarray, dict[str, Any]]:
    result = env.reset()
    if isinstance(result, tuple):
        obs, info = result
    else:
        obs, info = result, {}
    return obs, info


def _step(env: GymWeightTradingEnv, action) -> Tuple[np.ndarray, float, bool, dict[str, Any]]:
    result = env.step(action)
    if isinstance(result, tuple) and len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = bool(terminated or truncated)
    else:
        obs, reward, done, info = result
    return obs, float(reward), bool(done), info


def _build_rows():
    timestamps = [
        datetime(2023, 1, 3, 16, tzinfo=UTC),
        datetime(2023, 1, 4, 16, tzinfo=UTC),
        datetime(2023, 1, 5, 16, tzinfo=UTC),
    ]
    rows = []
    for idx, ts in enumerate(timestamps):
        rows.append(
            {
                "timestamp": ts,
                "panel": {
                    "AAPL": {
                        "close": 100.0 + idx,
                        "sma_fast": 99.0 + idx,
                        "sma_slow": 98.0 + idx,
                        "sma_diff": 1.0,
                        "sma_signal": 1.0,
                    },
                    "MSFT": {
                        "close": 200.0 + idx,
                        "sma_fast": 199.0 + idx,
                        "sma_slow": 198.0 + idx,
                        "sma_diff": 1.0,
                        "sma_signal": 0.0 if idx % 2 else 1.0,
                    },
                },
            }
        )
    return rows


def test_universe_env_interface_and_determinism():
    rows = _build_rows()
    columns = ("close", "sma_fast", "sma_slow", "sma_diff", "sma_signal")
    env = GymWeightTradingEnv(rows, config=SignalWeightEnvConfig(transaction_cost_bp=0.25), observation_columns=columns)
    obs, info = _reset(env)
    assert obs.shape == (len(columns) * 2 + 2,)
    assert env.action_space.shape == (2,)
    assert env.observation_space.shape == obs.shape
    assert not np.isnan(obs).any()
    action = np.array([0.6, 0.4], dtype=np.float32)
    obs2, reward, done, info = _step(env, action)
    assert obs2.shape == obs.shape
    assert reward != 0.0
    assert done is False
    assert set(info["weight_target"].keys()) == {"AAPL", "MSFT"}
    assert info["weight_realized"] == info["weight_target"]
    assert info["price_close"]["AAPL"] == rows[0]["panel"]["AAPL"]["close"]
    assert not np.isnan(obs2).any()

    # Deterministic behavior across resets with same actions
    env_copy = GymWeightTradingEnv(rows, config=SignalWeightEnvConfig(transaction_cost_bp=0.25), observation_columns=columns)
    _reset(env_copy)
    _, reward_copy, done_copy, info_copy = _step(env_copy, action)
    assert reward_copy == pytest.approx(reward)
    assert done_copy is done
    assert info_copy["weight_target"] == info["weight_target"]
