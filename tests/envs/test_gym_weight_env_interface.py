from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Tuple

import numpy as np

from research.envs.gym_weight_env import GymWeightTradingEnv


UTC = timezone.utc


def _reset(env: GymWeightTradingEnv) -> Tuple[np.ndarray, dict[str, Any]]:
    result = env.reset()
    if isinstance(result, tuple):
        obs, info = result
    else:
        obs, info = result, {}
    return obs, info


def _step(env: GymWeightTradingEnv, action: float) -> Tuple[np.ndarray, float, bool, dict[str, Any]]:
    result = env.step(action)
    if isinstance(result, tuple) and len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = bool(terminated or truncated)
    else:
        obs, reward, done, info = result
    return obs, float(reward), bool(done), info


def _build_rows():
    return [
        {
            "timestamp": datetime(2023, 1, 2, tzinfo=UTC),
            "close": 10.0,
            "sma_fast": 9.5,
            "sma_slow": 8.0,
            "sma_diff": 1.5,
            "sma_signal": 1.0,
        },
        {
            "timestamp": datetime(2023, 1, 3, tzinfo=UTC),
            "close": 11.0,
            "sma_fast": 9.7,
            "sma_slow": 8.5,
            "sma_diff": 1.2,
            "sma_signal": 1.0,
        },
        {
            "timestamp": datetime(2023, 1, 4, tzinfo=UTC),
            "close": 9.0,
            "sma_fast": 9.0,
            "sma_slow": 8.6,
            "sma_diff": 0.4,
            "sma_signal": 0.0,
        },
    ]


def test_gym_weight_env_spaces_and_determinism():
    env = GymWeightTradingEnv(_build_rows())
    obs, info = _reset(env)
    assert obs.shape == (5,)
    assert obs.dtype == np.float32
    assert env.action_space.shape == (1,)
    assert env.observation_space.shape == (5,)
    assert env.action_space.contains(np.array([0.5], dtype=np.float32))

    obs, reward, done, info = _step(env, 1.5)
    assert info["timestamp"].isoformat() == "2023-01-02T00:00:00+00:00"
    assert math.isclose(info["weight_target"], 1.0, rel_tol=0, abs_tol=0)
    assert math.isclose(info["weight_realized"], 1.0, rel_tol=0, abs_tol=0)
    expected_value = (10000.0 - 1.0) * (1.0 + 0.1)
    assert math.isclose(info["portfolio_value"], expected_value, rel_tol=1e-12, abs_tol=1e-9)
    assert math.isclose(reward, math.log(expected_value / 10000.0), rel_tol=1e-12, abs_tol=1e-9)
    assert done is False
    assert obs.shape == (5,)

    obs, reward, done, info = _step(env, -5.0)
    assert done is True
    assert math.isclose(info["weight_target"], 0.0, rel_tol=0, abs_tol=0)
    assert math.isclose(info["weight_realized"], 0.0, rel_tol=0, abs_tol=0)
    expected_value = (expected_value - expected_value * 0.0001) * (1.0 + 0.0 * ((9.0 - 11.0) / 11.0))
    assert math.isclose(info["portfolio_value"], expected_value, rel_tol=1e-12, abs_tol=1e-9)
    assert math.isclose(reward, math.log(expected_value / ((10000.0 - 1.0) * 1.1)), rel_tol=1e-12, abs_tol=1e-9)
