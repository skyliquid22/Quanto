from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np

from research.training.ppo_trainer import EvaluationResult, evaluate


UTC = timezone.utc


class DummyModel:
    def predict(self, obs, deterministic=True):
        return np.asarray([0.5], dtype=np.float32), None


class DummyEnv:
    def __init__(self):
        base = datetime(2023, 1, 2, tzinfo=UTC)
        self.timestamps = [base + timedelta(days=idx) for idx in range(3)]
        self.rewards = [0.01, -0.005]
        self.inner_env = SimpleNamespace(
            symbols=("AAA",),
            portfolio_value=100.0,
            current_row={"timestamp": self.timestamps[0]},
            current_weights=[0.0],
        )
        self._step = 0

    def reset(self):
        self._step = 0
        self.inner_env.current_row = {"timestamp": self.timestamps[0]}
        self.inner_env.portfolio_value = 100.0
        return np.asarray([0.0], dtype=np.float32), {}

    def step(self, action):
        reward = self.rewards[self._step]
        info = {
            "timestamp": self.timestamps[self._step + 1],
            "weight_realized": {"AAA": float(action[0]) if isinstance(action, (list, np.ndarray)) else float(action)},
            "weight_target": {"AAA": 0.5},
            "price_close": {"AAA": 100.0},
            "portfolio_value": 100.0 * (1.0 + reward),
            "cost_paid": 0.0,
            "regime_features": [0.2 + 0.1 * self._step, 0.1],
            "regime_feature_names": ["market_vol_20d", "market_trend_20d"],
        }
        self._step += 1
        self.inner_env.current_row = {"timestamp": self.timestamps[self._step]}
        done = self._step >= len(self.rewards)
        return np.asarray([0.0], dtype=np.float32), reward, done, info


def test_ppo_evaluate_records_regime_vectors():
    env = DummyEnv()
    model = DummyModel()
    result: EvaluationResult = evaluate(model, env)
    assert result.regime_feature_names == ("market_vol_20d", "market_trend_20d")
    assert result.regime_features == [(0.2, 0.1), (0.30000000000000004, 0.1)]
