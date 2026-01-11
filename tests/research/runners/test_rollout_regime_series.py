from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace

from research.eval.evaluate import EvalSeries, EvaluationMetadata, MetricConfig, evaluation_payload, from_rollout
from research.experiments import runner as experiment_runner
from research.runners.rollout import run_rollout


UTC = timezone.utc


class FakePolicy:
    def decide(self, *_args, **_kwargs) -> float:
        return 0.5


class FakeEnv:
    def __init__(self):
        self.symbols = ("AAA", "BBB")
        base = datetime(2023, 1, 2, tzinfo=UTC)
        self.rows = [
            {"timestamp": base, "panel": {"AAA": {}, "BBB": {}}},
            {"timestamp": base + timedelta(days=1), "panel": {"AAA": {}, "BBB": {}}},
            {"timestamp": base + timedelta(days=2), "panel": {"AAA": {}, "BBB": {}}},
        ]
        self.weight_path = [
            [0.0, 0.0],
            [0.6, 0.4],
            [0.4, 0.6],
        ]
        self.rewards = [0.01, -0.005]
        self.regime_values = [
            [0.2, 0.1],
            [0.3, 0.05],
        ]
        self.portfolio_value = 100.0
        self.num_assets = len(self.symbols)

    def reset(self):
        self.idx = 0
        self.portfolio_value = 100.0
        self.current_weights = self.weight_path[0][:]
        self.current_row = self.rows[0]
        return None

    def step(self, _action):
        reward = self.rewards[self.idx]
        realized = self.weight_path[self.idx + 1]
        self.portfolio_value *= 1.0 + reward
        info = {
            "timestamp": self.rows[self.idx + 1]["timestamp"],
            "price_close": {"AAA": 100.0, "BBB": 100.0},
            "weight_realized": {"AAA": realized[0], "BBB": realized[1]},
            "weight_target": {"AAA": realized[0], "BBB": realized[1]},
            "portfolio_value": self.portfolio_value,
            "cost_paid": 0.0,
            "reward": reward,
            "regime_features": self.regime_values[self.idx],
            "regime_feature_names": ["market_vol_20d", "market_trend_20d"],
        }
        self.idx += 1
        self.current_weights = realized[:]
        self.current_row = self.rows[self.idx]
        done = self.idx >= len(self.rewards)
        return None, reward, done, info


def test_rollout_persists_regime_series(tmp_path):
    env = FakeEnv()
    policy = FakePolicy()
    result = run_rollout(env, policy, inputs_used={}, metadata={})
    assert result.regime_feature_names == ("market_vol_20d", "market_trend_20d")
    assert len(result.regime_features) == len(result.log_returns)

    series = from_rollout(
        timestamps=result.timestamps,
        account_values=result.account_values,
        weights=result.weights,
        transaction_costs=result.transaction_costs,
        symbols=result.symbols,
        rollout_metadata=result.metadata,
        regime_features=result.regime_features,
        regime_feature_names=result.regime_feature_names,
    )
    metadata = EvaluationMetadata(
        symbols=result.symbols,
        start_date="2023-01-02",
        end_date="2023-01-04",
        interval="daily",
        feature_set="core_v1_regime",
        policy_id="test_policy",
        run_id="test_run",
    )
    payload = evaluation_payload(series, metadata, inputs_used={}, config=MetricConfig())
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    spec = SimpleNamespace(experiment_id="exp_test")
    experiment_runner._write_rollout_artifact(spec, payload, runs_dir)

    rollout_path = runs_dir / "rollout.json"
    data = json.loads(rollout_path.read_text(encoding="utf-8"))
    regime_section = data["series"].get("regime")
    assert regime_section is not None
    assert regime_section["feature_names"] == ["market_vol_20d", "market_trend_20d"]
    assert len(regime_section["values"]) == len(data["series"]["returns"])
