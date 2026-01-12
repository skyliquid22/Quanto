from __future__ import annotations

import math

import research.training.ppo_trainer as ppo_trainer
from research.training.ppo_trainer import RewardAdapterEnv
from research.training.reward_registry import create_reward, reward_versions


# Prefer gymnasium if available, otherwise fall back to gym; tests still work without either.
_IS_GYMNASIUM = False
try:  # pragma: no cover - optional deps for env compatibility
    import gymnasium as _gym

    _IS_GYMNASIUM = True
except Exception:  # pragma: no cover
    try:
        import gym as _gym  # type: ignore
    except Exception:
        _gym = None

if _gym is not None:
    try:
        _spaces = _gym.spaces  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover
        _spaces = None
else:
    _spaces = None


class _DummyEnv((_gym.Env if _gym is not None else object)):  # type: ignore[misc]
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        if _gym is not None:
            super().__init__()
        self.inner_env = self
        self.current_row = {"timestamp": "2023-01-01T00:00:00"}
        self.current_weights = (0.0,)
        self.symbols = ("AAA",)
        self._value = 100.0
        if _spaces is not None:
            self.observation_space = _spaces.Box(0.0, 1.0, shape=(1,), dtype=float)
            self.action_space = _spaces.Box(-1.0, 1.0, shape=(1,), dtype=float)
        else:
            self.observation_space = None
            self.action_space = None

    def reset(self, *args, **kwargs):
        self._value = 100.0
        self.current_weights = (0.0,)
        obs = 0.0
        info = {"timestamp": self.current_row["timestamp"]}
        if _IS_GYMNASIUM:
            return obs, info
        return obs

    def step(self, action):
        prev = self._value
        target = float(action)
        self._value *= 1.01
        self.current_weights = (target,)
        reward = math.log(self._value / prev)
        info = {
            "timestamp": self.current_row["timestamp"],
            "weight_realized": {"AAA": target},
            "weight_target": {"AAA": target},
            "price_close": {"AAA": 100.0},
            "portfolio_value": self._value,
            "cost_paid": abs(target) * 0.5,
        }
        if _IS_GYMNASIUM:
            terminated = True
            truncated = False
            return 0.0, reward, terminated, truncated, info
        if _gym is not None:
            return 0.0, reward, True, info
        return 0.0, reward, True, info


def test_reward_versions_registered():
    versions = reward_versions()
    assert "reward_v1" in versions
    assert "reward_v2" in versions
    first = create_reward("reward_v2")
    second = create_reward("reward_v2")
    assert first is not second


def test_reward_adapter_penalizes_turnover():
    env = _DummyEnv()
    adapter = RewardAdapterEnv(env, create_reward("reward_v2"))
    adapter.reset()
    adapter.step(0.2)  # warmup to establish last weights
    result = adapter.step(0.8)
    if isinstance(result, tuple) and len(result) == 5:
        _, shaped_reward, _, _, info = result
    else:
        _, shaped_reward, _, info = result
    assert info["reward_version"] == "reward_v2"
    assert shaped_reward < info["base_reward"]


def test_train_ppo_uses_default_reward_when_unset(monkeypatch):
    env = _DummyEnv()
    captured = {}

    class _FakePPO:
        def __init__(self, policy, inner_env, **kwargs):
            captured["env"] = inner_env

        def learn(self, total_timesteps):
            captured["timesteps"] = total_timesteps
            return self

    monkeypatch.setattr(ppo_trainer, "PPO", _FakePPO)
    model = ppo_trainer.train_ppo(env, total_timesteps=5, reward_version=None)
    assert isinstance(model, _FakePPO)
    wrapped_env = captured["env"]
    assert isinstance(wrapped_env, RewardAdapterEnv)
    assert wrapped_env.reward_version == ppo_trainer.DEFAULT_REWARD_VERSION
    assert captured["timesteps"] == 5


def test_train_ppo_uses_custom_reward_version(monkeypatch):
    env = _DummyEnv()
    captured = {}

    class _FakePPO:
        def __init__(self, policy, inner_env, **kwargs):
            captured["env"] = inner_env

        def learn(self, total_timesteps):
            captured["timesteps"] = total_timesteps
            return self

    monkeypatch.setattr(ppo_trainer, "PPO", _FakePPO)
    model = ppo_trainer.train_ppo(env, total_timesteps=3, reward_version="reward_v2")
    assert isinstance(model, _FakePPO)
    wrapped_env = captured["env"]
    assert isinstance(wrapped_env, RewardAdapterEnv)
    assert wrapped_env.reward_version == "reward_v2"
    assert captured["timesteps"] == 3
