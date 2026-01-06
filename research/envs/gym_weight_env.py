"""Gym/Gymnasium wrapper for the deterministic signal weight environment."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from research.envs.signal_weight_env import SignalWeightEnvConfig, SignalWeightTradingEnv

GYM_STYLE = "none"

try:  # pragma: no cover - optional dependency
    import gymnasium as _gym  # type: ignore

    from gymnasium import spaces as _spaces  # type: ignore

    GYM_STYLE = "gymnasium"
except Exception:  # pragma: no cover - try legacy gym
    try:
        import gym as _gym  # type: ignore

        from gym import spaces as _spaces  # type: ignore

        GYM_STYLE = "gym"
    except Exception:  # pragma: no cover - fall back to local stubs
        _gym = None
        _spaces = None
        GYM_STYLE = "none"


if _gym and hasattr(_gym, "Env"):
    GymBase = _gym.Env  # type: ignore[assignment]
else:  # pragma: no cover - used when gym is unavailable

    class GymBase:  # type: ignore[override]
        metadata: dict[str, Any] = {}

        def reset(self, *args, **kwargs):  # noqa: D401
            raise NotImplementedError("Gym is not installed")

        def step(self, *args, **kwargs):
            raise NotImplementedError("Gym is not installed")


if _spaces is not None:
    Box = _spaces.Box  # type: ignore[assignment]
else:  # pragma: no cover - used in CI without gym deps

    class Box:
        """Minimal Box space stub providing the attributes used in tests."""

        def __init__(self, low, high, shape, dtype=np.float32):
            arr_low = np.full(shape, low, dtype=dtype)
            arr_high = np.full(shape, high, dtype=dtype)
            self.low = arr_low
            self.high = arr_high
            self.shape = tuple(shape)
            self.dtype = np.dtype(dtype)

        def contains(self, sample) -> bool:
            arr = np.asarray(sample, dtype=self.dtype)
            if arr.shape != self.shape:
                return False
            return np.all(arr >= self.low) and np.all(arr <= self.high)


class GymWeightTradingEnv(GymBase):
    """Adapters `SignalWeightTradingEnv` to a Gym-compatible interface."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]],
        config: SignalWeightEnvConfig | None = None,
        observation_columns: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self._inner = SignalWeightTradingEnv(rows, config=config, observation_columns=observation_columns)
        obs_dim = len(self._inner.observation_columns)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self._return_info = GYM_STYLE == "gymnasium"

    @property
    def inner_env(self) -> SignalWeightTradingEnv:
        return self._inner

    def reset(self, seed: int | None = None, options: Mapping[str, Any] | None = None):
        if seed is not None:  # pragma: no cover - deterministic env, but maintain API
            np.random.seed(seed)
        obs = np.asarray(self._inner.reset(), dtype=np.float32)
        info = {"timestamp": self._inner.current_row["timestamp"]}
        return (obs, info) if self._return_info else obs

    def step(self, action):
        clipped_action = float(np.clip(action, 0.0, 1.0))
        result = self._inner.step(clipped_action)
        obs, reward, done, info = result
        obs_array = np.asarray(obs, dtype=np.float32)
        info_dict = dict(info)
        if "weight_target" not in info_dict:
            info_dict["weight_target"] = clipped_action
        if "weight_realized" not in info_dict:
            info_dict["weight_realized"] = info_dict["weight_target"]
        if self._return_info:
            terminated = bool(done)
            truncated = False
            return obs_array, float(reward), terminated, truncated, info_dict
        return obs_array, float(reward), bool(done), info_dict


__all__ = ["GymWeightTradingEnv"]
