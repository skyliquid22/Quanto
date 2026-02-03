"""SAC training/evaluation helpers for the weight-trading environment."""

from __future__ import annotations

from typing import Any, Mapping

try:  # pragma: no cover - stable_baselines3 is optional
    from stable_baselines3 import SAC  # type: ignore
except Exception:  # pragma: no cover
    SAC = None  # type: ignore[assignment]

from research.training.ppo_trainer import (
    DEFAULT_REWARD_VERSION,
    RewardAdapterEnv,
    EvaluationResult,
    evaluate as evaluate_model,
)
from research.training.reward_registry import create_reward


def train_sac(
    env,
    *,
    total_timesteps: int,
    seed: int | None = None,
    learning_rate: float | None = None,
    policy: str = "MlpPolicy",
    policy_kwargs: Mapping[str, Any] | None = None,
    reward_version: str | None = DEFAULT_REWARD_VERSION,
    **sac_kwargs: Any,
):
    """Train SAC using stable-baselines3 when available."""

    if SAC is None:  # pragma: no cover - exercised in orchestration tests
        raise RuntimeError("stable_baselines3 is required for SAC training")
    resolved_reward_version = reward_version or DEFAULT_REWARD_VERSION
    reward_fn = create_reward(resolved_reward_version)
    wrapped_env = RewardAdapterEnv(env, reward_fn)
    kwargs = dict(sac_kwargs)
    if learning_rate is not None:
        kwargs["learning_rate"] = learning_rate
    model = SAC(policy, wrapped_env, seed=seed, policy_kwargs=policy_kwargs, verbose=0, **kwargs)
    model.learn(total_timesteps=total_timesteps)
    return model


def evaluate(model, env) -> EvaluationResult:
    """Run a deterministic rollout using the trained model."""

    return evaluate_model(model, env)


__all__ = ["train_sac", "evaluate", "DEFAULT_REWARD_VERSION"]
