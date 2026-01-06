"""PPO training/evaluation helpers for the weight-trading environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np

try:  # pragma: no cover - stable_baselines3 is optional
    from stable_baselines3 import PPO  # type: ignore
except Exception:  # pragma: no cover - training should fail fast later
    PPO = None  # type: ignore[assignment]


@dataclass(frozen=True)
class EvaluationResult:
    metrics: Dict[str, float]
    timestamps: list[str]
    account_values: list[float]
    weights: list[float]
    log_returns: list[float]
    steps: list[Dict[str, float]]


def train_ppo(
    env,
    *,
    total_timesteps: int,
    seed: int | None = None,
    learning_rate: float | None = None,
    policy: str = "MlpPolicy",
    policy_kwargs: Mapping[str, Any] | None = None,
    **ppo_kwargs: Any,
):
    """Train PPO using stable-baselines3 when available."""

    if PPO is None:  # pragma: no cover - exercised in orchestration tests
        raise RuntimeError("stable_baselines3 is required for PPO training")
    kwargs = dict(ppo_kwargs)
    if learning_rate is not None:
        kwargs["learning_rate"] = learning_rate
    model = PPO(policy, env, seed=seed, policy_kwargs=policy_kwargs, verbose=0, **kwargs)
    model.learn(total_timesteps=total_timesteps)
    return model


def evaluate(model, env) -> EvaluationResult:
    """Run a deterministic rollout using the trained model."""

    inner_env = getattr(env, "inner_env", None)
    if inner_env is None:
        raise ValueError("Evaluation environment must expose an inner_env attribute.")

    obs, _ = _reset_env(env)
    timeline = [inner_env.current_row["timestamp"].isoformat()]
    account_values = [float(inner_env.portfolio_value)]
    weights = [float(inner_env.current_weight)]
    log_returns: list[float] = []
    steps: list[Dict[str, float]] = []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = _step_env(env, action)
        timestamp = info.get("timestamp")
        log_entry = {
            "timestamp": timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp),
            "price_close": float(info.get("price_close", 0.0)),
            "weight_target": float(info.get("weight_target", 0.0)),
            "weight_realized": float(info.get("weight_realized", info.get("weight_target", 0.0))),
            "portfolio_value": float(info.get("portfolio_value", 0.0)),
            "cost_paid": float(info.get("cost_paid", 0.0)),
            "reward": float(reward),
        }
        steps.append(log_entry)
        log_returns.append(float(reward))
        account_values.append(float(log_entry["portfolio_value"]))
        weights.append(float(log_entry["weight_realized"]))
        timeline.append(inner_env.current_row["timestamp"].isoformat())

    metrics = _compute_metrics(account_values, log_returns, steps, weights)
    return EvaluationResult(
        metrics=metrics,
        timestamps=timeline,
        account_values=account_values,
        weights=weights,
        log_returns=log_returns,
        steps=steps,
    )


def _reset_env(env):
    result = env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        obs, info = result
    else:
        obs, info = result, {}
    return np.asarray(obs, dtype=np.float32), info


def _step_env(env, action):
    result = env.step(action)
    if isinstance(result, tuple) and len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = bool(terminated or truncated)
    else:
        obs, reward, done, info = result
    return np.asarray(obs, dtype=np.float32), float(reward), bool(done), info


def _compute_metrics(
    account_values: Sequence[float],
    log_returns: Sequence[float],
    logs: Sequence[Mapping[str, float]],
    weights: Sequence[float],
) -> Dict[str, float]:
    if not account_values:
        return {key: 0.0 for key in ("total_return", "annualized_return", "annualized_vol", "sharpe", "max_drawdown")}
    start_value = account_values[0]
    end_value = account_values[-1]
    total_return = (end_value / start_value - 1.0) if start_value else 0.0
    num_periods = max(len(log_returns), 1)
    annualized_return = (1.0 + total_return) ** (252 / num_periods) - 1.0 if num_periods else 0.0
    annualized_vol = _annualized_volatility(log_returns)
    sharpe = annualized_return / annualized_vol if annualized_vol else 0.0
    max_drawdown = _max_drawdown(account_values)
    avg_cost = sum(entry.get("cost_paid", 0.0) for entry in logs) / len(logs) if logs else 0.0
    turnover = sum(abs(weights[idx] - weights[idx - 1]) for idx in range(1, len(weights)))
    metrics = {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "annualized_vol": float(annualized_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "avg_cost": float(avg_cost),
        "turnover": float(turnover),
        "final_value": float(end_value),
        "num_steps": float(len(log_returns)),
    }
    return metrics


def _annualized_volatility(log_returns: Sequence[float]) -> float:
    if not log_returns:
        return 0.0
    mean = sum(log_returns) / len(log_returns)
    variance = sum((value - mean) ** 2 for value in log_returns) / len(log_returns)
    return (variance ** 0.5) * (252 ** 0.5)


def _max_drawdown(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        drawdown = (value - peak) / peak if peak else 0.0
        worst = min(worst, drawdown)
    return abs(worst)


__all__ = ["EvaluationResult", "evaluate", "train_ppo"]
