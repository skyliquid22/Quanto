"""PPO training/evaluation helpers for the weight-trading environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - stable_baselines3 is optional
    from stable_baselines3 import PPO  # type: ignore
except Exception:  # pragma: no cover - training should fail fast later
    PPO = None  # type: ignore[assignment]

try:  # pragma: no cover - gym/gymnasium optional depending on env
    import gymnasium as _gym_api  # type: ignore
except Exception:  # pragma: no cover
    try:
        import gym as _gym_api  # type: ignore
    except Exception:  # pragma: no cover
        _gym_api = None

if _gym_api is not None and hasattr(_gym_api, "Wrapper"):
    _RewardWrapperBase = _gym_api.Wrapper  # type: ignore[assignment]
else:  # pragma: no cover - fallback shims when gym absent

    class _RewardWrapperBase:  # type: ignore[override]
        def __init__(self, env) -> None:
            self.env = env
            self.observation_space = getattr(env, "observation_space", None)
            self.action_space = getattr(env, "action_space", None)

        def reset(self, *args, **kwargs):
            return self.env.reset(*args, **kwargs)

        def step(self, action):
            return self.env.step(action)

from research.training.reward_registry import RewardFunction, create_reward
from research.regime import RegimeState


@dataclass(frozen=True)
class EvaluationResult:
    metrics: Dict[str, float]
    timestamps: list[str]
    account_values: list[float]
    weights: list[Dict[str, float]]
    log_returns: list[float]
    steps: list[Dict[str, object]]
    symbols: Tuple[str, ...]
    regime_features: list[Tuple[float, ...]] = field(default_factory=list)
    regime_feature_names: Tuple[str, ...] = field(default_factory=tuple)


class RewardAdapterEnv(_RewardWrapperBase):
    """Wrapper that rewrites rewards according to the configured reward version."""

    def __init__(self, env, reward_fn: RewardFunction):
        super().__init__(env)
        self._reward_fn = reward_fn
        self.reward_version = reward_fn.name
        inner = getattr(env, "inner_env", None)
        if inner is None and hasattr(env, "env"):
            inner = getattr(env, "env")
        self.inner_env = inner or env
        self._step_index = 0

    def reset(self, *args, **kwargs):
        self._reward_fn.reset()
        self._step_index = 0
        return super().reset(*args, **kwargs)

    def step(self, action):
        result = super().step(action)
        shaped = self._apply_reward(result)
        self._step_index += 1
        return shaped

    def _apply_reward(self, result):
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, info = result
            shaped_reward, info_payload = self._reshape_reward(reward, info)
            return obs, shaped_reward, terminated, truncated, info_payload
        obs, reward, done, info = result
        shaped_reward, info_payload = self._reshape_reward(reward, info)
        return obs, shaped_reward, done, info_payload

    def _reshape_reward(self, reward: float, info: Mapping[str, Any]):
        info_dict = dict(info or {})
        shaped_reward, components = self._reward_fn.compute(
            base_reward=float(reward),
            info=info_dict,
            step_index=self._step_index,
        )
        info_dict["reward_version"] = self.reward_version
        info_dict["base_reward"] = float(reward)
        info_dict["reward_components"] = components
        return float(shaped_reward), info_dict

    def __getattr__(self, item: str):
        return getattr(self.env, item)


DEFAULT_REWARD_VERSION = "reward_v1"


def train_ppo(
    env,
    *,
    total_timesteps: int,
    seed: int | None = None,
    learning_rate: float | None = None,
    policy: str = "MlpPolicy",
    policy_kwargs: Mapping[str, Any] | None = None,
    reward_version: str | None = DEFAULT_REWARD_VERSION,
    **ppo_kwargs: Any,
):
    """Train PPO using stable-baselines3 when available."""

    if PPO is None:  # pragma: no cover - exercised in orchestration tests
        raise RuntimeError("stable_baselines3 is required for PPO training")
    resolved_reward_version = reward_version or DEFAULT_REWARD_VERSION
    reward_fn = create_reward(resolved_reward_version)
    wrapped_env = RewardAdapterEnv(env, reward_fn)
    kwargs = dict(ppo_kwargs)
    if learning_rate is not None:
        kwargs["learning_rate"] = learning_rate
    model = PPO(policy, wrapped_env, seed=seed, policy_kwargs=policy_kwargs, verbose=0, **kwargs)
    model.learn(total_timesteps=total_timesteps)
    return model


def evaluate(model, env) -> EvaluationResult:
    """Run a deterministic rollout using the trained model."""

    inner_env = getattr(env, "inner_env", None)
    if inner_env is None:
        raise ValueError("Evaluation environment must expose an inner_env attribute.")

    obs, _ = _reset_env(env)
    symbol_order = tuple(getattr(inner_env, "symbols", ())) or _infer_symbols(inner_env.current_row)
    timeline = [inner_env.current_row["timestamp"].isoformat()]
    account_values = [float(inner_env.portfolio_value)]
    weights = [_weights_dict(inner_env.current_weights, symbol_order)]
    log_returns: list[float] = []
    steps: list[Dict[str, object]] = []
    regime_series: list[Tuple[float, ...]] = []
    regime_feature_names: list[str] = []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = _step_env(env, action)
        timestamp = info.get("timestamp")
        realized_weights = _weights_dict(info.get("weight_realized"), symbol_order)
        target_weights = _weights_dict(info.get("weight_target"), symbol_order)
        price_payload = _price_dict(info.get("price_close"), symbol_order)
        log_entry = {
            "timestamp": timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp),
            "price_close": _maybe_scalar(price_payload, symbol_order),
            "weight_target": _maybe_scalar(target_weights, symbol_order),
            "weight_realized": _maybe_scalar(realized_weights, symbol_order),
            "portfolio_value": float(info.get("portfolio_value", 0.0)),
            "cost_paid": float(info.get("cost_paid", 0.0)),
            "reward": float(reward),
        }
        steps.append(log_entry)
        log_returns.append(float(reward))
        regime_snapshot = _extract_regime_snapshot(info)
        if regime_snapshot:
            values, names = regime_snapshot
            if not regime_feature_names and names:
                regime_feature_names.extend(names)
            if regime_feature_names:
                normalized = _normalize_regime_values(values, len(regime_feature_names))
                regime_series.append(normalized)
        account_values.append(float(log_entry["portfolio_value"]))
        weights.append(realized_weights)
        timeline.append(inner_env.current_row["timestamp"].isoformat())

    metrics = _compute_metrics(account_values, log_returns, steps, weights, symbol_order)
    return EvaluationResult(
        metrics=metrics,
        timestamps=timeline,
        account_values=account_values,
        weights=weights,
        log_returns=log_returns,
        steps=steps,
        symbols=symbol_order,
        regime_features=regime_series,
        regime_feature_names=tuple(regime_feature_names),
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


def _extract_regime_snapshot(info: Mapping[str, object]) -> Tuple[Sequence[float], Sequence[str]] | None:
    raw_values = info.get("regime_features")
    names = info.get("regime_feature_names")
    state = info.get("regime_state")
    if (raw_values is None or names is None) and isinstance(state, RegimeState):
        raw_values = raw_values or state.features
        names = names or state.feature_names
    if raw_values is None or names is None:
        return None
    normalized_names = [str(name) for name in names if str(name)]
    if not normalized_names:
        return None
    return raw_values, normalized_names


def _normalize_regime_values(values: Sequence[float], width: int) -> Tuple[float, ...]:
    if width <= 0:
        return tuple()
    row = [0.0] * width
    for idx in range(width):
        try:
            row[idx] = float(values[idx])
        except (IndexError, TypeError, ValueError):
            row[idx] = 0.0
    return tuple(row)


def _compute_metrics(
    account_values: Sequence[float],
    log_returns: Sequence[float],
    logs: Sequence[Mapping[str, float]],
    weights: Sequence[Mapping[str, float]],
    symbol_order: Sequence[str],
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
    turnover = 0.0
    for idx in range(1, len(weights)):
        prev = weights[idx - 1]
        curr = weights[idx]
        for symbol in symbol_order:
            turnover += abs(curr.get(symbol, 0.0) - prev.get(symbol, 0.0))
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


def _weights_dict(payload: object, symbol_order: Sequence[str]) -> Dict[str, float]:
    if isinstance(payload, Mapping):
        return {symbol: float(payload.get(symbol, 0.0)) for symbol in symbol_order}
    if isinstance(payload, (list, tuple)):
        values = list(payload)
        if len(values) == 1 and len(symbol_order) > 1:
            values = values * len(symbol_order)
        return {symbol: float(values[idx]) for idx, symbol in enumerate(symbol_order)}
    if len(symbol_order) == 1:
        return {symbol_order[0]: float(payload or 0.0)}
    raise ValueError("weight payload missing symbol context")


def _price_dict(payload: object, symbol_order: Sequence[str]) -> Dict[str, float]:
    if isinstance(payload, Mapping):
        return {symbol: float(payload.get(symbol, 0.0)) for symbol in symbol_order}
    if len(symbol_order) == 1:
        return {symbol_order[0]: float(payload or 0.0)}
    raise ValueError("price payload missing symbol context")


def _maybe_scalar(payload: Mapping[str, float], symbol_order: Sequence[str]) -> object:
    if len(symbol_order) == 1:
        return float(payload[symbol_order[0]])
    return dict(payload)


def _infer_symbols(row: Mapping[str, object]) -> Tuple[str, ...]:
    panel = row.get("panel")
    if isinstance(panel, Mapping):
        return tuple(sorted(panel.keys()))
    symbol = str(row.get("symbol") or "asset")
    return (symbol,)


__all__ = ["EvaluationResult", "evaluate", "train_ppo"]
