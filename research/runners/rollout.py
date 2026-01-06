"""Deterministic rollout runner producing FinRL-style monitoring artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict, List, Mapping

from research.envs.signal_weight_env import SignalWeightTradingEnv
from research.policies.sma_weight_policy import SMAWeightPolicy

ANNUALIZATION_DAYS = 252


@dataclass(frozen=True)
class RolloutResult:
    timestamps: List[str]
    account_values: List[float]
    weights: List[float]
    log_returns: List[float]
    steps: List[Dict[str, float]]
    metrics: Dict[str, float]
    inputs_used: Dict[str, str]


def run_rollout(
    env: SignalWeightTradingEnv,
    policy: SMAWeightPolicy,
    *,
    inputs_used: Mapping[str, str],
) -> RolloutResult:
    env.reset()

    timeline = [env.current_row["timestamp"].isoformat()]  # type: ignore[index]
    account_values = [float(env.portfolio_value)]
    weights = [float(env.current_weight)]
    log_returns: List[float] = []
    logs: List[Dict[str, float]] = []

    done = False
    while not done:
        row = env.current_row
        weight = policy.decide(row)
        _, reward, done, info = env.step(weight)
        log_returns.append(float(reward))
        log_entry = {
            "timestamp": info["timestamp"].isoformat(),  # type: ignore[attr-defined]
            "price_close": float(info["price_close"]),
            "weight_target": float(info["weight_target"]),
            "weight_realized": float(info["weight_realized"]),
            "portfolio_value": float(info["portfolio_value"]),
            "cost_paid": float(info["cost_paid"]),
            "reward": float(info["reward"]),
        }
        logs.append(log_entry)
        account_values.append(float(info["portfolio_value"]))
        weights.append(float(info["weight_realized"]))
        next_timestamp = env.current_row["timestamp"].isoformat()  # type: ignore[index]
        timeline.append(next_timestamp)

    metrics = _compute_metrics(account_values, log_returns, logs, weights)
    return RolloutResult(
        timestamps=timeline,
        account_values=account_values,
        weights=weights,
        log_returns=log_returns,
        steps=logs,
        metrics=metrics,
        inputs_used={key: value for key, value in inputs_used.items()},
    )


def _compute_metrics(
    account_values: List[float],
    log_returns: List[float],
    logs: List[Dict[str, float]],
    weights: List[float],
) -> Dict[str, float]:
    if not account_values:
        return {key: 0.0 for key in ("total_return", "annualized_return", "annualized_vol", "sharpe", "max_drawdown")}
    start_value = account_values[0]
    end_value = account_values[-1]
    total_return = (end_value / start_value - 1.0) if start_value else 0.0
    num_periods = max(len(log_returns), 1)
    annualized_return = (1.0 + total_return) ** (ANNUALIZATION_DAYS / num_periods) - 1.0 if num_periods else 0.0
    annualized_vol = _annualized_volatility(log_returns)
    sharpe = annualized_return / annualized_vol if annualized_vol else 0.0
    max_drawdown = _max_drawdown(account_values)
    avg_cost = sum(entry["cost_paid"] for entry in logs) / len(logs) if logs else 0.0
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


def _annualized_volatility(log_returns: List[float]) -> float:
    if not log_returns:
        return 0.0
    mean = sum(log_returns) / len(log_returns)
    variance = sum((value - mean) ** 2 for value in log_returns) / len(log_returns)
    daily_vol = sqrt(variance)
    return daily_vol * sqrt(ANNUALIZATION_DAYS)


def _max_drawdown(values: List[float]) -> float:
    if not values:
        return 0.0
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        drawdown = (value - peak) / peak if peak else 0.0
        worst = min(worst, drawdown)
    return abs(worst)


__all__ = ["RolloutResult", "run_rollout"]
