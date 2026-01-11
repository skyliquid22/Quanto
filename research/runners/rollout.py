"""Deterministic rollout runner producing FinRL-style monitoring artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict, List, Mapping, Sequence, Tuple

from research.envs.signal_weight_env import SignalWeightTradingEnv
from research.policies.sma_weight_policy import SMAWeightPolicy
from research.regime import RegimeState

ANNUALIZATION_DAYS = 252


@dataclass(frozen=True)
class RolloutResult:
    timestamps: List[str]
    account_values: List[float]
    weights: List[Dict[str, float]]
    log_returns: List[float]
    steps: List[Dict[str, object]]
    metrics: Dict[str, float]
    inputs_used: Dict[str, str]
    symbols: Tuple[str, ...]
    transaction_costs: List[float]
    metadata: Dict[str, object]
    regime_features: List[Tuple[float, ...]]
    regime_feature_names: Tuple[str, ...]
    regime_features: List[Tuple[float, ...]]
    regime_feature_names: Tuple[str, ...]


def run_rollout(
    env: SignalWeightTradingEnv,
    policy: SMAWeightPolicy,
    *,
    inputs_used: Mapping[str, str],
    metadata: Mapping[str, object] | None = None,
) -> RolloutResult:
    env.reset()
    symbol_order = tuple(getattr(env, "symbols", ())) or _infer_symbol_order(env.current_row)

    timeline = [env.current_row["timestamp"].isoformat()]  # type: ignore[index]
    account_values = [float(env.portfolio_value)]
    weights = [_weights_by_symbol(env.current_weights, symbol_order)]
    log_returns: List[float] = []
    logs: List[Dict[str, object]] = []
    transaction_costs: List[float] = []
    regime_feature_names: List[str] = list(getattr(env, "_regime_feature_columns", ()) or [])
    regime_series: List[Tuple[float, ...]] = []

    done = False
    while not done:
        row = env.current_row
        action_vector = _decide_actions(policy, row, symbol_order)
        action_payload = action_vector[0] if len(action_vector) == 1 else action_vector
        _, reward, done, info = env.step(action_payload)
        realized_weights = _weights_by_symbol(info.get("weight_realized"), symbol_order)
        log_returns.append(float(reward))
        price_payload = _prices_by_symbol(info.get("price_close"), symbol_order)
        target_weights = _weights_by_symbol(info.get("weight_target"), symbol_order)
        weight_entry = _scalar_or_mapping(realized_weights, symbol_order)
        log_entry = {
            "timestamp": info["timestamp"].isoformat(),  # type: ignore[attr-defined]
            "price_close": _scalar_or_mapping(price_payload, symbol_order),
            "weight_target": _scalar_or_mapping(target_weights, symbol_order),
            "weight_realized": weight_entry,
            "portfolio_value": float(info["portfolio_value"]),
            "cost_paid": float(info["cost_paid"]),
            "reward": float(info["reward"]),
        }
        logs.append(log_entry)
        transaction_costs.append(float(info["cost_paid"]))
        _capture_regime_snapshot(info, regime_series, regime_feature_names)
        account_values.append(float(info["portfolio_value"]))
        weights.append(realized_weights)
        next_timestamp = env.current_row["timestamp"].isoformat()  # type: ignore[index]
        timeline.append(next_timestamp)

    metrics = _compute_metrics(account_values, log_returns, logs, weights, symbol_order)
    ordered_inputs = {key: inputs_used[key] for key in sorted(inputs_used)}
    metadata_payload = _normalize_metadata(metadata)
    return RolloutResult(
        timestamps=timeline,
        account_values=account_values,
        weights=weights,
        log_returns=log_returns,
        steps=logs,
        metrics=metrics,
        inputs_used=ordered_inputs,
        symbols=symbol_order,
        transaction_costs=transaction_costs,
        metadata=metadata_payload,
        regime_features=regime_series,
        regime_feature_names=tuple(regime_feature_names),
    )


def _compute_metrics(
    account_values: List[float],
    log_returns: List[float],
    logs: List[Dict[str, object]],
    weights: List[Dict[str, float]],
    symbol_order: Sequence[str],
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
    avg_cost = sum(float(entry["cost_paid"]) for entry in logs) / len(logs) if logs else 0.0
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


def _decide_actions(policy: SMAWeightPolicy, row: Mapping[str, object], symbol_order: Sequence[str]) -> List[float]:
    panel = row.get("panel")
    if isinstance(panel, Mapping) and symbol_order:
        actions = []
        for symbol in symbol_order:
            features = panel.get(symbol)
            if not isinstance(features, Mapping):
                raise ValueError("panel entries must be mappings")
            actions.append(float(policy.decide(features)))
        return actions
    return [float(policy.decide(row))]


def _weights_by_symbol(payload: object, symbol_order: Sequence[str]) -> Dict[str, float]:
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


def _prices_by_symbol(payload: object, symbol_order: Sequence[str]) -> Dict[str, float]:
    if isinstance(payload, Mapping):
        return {symbol: float(payload.get(symbol, 0.0)) for symbol in symbol_order}
    if len(symbol_order) == 1:
        return {symbol_order[0]: float(payload or 0.0)}
    raise ValueError("price payload missing symbol context")


def _scalar_or_mapping(payload: Mapping[str, float], symbol_order: Sequence[str]) -> object:
    if len(symbol_order) == 1:
        return float(payload[symbol_order[0]])
    return dict(payload)


def _infer_symbol_order(row: Mapping[str, object]) -> Tuple[str, ...]:
    panel = row.get("panel")
    if isinstance(panel, Mapping):
        return tuple(sorted(panel.keys()))
    symbol = str(row.get("symbol") or "asset")
    return (symbol,)


def _capture_regime_snapshot(
    info: Mapping[str, object],
    series: List[Tuple[float, ...]],
    name_list: List[str],
) -> None:
    if not name_list:
        inferred = _infer_regime_names_from_info(info)
        if inferred:
            name_list.extend(inferred)
    if not name_list:
        return
    width = len(name_list)
    raw_values = info.get("regime_features")
    if raw_values is None:
        state = info.get("regime_state")
        if isinstance(state, RegimeState):
            raw_values = state.features
    normalized = _normalize_regime_values(raw_values, width)
    series.append(normalized)


def _infer_regime_names_from_info(info: Mapping[str, object]) -> List[str]:
    names = info.get("regime_feature_names")
    if isinstance(names, Sequence):
        normalized = [str(name) for name in names if str(name)]
        if normalized:
            return normalized
    state = info.get("regime_state")
    if isinstance(state, RegimeState):
        return [str(name) for name in state.feature_names]
    return []


def _normalize_regime_values(values: object, width: int) -> Tuple[float, ...]:
    if not width:
        return tuple()
    row = [0.0] * width
    if isinstance(values, Sequence):
        for idx in range(width):
            try:
                row[idx] = float(values[idx])
            except (IndexError, TypeError, ValueError):
                row[idx] = 0.0
    return tuple(row)


def _normalize_metadata(payload: Mapping[str, object] | None) -> Dict[str, object]:
    if not payload:
        return {}
    ordered = {}
    for key in sorted(payload):
        ordered[str(key)] = payload[key]
    return ordered


__all__ = ["RolloutResult", "run_rollout"]
