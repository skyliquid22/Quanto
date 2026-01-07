"""Hierarchical policy wrapper combining controllers, allocators, and projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from research.envs.signal_weight_env import SignalWeightTradingEnv
from research.hierarchy.allocator_registry import Allocator
from research.hierarchy.controller import ModeController
from research.hierarchy.modes import DEFAULT_MODE
from research.regime import RegimeState
from research.risk import RiskConfig, project_weights


@dataclass(frozen=True)
class HierarchicalRolloutResult:
    timestamps: List[str]
    account_values: List[float]
    weights: List[Dict[str, float]]
    log_returns: List[float]
    transaction_costs: List[float]
    modes: List[str]
    symbols: Tuple[str, ...]
    regime_features: List[Tuple[float, ...]]
    regime_feature_names: Tuple[str, ...]
    logs: List[Dict[str, Any]]


class HierarchicalPolicy:
    """Combines a discrete controller and allocator registry with risk projection."""

    def __init__(
        self,
        controller: ModeController,
        allocators: Mapping[str, Allocator],
        risk: RiskConfig,
    ) -> None:
        self.controller = controller
        self.allocators = dict(allocators)
        self.risk_config = risk
        self._context: Dict[str, Any] = {}

    def reset(self) -> None:
        self.controller.reset()
        self._context = {}

    def set_allocator_context(self, context: Mapping[str, Any]) -> None:
        self._context = dict(context)

    def step(
        self,
        obs: Sequence[float],
        t: int,
        dates: Sequence[object] | np.ndarray,
        regime_features: np.ndarray | Sequence[Sequence[float]] | None,
        prev_w: Sequence[float],
        prev_mode: str | None,
    ) -> tuple[np.ndarray, str, np.ndarray]:
        obs_vector = np.asarray(obs, dtype=float).reshape(-1)
        prev_weights = np.asarray(prev_w, dtype=float)
        context = dict(self._context)
        if "num_assets" not in context:
            context["num_assets"] = len(prev_weights) if prev_weights.size else context.get("num_assets", 1)
        mode = self.controller.select_mode(t, dates, regime_features, prev_mode)
        allocator = self.allocators.get(mode) or self.allocators.get(DEFAULT_MODE)
        if allocator is None:
            raise ValueError("No allocator configured for mode '{mode}'".format(mode=mode))
        raw_action = np.asarray(allocator.act(obs_vector, context=context), dtype=float).reshape(-1)
        if prev_weights.size == 0:
            prev_weights = np.zeros_like(raw_action)
        weights = np.asarray(project_weights(raw_action, prev_weights, self.risk_config), dtype=float)
        return weights, mode, raw_action


def run_hierarchical_rollout(
    env: SignalWeightTradingEnv,
    policy: HierarchicalPolicy,
    *,
    rows: Sequence[Mapping[str, object]],
) -> HierarchicalRolloutResult:
    policy.reset()
    observation = env.reset()
    symbol_order = tuple(env.symbols)
    timestamps = [env.current_row["timestamp"].isoformat()]  # type: ignore[index]
    account_values = [float(env.portfolio_value)]
    weights = [_weights_by_symbol(env.current_weights, symbol_order)]
    log_returns: List[float] = []
    transaction_costs: List[float] = []
    modes: List[str] = []
    logs: List[Dict[str, Any]] = []
    regime_feature_names: Tuple[str, ...] = _infer_regime_names(rows)
    regime_series: List[Tuple[float, ...]] = []

    dates = np.asarray([row["timestamp"] for row in rows], dtype=object)
    regime_matrix = _build_regime_matrix(rows, len(regime_feature_names))
    prev_weights = np.asarray(env.current_weights, dtype=float)
    prev_mode: str | None = None
    step_index = 0
    done = False

    while not done:
        current_row = env.current_row
        policy.set_allocator_context(
            {
                "panel": current_row.get("panel", {}),
                "symbol_order": symbol_order,
                "num_assets": env.num_assets,
                "timestamp": current_row.get("timestamp"),
            }
        )
        projected, mode, raw_action = policy.step(
            observation,
            step_index,
            dates,
            regime_matrix,
            prev_weights,
            prev_mode,
        )
        observation, reward, done, info = env.step(raw_action.tolist(), mode=mode)
        realized_weights = _weights_by_symbol(info.get("weight_realized"), symbol_order)
        weights.append(realized_weights)
        account_values.append(float(info["portfolio_value"]))
        transaction_costs.append(float(info["cost_paid"]))
        timestamps.append(env.current_row["timestamp"].isoformat())  # type: ignore[index]
        log_returns.append(float(reward))
        modes.append(mode)
        regime_payload = tuple(float(value) for value in info.get("regime_features", []) or [0.0] * len(regime_feature_names))
        if len(regime_payload) != len(regime_feature_names):
            regime_payload = tuple([0.0] * len(regime_feature_names))
        if len(regime_feature_names) > 0:
            regime_series.append(regime_payload)
        logs.append(
            {
                "timestamp": info["timestamp"].isoformat(),  # type: ignore[attr-defined]
                "weight_realized": realized_weights,
                "weight_target": realized_weights,
                "portfolio_value": float(info["portfolio_value"]),
                "cost_paid": float(info["cost_paid"]),
                "reward": float(info["reward"]),
                "mode": mode,
            }
        )
        prev_mode = mode
        prev_weights = np.asarray(env.current_weights, dtype=float)
        step_index += 1

    return HierarchicalRolloutResult(
        timestamps=timestamps,
        account_values=account_values,
        weights=weights,
        log_returns=log_returns,
        transaction_costs=transaction_costs,
        modes=modes,
        symbols=symbol_order,
        regime_features=regime_series,
        regime_feature_names=regime_feature_names,
        logs=logs,
    )


def _weights_by_symbol(payload: object, symbol_order: Sequence[str]) -> Dict[str, float]:
    if isinstance(payload, Mapping):
        return {symbol: float(payload.get(symbol, 0.0)) for symbol in symbol_order}
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        values = list(payload)
        if len(values) == 1 and len(symbol_order) > 1:
            values = values * len(symbol_order)
        if len(values) != len(symbol_order):
            raise ValueError("Weight payload dimension mismatch")
        return {symbol: float(values[idx]) for idx, symbol in enumerate(symbol_order)}
    if len(symbol_order) == 1:
        return {symbol_order[0]: float(payload or 0.0)}
    raise ValueError("Weight payload must include symbol context")


def _infer_regime_names(rows: Sequence[Mapping[str, object]]) -> Tuple[str, ...]:
    for row in rows:
        state = row.get("regime_state")
        if isinstance(state, RegimeState):
            return tuple(state.feature_names)
    return tuple()


def _build_regime_matrix(rows: Sequence[Mapping[str, object]], width: int) -> np.ndarray:
    if width <= 0:
        return np.asarray([], dtype=float)
    snapshots: List[List[float]] = []
    for row in rows:
        state = row.get("regime_state")
        if isinstance(state, RegimeState):
            snapshots.append([float(value) for value in state.features])
        else:
            snapshots.append([0.0] * width)
    return np.asarray(snapshots, dtype=float)


__all__ = ["HierarchicalPolicy", "HierarchicalRolloutResult", "run_hierarchical_rollout"]
