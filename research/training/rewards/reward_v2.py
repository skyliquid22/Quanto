"""Stable log-return shaping that penalizes churn and worsening drawdowns."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence, Tuple, Dict, Iterable

from research.training.reward_registry import RewardFunction, RewardSpec, register_reward

EPS = 1e-12


def _sanitize_vector(values: Iterable[float] | None) -> Tuple[float, ...] | None:
    if values is None:
        return None
    data = tuple(float(entry) for entry in values)
    if not data:
        return None
    return data


class StableRewardV2:
    """Reward shaping that gently discourages churn and worsening drawdowns."""

    name = "reward_v2"

    DEFAULT_VALUE_KEYS = ("portfolio_value", "value", "equity", "account_value", "nav")
    DEFAULT_PREV_VALUE_KEYS = ("prev_portfolio_value", "prev_value", "prev_equity")
    DEFAULT_WEIGHT_KEYS = ("weight_realized", "weights", "current_weights", "target_weights", "portfolio_weights")
    DEFAULT_PREV_WEIGHT_KEYS = ("prev_weights", "last_weights", "previous_weights")
    DEFAULT_COST_KEYS = ("cost_paid", "transaction_cost", "cost", "fees_paid")

    def __init__(
        self,
        *,
        turnover_scale: float = 0.10,
        drawdown_scale: float = 0.50,
        cost_scale: float = 0.00,
        clip_reward: Tuple[float, float] | None = None,
        value_keys: Sequence[str] | None = None,
        prev_value_keys: Sequence[str] | None = None,
        weight_keys: Sequence[str] | None = None,
        prev_weight_keys: Sequence[str] | None = None,
        cost_keys: Sequence[str] | None = None,
    ) -> None:
        self.turnover_scale = float(turnover_scale)
        self.drawdown_scale = float(drawdown_scale)
        self.cost_scale = float(cost_scale)
        self.clip_reward = tuple(clip_reward) if clip_reward is not None else None
        self.value_keys = tuple(value_keys or self.DEFAULT_VALUE_KEYS)
        self.prev_value_keys = tuple(prev_value_keys or self.DEFAULT_PREV_VALUE_KEYS)
        self.weight_keys = tuple(weight_keys or self.DEFAULT_WEIGHT_KEYS)
        self.prev_weight_keys = tuple(prev_weight_keys or self.DEFAULT_PREV_WEIGHT_KEYS)
        self.cost_keys = tuple(cost_keys or self.DEFAULT_COST_KEYS)
        self._last_weights: Tuple[float, ...] | None = None
        self._peak_value: float | None = None
        self._last_drawdown: float = 0.0

    def reset(self) -> None:
        self._last_weights = None
        self._peak_value = None
        self._last_drawdown = 0.0

    def compute(
        self,
        *,
        base_reward: float,
        info: Mapping[str, Any],
        step_index: int,
    ) -> Tuple[float, Dict[str, float]]:
        base = float(base_reward)
        current_weights = self._extract_weights(info, self.weight_keys)
        prev_weights = self._extract_weights(info, self.prev_weight_keys)
        if prev_weights is None:
            prev_weights = self._last_weights

        turnover = self._compute_turnover(current_weights, prev_weights)
        turnover_penalty = turnover * self.turnover_scale

        current_value = self._extract_float(info, self.value_keys)
        drawdown, drawdown_increase = self._compute_drawdown(current_value)
        drawdown_penalty = drawdown_increase * self.drawdown_scale

        cost_penalty, cost_ratio = self._compute_cost_penalty(base, info, current_value)

        shaped = base - turnover_penalty - drawdown_penalty - cost_penalty
        if self.clip_reward is not None:
            lower, upper = self.clip_reward
            shaped = max(lower, min(upper, shaped))

        self._last_weights = current_weights if current_weights is not None else self._last_weights

        components = {
            "base": base,
            "turnover": turnover,
            "turnover_penalty": turnover_penalty,
            "drawdown": drawdown,
            "drawdown_increase": drawdown_increase,
            "drawdown_penalty": drawdown_penalty,
            "cost_fraction": cost_ratio,
            "cost_penalty": cost_penalty,
            "final_reward": shaped,
        }
        return shaped, components

    def _compute_turnover(
        self,
        current: Tuple[float, ...] | None,
        previous: Tuple[float, ...] | None,
    ) -> float:
        if current is None or previous is None or len(current) != len(previous):
            return 0.0
        delta = sum(abs(current[idx] - previous[idx]) for idx in range(len(current)))
        return 0.5 * delta

    def _compute_drawdown(self, value: float | None) -> Tuple[float, float]:
        if value is None or not math.isfinite(value):
            return self._last_drawdown, 0.0
        if self._peak_value is None or value >= self._peak_value:
            self._peak_value = max(value, EPS)
            self._last_drawdown = 0.0
            return 0.0, 0.0
        if self._peak_value <= EPS:
            return 0.0, 0.0
        drawdown = max(0.0, (self._peak_value - value) / self._peak_value)
        increase = max(0.0, drawdown - self._last_drawdown)
        self._last_drawdown = drawdown
        return drawdown, increase

    def _compute_cost_penalty(
        self,
        base_reward: float,
        info: Mapping[str, Any],
        current_value: float | None,
    ) -> Tuple[float, float]:
        if self.cost_scale <= EPS:
            return 0.0, 0.0
        cost_amount = self._extract_float(info, self.cost_keys) or 0.0
        if cost_amount <= 0.0:
            return 0.0, 0.0
        prev_value = self._extract_float(info, self.prev_value_keys)
        if prev_value is None and current_value is not None and math.isfinite(base_reward):
            ratio = math.exp(base_reward)
            if ratio > EPS:
                prev_value = current_value / ratio
        if prev_value is None or prev_value <= EPS:
            return 0.0, 0.0
        fraction = cost_amount / prev_value
        return fraction * self.cost_scale, fraction

    def _extract_weights(
        self,
        info: Mapping[str, Any],
        keys: Sequence[str],
    ) -> Tuple[float, ...] | None:
        for key in keys:
            payload = info.get(key)
            if payload is None:
                continue
            if isinstance(payload, Mapping):
                ordered = (payload[key] for key in sorted(payload))
                return _sanitize_vector(ordered)
            if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
                return _sanitize_vector(payload)
            try:
                value = float(payload)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            return (value,)
        return None

    def _extract_float(self, info: Mapping[str, Any], keys: Sequence[str]) -> float | None:
        for key in keys:
            value = info.get(key)
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                return number
        return None


def _factory() -> RewardFunction:
    return StableRewardV2()


register_reward(
    RewardSpec(
        name=StableRewardV2.name,
        description="Log-return reward with turnover and drawdown penalties.",
        factory=_factory,
    )
)


__all__ = ["StableRewardV2"]
