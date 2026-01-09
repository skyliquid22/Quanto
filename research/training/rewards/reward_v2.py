"""Risk-aware reward shaping penalizing turnover, cost, and drawdowns."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Tuple

from research.training.reward_registry import RewardFunction, RewardSpec, register_reward

EPS = 1e-12


class StabilityAwareReward:
    """Reward shaping emphasizing stability over raw log returns."""

    name = "reward_v2"

    def __init__(
        self,
        *,
        turnover_scale: float = 0.5,
        cost_scale: float = 1.0,
        drawdown_scale: float = 0.5,
    ) -> None:
        self.turnover_scale = float(turnover_scale)
        self.cost_scale = float(cost_scale)
        self.drawdown_scale = float(drawdown_scale)
        self._last_weights: Tuple[float, ...] | None = None
        self._peak_value: float | None = None

    def reset(self) -> None:
        self._last_weights = None
        self._peak_value = None

    def compute(
        self,
        *,
        base_reward: float,
        info: Mapping[str, Any],
        step_index: int,
    ) -> Tuple[float, Dict[str, float]]:
        base = float(base_reward)
        weights = self._coerce_weights(info.get("weight_realized"))
        turnover_penalty = self._turnover_penalty(weights)
        cost_penalty = self._cost_penalty(base, info)
        drawdown_penalty = self._drawdown_penalty(info)
        shaped = base - turnover_penalty - cost_penalty - drawdown_penalty
        components = {
            "base": base,
            "turnover_penalty": float(turnover_penalty),
            "cost_penalty": float(cost_penalty),
            "drawdown_penalty": float(drawdown_penalty),
        }
        return shaped, components

    def _coerce_weights(self, payload: Any) -> Tuple[float, ...]:
        if payload is None:
            if self._last_weights is not None:
                return self._last_weights
            return (0.0,)
        if isinstance(payload, Mapping):
            ordered = tuple(float(payload[key]) for key in sorted(payload))
        else:
            try:
                ordered = tuple(float(value) for value in payload)  # type: ignore[arg-type]
            except TypeError:
                ordered = (float(payload),)
        if not ordered:
            ordered = (0.0,)
        return ordered

    def _turnover_penalty(self, weights: Tuple[float, ...]) -> float:
        if self._last_weights is None or len(self._last_weights) != len(weights):
            penalty = 0.0
        else:
            penalty = sum(abs(weights[idx] - self._last_weights[idx]) for idx in range(len(weights)))
        self._last_weights = weights
        return penalty * self.turnover_scale

    def _cost_penalty(self, base_reward: float, info: Mapping[str, Any]) -> float:
        next_value = float(info.get("portfolio_value", 0.0))
        ratio = math.exp(base_reward) if math.isfinite(base_reward) else 1.0
        prev_value = next_value / (ratio if ratio > EPS else 1.0)
        cost_paid = float(info.get("cost_paid", 0.0))
        cost_rate = cost_paid / prev_value if prev_value > EPS else 0.0
        return cost_rate * self.cost_scale

    def _drawdown_penalty(self, info: Mapping[str, Any]) -> float:
        value = float(info.get("portfolio_value", 0.0))
        if self._peak_value is None or value > self._peak_value:
            self._peak_value = value
            return 0.0
        if self._peak_value <= EPS:
            return 0.0
        drawdown = max(0.0, (self._peak_value - value) / self._peak_value)
        return drawdown * self.drawdown_scale


def _factory() -> RewardFunction:
    return StabilityAwareReward()


register_reward(
    RewardSpec(
        name=StabilityAwareReward.name,
        description="Log-return reward penalized by turnover, cost, and drawdown.",
        factory=_factory,
    )
)


__all__ = ["StabilityAwareReward"]
