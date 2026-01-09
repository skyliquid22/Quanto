"""Baseline reward definition mirroring the environment's native log-return signal."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from research.training.reward_registry import RewardSpec, register_reward, RewardFunction


class LogReturnReward:
    """Identity reward used for reproducibility baselines."""

    name = "reward_v1"

    def reset(self) -> None:
        return None

    def compute(
        self,
        *,
        base_reward: float,
        info: Mapping[str, Any],
        step_index: int,
    ) -> Tuple[float, Dict[str, float]]:
        reward = float(base_reward)
        components = {
            "base": reward,
            "turnover_penalty": 0.0,
            "drawdown_penalty": 0.0,
            "cost_penalty": 0.0,
        }
        # Pass through any metadata without modification for downstream inspection.
        extra = info.get("reward_components")
        if isinstance(extra, Mapping):
            components.update({str(key): float(extra[key]) for key in extra if key not in components})
        return reward, components


def _factory() -> RewardFunction:
    return LogReturnReward()


register_reward(
    RewardSpec(
        name=LogReturnReward.name,
        description="Raw environment log-return reward.",
        factory=_factory,
    )
)


__all__ = ["LogReturnReward"]
