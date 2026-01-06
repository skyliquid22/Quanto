"""Deterministic SMA-to-weight policy used by baseline rollouts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class SMAWeightPolicyConfig:
    fast_key: str = "sma_fast"
    slow_key: str = "sma_slow"
    mode: str = "hard"  # either \"hard\" or \"sigmoid\"
    sigmoid_scale: float = 5.0

    def __post_init__(self) -> None:
        if self.mode not in {"hard", "sigmoid"}:
            raise ValueError("mode must be 'hard' or 'sigmoid'")
        if self.sigmoid_scale <= 0:
            raise ValueError("sigmoid_scale must be positive")


class SMAWeightPolicy:
    """Map SMA relationships to continuous weights deterministically."""

    def __init__(self, config: SMAWeightPolicyConfig | None = None) -> None:
        self.config = config or SMAWeightPolicyConfig()

    @property
    def mode(self) -> str:
        return self.config.mode

    def decide(self, features: Mapping[str, object]) -> float:
        fast = features.get(self.config.fast_key)
        slow = features.get(self.config.slow_key)
        if fast is None or slow is None:
            return 0.0
        diff = float(fast) - float(slow)
        if self.config.mode == "hard":
            return 1.0 if diff > 0 else 0.0
        return self._sigmoid_weight(diff)

    def _sigmoid_weight(self, diff: float) -> float:
        value = 1.0 / (1.0 + math.exp(-self.config.sigmoid_scale * diff))
        return max(0.0, min(1.0, value))


__all__ = ["SMAWeightPolicy", "SMAWeightPolicyConfig"]
