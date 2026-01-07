"""Allocator registry wiring low-level policies for hierarchical control."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from research.policies.sma_weight_policy import SMAWeightPolicy, SMAWeightPolicyConfig

try:  # pragma: no cover - optional dependency for PPO allocators
    from stable_baselines3 import PPO as SB3PPO  # type: ignore
except Exception:  # pragma: no cover
    SB3PPO = None  # type: ignore[assignment]


class Allocator(Protocol):
    """Protocol for low-level allocators that produce raw portfolio actions."""

    def act(self, obs: np.ndarray, *, context: Mapping[str, Any]) -> np.ndarray:  # pragma: no cover - protocol
        ...


@dataclass
class EqualWeightAllocator:
    num_assets: int

    def act(self, _: np.ndarray, *, context: Mapping[str, Any]) -> np.ndarray:
        assets = int(context.get("num_assets", self.num_assets)) or self.num_assets
        if assets <= 0:
            raise ValueError("EqualWeightAllocator requires at least one asset")
        weight = 1.0 / assets
        return np.full(assets, weight, dtype=float)


class SMAAllocator:
    def __init__(
        self,
        *,
        mode: str = "hard",
        sigmoid_scale: float = 5.0,
        fast_window: int = 20,
        slow_window: int = 50,
    ) -> None:
        fast = int(fast_window)
        slow = int(slow_window)
        if fast <= 0 or slow <= 0:
            raise ValueError("SMA allocator windows must be positive")
        if fast >= slow:
            raise ValueError("SMA allocator requires fast_window < slow_window")
        self._policy = SMAWeightPolicy(
            SMAWeightPolicyConfig(mode=mode, sigmoid_scale=sigmoid_scale)
        )
        self.fast_window = fast
        self.slow_window = slow

    def act(self, _: np.ndarray, *, context: Mapping[str, Any]) -> np.ndarray:
        panel = context.get("panel")
        symbol_order: Sequence[str] = tuple(context.get("symbol_order") or ())
        if not symbol_order:
            raise ValueError("SMA allocator requires symbol_order context")
        if not isinstance(panel, Mapping):
            raise ValueError("SMA allocator requires per-symbol feature panel")
        weights: list[float] = []
        for symbol in symbol_order:
            features = panel.get(symbol)
            if not isinstance(features, Mapping):
                raise ValueError("Panel entries must be mappings of features")
            weights.append(float(self._policy.decide(features)))
        return np.asarray(weights, dtype=float)


class PPOAllocator:
    def __init__(self, checkpoint: str) -> None:
        if SB3PPO is None:  # pragma: no cover - optional dependency
            raise RuntimeError("stable_baselines3 is required for PPO allocators")
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"PPO checkpoint not found: {checkpoint}")
        self._model = SB3PPO.load(str(path))

    def act(self, obs: np.ndarray, *, context: Mapping[str, Any]) -> np.ndarray:
        action, _ = self._model.predict(obs, deterministic=True)
        arr = np.asarray(action, dtype=float).reshape(-1)
        num_assets = int(context.get("num_assets", arr.size))
        if arr.size == 1 and num_assets > 1:
            arr = np.repeat(arr, num_assets)
        if arr.size != num_assets:
            raise ValueError("PPO allocator action dimension mismatch")
        return arr


@dataclass
class DefensiveCashAllocator:
    num_assets: int
    target_exposure: float = 0.0

    def __post_init__(self) -> None:
        if self.num_assets <= 0:
            raise ValueError("defensive_cash allocator requires at least one asset")
        exposure = float(self.target_exposure)
        if exposure < 0.0 or exposure > 1.0:
            raise ValueError("target_exposure must be within [0, 1]")
        object.__setattr__(self, "target_exposure", exposure)

    def act(self, _: np.ndarray, *, context: Mapping[str, Any]) -> np.ndarray:
        assets = int(context.get("num_assets", self.num_assets)) or self.num_assets
        if assets <= 0:
            raise ValueError("defensive_cash allocator requires positive asset count")
        exposure = self.target_exposure
        if exposure <= 0.0:
            return np.zeros(assets, dtype=np.float32)
        per_asset = exposure / assets
        return np.full(assets, per_asset, dtype=np.float32)


def build_allocator(allocator_config: Mapping[str, Any], *, num_assets: int) -> Allocator:
    """Instantiate an allocator from the serialized configuration."""

    if not isinstance(allocator_config, Mapping):
        raise TypeError("allocator config must be a mapping")
    alloc_type = str(allocator_config.get("type", "")).strip().lower()
    if not alloc_type:
        raise ValueError("allocator config must include a 'type' field")
    if alloc_type == "equal_weight":
        return EqualWeightAllocator(num_assets=num_assets)
    if alloc_type == "sma":
        mode = str(allocator_config.get("mode", "hard"))
        sigmoid_scale = float(allocator_config.get("sigmoid_scale", 5.0))
        fast_window = int(allocator_config.get("fast_window", 20))
        slow_window = int(allocator_config.get("slow_window", 50))
        return SMAAllocator(
            mode=mode,
            sigmoid_scale=sigmoid_scale,
            fast_window=fast_window,
            slow_window=slow_window,
        )
    if alloc_type == "ppo":
        checkpoint = allocator_config.get("checkpoint")
        if not checkpoint:
            raise ValueError("ppo allocator requires a checkpoint path")
        return PPOAllocator(str(checkpoint))
    if alloc_type == "defensive_cash":
        exposure = float(allocator_config.get("target_exposure", 0.0))
        return DefensiveCashAllocator(num_assets=num_assets, target_exposure=exposure)
    raise ValueError(f"Unsupported allocator type '{alloc_type}'")


__all__ = ["Allocator", "build_allocator"]
