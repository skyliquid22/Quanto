"""Versioned reward registry and factories used by PPO training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Protocol, Tuple


class RewardFunction(Protocol):
    """Protocol describing the methods exposed by concrete reward functions."""

    name: str

    def reset(self) -> None:  # pragma: no cover - exercised through adapters
        ...

    def compute(self, *, base_reward: float, info: Mapping[str, Any], step_index: int) -> Tuple[float, Dict[str, float]]:
        ...


@dataclass(frozen=True)
class RewardSpec:
    """Metadata describing a registered reward version."""

    name: str
    description: str
    factory: Callable[[], RewardFunction]


_REGISTRY: Dict[str, RewardSpec] = {}


def register_reward(spec: RewardSpec) -> None:
    """Register (or overwrite) a reward spec."""

    key = _normalize_key(spec.name)
    if not key:
        raise ValueError("reward spec name must be non-empty")
    _REGISTRY[key] = RewardSpec(name=spec.name, description=spec.description, factory=spec.factory)


def get_reward_spec(name: str) -> RewardSpec:
    """Return the registered spec for the requested reward version."""

    key = _normalize_key(name)
    if key not in _REGISTRY:
        raise KeyError(f"Unknown reward version '{name}'")
    return _REGISTRY[key]


def create_reward(name: str | None) -> RewardFunction:
    """Instantiate a reward function by version identifier (defaults to reward_v1)."""

    resolved = name or "reward_v1"
    spec = get_reward_spec(resolved)
    return spec.factory()


def reward_versions() -> Tuple[str, ...]:
    """Return the sorted tuple of registered reward identifiers."""

    return tuple(sorted(spec.name for spec in _REGISTRY.values()))


def reward_specs() -> Tuple[RewardSpec, ...]:
    """Return all registered specs (mostly for inspection/testing)."""

    ordered = sorted(_REGISTRY.values(), key=lambda spec: spec.name)
    return tuple(ordered)


def _normalize_key(name: str) -> str:
    token = str(name or "").strip().lower()
    return token


# Ensure the default reward versions are registered when the registry module is imported.
def _bootstrap_default_rewards() -> None:  # pragma: no cover - import side-effect
    from research.training.rewards import reward_v1, reward_v2  # noqa: F401


_bootstrap_default_rewards()


__all__ = [
    "RewardFunction",
    "RewardSpec",
    "create_reward",
    "get_reward_spec",
    "register_reward",
    "reward_specs",
    "reward_versions",
]
