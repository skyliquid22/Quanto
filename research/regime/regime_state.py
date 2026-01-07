"""Lightweight container for per-timestep regime context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RegimeState:
    """Snapshot of regime feature values for a given timestep."""

    features: np.ndarray
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        normalized = np.asarray(self.features, dtype="float64")
        if normalized.ndim != 1:
            normalized = normalized.reshape(-1)
        object.__setattr__(self, "features", normalized)
        if len(self.feature_names) != len(normalized):
            raise ValueError("feature_names length must match features dimension")

    def as_dict(self) -> dict[str, float]:
        return {name: float(self.features[idx]) for idx, name in enumerate(self.feature_names)}

    def as_tuple(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.features)


def build_regime_state(values: Sequence[float], feature_names: Sequence[str]) -> RegimeState:
    return RegimeState(features=np.asarray(values, dtype="float64"), feature_names=tuple(feature_names))


__all__ = ["RegimeState", "build_regime_state"]
