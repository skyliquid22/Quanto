"""Deterministic risk projection enforcing exposure constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

PROJECTION_TOLERANCE = 1e-12


@dataclass(frozen=True)
class RiskConfig:
    """Declarative configuration controlling weight projections."""

    long_only: bool = True
    max_weight: float | None = 1.0
    exposure_cap: float | None = 1.0
    min_cash: float | None = 0.0
    max_turnover_1d: float | None = None

    def __post_init__(self) -> None:
        if self.max_weight is not None and self.max_weight <= 0:
            raise ValueError("max_weight must be positive when provided")
        if self.exposure_cap is not None and self.exposure_cap <= 0:
            raise ValueError("exposure_cap must be positive when provided")
        if self.min_cash is not None:
            if self.min_cash < 0 or self.min_cash > 1:
                raise ValueError("min_cash must be between 0 and 1 inclusive when provided")
        if self.max_turnover_1d is not None and self.max_turnover_1d < 0:
            raise ValueError("max_turnover_1d must be non-negative when provided")

    def to_dict(self) -> dict[str, float | bool | None]:
        return {
            "long_only": bool(self.long_only),
            "max_weight": None if self.max_weight is None else float(self.max_weight),
            "exposure_cap": None if self.exposure_cap is None else float(self.exposure_cap),
            "min_cash": None if self.min_cash is None else float(self.min_cash),
            "max_turnover_1d": None if self.max_turnover_1d is None else float(self.max_turnover_1d),
        }


def project_weights(
    raw: Sequence[float] | Iterable[float],
    prev_weights: Sequence[float] | Iterable[float] | None,
    risk_config: RiskConfig | None = None,
) -> List[float]:
    """Project raw weights onto the feasible set defined by risk_config."""

    cfg = risk_config or RiskConfig()
    proposed = _coerce_vector(raw)
    previous = _coerce_vector(prev_weights) if prev_weights is not None else [0.0] * len(proposed)
    if len(previous) != len(proposed):
        raise ValueError("prev_weights must match the shape of raw weights")
    target = _apply_core_constraints(proposed, cfg)
    cap = _sanitize_cap(cfg.max_turnover_1d)
    if cap is not None:
        delta = [target[idx] - previous[idx] for idx in range(len(target))]
        total_turnover = sum(abs(value) for value in delta)
        if total_turnover > cap + PROJECTION_TOLERANCE:
            if cap <= PROJECTION_TOLERANCE:
                target = list(previous)
            else:
                alpha = cap / total_turnover
                adjusted = [previous[idx] + alpha * delta[idx] for idx in range(len(delta))]
                target = _apply_core_constraints(adjusted, cfg)
    return _sanitize(target)


def _apply_core_constraints(weights: Sequence[float], cfg: RiskConfig) -> List[float]:
    clipped = list(weights)
    if cfg.long_only:
        clipped = [max(value, 0.0) for value in clipped]
    if cfg.max_weight is not None:
        limit = max(cfg.max_weight, 0.0)
        clipped = [min(value, limit) for value in clipped]
    exposure_cap = _effective_exposure_cap(cfg)
    total = sum(clipped)
    if exposure_cap is not None and total > exposure_cap and total > PROJECTION_TOLERANCE:
        scale = exposure_cap / total
        clipped = [value * scale for value in clipped]
    return clipped


def _effective_exposure_cap(cfg: RiskConfig) -> float | None:
    caps: List[float] = []
    if cfg.exposure_cap is not None:
        caps.append(max(cfg.exposure_cap, 0.0))
    if cfg.min_cash is not None:
        caps.append(max(0.0, 1.0 - cfg.min_cash))
    if not caps:
        return None
    return min(caps)


def _sanitize_cap(cap: float | None) -> float | None:
    if cap is None:
        return None
    if cap <= PROJECTION_TOLERANCE:
        return 0.0
    return float(cap)


def _coerce_vector(values: Sequence[float] | Iterable[float] | None) -> List[float]:
    if values is None:
        return []
    if isinstance(values, Sequence):
        return [float(entry) for entry in values]
    return [float(entry) for entry in values]


def _sanitize(values: Sequence[float]) -> List[float]:
    sanitized: List[float] = []
    for value in values:
        if abs(value) <= PROJECTION_TOLERANCE:
            sanitized.append(0.0)
        else:
            sanitized.append(float(value))
    return sanitized


__all__ = ["PROJECTION_TOLERANCE", "RiskConfig", "project_weights"]

