"""Canonical mode definitions for hierarchical controllers."""

from __future__ import annotations

from typing import Iterable, Tuple

MODE_RISK_ON = "risk_on"
MODE_NEUTRAL = "neutral"
MODE_DEFENSIVE = "defensive"


def available_modes() -> Tuple[str, ...]:
    """Return the ordered tuple of supported mode labels."""

    return (MODE_RISK_ON, MODE_NEUTRAL, MODE_DEFENSIVE)


DEFAULT_MODE = MODE_NEUTRAL


def normalize_mode(name: str) -> str:
    """Normalize and validate a mode label."""

    token = str(name or "").strip().lower()
    if not token:
        raise ValueError("mode must be provided")
    if token not in available_modes():
        raise ValueError(f"Unsupported mode '{name}'.")
    return token


def ensure_mode_inventory(modes: Iterable[str]) -> Tuple[str, ...]:
    """Validate that the provided modes cover the required inventory."""

    normalized = tuple(dict.fromkeys(normalize_mode(mode) for mode in modes))
    required = set(available_modes())
    if not required.issubset(normalized):
        missing = sorted(required - set(normalized))
        raise ValueError(f"allocator_by_mode missing required modes: {missing}")
    return normalized


__all__ = [
    "MODE_RISK_ON",
    "MODE_NEUTRAL",
    "MODE_DEFENSIVE",
    "DEFAULT_MODE",
    "available_modes",
    "normalize_mode",
    "ensure_mode_inventory",
]
