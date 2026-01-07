"""Deterministic regime-aware mode controller."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Mapping, Sequence

import numpy as np

from research.features.regime_features_v1 import REGIME_FEATURE_COLUMNS
from research.hierarchy.modes import DEFAULT_MODE, normalize_mode


_FEATURE_INDEX = {name: idx for idx, name in enumerate(REGIME_FEATURE_COLUMNS)}


@dataclass(frozen=True)
class ControllerConfig:
    """Configuration options governing discrete mode selection."""

    update_frequency: str
    min_hold_steps: int = 5
    vol_threshold_high: float = 0.0
    trend_threshold_high: float = 0.0
    dispersion_threshold_high: float = 0.0
    fallback_mode: str = DEFAULT_MODE
    every_k_steps: int | None = None

    def __post_init__(self) -> None:
        freq = str(self.update_frequency or "").strip().lower()
        if freq not in {"weekly", "monthly", "every_k_steps"}:
            raise ValueError("update_frequency must be 'weekly', 'monthly', or 'every_k_steps'")
        object.__setattr__(self, "update_frequency", freq)
        if self.min_hold_steps < 0:
            raise ValueError("min_hold_steps must be non-negative")
        if freq == "every_k_steps":
            if self.every_k_steps is None:
                raise ValueError("every_k_steps requires 'k' to be provided in the config")
            if self.every_k_steps <= 0:
                raise ValueError("k must be positive when using every_k_steps")
        fallback = normalize_mode(self.fallback_mode)
        object.__setattr__(self, "fallback_mode", fallback)
        object.__setattr__(self, "every_k_steps", self.every_k_steps if freq == "every_k_steps" else None)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ControllerConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("controller_config must be a mapping")
        params = dict(payload)
        freq = str(params.get("update_frequency", "")).strip().lower()
        min_hold = int(params.get("min_hold_steps", 5))
        vol_threshold = float(params.get("vol_threshold_high", 0.0))
        trend_threshold = float(params.get("trend_threshold_high", 0.0))
        dispersion_threshold = float(params.get("dispersion_threshold_high", 0.0))
        fallback = str(params.get("fallback_mode", DEFAULT_MODE))
        every_k = params.get("k") if freq == "every_k_steps" else params.get("every_k_steps")
        every_k_value = int(every_k) if every_k is not None else None
        return cls(
            update_frequency=freq,
            min_hold_steps=min_hold,
            vol_threshold_high=vol_threshold,
            trend_threshold_high=trend_threshold,
            dispersion_threshold_high=dispersion_threshold,
            fallback_mode=fallback,
            every_k_steps=every_k_value,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "update_frequency": self.update_frequency,
            "min_hold_steps": int(self.min_hold_steps),
            "vol_threshold_high": float(self.vol_threshold_high),
            "trend_threshold_high": float(self.trend_threshold_high),
            "dispersion_threshold_high": float(self.dispersion_threshold_high),
            "fallback_mode": self.fallback_mode,
        }
        if self.every_k_steps is not None:
            payload["every_k_steps"] = int(self.every_k_steps)
        return payload


class ModeController:
    """Rule-based controller that selects discrete modes from regime context."""

    def __init__(self, *, config: ControllerConfig):
        self.config = config
        self._current_mode: str | None = None
        self._last_update_step: int = -1
        self._last_change_step: int = -1

    def reset(self) -> None:
        self._current_mode = None
        self._last_update_step = -1
        self._last_change_step = -1

    def select_mode(
        self,
        t: int,
        dates: Sequence[object] | np.ndarray,
        regime_features: np.ndarray | Sequence[Sequence[float]] | None,
        prev_mode: str | None,
    ) -> str:
        if t < 0:
            raise ValueError("t must be non-negative")
        if self._current_mode is None and prev_mode is not None:
            self._current_mode = normalize_mode(prev_mode)
            self._last_change_step = t
        hold_active = self._is_hold_active(t)
        update_due = self._is_update_due(t, dates)
        candidate = self._current_mode or normalize_mode(prev_mode or self.config.fallback_mode)
        if update_due and not hold_active:
            row = _row_at(regime_features, t)
            candidate = self._determine_mode(row)
        self._last_update_step = t
        normalized = normalize_mode(candidate)
        if self._current_mode is None:
            self._current_mode = normalized
            self._last_change_step = t
        elif normalized != self._current_mode and not hold_active:
            self._current_mode = normalized
            self._last_change_step = t
        return self._current_mode

    def _determine_mode(self, row: np.ndarray | None) -> str:
        if row is None or row.size == 0:
            return self.config.fallback_mode
        vol = _extract_feature(row, "market_vol_20d")
        trend = _extract_feature(row, "market_trend_20d")
        dispersion = _extract_feature(row, "dispersion_20d")
        if vol >= self.config.vol_threshold_high:
            return normalize_mode("defensive")
        if trend >= self.config.trend_threshold_high and dispersion >= self.config.dispersion_threshold_high:
            return normalize_mode("risk_on")
        return self.config.fallback_mode

    def _is_hold_active(self, t: int) -> bool:
        if self._current_mode is None:
            return False
        if self.config.min_hold_steps <= 0:
            return False
        if self._last_change_step < 0:
            return False
        return (t - self._last_change_step) < self.config.min_hold_steps

    def _is_update_due(self, t: int, dates: Sequence[object] | np.ndarray) -> bool:
        if self._last_update_step < 0:
            return True
        if self.config.update_frequency == "every_k_steps":
            assert self.config.every_k_steps is not None
            return (t - self._last_update_step) >= self.config.every_k_steps
        current = _resolve_date(dates, t)
        previous = _resolve_date(dates, self._last_update_step)
        if self.config.update_frequency == "weekly":
            return (current.isocalendar()[:2]) != (previous.isocalendar()[:2])
        if self.config.update_frequency == "monthly":
            return (current.year, current.month) != (previous.year, previous.month)
        return True


def _row_at(values: np.ndarray | Sequence[Sequence[float]] | None, idx: int) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        return arr
    if idx >= arr.shape[0]:
        idx = arr.shape[0] - 1
    return arr[idx]


def _extract_feature(row: np.ndarray, name: str) -> float:
    index = _FEATURE_INDEX.get(name)
    if index is None:
        return 0.0
    if row.ndim == 0:
        return float(row)
    if index >= row.shape[0]:
        return 0.0
    value = row[index]
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return 0.0


def _resolve_date(dates: Sequence[object] | np.ndarray, idx: int) -> date:
    if isinstance(dates, np.ndarray):
        if dates.size == 0:
            raise ValueError("dates cannot be empty")
        if idx >= len(dates):
            idx = len(dates) - 1
        value = dates[idx]
    else:
        if not dates:
            raise ValueError("dates cannot be empty")
        if idx >= len(dates):
            idx = len(dates) - 1
        value = dates[idx]
    ts = _coerce_datetime(value)
    return ts.date()


def _coerce_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    if isinstance(value, np.datetime64):
        return _datetime_from_numpy(value)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    raise TypeError("dates array must contain datetime-like objects")


def _datetime_from_numpy(value: np.datetime64) -> datetime:
    as_ns = value.astype("datetime64[ns]")
    return datetime.utcfromtimestamp(as_ns.astype("int64") / 1_000_000_000)


__all__ = ["ControllerConfig", "ModeController"]
