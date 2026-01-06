"""Simple SMA crossover signal generation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, List, Sequence


@dataclass(frozen=True)
class SMAStrategyConfig:
    fast_window: int = 20
    slow_window: int = 50

    def __post_init__(self) -> None:
        if self.fast_window <= 0 or self.slow_window <= 0:
            raise ValueError("SMA windows must be positive")
        if self.fast_window >= self.slow_window:
            raise ValueError("fast_window must be smaller than slow_window")


@dataclass(frozen=True)
class SMAStrategyResult:
    symbol: str
    timestamps: List[datetime]
    closes: List[float]
    fast_sma: List[float | None]
    slow_sma: List[float | None]
    signal: List[int]
    positions: List[int]


def run_sma_crossover(
    symbol: str,
    timestamps: Sequence[datetime],
    closes: Sequence[float],
    config: SMAStrategyConfig,
) -> SMAStrategyResult:
    if len(closes) != len(timestamps):
        raise ValueError("timestamps and closes must be the same length")
    fast = _simple_moving_average(closes, config.fast_window)
    slow = _simple_moving_average(closes, config.slow_window)
    signals = _compute_signals(fast, slow)
    positions = _shift_positions(signals)
    return SMAStrategyResult(
        symbol=symbol,
        timestamps=list(timestamps),
        closes=list(float(value) for value in closes),
        fast_sma=fast,
        slow_sma=slow,
        signal=signals,
        positions=positions,
    )


def _simple_moving_average(values: Sequence[float], window: int) -> List[float | None]:
    results: List[float | None] = [None] * len(values)
    if window <= 0:
        return results
    running_sum = 0.0
    history: Deque[float] = deque()
    for index, value in enumerate(values):
        numeric = float(value)
        running_sum += numeric
        history.append(numeric)
        if len(history) > window:
            running_sum -= history.popleft()
        if len(history) == window:
            results[index] = running_sum / window
    return results


def _compute_signals(fast: Sequence[float | None], slow: Sequence[float | None]) -> List[int]:
    signals: List[int] = []
    for fast_value, slow_value in zip(fast, slow):
        if fast_value is None or slow_value is None:
            signals.append(0)
        elif fast_value > slow_value:
            signals.append(1)
        else:
            signals.append(0)
    return signals


def _shift_positions(signals: Sequence[int]) -> List[int]:
    if not signals:
        return []
    positions: List[int] = [0] * len(signals)
    for idx in range(1, len(signals)):
        positions[idx] = signals[idx - 1]
    return positions


__all__ = ["SMAStrategyConfig", "SMAStrategyResult", "run_sma_crossover"]
