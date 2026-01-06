"""Continuous target-weight trading environment for deterministic rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Iterable, List, Mapping, MutableMapping, Sequence, Tuple


@dataclass(frozen=True)
class SignalWeightEnvConfig:
    """Configuration controlling portfolio mechanics."""

    initial_cash: float = 10_000.0
    transaction_cost_bp: float = 1.0
    allow_short: bool = False
    action_clip: Tuple[float, float] = (0.0, 1.0)

    def __post_init__(self) -> None:
        low, high = self.action_clip
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if self.transaction_cost_bp < 0:
            raise ValueError("transaction_cost_bp must be non-negative")
        if high < low:
            raise ValueError("action_clip upper bound cannot be less than lower bound")
        if not self.allow_short:
            if low < 0.0:
                raise ValueError("action_clip lower bound cannot be negative when allow_short=False")
            if high > 1.0:
                raise ValueError("action_clip upper bound cannot exceed 1.0 when allow_short=False")


class SignalWeightTradingEnv:
    """Minimal RL-style environment using canonical features as observations."""

    _default_feature_columns = ("close", "sma_fast", "sma_slow", "sma_diff")

    def __init__(
        self,
        rows: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
        config: SignalWeightEnvConfig | None = None,
        observation_columns: Sequence[str] | None = None,
    ) -> None:
        entries = [self._coerce_row(row) for row in rows]
        entries.sort(key=lambda item: item["timestamp"])
        if len(entries) < 2:
            raise ValueError("at least two rows are required to run the environment")
        self._rows: List[MutableMapping[str, object]] = entries
        self.config = config or SignalWeightEnvConfig()
        self._portfolio_value: float = self.config.initial_cash
        self._current_weight: float = 0.0
        self._step_index: int = 0
        self._done: bool = False
        feature_columns = tuple(str(col).strip() for col in (observation_columns or self._default_feature_columns))
        if not feature_columns:
            raise ValueError("observation_columns must include at least one feature")
        if "close" not in feature_columns:
            raise ValueError("observation_columns must include 'close' so the environment can value positions")
        self._feature_columns = feature_columns
        self._observation_columns = feature_columns + ("prev_weight",)

    @property
    def observation_columns(self) -> Tuple[str, ...]:
        return self._observation_columns

    @property
    def current_row(self) -> Mapping[str, object]:
        return self._rows[self._step_index]

    @property
    def portfolio_value(self) -> float:
        return self._portfolio_value

    @property
    def current_weight(self) -> float:
        return self._current_weight

    def reset(self) -> Tuple[float, ...]:
        self._portfolio_value = self.config.initial_cash
        self._current_weight = 0.0
        self._step_index = 0
        self._done = False
        return self._build_observation()

    def step(self, action: float) -> Tuple[Tuple[float, ...], float, bool, Mapping[str, object]]:
        if self._done:
            raise RuntimeError("cannot step once the environment is done; call reset() first")

        row = self._rows[self._step_index]
        next_row = self._rows[self._step_index + 1]
        price_now = float(row["close"])
        price_next = float(next_row["close"])
        prev_value = self._portfolio_value

        target_weight = self._clip_action(action)
        current_value = self._current_weight * prev_value
        target_value = target_weight * prev_value
        trade_value = target_value - current_value
        cost_rate = self.config.transaction_cost_bp / 10_000.0
        cost_paid = abs(trade_value) * cost_rate
        post_cost_value = prev_value - cost_paid
        pct_return = (price_next - price_now) / price_now if price_now else 0.0
        next_value = post_cost_value * (1.0 + target_weight * pct_return)
        if next_value <= 0:
            raise RuntimeError("portfolio value became non-positive; check inputs")

        reward = math.log(next_value / prev_value) if prev_value > 0 else 0.0
        self._portfolio_value = next_value
        self._current_weight = target_weight
        self._step_index += 1
        done = self._step_index >= len(self._rows) - 1
        self._done = done

        info = {
            "timestamp": row["timestamp"],
            "price_close": price_now,
            "weight_target": target_weight,
            "weight_realized": target_weight,
            "portfolio_value": next_value,
            "cost_paid": cost_paid,
            "reward": reward,
        }
        return self._build_observation(), reward, done, info

    def _clip_action(self, value: float) -> float:
        low, high = self.config.action_clip
        return max(low, min(high, float(value)))

    def _build_observation(self) -> Tuple[float, ...]:
        row = self._rows[self._step_index]
        values = []
        for column in self._feature_columns:
            value = row.get(column)
            if value is None:
                raise ValueError(f"Row missing required observation column '{column}'")
            values.append(float(value))
        values.append(float(self._current_weight))
        return tuple(values)

    @staticmethod
    def _coerce_row(raw: Mapping[str, object]) -> MutableMapping[str, object]:
        timestamp = raw.get("timestamp")
        if not isinstance(timestamp, datetime):
            raise TypeError("rows must include datetime timestamps")
        close = raw.get("close")
        if close is None:
            raise ValueError("rows must include close prices")
        normalized: MutableMapping[str, object] = dict(raw)
        normalized["timestamp"] = timestamp
        normalized["close"] = float(close)
        fast = normalized.get("sma_fast")
        slow = normalized.get("sma_slow")
        if "sma_diff" not in normalized and fast is not None and slow is not None:
            normalized["sma_diff"] = float(fast) - float(slow)
        if "sma_signal" not in normalized:
            normalized["sma_signal"] = float(raw.get("sma_signal", 0.0))
        return normalized


__all__ = ["SignalWeightEnvConfig", "SignalWeightTradingEnv"]
