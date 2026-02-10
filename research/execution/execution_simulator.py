"""Deterministic execution simulator for order-type gating using daily OHLC."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Sequence


ORDER_TYPES = ("market", "limit", "stop_loss", "trailing_stop")


@dataclass(frozen=True)
class ExecutionSimConfig:
    """Configuration for the execution simulator wrapper."""

    enabled: bool = False
    fill_on_next_bar: bool = True
    range_shrink_pct: float = 0.15
    slippage_bps: float = 0.0
    allow_partial_fills: bool = False
    default_order_type: str = "market"
    reward_slippage_scale: float = 0.1
    reward_missed_fill_scale: float = 0.1

    def __post_init__(self) -> None:
        if self.range_shrink_pct < 0 or self.range_shrink_pct >= 0.5:
            raise ValueError("range_shrink_pct must be in [0, 0.5).")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative.")
        if self.default_order_type not in ORDER_TYPES:
            raise ValueError(f"default_order_type must be one of {ORDER_TYPES}.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "fill_on_next_bar": bool(self.fill_on_next_bar),
            "range_shrink_pct": float(self.range_shrink_pct),
            "slippage_bps": float(self.slippage_bps),
            "allow_partial_fills": bool(self.allow_partial_fills),
            "default_order_type": self.default_order_type,
            "reward_slippage_scale": float(self.reward_slippage_scale),
            "reward_missed_fill_scale": float(self.reward_missed_fill_scale),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ExecutionSimConfig:
        if isinstance(payload, ExecutionSimConfig):
            return payload
        return cls(
            enabled=bool(payload.get("enabled", False)),
            fill_on_next_bar=bool(payload.get("fill_on_next_bar", True)),
            range_shrink_pct=float(payload.get("range_shrink_pct", 0.15)),
            slippage_bps=float(payload.get("slippage_bps", 0.0)),
            allow_partial_fills=bool(payload.get("allow_partial_fills", False)),
            default_order_type=str(payload.get("default_order_type", "market")),
            reward_slippage_scale=float(payload.get("reward_slippage_scale", 0.1)),
            reward_missed_fill_scale=float(payload.get("reward_missed_fill_scale", 0.1)),
        )


@dataclass(frozen=True)
class ExecutionSimResult:
    executed_weights: Sequence[float]
    execution_slippage_bps: float
    missed_fill_ratio: float
    unfilled_notional: float
    order_type_counts: Dict[str, int]

    def to_info(self) -> Dict[str, Any]:
        return {
            "execution_slippage_bps": float(self.execution_slippage_bps),
            "missed_fill_ratio": float(self.missed_fill_ratio),
            "unfilled_notional": float(self.unfilled_notional),
            "order_type_counts": dict(self.order_type_counts),
        }


def resolve_execution_sim_config(value: Any) -> ExecutionSimConfig | None:
    if value is None:
        return None
    if isinstance(value, ExecutionSimConfig):
        return value
    if isinstance(value, Mapping):
        return ExecutionSimConfig.from_mapping(value)
    raise TypeError("execution_sim must be a mapping or ExecutionSimConfig instance.")


def _normalize_offset(value: Any) -> float:
    if value is None:
        return 0.0
    offset = float(value)
    if abs(offset) > 1.0:
        return offset / 10_000.0
    return offset


def compute_effective_band(
    high: float,
    low: float,
    *,
    range_shrink_pct: float,
    slippage_bps: float,
) -> tuple[float, float]:
    upper = float(high)
    lower = float(low)
    if upper < lower:
        upper, lower = lower, upper
    mid = (upper + lower) / 2.0
    half = (upper - lower) / 2.0
    shrink = max(0.0, min(range_shrink_pct, 0.4999))
    span = (1.0 - shrink) * half
    effective_high = mid + span
    effective_low = mid - span
    if slippage_bps:
        slip = slippage_bps / 10_000.0
        effective_high *= 1.0 - slip
        effective_low *= 1.0 + slip
    if effective_high < effective_low:
        effective_high, effective_low = effective_low, effective_high
    return effective_high, effective_low


def attach_price_panel(
    rows: Sequence[MutableMapping[str, object]],
    slices: Mapping[str, Any],
    symbol_order: Sequence[str],
) -> Sequence[MutableMapping[str, object]]:
    if not rows:
        return rows
    try:  # pragma: no cover - depends on pandas
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pandas is required to attach price panels") from exc

    ts_list = []
    for row in rows:
        ts = pd.Timestamp(row["timestamp"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        ts_list.append(ts)
    timestamps = pd.DatetimeIndex(ts_list)
    price_frames: Dict[str, "pd.DataFrame"] = {}
    for symbol in symbol_order:
        slice_data = slices.get(symbol)
        if slice_data is None or slice_data.frame is None or slice_data.frame.empty:
            raise ValueError(f"Missing canonical OHLC data for symbol {symbol}")
        frame = slice_data.frame.copy()
        for column in ("open", "high", "low", "close"):
            if column not in frame.columns:
                raise ValueError(f"Canonical slice missing '{column}' for {symbol}")
        frame = frame[["open", "high", "low", "close"]].copy()
        frame.index = pd.to_datetime(frame.index, utc=True)
        aligned = frame.reindex(timestamps)
        if aligned.isna().any().any():
            missing = aligned.isna().any(axis=1)
            first_missing = aligned.index[missing][0] if missing.any() else None
            raise ValueError(f"Missing OHLC data for {symbol} at {first_missing}")
        price_frames[symbol] = aligned

    for idx, ts in enumerate(timestamps):
        panel: Dict[str, Dict[str, float]] = {}
        for symbol in symbol_order:
            row = price_frames[symbol].loc[ts]
            panel[symbol] = {
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
        rows[idx]["price_panel"] = panel
    return rows


class ExecutionSimulator:
    """Deterministic order-type gating using daily OHLC."""

    def __init__(self, symbols: Sequence[str], config: ExecutionSimConfig) -> None:
        self._symbols = tuple(symbols)
        self._config = config
        self._trailing_anchor: Dict[str, float | None] = {symbol: None for symbol in self._symbols}

    def resolve_step(
        self,
        *,
        prev_weights: Sequence[float],
        target_weights: Sequence[float],
        prev_value: float,
        price_panel: Mapping[str, Mapping[str, float]],
        next_price_panel: Mapping[str, Mapping[str, float]],
        execution_action: Mapping[str, Any] | None = None,
    ) -> ExecutionSimResult:
        if not self._config.enabled:
            return ExecutionSimResult(
                executed_weights=target_weights,
                execution_slippage_bps=0.0,
                missed_fill_ratio=0.0,
                unfilled_notional=0.0,
                order_type_counts={},
            )

        if isinstance(execution_action, str):
            execution_action = {"order_type": execution_action}

        total_target_notional = 0.0
        unfilled_notional = 0.0
        slippage_notional = 0.0
        filled_notional = 0.0
        order_type_counts: Counter[str] = Counter()
        executed = list(prev_weights)

        for idx, symbol in enumerate(self._symbols):
            current_weight = float(prev_weights[idx])
            target_weight = float(target_weights[idx])
            delta = target_weight - current_weight
            if abs(delta) <= 1e-12:
                self._update_trailing_anchor(symbol, price_panel, current_weight)
                continue

            total_target_notional += abs(delta) * prev_value
            instruction = self._resolve_instruction(execution_action, symbol)
            order_type = instruction["order_type"]
            limit_offset = instruction["limit_offset"]
            stop_distance = instruction["stop_distance"]
            trailing_distance = instruction["trailing_distance"]

            side = "buy" if delta > 0 else "sell"
            if side == "buy" and order_type in {"stop_loss", "trailing_stop"}:
                order_type = "market"
            order_type_counts[order_type] += 1

            ref_price = float(price_panel[symbol]["close"])
            fill_panel = next_price_panel if self._config.fill_on_next_bar else price_panel
            next_prices = fill_panel[symbol]
            next_open = float(next_prices["open"])
            next_high = float(next_prices["high"])
            next_low = float(next_prices["low"])

            effective_high, effective_low = compute_effective_band(
                next_high,
                next_low,
                range_shrink_pct=self._config.range_shrink_pct,
                slippage_bps=self._config.slippage_bps,
            )

            filled, fill_price = self._evaluate_fill(
                order_type=order_type,
                side=side,
                ref_price=ref_price,
                limit_offset=limit_offset,
                stop_distance=stop_distance,
                trailing_distance=trailing_distance,
                next_open=next_open,
                effective_high=effective_high,
                effective_low=effective_low,
                trailing_anchor=self._trailing_anchor.get(symbol),
            )

            if not filled:
                unfilled_notional += abs(delta) * prev_value
                self._update_trailing_anchor(symbol, price_panel, current_weight)
                continue

            executed[idx] = target_weight
            fill_notional = abs(delta) * prev_value
            filled_notional += fill_notional
            slippage = abs(fill_price - ref_price) / ref_price if ref_price else 0.0
            slippage_notional += slippage * fill_notional
            self._update_trailing_anchor(symbol, next_price_panel, executed[idx], use_next=True)

        missed_fill_ratio = (
            unfilled_notional / total_target_notional if total_target_notional > 0 else 0.0
        )
        execution_slippage_bps = (
            (slippage_notional / filled_notional) * 10_000.0 if filled_notional > 0 else 0.0
        )

        return ExecutionSimResult(
            executed_weights=tuple(executed),
            execution_slippage_bps=execution_slippage_bps,
            missed_fill_ratio=missed_fill_ratio,
            unfilled_notional=unfilled_notional,
            order_type_counts=dict(order_type_counts),
        )

    def reset(self) -> None:
        self._trailing_anchor = {symbol: None for symbol in self._symbols}

    def _resolve_instruction(
        self,
        execution_action: Mapping[str, Any] | None,
        symbol: str,
    ) -> Dict[str, Any]:
        order_type = self._config.default_order_type
        limit_offset = 0.0
        stop_distance = 0.0
        trailing_distance = 0.0
        if execution_action:
            order_type = str(execution_action.get("order_type", order_type))
            limit_offset = _normalize_offset(execution_action.get("limit_offset", limit_offset))
            stop_distance = _normalize_offset(execution_action.get("stop_distance", stop_distance))
            trailing_distance = _normalize_offset(execution_action.get("trailing_distance", trailing_distance))
            per_symbol = execution_action.get("per_symbol")
            if isinstance(per_symbol, Mapping) and symbol in per_symbol:
                symbol_action = per_symbol[symbol] or {}
                order_type = str(symbol_action.get("order_type", order_type))
                limit_offset = _normalize_offset(symbol_action.get("limit_offset", limit_offset))
                stop_distance = _normalize_offset(symbol_action.get("stop_distance", stop_distance))
                trailing_distance = _normalize_offset(symbol_action.get("trailing_distance", trailing_distance))
        if order_type not in ORDER_TYPES:
            raise ValueError(f"Unsupported order_type '{order_type}'")
        return {
            "order_type": order_type,
            "limit_offset": abs(limit_offset),
            "stop_distance": abs(stop_distance),
            "trailing_distance": abs(trailing_distance),
        }

    @staticmethod
    def _evaluate_fill(
        *,
        order_type: str,
        side: str,
        ref_price: float,
        limit_offset: float,
        stop_distance: float,
        trailing_distance: float,
        next_open: float,
        effective_high: float,
        effective_low: float,
        trailing_anchor: float | None,
    ) -> tuple[bool, float]:
        if order_type == "market":
            return True, next_open

        if order_type == "limit":
            if side == "buy":
                limit_price = ref_price * (1.0 - limit_offset)
                return (effective_low <= limit_price), limit_price
            limit_price = ref_price * (1.0 + limit_offset)
            return (effective_high >= limit_price), limit_price

        if order_type == "stop_loss":
            if side != "sell":
                return True, next_open
            stop_price = ref_price * (1.0 - stop_distance)
            return (effective_low <= stop_price), stop_price

        if order_type == "trailing_stop":
            if side != "sell":
                return True, next_open
            anchor = trailing_anchor if trailing_anchor is not None else ref_price
            stop_price = anchor * (1.0 - trailing_distance)
            return (effective_low <= stop_price), stop_price

        raise ValueError(f"Unsupported order_type '{order_type}'")

    def _update_trailing_anchor(
        self,
        symbol: str,
        price_panel: Mapping[str, Mapping[str, float]],
        weight: float,
    ) -> None:
        if weight <= 0:
            self._trailing_anchor[symbol] = None
            return
        high = float(price_panel[symbol]["high"])
        anchor = self._trailing_anchor.get(symbol)
        if anchor is None:
            self._trailing_anchor[symbol] = high
        else:
            self._trailing_anchor[symbol] = max(anchor, high)


__all__ = [
    "ExecutionSimConfig",
    "ExecutionSimResult",
    "ExecutionSimulator",
    "attach_price_panel",
    "compute_effective_band",
    "resolve_execution_sim_config",
]
