"""Execution metrics recorder."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from research.execution.types import Fill, Order


@dataclass
class ExecutionMetricsRecorder:
    """Aggregates per-step execution statistics."""

    _orders_requested: int = 0
    _orders_submitted: int = 0
    _orders_rejected: int = 0
    _requested_qty: int = 0
    _filled_qty: int = 0
    _fills_notional: float = 0.0
    _fees: float = 0.0
    _slippage_bps: list[float] = field(default_factory=list)
    _turnover: float = 0.0
    _halts: list[str] = field(default_factory=list)

    def record_step(
        self,
        *,
        compiled_orders: Iterable[Order],
        submitted_orders: Iterable[Order],
        rejected_orders: Iterable[Order],
        fills: Iterable[Fill],
        reference_prices: Mapping[str, float],
        side_lookup: Mapping[str, str],
    ) -> None:
        compiled_orders = list(compiled_orders)
        submitted_orders = list(submitted_orders)
        rejected_orders = list(rejected_orders)
        fills = list(fills)
        self._orders_requested += len(compiled_orders)
        self._orders_submitted += len(submitted_orders)
        self._orders_rejected += len(rejected_orders)
        self._requested_qty += sum(abs(order.qty) for order in compiled_orders)
        filled_qty = sum(abs(fill.qty) for fill in fills)
        self._filled_qty += filled_qty
        self._fees += sum(fill.fees for fill in fills)
        self._fills_notional += sum(abs(fill.qty) * fill.price for fill in fills)
        for fill in fills:
            price = reference_prices.get(fill.symbol)
            side = side_lookup.get(fill.client_order_id)
            if not price or price <= 0 or not side:
                continue
            direction = 1.0 if side == "BUY" else -1.0
            slippage = ((fill.price - price) / price) * 10_000.0 * direction
            self._slippage_bps.append(float(slippage))

    def record_turnover(self, value: float) -> None:
        self._turnover += max(float(value), 0.0)

    def record_halt(self, reason: str | None) -> None:
        if reason:
            self._halts.append(reason)

    def summary(self) -> dict[str, object]:
        total_orders = max(1, self._orders_requested)
        fill_rate = self._filled_qty / max(1, self._requested_qty) if self._requested_qty else 0.0
        reject_rate = self._orders_rejected / total_orders
        slippage_avg = sum(self._slippage_bps) / len(self._slippage_bps) if self._slippage_bps else 0.0
        return {
            "fill_rate": float(fill_rate),
            "reject_rate": float(reject_rate),
            "avg_slippage_bps": float(slippage_avg),
            "total_fees": float(self._fees),
            "turnover_realized": float(self._fills_notional),
            "latency_buckets_sec": {},
            "halts_count": len(self._halts),
            "halt_reasons": list(self._halts),
        }

    def write(self, path: Path) -> Path:
        payload = self.summary()
        path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        return path


__all__ = ["ExecutionMetricsRecorder"]
