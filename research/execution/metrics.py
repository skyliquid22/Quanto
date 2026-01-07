"""Execution metrics recorder."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

from research.execution.types import Fill, Order

_REGIME_BUCKETS = ("high_vol", "mid_vol", "low_vol")


@dataclass
class _RegimeExecutionStats:
    submitted: int = 0
    filled: int = 0
    rejected: int = 0
    slippage_weighted_sum: float = 0.0
    slippage_notional: float = 0.0
    slippage_samples: list[tuple[float, float]] = field(default_factory=list)

    def record(
        self,
        *,
        submitted: int,
        filled: int,
        rejected: int,
        slippage_samples: Sequence[tuple[float, float]],
    ) -> None:
        self.submitted += int(submitted)
        self.filled += int(filled)
        self.rejected += int(rejected)
        for value, notional in slippage_samples:
            if notional <= 0:
                continue
            self.slippage_weighted_sum += value * notional
            self.slippage_notional += notional
            self.slippage_samples.append((value, notional))

    def summary(self) -> dict[str, float]:
        total_submitted = max(self.submitted, 1)
        total_attempts = max(self.submitted + self.rejected, 1)
        return {
            "fill_rate": float(self.filled / total_submitted),
            "reject_rate": float(self.rejected / total_attempts),
            "avg_slippage_bps": float(_weighted_average(self.slippage_weighted_sum, self.slippage_notional)),
            "p95_slippage_bps": float(_weighted_quantile(self.slippage_samples, 0.95) or 0.0),
        }


@dataclass
class ExecutionMetricsRecorder:
    """Aggregates per-step execution statistics."""

    _orders_submitted: int = 0
    _orders_rejected: int = 0
    _orders_filled: int = 0
    _fees: float = 0.0
    _turnover: float = 0.0
    _halts: list[str] = field(default_factory=list)
    _halt_events: int = 0
    _slippage_weighted_sum: float = 0.0
    _slippage_notional: float = 0.0
    _slippage_samples: list[tuple[float, float]] = field(default_factory=list)
    _order_qty: Dict[str, float] = field(default_factory=dict)
    _order_fills: Dict[str, float] = field(default_factory=dict)
    _regime_stats: Dict[str, _RegimeExecutionStats] = field(default_factory=dict)

    def record_step(
        self,
        *,
        compiled_orders: Iterable[Order],
        submitted_orders: Iterable[Order],
        rejected_orders: Iterable[Order],
        fills: Iterable[Fill],
        reference_prices: Mapping[str, float],
        side_lookup: Mapping[str, str],
        regime_bucket: str | None = None,
    ) -> None:
        compiled_orders = list(compiled_orders)
        submitted_orders = list(submitted_orders)
        rejected_orders = list(rejected_orders)
        fills = list(fills)
        self._orders_submitted += len(submitted_orders)
        self._orders_rejected += len(rejected_orders)
        for order in submitted_orders:
            self._order_qty[order.client_order_id] = self._order_qty.get(order.client_order_id, 0.0) + abs(order.qty)

        filled_by_order: Dict[str, float] = {}
        slippage_samples: list[tuple[float, float]] = []
        for fill in fills:
            reference_price = reference_prices.get(fill.symbol) or fill.price
            side = side_lookup.get(fill.client_order_id)
            if reference_price and reference_price > 0 and side:
                direction = 1.0 if side == "BUY" else -1.0
                slippage = ((fill.price - reference_price) / reference_price) * 10_000.0 * direction
                notional = abs(fill.qty) * reference_price
                if notional > 0:
                    self._slippage_weighted_sum += slippage * notional
                    self._slippage_notional += notional
                    self._slippage_samples.append((float(slippage), float(notional)))
                    slippage_samples.append((float(slippage), float(notional)))
            filled_by_order[fill.client_order_id] = filled_by_order.get(fill.client_order_id, 0.0) + abs(fill.qty)
        for client_id, qty in filled_by_order.items():
            self._order_fills[client_id] = self._order_fills.get(client_id, 0.0) + qty

        self._orders_filled += len(filled_by_order)
        self._fees += sum(float(fill.fees) for fill in fills)

        if regime_bucket:
            stats = self._regime_stats.setdefault(regime_bucket, _RegimeExecutionStats())
            stats.record(
                submitted=len(submitted_orders),
                filled=len(filled_by_order),
                rejected=len(rejected_orders),
                slippage_samples=slippage_samples,
            )

    def record_turnover(self, value: float) -> None:
        self._turnover += max(float(value), 0.0)

    def record_halt(self, reason: str | None) -> None:
        self._halt_events += 1
        if reason:
            self._halts.append(reason)

    def summary(self) -> dict[str, object]:
        return self.snapshot()["summary"]

    def snapshot(self) -> dict[str, object]:
        summary = {
            "fill_rate": float(self._orders_filled / max(self._orders_submitted, 1)),
            "reject_rate": float(self._orders_rejected / max(self._orders_submitted + self._orders_rejected, 1)),
            "avg_slippage_bps": float(_weighted_average(self._slippage_weighted_sum, self._slippage_notional)),
            "p95_slippage_bps": float(_weighted_quantile(self._slippage_samples, 0.95) or 0.0),
            "total_fees": float(self._fees),
            "turnover_realized": float(self._turnover),
            "execution_halts": self._halt_events,
            "halt_reasons": list(self._halts),
            "order_latency_ms": {},
            "partial_fill_rate": float(self._partial_fill_rate()),
        }
        regime_metrics = {
            bucket: stats.summary()
            for bucket, stats in sorted(self._regime_stats.items())
            if bucket in _REGIME_BUCKETS
        }
        return {
            "summary": summary,
            "regime": regime_metrics,
        }

    def write(self, path: Path) -> Path:
        payload = self.snapshot()
        path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        return path

    def _partial_fill_rate(self) -> float:
        partial = 0
        for client_id, requested in self._order_qty.items():
            filled = self._order_fills.get(client_id, 0.0)
            if 0.0 < filled < requested:
                partial += 1
        return partial / max(self._orders_submitted, 1)


def _weighted_average(weighted_sum: float, total_weight: float) -> float:
    if total_weight <= 0:
        return 0.0
    return weighted_sum / total_weight


def _weighted_quantile(
    samples: Sequence[tuple[float, float]],
    percentile: float,
) -> float | None:
    if not samples:
        return None
    total_weight = sum(max(weight, 0.0) for _, weight in samples)
    if total_weight <= 0:
        return None
    cutoff = percentile * total_weight
    cumulative = 0.0
    for value, weight in sorted(samples, key=lambda entry: entry[0]):
        w = max(weight, 0.0)
        cumulative += w
        if cumulative >= cutoff:
            return value
    return samples[-1][0]


__all__ = ["ExecutionMetricsRecorder"]
