"""Order helper utilities."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from research.execution.types import Order

TOLERANCE = 1e-9
ORDER_SIDE_RANK = {"SELL": 0, "BUY": 1}


@dataclass
class OrderCandidate:
    """Intermediate representation used by the compiler."""

    symbol: str
    side: str
    qty: int
    price: float
    delta_notional: float

    @property
    def notional(self) -> float:
        return float(self.qty) * float(self.price)


def deterministic_order_id(
    *,
    run_id: str,
    step_index: int,
    symbol: str,
    side: str,
    qty: int,
) -> str:
    """Derive a deterministic order identifier from deterministic inputs."""

    payload = f"{run_id}|{step_index}|{symbol}|{side}|{qty}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{symbol}-{digest[:12]}"


def sort_orders(orders: Iterable[Order]) -> list[Order]:
    """Sort orders by symbol and side for deterministic audits."""

    return sorted(orders, key=lambda order: (order.symbol, ORDER_SIDE_RANK.get(order.side, 9), order.client_order_id))


def stable_symbol_weights(
    target_weights: Mapping[str, float],
    symbol_order: Sequence[str],
) -> dict[str, float]:
    """Emit weights mapped to the canonical symbol ordering."""

    ordered: dict[str, float] = {}
    for symbol in symbol_order:
        ordered[symbol] = float(target_weights.get(symbol, 0.0))
    for symbol, weight in target_weights.items():
        if symbol not in ordered:
            ordered[str(symbol)] = float(weight)
    return ordered


def floor_shares(value: float) -> int:
    """Deterministic rounding rule used by the compiler."""

    if abs(value) < TOLERANCE:
        return 0
    return int(math.floor(abs(value)))


__all__ = [
    "OrderCandidate",
    "TOLERANCE",
    "deterministic_order_id",
    "floor_shares",
    "sort_orders",
    "stable_symbol_weights",
]
