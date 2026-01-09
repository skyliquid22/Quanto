"""Broker reconciliation helpers used by paper execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from research.execution.types import AccountSnapshot, PositionSnapshot
from research.paper.config import ReconciliationConfig


@dataclass
class ReconciliationResult:
    """Outcome of reconciling internal vs broker state."""

    matched: bool
    reason: str | None = None
    position_deltas: dict[str, float] = field(default_factory=dict)
    cash_delta: float = 0.0


def normalize_positions(positions: Sequence[PositionSnapshot]) -> dict[str, float]:
    """Convert broker positions into a symbol -> qty mapping."""

    normalized: dict[str, float] = {}
    for entry in positions:
        normalized[str(entry.symbol)] = float(entry.qty)
    return normalized


class PaperReconciler:
    """Validates broker snapshots against internal holdings."""

    def __init__(self, config: ReconciliationConfig) -> None:
        self._config = config

    def reconcile(
        self,
        *,
        internal_holdings: Mapping[str, float],
        broker_positions: Sequence[PositionSnapshot],
        cash: float,
        account: AccountSnapshot,
    ) -> ReconciliationResult:
        normalized = normalize_positions(broker_positions)
        deltas: dict[str, float] = {}
        for symbol, qty in internal_holdings.items():
            broker_qty = normalized.get(symbol, 0.0)
            delta = float(broker_qty) - float(qty)
            if abs(delta) > self._config.position_tolerance_shares:
                deltas[str(symbol)] = delta
        for symbol, qty in normalized.items():
            if symbol not in internal_holdings and abs(qty) > self._config.position_tolerance_shares:
                deltas[str(symbol)] = qty
        cash_delta = float(account.cash) - float(cash)
        matched = not deltas and abs(cash_delta) <= self._config.cash_tolerance_usd
        reason = None
        if deltas:
            reasons = ", ".join(sorted(f"{symbol}:{delta:+.4f}" for symbol, delta in deltas.items()))
            reason = f"position_mismatch[{reasons}]"
        elif abs(cash_delta) > self._config.cash_tolerance_usd:
            reason = f"cash_mismatch[{cash_delta:+.2f}]"
        return ReconciliationResult(
            matched=matched,
            reason=reason,
            position_deltas=deltas,
            cash_delta=cash_delta,
        )


__all__ = ["PaperReconciler", "ReconciliationResult", "normalize_positions"]
