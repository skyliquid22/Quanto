"""Abstract broker adapter used by execution modes."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from research.execution.types import (
    AccountSnapshot,
    Fill,
    Order,
    OrderStatusView,
    PositionSnapshot,
    SubmissionResult,
)


class BrokerAdapter:
    """Interface implemented by broker-specific adapters."""

    name = "abstract"

    def submit_orders(self, orders: Sequence[Order], *, as_of: str) -> SubmissionResult:  # pragma: no cover - interface
        raise NotImplementedError

    def poll_orders(self, broker_order_ids: Sequence[str], *, as_of: str) -> list[OrderStatusView]:  # pragma: no cover
        raise NotImplementedError

    def fetch_fills(self, broker_order_ids: Sequence[str], *, as_of: str) -> list[Fill]:  # pragma: no cover
        raise NotImplementedError

    def get_account(self, *, as_of: str) -> AccountSnapshot:  # pragma: no cover
        raise NotImplementedError

    def get_positions(self, *, as_of: str) -> list[PositionSnapshot]:  # pragma: no cover
        raise NotImplementedError

    def update_market_data(self, prices: Mapping[str, float]) -> None:  # pragma: no cover - optional hook
        """Hook for brokers needing current price context."""

    def normalize_error(self, exc: Exception) -> str:
        return f"{self.name}:{exc}"


__all__ = ["BrokerAdapter"]
