"""Deterministic simulation broker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from research.execution.broker_base import BrokerAdapter
from research.execution.types import (
    AccountSnapshot,
    Fill,
    Order,
    OrderStatusView,
    PositionSnapshot,
    SubmissionResult,
)


@dataclass(frozen=True)
class SimBrokerConfig:
    """Static fill / fee model."""

    slippage_bps: float = 0.0
    fee_bps: float = 0.0
    fee_per_order: float = 0.0


class SimBrokerAdapter(BrokerAdapter):
    """Fills all orders deterministically at the provided price source."""

    name = "sim"

    def __init__(self, config: SimBrokerConfig | None = None) -> None:
        self._config = config or SimBrokerConfig()
        self._prices: dict[str, float] = {}
        self._fills: list[Fill] = []
        self._last_orders: dict[str, Order] = {}

    def update_market_data(self, prices: Mapping[str, float]) -> None:
        self._prices = {symbol: float(price) for symbol, price in prices.items()}

    def submit_orders(self, orders: Sequence[Order], *, as_of: str) -> SubmissionResult:
        accepted: list[Order] = []
        rejected: list[Order] = []
        errors: list[str] = []
        self._fills = []
        for order in orders:
            price = self._prices.get(order.symbol)
            if price is None or price <= 0:
                order.status = "REJECTED"
                order.reject_reason = "MISSING_PRICE"
                rejected.append(order)
                continue
            order.submitted_at = as_of
            order.status = "FILLED"
            order.broker_order_id = order.client_order_id
            exec_price = self._apply_slippage(price, order.side)
            fees = self._compute_fees(exec_price, order.qty)
            fill = Fill(
                client_order_id=order.client_order_id,
                symbol=order.symbol,
                qty=int(order.qty),
                price=float(exec_price),
                fees=float(fees),
                filled_at=as_of,
                broker_trade_id=f"{order.client_order_id}-fill",
            )
            self._fills.append(fill)
            accepted.append(order)
            self._last_orders[order.client_order_id] = order
        return SubmissionResult(accepted=accepted, rejected=rejected, errors=errors)

    def poll_orders(self, broker_order_ids: Sequence[str], *, as_of: str) -> list[OrderStatusView]:
        statuses: list[OrderStatusView] = []
        for order_id in broker_order_ids:
            ref = self._last_orders.get(order_id)
            if ref is None:
                continue
            statuses.append(
                OrderStatusView(
                    client_order_id=ref.client_order_id,
                    broker_order_id=ref.broker_order_id,
                    status=ref.status,
                    filled_qty=ref.qty if ref.status == "FILLED" else 0,
                    remaining_qty=0 if ref.status == "FILLED" else ref.qty,
                    reject_reason=ref.reject_reason,
                )
            )
        return statuses

    def fetch_fills(self, broker_order_ids: Sequence[str], *, as_of: str) -> list[Fill]:
        if not broker_order_ids:
            return []
        lookup = set(broker_order_ids)
        return [fill for fill in self._fills if fill.client_order_id in lookup]

    def get_account(self, *, as_of: str) -> AccountSnapshot:
        return AccountSnapshot(as_of=as_of, equity=0.0, cash=0.0)

    def get_positions(self, *, as_of: str) -> list[PositionSnapshot]:
        return []

    def _apply_slippage(self, price: float, side: str) -> float:
        slippage = max(self._config.slippage_bps, 0.0) / 10_000.0
        if side == "BUY":
            return price * (1.0 + slippage)
        return price * (1.0 - slippage)

    def _compute_fees(self, price: float, qty: int) -> float:
        notional = float(abs(qty)) * float(price)
        return notional * max(self._config.fee_bps, 0.0) / 10_000.0 + max(self._config.fee_per_order, 0.0)


__all__ = ["SimBrokerAdapter", "SimBrokerConfig"]
