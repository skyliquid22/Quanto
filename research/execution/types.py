"""Core dataclasses shared across execution modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Literal, Mapping, MutableMapping, Sequence

OrderSide = Literal["BUY", "SELL"]
OrderStatus = Literal["NEW", "SUBMITTED", "PARTIALLY_FILLED", "FILLED", "REJECTED", "CANCELED"]
OrderType = Literal["MARKET"]
TimeInForce = Literal["DAY"]


def _isoformat(value: datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return value.isoformat()


@dataclass
class Order:
    """Single-leg market order used by execution controllers."""

    client_order_id: str
    symbol: str
    side: OrderSide
    qty: int
    order_type: OrderType = "MARKET"
    time_in_force: TimeInForce = "DAY"
    submitted_at: str | None = None
    status: OrderStatus = "NEW"
    broker_order_id: str | None = None
    reject_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "broker_order_id": self.broker_order_id,
            "client_order_id": self.client_order_id,
            "order_type": self.order_type,
            "qty": int(self.qty),
            "reject_reason": self.reject_reason,
            "side": self.side,
            "status": self.status,
            "symbol": self.symbol,
            "submitted_at": _isoformat(self.submitted_at),
            "time_in_force": self.time_in_force,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "Order":
        return cls(
            client_order_id=str(payload["client_order_id"]),
            symbol=str(payload["symbol"]),
            side=str(payload["side"]),
            qty=int(payload["qty"]),
            order_type=str(payload.get("order_type") or "MARKET"),
            time_in_force=str(payload.get("time_in_force") or "DAY"),
            submitted_at=payload.get("submitted_at"),
            status=str(payload.get("status") or "NEW"),
            broker_order_id=payload.get("broker_order_id"),
            reject_reason=payload.get("reject_reason"),
        )


@dataclass
class Fill:
    """Normalized broker fill."""

    client_order_id: str
    symbol: str
    qty: int
    price: float
    fees: float
    filled_at: str
    broker_trade_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "broker_trade_id": self.broker_trade_id,
            "client_order_id": self.client_order_id,
            "fees": float(self.fees),
            "filled_at": self.filled_at,
            "price": float(self.price),
            "qty": int(self.qty),
            "symbol": self.symbol,
        }


@dataclass
class OrderStatusView:
    """Snapshot returned during polling."""

    client_order_id: str
    broker_order_id: str | None
    status: OrderStatus
    filled_qty: int
    remaining_qty: int
    reject_reason: str | None = None


@dataclass
class SubmissionResult:
    """Result of submitting a batch of orders."""

    accepted: list[Order] = field(default_factory=list)
    rejected: list[Order] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class AccountSnapshot:
    """Broker account info normalized for audits."""

    as_of: str
    equity: float
    cash: float
    buying_power: float | None = None
    currency: str = "USD"


@dataclass
class PositionSnapshot:
    """Broker position info normalized for audits."""

    symbol: str
    qty: float
    avg_price: float
    market_price: float
    market_value: float


@dataclass
class RiskCheckResult:
    """Outcome of a deterministic execution risk check."""

    approved: list[Order]
    rejected: list[Order]
    halted: bool
    halt_reason: str | None
    snapshot: dict[str, float]


@dataclass
class ExecutionStepResult:
    """Aggregated results returned by the execution controller."""

    update: "PortfolioUpdate"
    compiled_orders: list[Order]
    orders_submitted: list[Order]
    orders_rejected: list[Order]
    open_orders: list[Order]
    fills: list[Fill]
    risk_snapshot: dict[str, float]
    broker_errors: list[str]
    halted: bool
    halt_reason: str | None


# Late import to avoid circular dependency when typing PortfolioUpdate in dataclasses
from research.shadow.portfolio import PortfolioUpdate  # noqa: E402

__all__ = [
    "AccountSnapshot",
    "ExecutionStepResult",
    "Fill",
    "Order",
    "OrderSide",
    "OrderStatus",
    "OrderStatusView",
    "OrderType",
    "PositionSnapshot",
    "RiskCheckResult",
    "SubmissionResult",
    "TimeInForce",
]
