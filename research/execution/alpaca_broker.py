"""Alpaca paper-trading REST adapter."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Sequence

import httpx

from research.execution.broker_base import BrokerAdapter
from research.execution.types import (
    AccountSnapshot,
    Fill,
    Order,
    OrderStatusView,
    PositionSnapshot,
    SubmissionResult,
)

# Alpaca order-status → internal status mapping.
_STATUS_MAP: dict[str, str] = {
    "new": "SUBMITTED",
    "accepted": "SUBMITTED",
    "pending_new": "SUBMITTED",
    "accepted_for_bidding": "SUBMITTED",
    "held": "SUBMITTED",
    "partially_filled": "PARTIALLY_FILLED",
    "filled": "FILLED",
    "done_for_day": "CANCELED",
    "canceled": "CANCELED",
    "expired": "CANCELED",
    "replaced": "CANCELED",
    "stopped": "CANCELED",
    "suspended": "REJECTED",
    "rejected": "REJECTED",
    "calculated": "SUBMITTED",
    "pending_cancel": "SUBMITTED",
    "pending_replace": "SUBMITTED",
}


def _map_status(alpaca_status: str) -> str:
    return _STATUS_MAP.get(str(alpaca_status).lower(), "SUBMITTED")


def _auth_headers(api_key: str, secret_key: str) -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }


def _raise_for_status(response: httpx.Response) -> None:
    """Raise with a human-readable message that includes the response body."""
    if response.is_error:
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        raise httpx.HTTPStatusError(
            f"Alpaca API error {response.status_code}: {detail}",
            request=response.request,
            response=response,
        )


@dataclass(frozen=True)
class AlpacaBrokerConfig:
    """Configuration required by the Alpaca adapter."""

    api_key: str
    secret_key: str
    base_url: str

    @classmethod
    def from_env(cls) -> "AlpacaBrokerConfig":
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        if not api_key or not secret_key:
            raise RuntimeError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be configured for alpaca execution mode."
            )
        return cls(api_key=api_key, secret_key=secret_key, base_url=base_url)


class AlpacaBrokerAdapter(BrokerAdapter):
    """Alpaca paper-trading REST adapter.

    All requests target the paper trading environment.  The ``base_url``
    must contain the string ``"paper"`` — the adapter refuses to connect to
    live endpoints.
    """

    name = "alpaca"

    def __init__(self, config: AlpacaBrokerConfig | None = None) -> None:
        resolved = config or AlpacaBrokerConfig.from_env()
        if not resolved.api_key or not resolved.secret_key:
            raise RuntimeError("Alpaca adapter requires non-empty API credentials.")
        base_url = str(resolved.base_url or "")
        if "paper" not in base_url:
            raise RuntimeError(
                "Alpaca adapter is locked to paper trading; refusing non-paper base URL."
            )
        self._config = resolved
        self._base_url = base_url.rstrip("/")
        self._headers = _auth_headers(resolved.api_key, resolved.secret_key)

    # ------------------------------------------------------------------
    # BrokerAdapter interface
    # ------------------------------------------------------------------

    def submit_orders(self, orders: Sequence[Order], *, as_of: str) -> SubmissionResult:
        result = SubmissionResult()
        if not orders:
            return result
        with httpx.Client(timeout=30.0) as client:
            for order in orders:
                try:
                    submitted = self._submit_one(order, client)
                    result.accepted.append(submitted)
                except httpx.HTTPStatusError as exc:
                    order.status = "REJECTED"
                    order.reject_reason = str(exc)
                    result.rejected.append(order)
                    result.errors.append(str(exc))
        return result

    def poll_orders(
        self, broker_order_ids: Sequence[str], *, as_of: str
    ) -> list[OrderStatusView]:
        if not broker_order_ids:
            return []
        views: list[OrderStatusView] = []
        with httpx.Client(timeout=30.0) as client:
            for broker_id in broker_order_ids:
                payload = self._get_order(broker_id, client)
                views.append(_parse_order_status_view(payload))
        return views

    def fetch_fills(
        self, broker_order_ids: Sequence[str], *, as_of: str
    ) -> list[Fill]:
        if not broker_order_ids:
            return []
        fills: list[Fill] = []
        with httpx.Client(timeout=30.0) as client:
            for broker_id in broker_order_ids:
                payload = self._get_order(broker_id, client)
                fill = _parse_fill(payload)
                if fill is not None:
                    fills.append(fill)
        return fills

    def get_account(self, *, as_of: str) -> AccountSnapshot:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                f"{self._base_url}/v2/account",
                headers=self._headers,
            )
            _raise_for_status(response)
            payload = response.json()
        return AccountSnapshot(
            as_of=as_of,
            equity=float(payload.get("equity") or 0.0),
            cash=float(payload.get("cash") or 0.0),
            buying_power=float(payload.get("buying_power") or 0.0),
            currency=str(payload.get("currency") or "USD"),
        )

    def get_positions(self, *, as_of: str) -> list[PositionSnapshot]:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                f"{self._base_url}/v2/positions",
                headers=self._headers,
            )
            _raise_for_status(response)
            payload = response.json()
        positions: list[PositionSnapshot] = []
        for item in payload or []:
            positions.append(
                PositionSnapshot(
                    symbol=str(item.get("symbol") or ""),
                    qty=float(item.get("qty") or 0.0),
                    avg_price=float(item.get("avg_entry_price") or 0.0),
                    market_price=float(item.get("current_price") or 0.0),
                    market_value=float(item.get("market_value") or 0.0),
                )
            )
        return positions

    def normalize_error(self, exc: Exception) -> str:
        if isinstance(exc, httpx.HTTPStatusError):
            try:
                detail = exc.response.json()
                return f"alpaca_http_{exc.response.status_code}: {detail.get('message', str(detail))}"
            except Exception:
                return f"alpaca_http_{exc.response.status_code}"
        return f"alpaca_error: {exc}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _submit_one(self, order: Order, client: httpx.Client) -> Order:
        body: dict[str, Any] = {
            "symbol": order.symbol,
            "qty": str(int(order.qty)),
            "side": order.side.lower(),
            "type": "market",
            "time_in_force": "day",
            "client_order_id": order.client_order_id,
        }
        response = client.post(
            f"{self._base_url}/v2/orders",
            json=body,
            headers=self._headers,
        )
        _raise_for_status(response)
        payload = response.json()
        order.broker_order_id = str(payload.get("id") or "")
        order.status = _map_status(str(payload.get("status") or "new"))
        order.submitted_at = str(payload.get("submitted_at") or "")
        return order

    def _get_order(self, broker_order_id: str, client: httpx.Client) -> dict[str, Any]:
        response = client.get(
            f"{self._base_url}/v2/orders/{broker_order_id}",
            headers=self._headers,
        )
        _raise_for_status(response)
        return response.json()


def _parse_order_status_view(payload: dict[str, Any]) -> OrderStatusView:
    alpaca_status = str(payload.get("status") or "new")
    internal_status = _map_status(alpaca_status)
    filled_qty = int(float(payload.get("filled_qty") or 0))
    ordered_qty = int(float(payload.get("qty") or 0))
    remaining = max(0, ordered_qty - filled_qty)
    reject_reason: str | None = None
    if internal_status == "REJECTED":
        reject_reason = str(payload.get("failed_at") or alpaca_status)
    return OrderStatusView(
        client_order_id=str(payload.get("client_order_id") or ""),
        broker_order_id=str(payload.get("id") or ""),
        status=internal_status,
        filled_qty=filled_qty,
        remaining_qty=remaining,
        reject_reason=reject_reason,
    )


def _parse_fill(payload: dict[str, Any]) -> Fill | None:
    """Return a Fill if the order is fully or partially filled, else None."""
    filled_qty = int(float(payload.get("filled_qty") or 0))
    if filled_qty <= 0:
        return None
    filled_avg_price = float(payload.get("filled_avg_price") or 0.0)
    if filled_avg_price <= 0.0:
        return None
    return Fill(
        client_order_id=str(payload.get("client_order_id") or ""),
        symbol=str(payload.get("symbol") or ""),
        qty=filled_qty,
        price=filled_avg_price,
        fees=0.0,  # Alpaca paper trading has no commissions.
        filled_at=str(payload.get("filled_at") or payload.get("updated_at") or ""),
        broker_trade_id=str(payload.get("id") or ""),
    )


__all__ = ["AlpacaBrokerAdapter", "AlpacaBrokerConfig"]
