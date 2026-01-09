"""Paper Alpaca adapter skeleton used for integration tests."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

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
            raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be configured for alpaca execution mode.")
        return cls(api_key=api_key, secret_key=secret_key, base_url=base_url)


class AlpacaBrokerAdapter(BrokerAdapter):
    """Fail-fast adapter placeholder until live connectivity is enabled."""

    name = "alpaca"

    def __init__(self, config: AlpacaBrokerConfig | None = None) -> None:
        resolved = config or AlpacaBrokerConfig.from_env()
        if not resolved.api_key or not resolved.secret_key:
            raise RuntimeError("Alpaca adapter requires non-empty API credentials.")
        base_url = str(resolved.base_url or "")
        if "paper" not in base_url:
            raise RuntimeError("Alpaca adapter is locked to paper trading; refusing non-paper base URL.")
        self._config = resolved

    def submit_orders(self, orders: Sequence[Order], *, as_of: str) -> SubmissionResult:
        raise RuntimeError("Alpaca adapter is disabled in this environment; configure live credentials to proceed.")

    def poll_orders(self, broker_order_ids: Sequence[str], *, as_of: str) -> list[OrderStatusView]:
        raise RuntimeError("Alpaca adapter disabled; polling not available.")

    def fetch_fills(self, broker_order_ids: Sequence[str], *, as_of: str) -> list[Fill]:
        raise RuntimeError("Alpaca adapter disabled; fill retrieval not available.")

    def get_account(self, *, as_of: str) -> AccountSnapshot:
        raise RuntimeError("Alpaca adapter disabled; account snapshots unavailable.")

    def get_positions(self, *, as_of: str) -> list[PositionSnapshot]:
        raise RuntimeError("Alpaca adapter disabled; positions unavailable.")


__all__ = ["AlpacaBrokerAdapter", "AlpacaBrokerConfig"]
