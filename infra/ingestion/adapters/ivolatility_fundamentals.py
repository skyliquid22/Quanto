"""Stub adapter for iVolatility fundamentals ingestion."""

from __future__ import annotations

from typing import Mapping, Sequence

from ..ivolatility_client import IvolatilityClient
from .polygon_fundamentals import (
    FundamentalsAdapterResult,
    FundamentalsIngestionRequest,
)


class IvolatilityFundamentalsUnsupported(RuntimeError):
    """Raised when callers attempt to use unsupported fundamentals ingestion."""


class IvolatilityFundamentalsAdapter:
    """Fail-fast adapter that documents the unsupported fundamentals workflow."""

    def __init__(
        self,
        *,
        client: IvolatilityClient | None = None,
        vendor: str = "ivolatility",
        supported: bool = False,
    ) -> None:
        self.client = client
        self.vendor = vendor
        self.supported = supported
        if supported and client is None:
            raise ValueError("A client must be provided when supported=True")

    def fetch_raw(self, request: FundamentalsIngestionRequest) -> FundamentalsAdapterResult:
        self._raise_not_supported()
        raise AssertionError("unreachable")  # pragma: no cover - guarded by _raise_not_supported

    def normalize_records(self, payload: Sequence[Mapping[str, object]]):
        self._raise_not_supported()
        raise AssertionError("unreachable")  # pragma: no cover - guarded by _raise_not_supported

    def _raise_not_supported(self) -> None:
        raise IvolatilityFundamentalsUnsupported(
            "iVolatility Backtest API Plus does not expose fundamentals data. "
            "Disable fundamentals ingestion for source_vendor=ivolatility or "
            "source the domain from a supported vendor."
        )


__all__ = ["IvolatilityFundamentalsAdapter", "IvolatilityFundamentalsUnsupported"]
