"""Vendor-specific ingestion adapters."""

from .polygon_equity import EquityIngestionRequest, PolygonEquityAdapter, PolygonRESTClient, RateLimitError
from .polygon_fundamentals import (
    FundamentalsAdapterResult,
    FundamentalsIngestionRequest,
    PolygonFundamentalsAdapter,
    PolygonFundamentalsRESTClient,
)
from .polygon_options import (
    OptionReferenceIngestionRequest,
    OptionTimeseriesIngestionRequest,
    PolygonOptionsAdapter,
    PolygonOptionsRESTClient,
)

__all__ = [
    "EquityIngestionRequest",
    "PolygonEquityAdapter",
    "PolygonRESTClient",
    "RateLimitError",
    "FundamentalsAdapterResult",
    "FundamentalsIngestionRequest",
    "PolygonFundamentalsAdapter",
    "PolygonFundamentalsRESTClient",
    "OptionReferenceIngestionRequest",
    "OptionTimeseriesIngestionRequest",
    "PolygonOptionsAdapter",
    "PolygonOptionsRESTClient",
]
