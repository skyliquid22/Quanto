"""Vendor-specific ingestion adapters."""

from .polygon_equity import EquityIngestionRequest, PolygonEquityAdapter, PolygonRESTClient, RateLimitError
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
    "OptionReferenceIngestionRequest",
    "OptionTimeseriesIngestionRequest",
    "PolygonOptionsAdapter",
    "PolygonOptionsRESTClient",
]
