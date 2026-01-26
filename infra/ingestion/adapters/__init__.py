"""Vendor-specific ingestion adapters."""

from .ivolatility_equity import IvolatilityEquityAdapter
from .ivolatility_fundamentals import IvolatilityFundamentalsAdapter, IvolatilityFundamentalsUnsupported
from .ivolatility_options import IvolatilityOptionsAdapter
from .ivolatility_options_surface import (
    IvolatilityOptionsSurfaceAdapter,
    OptionsSurfaceIngestionRequest,
    OptionsSurfaceStorage,
)
from .financialdatasets_fundamentals import (
    FinancialDatasetsAdapter,
    FinancialDatasetsAdapterResult,
    FinancialDatasetsRESTClient,
)
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
    "IvolatilityEquityAdapter",
    "IvolatilityOptionsAdapter",
    "IvolatilityFundamentalsAdapter",
    "IvolatilityFundamentalsUnsupported",
    "IvolatilityOptionsSurfaceAdapter",
    "OptionsSurfaceIngestionRequest",
    "OptionsSurfaceStorage",
    "FinancialDatasetsAdapter",
    "FinancialDatasetsAdapterResult",
    "FinancialDatasetsRESTClient",
]
