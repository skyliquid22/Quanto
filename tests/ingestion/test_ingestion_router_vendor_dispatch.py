"""Router dispatch tests covering the new vendor matrix."""

from infra.ingestion.adapters import IvolatilityEquityAdapter, PolygonEquityAdapter
from infra.ingestion.router import IngestionRouter
import pytest


def test_router_selects_polygon_equity_adapter() -> None:
    router = IngestionRouter()
    route = router.resolve_vendor_adapter("equity_ohlcv", "polygon", "rest")
    assert route.adapter is PolygonEquityAdapter
    assert route.mode == "rest"


def test_router_selects_ivolatility_equity_adapter() -> None:
    router = IngestionRouter()
    route = router.resolve_vendor_adapter("equity_ohlcv", "ivolatility", "rest")
    assert route.adapter is IvolatilityEquityAdapter
    assert route.vendor == "ivolatility"


def test_router_rejects_unsupported_mode() -> None:
    router = IngestionRouter()
    with pytest.raises(ValueError):
        router.resolve_vendor_adapter("option_open_interest", "ivolatility", "flat_file")
