"""Regression test: _instantiate_options_adapter must accept vendor_params (6 args).

Bug: scripts/ingest.py called _instantiate_options_adapter with 5 positional args
but the function expects 6 (missing vendor_params). This caused a TypeError at
runtime for any --domain option_* invocation through the unified CLI.
"""

from scripts.ingest import _instantiate_options_adapter
from infra.ingestion.router import AdapterRoute, IngestionRouter


def test_instantiate_options_adapter_polygon_flat_file() -> None:
    """Polygon flat_file mode needs no API key — exercises the 6-arg call."""
    router = IngestionRouter()
    route = router.resolve_vendor_adapter("option_contract_reference", "polygon", "flat_file")
    adapter, callbacks = _instantiate_options_adapter(
        route,
        "polygon",
        {},           # rest_cfg
        {},           # flat_cfg
        "flat_file",  # mode
        {},           # vendor_params — the previously missing arg
    )
    assert adapter is not None
    assert isinstance(callbacks, list)
