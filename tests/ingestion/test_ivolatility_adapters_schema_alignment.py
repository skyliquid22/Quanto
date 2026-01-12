from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import guard
    sys.path.insert(0, str(PROJECT_ROOT))

from infra.ingestion.adapters import (
    EquityIngestionRequest,
    FundamentalsIngestionRequest,
    IvolatilityEquityAdapter,
    IvolatilityFundamentalsAdapter,
    IvolatilityFundamentalsUnsupported,
    IvolatilityOptionsAdapter,
    OptionReferenceIngestionRequest,
    OptionTimeseriesIngestionRequest,
)
from infra.validation import validate_records


class _StaticClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def fetch(self, endpoint, params):
        symbol = params.get("symbol") or params.get("option_symbol") or ""
        date_value = params.get("date")
        key = (endpoint, symbol, date_value)
        payload = self.responses.get(key)
        if payload is None:
            key = (endpoint, symbol)
            payload = self.responses.get(key)
        if payload is None:
            key = (endpoint,)
            payload = self.responses.get(key, [])
        self.calls.append({"endpoint": endpoint, "params": dict(params)})
        return [dict(entry) for entry in payload]

    def fetch_async_dataset(self, endpoint, params):
        return self.fetch(endpoint, params)


def test_equity_adapter_produces_schema_aligned_records(tmp_path):
    responses = {
        (
            IvolatilityEquityAdapter.STOCK_PRICES_ENDPOINT,
            "AAPL",
            "2024-01-01",
        ): [
            {"date": "2024-01-01", "openPrice": 1, "highPrice": 2, "lowPrice": 1, "closePrice": 1.8, "volume": 900}
        ],
        (
            IvolatilityEquityAdapter.STOCK_PRICES_ENDPOINT,
            "AAPL",
            "2024-01-02",
        ): [
            {"date": "2024-01-02", "openPrice": 2, "highPrice": 3, "lowPrice": 1, "closePrice": 2.5, "volume": 1000}
        ],
    }
    client = _StaticClient(responses)
    adapter = IvolatilityEquityAdapter(client, config={"use_bulk": False})
    request = EquityIngestionRequest(
        symbols=("AAPL",),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 2),
        vendor="ivolatility",
    )

    normalized = asyncio.run(adapter.fetch_equity_ohlcv_rest(request))  # type: ignore[name-defined]
    validated, manifest = adapter.validate(
        normalized,
        run_id="test-equity",
        config={"manifest_base_path": tmp_path / "validation"},
    )

    assert len(validated) == 2
    assert manifest["domain"] == "equity_ohlcv"


def test_options_adapter_covers_reference_and_timeseries(tmp_path):
    responses = {
        (IvolatilityOptionsAdapter.SERIES_ENDPOINT, "AAPL", "2024-01-02"): [
            {
                "optionSymbol": "AAPL240119C00150000",
                "underlying_symbol": "AAPL",
                "expiration": "2024-01-19",
                "strike": 150,
                "callPut": "C",
                "multiplier": 100,
            }
        ],
        (IvolatilityOptionsAdapter.SINGLE_CONTRACT_ENDPOINT, "AAPL240119C00150000"): [
            {
                "option_symbol": "AAPL240119C00150000",
                "date": "2024-01-02",
                "open": 3.2,
                "high": 3.5,
                "low": 3.1,
                "close": 3.4,
                "volume": 42,
                "open_interest": 5000,
            }
        ],
    }
    client = _StaticClient(responses)
    adapter = IvolatilityOptionsAdapter(client)

    ref_request = OptionReferenceIngestionRequest(
        underlying_symbols=("AAPL",),
        as_of_date=date(2024, 1, 2),
        vendor="ivolatility",
    )
    ref_raw = adapter.fetch_contract_reference(ref_request)
    ref_records = adapter.normalize_contract_reference(ref_raw)
    validated_ref, _ = adapter.validate(
        "option_contract_reference",
        ref_records,
        run_id="test-reference",
        config={"manifest_base_path": tmp_path / "validation"},
    )
    assert validated_ref[0]["option_symbol"] == "AAPL240119C00150000"

    ts_request = OptionTimeseriesIngestionRequest(
        domain="option_contract_ohlcv",
        option_symbols=("AAPL240119C00150000",),
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 2),
        vendor="ivolatility",
    )
    ts_raw = adapter.fetch_option_ohlcv(ts_request)
    ts_records = adapter.normalize_option_ohlcv(ts_raw)
    validated_ts, _ = adapter.validate(
        "option_contract_ohlcv",
        ts_records,
        run_id="test-ohlcv",
        config={"manifest_base_path": tmp_path / "validation"},
    )
    assert validated_ts[0]["option_symbol"] == "AAPL240119C00150000"

    oi_request = OptionTimeseriesIngestionRequest(
        domain="option_open_interest",
        option_symbols=("AAPL240119C00150000",),
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 2),
        vendor="ivolatility",
    )
    oi_raw = adapter.fetch_option_open_interest(oi_request)
    oi_records = adapter.normalize_option_open_interest(oi_raw)
    validated_oi, _ = adapter.validate(
        "option_open_interest",
        oi_records,
        run_id="test-oi",
        config={"manifest_base_path": tmp_path / "validation"},
    )
    assert validated_oi[0]["open_interest"] == 5000.0


def test_fundamentals_adapter_fails_fast():
    adapter = IvolatilityFundamentalsAdapter()
    request = FundamentalsIngestionRequest(
        symbols=("AAPL",),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        vendor="ivolatility",
    )
    with pytest.raises(IvolatilityFundamentalsUnsupported):
        adapter.fetch_raw(request)
