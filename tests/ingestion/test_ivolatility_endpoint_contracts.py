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
    IvolatilityEquityAdapter,
    IvolatilityOptionsAdapter,
    OptionReferenceIngestionRequest,
    OptionTimeseriesIngestionRequest,
)


class _RecordingClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def fetch(self, endpoint, params):
        symbol = params.get("symbol") or params.get("option_symbol") or ""
        date_value = params.get("date")
        payload = []
        for key in (
            (endpoint, symbol, date_value),
            (endpoint, symbol),
            (endpoint,),
        ):
            if key in self.responses:
                payload = self.responses[key]
                break
        self.calls.append({"endpoint": endpoint, "params": dict(params)})
        return [dict(entry) for entry in payload]

    def fetch_async_dataset(self, endpoint, params):
        return self.fetch(endpoint, params)


def test_equity_bulk_endpoint_and_params():
    responses = {
        (IvolatilityEquityAdapter.BULK_ENDPOINT,): [
            {
                "symbol": "AAPL",
                "tradeDate": "2025-12-14",
                "OPEN_PRICE": 1,
                "HIGH": 2,
                "LOW": 1,
                "CLOSE_PRICE": 1.5,
                "STOCK_VOLUME": 1000,
            },
            {
                "symbol": "MSFT",
                "tradeDate": "2025-12-14",
                "OPEN_PRICE": 2,
                "HIGH": 3,
                "LOW": 2,
                "CLOSE_PRICE": 2.5,
                "STOCK_VOLUME": 900,
            },
        ]
    }
    client = _RecordingClient(responses)
    adapter = IvolatilityEquityAdapter(
        client,
        config={"stock_group": "ALL_USA", "use_bulk": True},
    )
    request = EquityIngestionRequest(
        symbols=("AAPL", "MSFT"),
        start_date=date(2025, 12, 14),
        end_date=date(2025, 12, 15),
        vendor="ivolatility",
    )

    records = asyncio.run(adapter.fetch_equity_ohlcv_rest(request))

    assert client.calls[0]["endpoint"] == IvolatilityEquityAdapter.BULK_ENDPOINT
    params = client.calls[0]["params"]
    assert params["stockGroup"] == "ALL_USA"
    assert params["from"] == "2025-12-14"
    assert params["to"] == "2025-12-15"
    assert "symbols" not in params
    assert {rec["symbol"] for rec in records} == {"AAPL", "MSFT"}


def test_equity_per_symbol_endpoint_and_params():
    responses = {
        (IvolatilityEquityAdapter.STOCK_PRICES_ENDPOINT, "AAPL", "2024-01-01"): [
            {"date": "2024-01-01", "open": 1, "high": 2, "low": 1, "close": 1.5, "volume": 100}
        ],
        (IvolatilityEquityAdapter.STOCK_PRICES_ENDPOINT, "AAPL", "2024-01-02"): [
            {"date": "2024-01-02", "open": 2, "high": 3, "low": 2, "close": 2.5, "volume": 200}
        ],
    }
    client = _RecordingClient(responses)
    adapter = IvolatilityEquityAdapter(client, config={"use_bulk": False})
    request = EquityIngestionRequest(
        symbols=("AAPL",),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 2),
        vendor="ivolatility",
    )

    asyncio.run(adapter.fetch_equity_ohlcv_rest(request))

    assert all(call["endpoint"] == IvolatilityEquityAdapter.STOCK_PRICES_ENDPOINT for call in client.calls)
    first_call = client.calls[0]["params"]
    assert first_call["symbol"] == "AAPL"
    assert first_call["date"] == "2024-01-01"


def test_equity_bulk_endpoint_uses_request_symbols_when_no_group():
    responses = {
        (IvolatilityEquityAdapter.BULK_ENDPOINT,): []
    }
    client = _RecordingClient(responses)
    adapter = IvolatilityEquityAdapter(client, config={"use_bulk": True})
    request = EquityIngestionRequest(
        symbols=("AAPL", "MSFT"),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 1),
        vendor="ivolatility",
    )

    asyncio.run(adapter.fetch_equity_ohlcv_rest(request))

    call = client.calls[0]
    assert call["endpoint"] == IvolatilityEquityAdapter.BULK_ENDPOINT
    params = call["params"]
    assert params["symbols"] == "AAPL,MSFT"


def test_option_series_endpoint_params():
    responses = {
        (IvolatilityOptionsAdapter.SERIES_ENDPOINT, "AAPL", "2024-06-14"): [
            {
                "optionSymbol": "AAPL240119P00150000",
                "underlying_symbol": "AAPL",
                "expiration": "2024-01-19",
                "strike": 150,
                "callPut": "P",
                "multiplier": 100,
            }
        ]
    }
    client = _RecordingClient(responses)
    adapter = IvolatilityOptionsAdapter(client)
    request = OptionReferenceIngestionRequest(
        underlying_symbols=("AAPL",),
        as_of_date=date(2024, 6, 14),
        vendor="ivolatility",
        options={
            "exp_from": "2024-01-01",
            "exp_to": "2024-01-31",
            "strike_from": 100,
            "strike_to": 200,
            "call_put": "P",
        },
    )

    adapter.fetch_contract_reference(request)

    call = client.calls[0]
    assert call["endpoint"] == IvolatilityOptionsAdapter.SERIES_ENDPOINT
    params = call["params"]
    assert params["symbol"] == "AAPL"
    assert params["date"] == "2024-06-14"
    assert params["expFrom"] == "2024-01-01"
    assert params["expTo"] == "2024-01-31"
    assert params["strikeFrom"] == "100"
    assert params["strikeTo"] == "200"
    assert params["callPut"] == "P"


def test_single_option_endpoint_params():
    responses = {
        (IvolatilityOptionsAdapter.SINGLE_CONTRACT_ENDPOINT, "AAPL240119C00150000"): [
            {
                "option_symbol": "AAPL240119C00150000",
                "date": "2024-01-02",
                "open": 1,
                "high": 2,
                "low": 1,
                "close": 1.5,
                "volume": 10,
                "open_interest": 500,
            }
        ]
    }
    client = _RecordingClient(responses)
    adapter = IvolatilityOptionsAdapter(client)
    request = OptionTimeseriesIngestionRequest(
        domain="option_contract_ohlcv",
        option_symbols=("AAPL240119C00150000",),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 3),
        vendor="ivolatility",
    )

    adapter.fetch_option_ohlcv(request)

    call = client.calls[0]
    assert call["endpoint"] == IvolatilityOptionsAdapter.SINGLE_CONTRACT_ENDPOINT
    params = call["params"]
    assert params["symbol"] == "AAPL240119C00150000"
    assert params["from"] == "2024-01-01"
    assert params["to"] == "2024-01-03"
