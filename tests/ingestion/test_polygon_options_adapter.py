from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timezone
from pathlib import Path
import sys
import zipfile

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import guard
    sys.path.insert(0, str(PROJECT_ROOT))

from infra.ingestion.adapters import (  # noqa: E402
    OptionReferenceIngestionRequest,
    OptionTimeseriesIngestionRequest,
    PolygonOptionsAdapter,
    RateLimitError,
)


class _StubOptionsRestClient:
    def __init__(self) -> None:
        self.reference_payloads: dict[str, list[object]] = {}
        self.ohlcv_payloads: dict[str, list[object]] = {}
        self.oi_payloads: dict[str, list[object]] = {}
        self.contract_calls: list[dict[str, object]] = []
        self.ohlcv_calls: list[dict[str, object]] = []
        self.oi_calls: list[dict[str, object]] = []

    async def fetch_option_contracts(self, underlying, as_of, *, page_url, page_size):
        self.contract_calls.append({"underlying": underlying, "page_url": page_url})
        queue = self.reference_payloads.setdefault(underlying, [])
        if not queue:
            return {"results": []}
        payload = queue.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return payload

    async def fetch_option_ohlcv(self, option_symbol, start, end, *, page_url, page_size):
        self.ohlcv_calls.append({"symbol": option_symbol, "page_url": page_url})
        queue = self.ohlcv_payloads.setdefault(option_symbol, [])
        if not queue:
            return {"results": []}
        payload = queue.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return payload

    async def fetch_option_open_interest(self, option_symbol, start, end, *, page_url, page_size):
        self.oi_calls.append({"symbol": option_symbol, "page_url": page_url})
        queue = self.oi_payloads.setdefault(option_symbol, [])
        if not queue:
            return {"results": []}
        payload = queue.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return payload

    async def aclose(self):  # pragma: no cover - compatibility
        return None


def test_reference_rest_handles_rate_limits(monkeypatch):
    client = _StubOptionsRestClient()
    client.reference_payloads["AAPL"] = [
        RateLimitError("rate", retry_after=0),
        {
            "results": [
                {
                    "contract_symbol": "OPT1",
                    "underlying_symbol": "AAPL",
                    "expiration_date": "2024-01-19",
                    "strike": 150,
                    "option_type": "C",
                    "multiplier": 100,
                }
            ],
            "next_url": None,
        },
    ]
    adapter = PolygonOptionsAdapter(rest_client=client, rest_config={"concurrency": 1})

    async def fast_sleep(delay):  # pragma: no cover - ensures fast tests
        return None

    monkeypatch.setattr("infra.ingestion.adapters.polygon_options.asyncio.sleep", fast_sleep)

    request = OptionReferenceIngestionRequest(
        underlying_symbols=("AAPL",),
        as_of_date=date(2024, 1, 12),
    )
    records = asyncio.run(adapter.fetch_contract_reference_rest(request))
    assert len(records) == 1
    assert records[0]["option_symbol"] == "OPT1"
    assert client.contract_calls[0]["page_url"] is None


def test_timeseries_rest_supports_ohlcv_and_open_interest():
    client = _StubOptionsRestClient()
    client.ohlcv_payloads["OPT1"] = [
        {
            "results": [
                {"t": 1704153600000, "open": 1.0, "high": 1.2, "low": 0.9, "close": 1.1, "volume": 10.0}
            ]
        }
    ]
    client.oi_payloads["OPT1"] = [
        {
            "results": [
                {"t": 1704153600000, "open_interest": 250},
            ]
        }
    ]
    adapter = PolygonOptionsAdapter(rest_client=client)

    ohlcv_request = OptionTimeseriesIngestionRequest(
        domain="option_contract_ohlcv",
        option_symbols=("OPT1",),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 2),
    )
    ohlcv_records = asyncio.run(adapter.fetch_option_ohlcv_rest(ohlcv_request))
    assert ohlcv_records[0]["option_symbol"] == "OPT1"
    assert ohlcv_records[0]["timestamp"].tzinfo == timezone.utc

    oi_request = OptionTimeseriesIngestionRequest(
        domain="option_open_interest",
        option_symbols=("OPT1",),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 2),
    )
    oi_records = asyncio.run(adapter.fetch_option_open_interest_rest(oi_request))
    assert oi_records[0]["open_interest"] == 250.0


def test_flat_file_streams_zip_payloads(tmp_path):
    ref_zip = tmp_path / "reference.zip"
    with zipfile.ZipFile(ref_zip, "w") as archive:
        archive.writestr(
            "reference.jsonl",
            json.dumps(
                {
                    "option_symbol": "OPT1",
                    "underlying_symbol": "AAPL",
                    "expiration_date": "2024-01-19",
                    "strike": 150,
                    "option_type": "C",
                    "multiplier": 100,
                }
            )
            + "\n",
        )
    ohlcv_zip = tmp_path / "ohlcv.zip"
    with zipfile.ZipFile(ohlcv_zip, "w") as archive:
        archive.writestr(
            "ohlcv.jsonl",
            json.dumps(
                {
                    "timestamp": "2024-01-01T00:00:00Z",
                    "option_symbol": "OPT1",
                    "open": 1,
                    "high": 2,
                    "low": 0.5,
                    "close": 1.5,
                    "volume": 10,
                }
            )
            + "\n",
        )
    oi_zip = tmp_path / "oi.zip"
    with zipfile.ZipFile(oi_zip, "w") as archive:
        archive.writestr(
            "oi.jsonl",
            json.dumps(
                {
                    "timestamp": "2024-01-01T00:00:00Z",
                    "option_symbol": "OPT1",
                    "open_interest": 250,
                }
            )
            + "\n",
        )

    adapter = PolygonOptionsAdapter(flat_file_config={"decompression_workers": 2})

    ref_request = OptionReferenceIngestionRequest(
        underlying_symbols=("AAPL",),
        as_of_date=date(2024, 1, 1),
        flat_file_uris=(str(ref_zip),),
    )
    ref_records = list(adapter.stream_reference_flat_files(ref_request))
    assert ref_records[0]["underlying_symbol"] == "AAPL"

    ohlcv_request = OptionTimeseriesIngestionRequest(
        domain="option_contract_ohlcv",
        option_symbols=("OPT1",),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 1),
        flat_file_uris=(str(ohlcv_zip),),
    )
    ohlcv_records = list(adapter.stream_option_ohlcv_flat_files(ohlcv_request))
    assert ohlcv_records[0]["timestamp"] == datetime(2024, 1, 1, tzinfo=timezone.utc)

    oi_request = OptionTimeseriesIngestionRequest(
        domain="option_open_interest",
        option_symbols=("OPT1",),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 1),
        flat_file_uris=(str(oi_zip),),
    )
    oi_records = list(adapter.stream_option_open_interest_flat_files(oi_request))
    assert oi_records[0]["open_interest"] == 250.0


def test_ohlcv_missing_price_raises_value_error():
    """float(None) bug: missing OHLC keys must raise ValueError, not TypeError."""
    client = _StubOptionsRestClient()
    client.ohlcv_payloads["OPT1"] = [
        {"results": [{"t": 1704153600000}]},  # no OHLC keys
    ]
    adapter = PolygonOptionsAdapter(rest_client=client)

    request = OptionTimeseriesIngestionRequest(
        domain="option_contract_ohlcv",
        option_symbols=("OPT1",),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 2),
    )

    with pytest.raises(ValueError, match="open"):
        asyncio.run(adapter.fetch_option_ohlcv_rest(request))


def test_ohlcv_missing_volume_defaults_to_zero():
    """Volume should default to 0.0 when absent from payload."""
    client = _StubOptionsRestClient()
    client.ohlcv_payloads["OPT1"] = [
        {"results": [{"t": 1704153600000, "open": 1.0, "high": 1.2, "low": 0.9, "close": 1.1}]},
    ]
    adapter = PolygonOptionsAdapter(rest_client=client)

    request = OptionTimeseriesIngestionRequest(
        domain="option_contract_ohlcv",
        option_symbols=("OPT1",),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 2),
    )

    records = asyncio.run(adapter.fetch_option_ohlcv_rest(request))
    assert records[0]["volume"] == 0.0


if __name__ == "__main__":  # pragma: no cover - debug helper
    pytest.main([__file__])
