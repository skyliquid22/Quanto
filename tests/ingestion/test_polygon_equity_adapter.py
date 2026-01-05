from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
import gzip
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import guard
    sys.path.insert(0, str(PROJECT_ROOT))

from infra.ingestion.adapters import EquityIngestionRequest, PolygonEquityAdapter, RateLimitError


class _FakeRestClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    async def fetch_aggregates(self, symbol, start, end, *, page_url, page_size):
        self.calls.append({"symbol": symbol, "page_url": page_url})
        if not self.payloads:
            return {}
        response = self.payloads.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_rest_adapter_handles_pagination_and_rate_limit(monkeypatch):
    payloads = [
        RateLimitError("limit", retry_after=0),
        {"results": [{"t": 1704067200000, "o": 1, "h": 2, "l": 1, "c": 2, "v": 10.0}], "next_url": "next"},
        {"results": [{"t": 1704153600000, "o": 2, "h": 3, "l": 2, "c": 3, "v": 20.0}], "next_url": None},
    ]
    client = _FakeRestClient(payloads)
    adapter = PolygonEquityAdapter(rest_client=client, rest_config={"concurrency": 1})

    async def fake_sleep(delay):  # pragma: no cover - ensures tests run fast
        return None

    monkeypatch.setattr("infra.ingestion.adapters.polygon_equity.asyncio.sleep", fake_sleep)

    request = EquityIngestionRequest(
        symbols=("AAPL",),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 2),
    )

    records = asyncio.run(adapter.fetch_equity_ohlcv_rest(request))
    assert len(records) == 2
    assert all(record["symbol"] == "AAPL" for record in records)
    assert records[0]["timestamp"].tzinfo == timezone.utc
    assert client.calls[0]["page_url"] is None
    assert client.calls[-1]["page_url"] == "next"


def test_flat_file_loader_handles_gzip_csv(tmp_path):
    csv_path = tmp_path / "bars.csv.gz"
    with gzip.open(csv_path, "wt", encoding="utf-8") as handle:
        handle.write("timestamp,symbol,open,high,low,close,volume\n")
        handle.write("2024-01-01T00:00:00Z,AAPL,1,2,1,2,10\n")
        handle.write("2024-01-02T00:00:00Z,MSFT,2,3,2,3,20\n")

    adapter = PolygonEquityAdapter(flat_file_config={"decompression_workers": 2})
    request = EquityIngestionRequest(
        symbols=("AAPL", "MSFT"),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 2),
        flat_file_uris=(str(csv_path),),
    )

    records = list(adapter.stream_flat_file_equity_bars(request))
    assert [record["symbol"] for record in records] == ["AAPL", "MSFT"]
    assert records[0]["timestamp"] == datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_flat_file_loader_preserves_file_order(tmp_path):
    paths = []
    for idx, symbol in enumerate(("AAA", "BBB")):
        path = tmp_path / f"{symbol}.csv.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write("timestamp,symbol,open,high,low,close,volume\n")
            handle.write(f"2024-01-0{idx+1}T00:00:00Z,{symbol},1,2,1,2,5\n")
        paths.append(str(path))

    adapter = PolygonEquityAdapter(flat_file_config={"decompression_workers": 4})
    request = EquityIngestionRequest(
        symbols=("AAA", "BBB"),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 2),
        flat_file_uris=tuple(paths),
    )

    records = list(adapter.stream_flat_file_equity_bars(request))
    assert [record["symbol"] for record in records] == ["AAA", "BBB"]
