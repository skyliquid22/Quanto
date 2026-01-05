from __future__ import annotations

import asyncio
from datetime import date
import gzip
import json
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import guard
    sys.path.insert(0, str(PROJECT_ROOT))

from infra.ingestion.adapters import FundamentalsIngestionRequest, PolygonFundamentalsAdapter


class _FakeRESTClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    async def fetch_fundamentals(
        self,
        symbol,
        *,
        statement_type,
        page_url,
        limit,
        start_date,
        end_date,
    ):
        self.calls.append({
            "symbol": symbol,
            "statement_type": statement_type,
            "page_url": page_url,
        })
        if not self.payloads:
            return {"results": [], "next_url": None}
        return self.payloads.pop(0)


def _baseline_entry(symbol: str, report_date: str, *, restated: bool = False) -> dict:
    return {
        "symbol": symbol,
        "report_date": report_date,
        "fiscal_period": "FY23",
        "revenue": 100.0,
        "net_income": 10.0,
        "eps": 1.0,
        "total_assets": 50.0,
        "total_liabilities": 20.0,
        "shareholder_equity": 30.0,
        "operating_income": 12.0,
        "free_cash_flow": 8.0,
        "shares_outstanding": 1000.0,
        "filing_id": f"filing-{symbol}-{report_date}",
        "filing_date": "2024-02-01",
        "statement_type": "annual",
        "restated": restated,
        "restatement_of": f"old-{symbol}" if restated else None,
    }


def test_rest_adapter_handles_pagination_and_restatements():
    payloads = [
        {"results": [_baseline_entry("AAPL", "2023-12-31", restated=True)], "next_url": "next"},
        {"results": [_baseline_entry("AAPL", "2022-12-31")], "next_url": None},
    ]
    client = _FakeRESTClient(payloads)
    adapter = PolygonFundamentalsAdapter(rest_client=client)
    request = FundamentalsIngestionRequest(
        symbols=("AAPL",),
        start_date=date(2022, 1, 1),
        end_date=date(2023, 12, 31),
        statement_types=("annual",),
    )

    result = asyncio.run(adapter.fetch_fundamentals_rest(request))

    assert len(result.records) == 2
    assert all(isinstance(record["report_date"], str) for record in result.records)
    assert any(filing.get("restated") for filing in result.filings)
    assert result.source_payloads[0]["kind"] == "rest_page"
    assert client.calls[0]["page_url"] is None
    assert client.calls[-1]["page_url"] == "next"


def test_flat_file_loader_processes_json_bundle(tmp_path):
    payload = [
        _baseline_entry("AAPL", "2023-12-31"),
        _baseline_entry("MSFT", "2023-12-31"),
    ]
    path = tmp_path / "fundamentals.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    adapter = PolygonFundamentalsAdapter()
    request = FundamentalsIngestionRequest(
        symbols=("AAPL", "MSFT"),
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
        flat_file_uris=(str(path),),
    )

    result = adapter.load_flat_file_fundamentals(request)

    assert sorted(record["symbol"] for record in result.records) == ["AAPL", "MSFT"]
    assert all(record["report_date"] == "2023-12-31" for record in result.records)
    assert result.source_payloads[0]["kind"] == "flat_file"


def test_missing_metric_raises_value_error():
    bad_entry = _baseline_entry("AAPL", "2023-12-31")
    bad_entry.pop("revenue")
    client = _FakeRESTClient([{ "results": [bad_entry], "next_url": None }])
    adapter = PolygonFundamentalsAdapter(rest_client=client)
    request = FundamentalsIngestionRequest(
        symbols=("AAPL",),
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
    )

    with pytest.raises(ValueError):
        asyncio.run(adapter.fetch_fundamentals_rest(request))
