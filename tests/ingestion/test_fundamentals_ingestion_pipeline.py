from __future__ import annotations

import json
from datetime import date
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import guard
    sys.path.insert(0, str(PROJECT_ROOT))

from infra.ingestion.adapters import FundamentalsAdapterResult, FundamentalsIngestionRequest
from infra.ingestion.fundamentals_pipeline import FundamentalsIngestionPipeline
from infra.ingestion.router import IngestionRouter
from infra.storage.raw_writer import RawFundamentalsWriter
from infra.validation import ValidationError


class _StubAdapter:
    def __init__(self, responses):
        self.responses = responses
        self.calls: list[str] = []

    async def fetch_fundamentals_rest(self, request):
        symbol = request.symbols[0]
        self.calls.append(symbol)
        return self.responses[symbol]

    def load_flat_file_fundamentals(self, request):  # pragma: no cover - rest-only stub
        raise RuntimeError("flat-file mode not supported in stub")


def _record(symbol: str, report_date: str, *, fiscal_period: str = "FY23") -> dict:
    return {
        "symbol": symbol,
        "report_date": report_date,
        "fiscal_period": fiscal_period,
        "revenue": 100.0,
        "net_income": 9.0,
        "eps": 1.0,
        "total_assets": 50.0,
        "total_liabilities": 20.0,
        "shareholder_equity": 30.0,
        "operating_income": 11.0,
        "free_cash_flow": 8.0,
        "shares_outstanding": 1000.0,
        "source_vendor": "polygon",
    }


def _adapter_result(symbol: str, *, restated: bool = False) -> FundamentalsAdapterResult:
    filings = [
        {
            "symbol": symbol,
            "statement_type": "annual",
            "filing_id": f"{symbol}-2023",
            "filing_date": "2024-02-01",
            "report_date": "2023-12-31",
            "restated": restated,
            "supersedes": f"{symbol}-2022" if restated else None,
            "restatement_note": f"Supersedes {symbol}-2022" if restated else None,
        }
    ]
    payloads = [
        {"kind": "rest_page", "hash": f"sha256:{symbol}", "symbol": symbol, "statement_type": "annual"}
    ]
    return FundamentalsAdapterResult(records=[_record(symbol, "2023-12-31")], filings=filings, source_payloads=payloads)


def test_pipeline_writes_manifests_and_respects_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr("infra.storage.parquet._PARQUET_AVAILABLE", False)
    raw_base = tmp_path / "raw"
    checkpoints = tmp_path / "checkpoints"

    responses = {
        "AAPL": _adapter_result("AAPL", restated=True),
        "MSFT": _adapter_result("MSFT", restated=False),
    }
    adapter = _StubAdapter(responses)
    router = IngestionRouter({"force_mode": "rest"})
    writer = RawFundamentalsWriter(base_path=raw_base)
    pipeline = FundamentalsIngestionPipeline(
        adapter=adapter,
        router=router,
        raw_writer=writer,
        manifest_base_dir=raw_base,
        checkpoint_dir=checkpoints,
    )

    request = FundamentalsIngestionRequest(
        symbols=("AAPL", "MSFT"),
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
    )

    manifest = pipeline.run(request, run_id="fundamentals-run")

    assert manifest["status"] == "succeeded"
    manifest_path = raw_base / "polygon" / "fundamentals" / "manifests" / "fundamentals-run.json"
    assert manifest_path.exists()
    stored_manifest = json.loads(manifest_path.read_text())
    assert stored_manifest["filing_lineage"]
    assert stored_manifest["restatements"]

    data_file = raw_base / "polygon" / "fundamentals" / "AAPL" / "2023" / "12" / "31.parquet"
    assert data_file.exists()

    initial_calls = list(adapter.calls)
    manifest_again = pipeline.run(request, run_id="fundamentals-run")
    assert manifest_again == manifest
    assert adapter.calls == initial_calls


def test_pipeline_validation_failure_emits_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr("infra.storage.parquet._PARQUET_AVAILABLE", False)
    raw_base = tmp_path / "raw"
    adapter = _StubAdapter({
        "AAPL": FundamentalsAdapterResult(
            records=[{k: v for k, v in _record("AAPL", "2023-12-31").items() if k != "net_income"}],
            filings=[],
            source_payloads=[{"kind": "rest_page", "hash": "sha256:AAPL"}],
        )
    })
    router = IngestionRouter({"force_mode": "rest"})
    writer = RawFundamentalsWriter(base_path=raw_base)
    pipeline = FundamentalsIngestionPipeline(
        adapter=adapter,
        router=router,
        raw_writer=writer,
        manifest_base_dir=raw_base,
        checkpoint_dir=tmp_path / "checkpoints",
    )
    request = FundamentalsIngestionRequest(
        symbols=("AAPL",),
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
    )

    with pytest.raises(ValidationError):
        pipeline.run(request, run_id="broken-run")

    manifest_path = raw_base / "polygon" / "fundamentals" / "manifests" / "broken-run.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text())
    assert payload["status"] == "failed"
    assert payload["failures"]


def test_raw_fundamentals_writer_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr("infra.storage.parquet._PARQUET_AVAILABLE", False)
    writer = RawFundamentalsWriter(base_path=tmp_path / "raw")
    records = [_record("AAPL", "2023-12-31")]

    writer.write_records("polygon", records)
    file_path = tmp_path / "raw" / "polygon" / "fundamentals" / "AAPL" / "2023" / "12" / "31.parquet"
    first_bytes = file_path.read_bytes()
    writer.write_records("polygon", records)
    assert file_path.read_bytes() == first_bytes


def test_raw_fundamentals_writer_orders_and_groups_by_fiscal_period(tmp_path, monkeypatch):
    monkeypatch.setattr("infra.storage.parquet._PARQUET_AVAILABLE", False)
    writer = RawFundamentalsWriter(base_path=tmp_path / "raw")
    records = [
        _record("AAPL", "2023-12-31", fiscal_period="FY23"),
        _record("AAPL", "2023-12-31", fiscal_period="FY22"),
    ]

    writer.write_records("polygon", records)
    file_path = tmp_path / "raw" / "polygon" / "fundamentals" / "AAPL" / "2023" / "12" / "31.parquet"
    payload = json.loads(file_path.read_text())
    assert [entry["fiscal_period"] for entry in payload] == ["FY22", "FY23"]


def test_raw_fundamentals_writer_raises_on_duplicate_keys(tmp_path, monkeypatch):
    monkeypatch.setattr("infra.storage.parquet._PARQUET_AVAILABLE", False)
    writer = RawFundamentalsWriter(base_path=tmp_path / "raw")
    base_record = _record("AAPL", "2023-12-31", fiscal_period="FY23")
    records = [base_record, {**base_record, "net_income": 12.0}]

    with pytest.raises(ValueError):
        writer.write_records("polygon", records)
