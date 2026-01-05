from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import guard
    sys.path.insert(0, str(PROJECT_ROOT))

from infra.ingestion.adapters import EquityIngestionRequest, PolygonEquityAdapter
from infra.ingestion.data_pipeline import EquityIngestionPipeline
from infra.ingestion.router import IngestionRouter
from infra.storage.raw_writer import RawEquityOHLCVWriter


class _StubRestClient:
    def __init__(self, payload):
        self.payload = payload

    async def fetch_aggregates(self, symbol, start, end, *, page_url, page_size):
        return self.payload


def _build_request():
    return EquityIngestionRequest(
        symbols=("AAPL",),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 1),
    )


def test_pipeline_rest_flow_writes_raw_layout(tmp_path, monkeypatch):
    rest_payload = {"results": [{"t": 1704067200000, "o": 1.0, "h": 2.0, "l": 1.0, "c": 2.0, "v": 10.0}], "next_url": None}
    adapter = PolygonEquityAdapter(rest_client=_StubRestClient(rest_payload))
    router = IngestionRouter({"force_mode": "rest"})
    raw_root = tmp_path / "raw"
    manifest_dir = tmp_path / "manifests"
    writer = RawEquityOHLCVWriter(base_path=raw_root)

    # Force JSON fallback for deterministic hashing in tests.
    monkeypatch.setattr("infra.storage.parquet._PARQUET_AVAILABLE", False)

    pipeline = EquityIngestionPipeline(adapter=adapter, router=router, raw_writer=writer, manifest_dir=manifest_dir)
    request = _build_request()

    manifest = pipeline.run(request, run_id="test-run")

    assert manifest["status"] == "succeeded"
    manifest_path = manifest_dir / "test-run.json"
    assert manifest_path.exists()
    stored_manifest = json.loads(manifest_path.read_text())
    assert stored_manifest["files_written"]

    data_file = raw_root / "polygon" / "equity_ohlcv" / "AAPL" / "daily" / "2024" / "01" / "01.parquet"
    assert data_file.exists()

    # Re-run with the same configuration to verify determinism.
    manifest2 = pipeline.run(request, run_id="test-run")
    assert manifest2 == manifest


def test_raw_writer_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr("infra.storage.parquet._PARQUET_AVAILABLE", False)
    writer = RawEquityOHLCVWriter(base_path=tmp_path / "raw")
    records = [
        {
            "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "symbol": "AAPL",
            "open": 1.0,
            "high": 2.0,
            "low": 1.0,
            "close": 2.0,
            "volume": 10.0,
            "source_vendor": "polygon",
        }
    ]

    writer.write_records("polygon", records)
    data_file = tmp_path / "raw" / "polygon" / "equity_ohlcv" / "AAPL" / "daily" / "2024" / "01" / "01.parquet"
    initial_bytes = data_file.read_bytes()
    writer.write_records("polygon", records)
    assert data_file.read_bytes() == initial_bytes
