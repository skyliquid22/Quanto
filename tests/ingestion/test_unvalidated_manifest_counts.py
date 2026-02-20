"""Tests that pipelines report honest validated counts in manifests.

Verifies the fix for qu-rewg.2: domains that skip validation must report
validated=0, not validated=len(records).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Sequence

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import guard
    sys.path.insert(0, str(PROJECT_ROOT))

from infra.ingestion.fundamentals_pipeline import FinancialDatasetsRawPipeline
from infra.ingestion.insiders_pipeline import InsiderTradesIngestionPipeline
from infra.ingestion.request import IngestionRequest


@dataclass
class _FakeAdapterResult:
    records: List[Dict[str, Any]]
    source_payloads: List[Dict[str, Any]]
    filings: List[Dict[str, Any]] = field(default_factory=list)


class _StubFinancialDatasetsAdapter:
    """Stub adapter returning canned records for any domain."""

    def __init__(self, records: List[Dict[str, Any]]) -> None:
        self._records = records

    async def fetch_company_facts_rest(self, symbols, **kwargs):
        return _FakeAdapterResult(records=list(self._records), source_payloads=[{"kind": "stub"}])

    async def fetch_financial_metrics_rest(self, symbols, **kwargs):
        return _FakeAdapterResult(records=list(self._records), source_payloads=[{"kind": "stub"}])

    async def fetch_financial_metrics_snapshot_rest(self, symbols, **kwargs):
        return _FakeAdapterResult(records=list(self._records), source_payloads=[{"kind": "stub"}])

    async def fetch_institutional_ownership_rest(self, symbols, **kwargs):
        return _FakeAdapterResult(records=list(self._records), source_payloads=[{"kind": "stub"}])

    async def fetch_news_rest(self, symbols, **kwargs):
        return _FakeAdapterResult(records=list(self._records), source_payloads=[{"kind": "stub"}])

    async def fetch_insider_trades_rest(self, symbols, **kwargs):
        return _FakeAdapterResult(records=list(self._records), source_payloads=[{"kind": "stub"}])

    async def aclose(self):
        pass


class _StubWriter:
    """Writer that records calls without touching the filesystem."""

    def __init__(self) -> None:
        self.written: List[Any] = []

    def _write(self, vendor, records):
        self.written.append(records)
        return {"files": [{"path": "/tmp/fake.parquet", "records": len(records)}]}

    write_company_facts = _write
    write_financial_metrics = _write
    write_financial_metrics_snapshot = _write
    write_financial_statements = _write
    write_institutional_ownership = _write
    write_news = _write
    write_insider_trades = _write


UNVALIDATED_FD_DOMAINS = [
    "company_facts",
    "financial_metrics",
    "financial_metrics_snapshot",
    "institutional_ownership",
    "news",
]


@pytest.mark.parametrize("domain", UNVALIDATED_FD_DOMAINS)
def test_financial_datasets_raw_pipeline_reports_validated_zero(tmp_path, domain):
    """Domains without canonical schemas must report validated=0."""
    records = [{"id": 1, "symbol": "AAPL", "data": "test"}]
    adapter = _StubFinancialDatasetsAdapter(records)
    writer = _StubWriter()
    pipeline = FinancialDatasetsRawPipeline(
        adapter=adapter,
        raw_writer=writer,
        manifest_base_dir=tmp_path,
    )
    request = IngestionRequest(
        domain=domain,
        vendor="financialdatasets",
        mode="rest",
        symbols=("AAPL",),
    )
    manifest = pipeline.run(request, run_id=f"test-{domain}")

    assert manifest["status"] == "succeeded"
    assert manifest["record_counts"]["requested"] == 1
    assert manifest["record_counts"]["validated"] == 0


def test_insider_trades_pipeline_reports_validated_zero(tmp_path):
    """InsiderTradesIngestionPipeline must report validated=0."""
    records = [{"id": 1, "symbol": "AAPL", "transaction_type": "P"}]
    adapter = _StubFinancialDatasetsAdapter(records)
    writer = _StubWriter()
    pipeline = InsiderTradesIngestionPipeline(
        adapter=adapter,
        raw_writer=writer,
        manifest_base_dir=tmp_path,
    )
    request = IngestionRequest(
        domain="insider_trades",
        vendor="financialdatasets",
        mode="rest",
        symbols=("AAPL",),
    )
    manifest = pipeline.run(request, run_id="test-insider-trades")

    assert manifest["status"] == "succeeded"
    assert manifest["record_counts"]["requested"] == 1
    assert manifest["record_counts"]["validated"] == 0


def test_manifest_persisted_with_honest_counts(tmp_path):
    """Manifest JSON on disk must have validated=0 for unvalidated domain."""
    records = [{"id": 1, "symbol": "MSFT"}]
    adapter = _StubFinancialDatasetsAdapter(records)
    writer = _StubWriter()
    pipeline = FinancialDatasetsRawPipeline(
        adapter=adapter,
        raw_writer=writer,
        manifest_base_dir=tmp_path,
    )
    request = IngestionRequest(
        domain="company_facts",
        vendor="financialdatasets",
        mode="rest",
        symbols=("MSFT",),
    )
    manifest = pipeline.run(request, run_id="persist-test")

    manifest_path = tmp_path / "financialdatasets" / "company_facts" / "manifests" / "persist-test.json"
    assert manifest_path.exists()
    stored = json.loads(manifest_path.read_text())
    assert stored["record_counts"]["validated"] == 0
    assert stored["record_counts"]["requested"] == 1
