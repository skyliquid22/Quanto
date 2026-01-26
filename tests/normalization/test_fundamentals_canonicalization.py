from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from infra.normalization import ReconciliationBuilder
from infra.normalization.lineage import compute_file_hash
from infra.storage.parquet import _PARQUET_AVAILABLE, write_parquet_atomic

pytestmark = pytest.mark.skipif(not _PARQUET_AVAILABLE, reason="pyarrow is required for reconciliation tests")

UTC = timezone.utc


def _write_manifest(base: Path, domain: str, run_id: str, vendor: str, *, file_hashes: list[str]) -> None:
    manifest_dir = base / domain
    manifest_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "domain": domain,
        "schema_version": "v1",
        "source_vendor": vendor,
        "run_id": run_id,
        "input_file_hashes": file_hashes,
        "total_records": 1,
        "valid_records": 1,
        "invalid_records": 0,
        "validation_status": "passed",
        "creation_timestamp": "2023-01-02T00:00:00Z",
    }
    (manifest_dir / f"{run_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_financial_statements_raw_can_canonicalize(tmp_path: Path):
    raw_root = tmp_path / "raw"
    manifest_root = tmp_path / "manifests"
    canonical_root = tmp_path / "canonical"
    metrics_root = tmp_path / "metrics"

    raw_path = raw_root / "financialdatasets" / "financial_statements" / "AAPL" / "2023" / "01" / "02.parquet"
    record = {
        "symbol": "AAPL",
        "report_date": date(2023, 1, 2),
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
        "statement_type": "all",
        "source_vendor": "financialdatasets",
    }
    write_parquet_atomic([record], raw_path)

    _write_manifest(
        manifest_root,
        "financial_statements",
        "fd_run",
        "financialdatasets",
        file_hashes=[compute_file_hash(raw_path)],
    )

    config = {
        "reconciliation": {
            "domains": {
                "fundamentals": {
                    "vendor_priority": ["financialdatasets"],
                    "fundamentals_of_record": "financialdatasets",
                }
            }
        }
    }
    builder = ReconciliationBuilder(
        config,
        raw_data_root=raw_root,
        canonical_root=canonical_root,
        validation_manifest_root_path=manifest_root,
        metrics_root=metrics_root,
        now=datetime(2023, 1, 3, tzinfo=UTC),
    )

    builder.run(domains=["fundamentals"], start_date=date(2023, 1, 2), end_date=date(2023, 1, 2), run_id="fdcanon")

    out_path = canonical_root / "fundamentals" / "AAPL" / "2023" / "01" / "02.parquet"
    assert out_path.exists()
