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
        "total_records": 4,
        "valid_records": 4,
        "invalid_records": 0,
        "validation_status": "passed",
        "creation_timestamp": "2023-01-02T00:00:00Z",
    }
    (manifest_dir / f"{run_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_records(builder: ReconciliationBuilder, path: Path) -> list[dict]:
    return builder._read_records(path)  # type: ignore[attr-defined]


def test_fundamentals_respects_record_vendor_and_restatements(tmp_path):
    raw_root = tmp_path / "raw"
    manifest_root = tmp_path / "manifests"
    canonical_root = tmp_path / "canonical"
    metrics_root = tmp_path / "metrics"

    factset_aapl = raw_root / "factset" / "fundamentals" / "AAPL" / "2023" / "01" / "02.parquet"
    sec_msft = raw_root / "sec" / "fundamentals" / "MSFT" / "2023" / "01" / "02.parquet"

    write_parquet_atomic(
        [
            {
                "symbol": "AAPL",
                "report_date": date(2023, 1, 2),
                "filing_date": date(2023, 2, 1),
                "filing_id": "AAPL-Q1",
                "statement_type": "income",
                "net_income": 100,
            },
            {
                "symbol": "AAPL",
                "report_date": date(2023, 1, 2),
                "filing_date": date(2023, 2, 15),
                "filing_id": "AAPL-Q1-RESTATED",
                "statement_type": "income",
                "net_income": 105,
            },
        ],
        factset_aapl,
    )
    write_parquet_atomic(
        [
            {
                "symbol": "MSFT",
                "report_date": date(2023, 1, 2),
                "filing_date": date(2023, 2, 5),
                "filing_id": "MSFT-Q1",
                "statement_type": "cashflow",
                "operating_cash_flow": 200,
            }
        ],
        sec_msft,
    )

    _write_manifest(manifest_root, "fundamentals", "factset_run", "factset", file_hashes=[compute_file_hash(factset_aapl)])
    _write_manifest(manifest_root, "fundamentals", "sec_run", "sec", file_hashes=[compute_file_hash(sec_msft)])

    config = {
        "reconciliation": {
            "domains": {
                "fundamentals": {
                    "vendor_priority": ["factset", "sec"],
                    "fundamentals_of_record": "factset",
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

    builder.run(domains=["fundamentals"], start_date=date(2023, 1, 2), end_date=date(2023, 1, 2), run_id="fundrun")

    aapl_path = canonical_root / "fundamentals" / "AAPL" / "2023" / "01" / "02.parquet"
    msft_path = canonical_root / "fundamentals" / "MSFT" / "2023" / "01" / "02.parquet"
    aapl_records = _load_records(builder, aapl_path)
    msft_records = _load_records(builder, msft_path)

    assert aapl_records[0]["filing_id"] == "AAPL-Q1-RESTATED"
    assert aapl_records[0]["reconcile_method"] == "restatement"
    assert msft_records[0]["reconcile_method"] == "fallback"
    assert msft_records[0]["fallback_source_vendor"] == "sec"
    assert msft_records[0]["primary_source_vendor"] == "factset"

    manifest_path = canonical_root / "manifests" / "fundamentals" / "fundrun.json"
    manifest_payload = json.loads(manifest_path.read_text())
    superseded = manifest_payload["lineage"]["metadata"]["superseded_filings"]
    assert "AAPL|2023-01-02|income" in superseded
    assert "AAPL-Q1" in superseded["AAPL|2023-01-02|income"]
