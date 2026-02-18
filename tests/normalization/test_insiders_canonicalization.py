from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from infra.normalization import ReconciliationBuilder
from infra.normalization.lineage import compute_file_hash
from infra.storage.parquet import _PARQUET_AVAILABLE

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


def test_insider_trades_raw_can_canonicalize(tmp_path: Path):
    raw_root = tmp_path / "raw"
    manifest_root = tmp_path / "manifests"
    canonical_root = tmp_path / "canonical"
    metrics_root = tmp_path / "metrics"

    raw_path = raw_root / "financialdatasets" / "insider_trades" / "AAPL" / "2023.csv"
    record = {
        "symbol": "AAPL",
        "filing_date": date(2023, 1, 3),
        "transaction_date": date(2023, 1, 2),
        "transaction_shares": 10,
        "transaction_value": 1000.0,
        "source_vendor": "financialdatasets",
    }
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame([record]).to_csv(raw_path, index=False)

    _write_manifest(
        manifest_root,
        "insider_trades",
        "fd_run",
        "financialdatasets",
        file_hashes=[compute_file_hash(raw_path)],
    )

    config = {
        "reconciliation": {
            "domains": {
                "insiders": {
                    "vendor_priority": ["financialdatasets"],
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
        now=datetime(2023, 1, 4, tzinfo=UTC),
    )

    builder.run(domains=["insiders"], start_date=date(2023, 1, 3), end_date=date(2023, 1, 3), run_id="inscanon")

    out_path = canonical_root / "insiders" / "AAPL" / "2023" / "01" / "03.parquet"
    assert out_path.exists()
