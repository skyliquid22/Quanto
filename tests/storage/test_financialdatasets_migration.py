from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from infra.storage.financialdatasets_policy import DomainPolicy
from scripts.migrate_financialdatasets_raw_layout import (
    _report_path,
    migrate_to_staging,
    promote_staging,
)


def _write_old_layout_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_migrate_financialdatasets_dry_run(tmp_path: Path) -> None:
    raw_root = tmp_path / ".quanto_data" / "raw"
    old_path = raw_root / "financialdatasets" / "financial_statements" / "AAPL" / "2024" / "01" / "02.csv"
    _write_old_layout_csv(old_path, [{"symbol": "AAPL", "report_date": "2024-01-02", "statement_type": "income"}])

    policies = {
        "financial_statements": DomainPolicy(
            policy="timeseries_csv_unsharded",
            date_priority=("report_date",),
            date_kind="date",
            dedup_keys=("ticker", "report_date", "statement_type"),
        )
    }

    report = migrate_to_staging(raw_root_path=raw_root, policies=policies, dry_run=True)
    staging_root = raw_root / "financialdatasets" / ".migration_staging"
    assert not staging_root.exists()
    assert report["domains"]["financial_statements"]["rows_read"] == 1


def test_migrate_financialdatasets_stage_and_promote(tmp_path: Path) -> None:
    raw_root = tmp_path / ".quanto_data" / "raw"
    vendor_root = raw_root / "financialdatasets"
    old_path = vendor_root / "financial_statements" / "AAPL" / "2024" / "01" / "02.csv"
    _write_old_layout_csv(old_path, [{"symbol": "AAPL", "report_date": "2024-01-02", "statement_type": "income"}])

    manifest_dir = vendor_root / "financial_statements" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "run1.json"
    manifest_payload = {
        "run_id": "run1",
        "vendor": "financialdatasets",
        "domain": "financial_statements",
        "files_written": [{"path": str(old_path), "hash": "sha256:old", "records": 1}],
        "symbols": ["AAPL"],
    }
    manifest_path.write_text(json.dumps(manifest_payload))

    policies = {
        "financial_statements": DomainPolicy(
            policy="timeseries_csv_unsharded",
            date_priority=("report_date",),
            date_kind="date",
            dedup_keys=("ticker", "report_date", "statement_type"),
        )
    }

    report = migrate_to_staging(raw_root_path=raw_root, policies=policies, dry_run=False)
    report_path = _report_path(vendor_root, None)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    staging_path = vendor_root / ".migration_staging" / "financial_statements" / "AAPL.csv"
    assert staging_path.exists()
    assert report["domains"]["financial_statements"]["rows_written"] == 1

    promote_report = promote_staging(
        raw_root_path=raw_root,
        policies=policies,
        force=False,
        report_path=report_path,
    )
    final_path = vendor_root / "financial_statements" / "AAPL.csv"
    assert final_path.exists()
    assert not old_path.exists()

    updated_manifest = json.loads(manifest_path.read_text())
    assert updated_manifest["files_written"][0]["path"] == str(final_path)
    assert promote_report["manifest_updates"]
