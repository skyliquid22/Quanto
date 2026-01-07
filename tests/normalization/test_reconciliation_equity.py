from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from infra.normalization import ReconciliationBuilder
from infra.normalization.lineage import compute_file_hash
from infra.storage.parquet import write_parquet_atomic


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
        "total_records": 2,
        "valid_records": 2,
        "invalid_records": 0,
        "validation_status": "passed",
        "creation_timestamp": "2023-01-02T00:00:00Z",
    }
    (manifest_dir / f"{run_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_records(builder: ReconciliationBuilder, path: Path) -> list[dict]:
    return builder._read_records(path)  # type: ignore[attr-defined]


def test_equity_reconciliation_prefers_primary_with_fallback(tmp_path):
    raw_root = tmp_path / "raw"
    manifest_root = tmp_path / "manifests"
    canonical_root = tmp_path / "canonical"
    metrics_root = tmp_path / "metrics"

    polygon_path = raw_root / "polygon" / "equity_ohlcv" / "AAPL" / "daily" / "2023" / "01" / "02.parquet"
    iex_aapl = raw_root / "iex" / "equity_ohlcv" / "AAPL" / "daily" / "2023" / "01" / "02.parquet"
    iex_msft = raw_root / "iex" / "equity_ohlcv" / "MSFT" / "daily" / "2023" / "01" / "02.parquet"

    write_parquet_atomic(
        [
            {
                "symbol": "AAPL",
                "timestamp": datetime(2023, 1, 2, 16, tzinfo=UTC),
                "open": 100,
                "high": 110,
                "low": 95,
                "close": 105,
                "volume": 1000,
            }
        ],
        polygon_path,
        use_pyarrow=False,
    )
    write_parquet_atomic(
        [
            {
                "symbol": "AAPL",
                "timestamp": datetime(2023, 1, 2, 16, tzinfo=UTC),
                "open": 101,
                "high": 111,
                "low": 96,
                "close": 106,
                "volume": 900,
            }
        ],
        iex_aapl,
        use_pyarrow=False,
    )
    write_parquet_atomic(
        [
            {
                "symbol": "MSFT",
                "timestamp": datetime(2023, 1, 2, 16, tzinfo=UTC),
                "open": 200,
                "high": 210,
                "low": 190,
                "close": 205,
                "volume": 500,
            }
        ],
        iex_msft,
        use_pyarrow=False,
    )

    _write_manifest(manifest_root, "equity_ohlcv", "polygon_run", "polygon", file_hashes=[compute_file_hash(polygon_path)])
    iex_hashes = [compute_file_hash(iex_aapl), compute_file_hash(iex_msft)]
    _write_manifest(manifest_root, "equity_ohlcv", "iex_run", "iex", file_hashes=iex_hashes)

    config = {
        "reconciliation": {
            "domains": {
                "equity_ohlcv": {
                    "vendor_priority": ["polygon", "iex"],
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

    builder.run(domains=["equity_ohlcv"], start_date=date(2023, 1, 2), end_date=date(2023, 1, 2), run_id="testrun")

    aapl_path = canonical_root / "equity_ohlcv" / "AAPL" / "daily" / "2023.parquet"
    msft_path = canonical_root / "equity_ohlcv" / "MSFT" / "daily" / "2023.parquet"
    aapl_records = _load_records(builder, aapl_path)
    msft_records = _load_records(builder, msft_path)

    assert aapl_records[0]["reconcile_method"] == "primary"
    assert aapl_records[0]["selected_source_vendor"] == "polygon"
    assert aapl_records[0]["fallback_source_vendor"] is None
    assert aapl_records[0]["input_file_hashes"]

    assert msft_records[0]["reconcile_method"] == "fallback"
    assert msft_records[0]["fallback_source_vendor"] == "iex"
    assert msft_records[0]["primary_source_vendor"] == "polygon"

    metrics_path = metrics_root / "equity_ohlcv_testrun.json"
    metrics_payload = json.loads(metrics_path.read_text())
    assert metrics_payload["fallback_count"] == 1
    assert metrics_payload["percent_missing_primary"] > 0

    manifest_path = canonical_root / "manifests" / "equity_ohlcv" / "testrun.json"
    manifest_payload = json.loads(manifest_path.read_text())
    assert manifest_payload["records_written"] == 2
    assert manifest_payload["lineage"]["metadata"]["manifest_paths"]["polygon"].endswith("polygon_run.json")
    assert not (canonical_root / "equity_ohlcv" / "AAPL" / "daily" / "2023" ).exists()


def test_equity_reconciliation_handles_yearly_raw_shards(tmp_path):
    raw_root = tmp_path / "raw"
    manifest_root = tmp_path / "manifests"
    canonical_root = tmp_path / "canonical"
    metrics_root = tmp_path / "metrics"

    yearly_path = raw_root / "polygon" / "equity_ohlcv" / "MSFT" / "daily" / "2023.parquet"
    records = []
    for day in range(3, 6):
        ts = datetime(2023, 1, day, 16, tzinfo=UTC)
        records.append(
            {
                "symbol": "MSFT",
                "t": int(ts.timestamp() * 1000),
                "open": 100 + day,
                "high": 105 + day,
                "low": 95 + day,
                "close": 102 + day,
                "volume": 1000 + day,
            }
        )
    write_parquet_atomic(records, yearly_path, use_pyarrow=False)
    file_hash = compute_file_hash(yearly_path)
    ingest_manifest_dir = raw_root / "polygon" / "equity_ohlcv" / "manifests"
    ingest_manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_payload = {
        "domain": "equity_ohlcv",
        "schema_version": "v1",
        "source_vendor": "polygon",
        "run_id": "polygon_yearly",
        "input_file_hashes": [file_hash],
        "total_records": len(records),
        "valid_records": len(records),
        "invalid_records": 0,
        "validation_status": "passed",
        "creation_timestamp": "2023-01-05T00:00:00Z",
    }
    (ingest_manifest_dir / "polygon_yearly.json").write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    config = {
        "reconciliation": {
            "domains": {
                "equity_ohlcv": {
                    "vendor_priority": ["polygon"],
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
        now=datetime(2023, 1, 7, tzinfo=UTC),
    )

    manifests = builder.run(
        domains=["equity_ohlcv"],
        start_date=date(2023, 1, 1),
        end_date=date(2023, 1, 10),
        run_id="yearly_run",
    )

    manifest_payload = manifests["equity_ohlcv"]
    assert manifest_payload["inputs"], "expected yearly shard to be discovered"
    assert manifest_payload["records_written"] == len(records)

    canonical_file = canonical_root / "equity_ohlcv" / "MSFT" / "daily" / "2023.parquet"
    assert canonical_file.exists()
