from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from infra.normalization import ReconciliationBuilder
from infra.normalization.lineage import compute_file_hash
from infra.storage.parquet import write_parquet_atomic

UTC = timezone.utc


def _write_manifest(base: Path, domain: str, run_id: str, vendor: str, *, file_hashes: list[str]) -> None:
    payload = {
        "domain": domain,
        "schema_version": "v1",
        "source_vendor": vendor,
        "run_id": run_id,
        "input_file_hashes": file_hashes,
        "total_records": 3,
        "valid_records": 3,
        "invalid_records": 0,
        "validation_status": "passed",
        "creation_timestamp": "2023-01-02T00:00:00Z",
    }
    manifest_dir = base / domain
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / f"{run_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_records(builder: ReconciliationBuilder, path: Path) -> list[dict]:
    return builder._read_records(path)  # type: ignore[attr-defined]


def test_options_enforces_single_vendor_per_underlying(tmp_path):
    raw_root = tmp_path / "raw"
    manifest_root = tmp_path / "manifests"
    canonical_root = tmp_path / "canonical"
    metrics_root = tmp_path / "metrics"

    occ_spy = raw_root / "occ" / "option_contract_ohlcv" / "SPY2301C00100000" / "daily" / "2023" / "01" / "02.parquet"
    opra_spy = raw_root / "opra" / "option_contract_ohlcv" / "SPY2301C00100000" / "daily" / "2023" / "01" / "02.parquet"
    opra_qqq = raw_root / "opra" / "option_contract_ohlcv" / "QQQ2301C00100000" / "daily" / "2023" / "01" / "02.parquet"

    write_parquet_atomic(
        [
            {
                "option_symbol": "SPY2301C00100000",
                "underlying_symbol": "SPY",
                "timestamp": datetime(2023, 1, 2, 15, tzinfo=UTC),
                "open": 2.5,
                "high": 3.0,
                "low": 2.0,
                "close": 2.8,
                "volume": 150,
            }
        ],
        occ_spy,
        use_pyarrow=False,
    )
    write_parquet_atomic(
        [
            {
                "option_symbol": "SPY2301C00100000",
                "underlying_symbol": "SPY",
                "timestamp": datetime(2023, 1, 2, 15, tzinfo=UTC),
                "open": 2.6,
                "high": 3.1,
                "low": 2.1,
                "close": 2.7,
                "volume": 200,
            }
        ],
        opra_spy,
        use_pyarrow=False,
    )
    write_parquet_atomic(
        [
            {
                "option_symbol": "QQQ2301C00100000",
                "underlying_symbol": "QQQ",
                "timestamp": datetime(2023, 1, 2, 15, tzinfo=UTC),
                "open": 1.5,
                "high": 1.8,
                "low": 1.2,
                "close": 1.7,
                "volume": 80,
            }
        ],
        opra_qqq,
        use_pyarrow=False,
    )

    _write_manifest(manifest_root, "option_contract_ohlcv", "occ_run", "occ", file_hashes=[compute_file_hash(occ_spy)])
    _write_manifest(
        manifest_root,
        "option_contract_ohlcv",
        "opra_run",
        "opra",
        file_hashes=[compute_file_hash(opra_spy), compute_file_hash(opra_qqq)],
    )

    config = {
        "reconciliation": {
            "domains": {
                "option_contract_ohlcv": {
                    "vendor_priority": ["occ", "opra"],
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

    builder.run(domains=["option_contract_ohlcv"], start_date=date(2023, 1, 2), end_date=date(2023, 1, 2), run_id="optionsrun")

    spy_path = canonical_root / "option_contract_ohlcv" / "SPY2301C00100000" / "daily" / "2023" / "01" / "02.parquet"
    qqq_path = canonical_root / "option_contract_ohlcv" / "QQQ2301C00100000" / "daily" / "2023" / "01" / "02.parquet"
    spy_records = _load_records(builder, spy_path)
    qqq_records = _load_records(builder, qqq_path)

    assert spy_records[0]["selected_source_vendor"] == "occ"
    assert spy_records[0]["fallback_source_vendor"] is None
    # QQQ obtains coverage solely from fallback vendor
    assert qqq_records[0]["reconcile_method"] == "fallback"
    assert qqq_records[0]["fallback_source_vendor"] == "opra"
    assert qqq_records[0]["primary_source_vendor"] == "occ"

    metrics_path = metrics_root / "option_contract_ohlcv_optionsrun.json"
    metrics_payload = json.loads(metrics_path.read_text())
    assert metrics_payload["fallback_count"] == 1
    assert metrics_payload["percent_missing_primary"] > 0
