from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import guard
    sys.path.insert(0, str(PROJECT_ROOT))

from infra.storage.parquet import _PARQUET_AVAILABLE
from infra.storage.raw_writer import RawEquityOHLCVWriter, RawOptionsWriter

pytestmark = pytest.mark.skipif(not _PARQUET_AVAILABLE, reason="pyarrow is required for raw writer tests")


def test_equity_writes_are_idempotent(tmp_path):
    records = [
        {
            "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "symbol": "AAPL",
            "open": 1.0,
            "high": 2.0,
            "low": 1.0,
            "close": 1.5,
            "volume": 100.0,
            "source_vendor": "ivolatility",
        }
    ]
    writer = RawEquityOHLCVWriter(base_path=tmp_path / "raw")

    first = writer.write_records("ivolatility", records)
    first_bytes = _read_files(first)
    second = writer.write_records("ivolatility", records)
    second_bytes = _read_files(second)

    assert _normalize_manifest(first) == _normalize_manifest(second)
    assert first_bytes == second_bytes


def test_options_writes_are_idempotent(tmp_path):
    writer = RawOptionsWriter(base_path=tmp_path / "raw")
    reference_records = [
        {
            "option_symbol": "AAPL240119C00150000",
            "underlying_symbol": "AAPL",
            "expiration_date": date(2024, 1, 19),
            "strike": 150.0,
            "option_type": "call",
            "multiplier": 100.0,
            "source_vendor": "ivolatility",
        }
    ]
    ohlcv_records = [
        {
            "timestamp": datetime(2024, 1, 2, tzinfo=timezone.utc),
            "option_symbol": "AAPL240119C00150000",
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.05,
            "volume": 25.0,
            "source_vendor": "ivolatility",
        }
    ]
    oi_records = [
        {
            "timestamp": datetime(2024, 1, 2, tzinfo=timezone.utc),
            "option_symbol": "AAPL240119C00150000",
            "open_interest": 500.0,
            "source_vendor": "ivolatility",
        }
    ]

    first_reference = writer.write_contract_reference(
        "ivolatility",
        reference_records,
        snapshot_date=date(2024, 1, 2),
    )
    second_reference = writer.write_contract_reference(
        "ivolatility",
        reference_records,
        snapshot_date=date(2024, 1, 2),
    )
    assert _normalize_manifest(first_reference) == _normalize_manifest(second_reference)

    first_ohlcv = writer.write_option_ohlcv("ivolatility", ohlcv_records)
    bytes_ohlcv_first = _read_files(first_ohlcv)
    second_ohlcv = writer.write_option_ohlcv("ivolatility", ohlcv_records)
    bytes_ohlcv_second = _read_files(second_ohlcv)
    assert _normalize_manifest(first_ohlcv) == _normalize_manifest(second_ohlcv)
    assert bytes_ohlcv_first == bytes_ohlcv_second

    first_oi = writer.write_option_open_interest("ivolatility", oi_records)
    bytes_oi_first = _read_files(first_oi)
    second_oi = writer.write_option_open_interest("ivolatility", oi_records)
    bytes_oi_second = _read_files(second_oi)
    assert _normalize_manifest(first_oi) == _normalize_manifest(second_oi)
    assert bytes_oi_first == bytes_oi_second


def _normalize_manifest(manifest):
    files = sorted(manifest.get("files", []), key=lambda item: item["path"])
    return {"files": files, "total_files": manifest.get("total_files")}


def _read_files(manifest):
    contents = {}
    for entry in manifest.get("files", []):
        path = Path(entry["path"])
        contents[path] = path.read_bytes()
    return contents
