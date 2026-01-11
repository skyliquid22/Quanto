from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import guard
    sys.path.insert(0, str(PROJECT_ROOT))

from infra.storage.raw_writer import RawEquityOHLCVWriter
from infra.storage.parquet import _PARQUET_AVAILABLE

pytestmark = pytest.mark.skipif(not _PARQUET_AVAILABLE, reason="pyarrow is required for parquet layout tests")


@pytest.mark.parametrize("vendor", ["polygon", "test_vendor"])
def test_equity_raw_paths_are_vendor_partitioned(tmp_path, vendor) -> None:
    base_path = tmp_path / "raw"
    writer = RawEquityOHLCVWriter(base_path=base_path)
    records = [
        {
            "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "symbol": "AAPL",
            "open": 1.0,
            "high": 2.0,
            "low": 1.0,
            "close": 2.0,
            "volume": 10.0,
        },
        {
            "timestamp": datetime(2024, 1, 2, tzinfo=timezone.utc),
            "symbol": "AAPL",
            "open": 1.0,
            "high": 2.0,
            "low": 1.0,
            "close": 2.0,
            "volume": 11.0,
        },
    ]

    result = writer.write_records(vendor, records)
    assert result["total_files"] == 1

    emitted_paths = [Path(entry["path"]) for entry in result["files"]]
    for path in emitted_paths:
        rel_path = path.relative_to(base_path)
        _assert_equity_layout(rel_path, vendor)


def _assert_equity_layout(rel_path: Path, vendor: str) -> None:
    parts = rel_path.parts
    assert parts[0] == vendor
    assert parts[1] == "equity_ohlcv"
    assert len(parts) == 5, f"unexpected layout segments: {parts}"
    symbol = parts[2]
    assert symbol
    assert parts[3] == "daily"
    shard = parts[4]
    assert shard.endswith(".parquet")
    year = shard.replace(".parquet", "")
    assert year.isdigit() and len(year) == 4
