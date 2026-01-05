from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import guard
    sys.path.insert(0, str(PROJECT_ROOT))

from infra.storage.raw_writer import RawEquityOHLCVWriter


@pytest.mark.parametrize("vendor", ["polygon", "test_vendor"])
def test_equity_raw_paths_are_vendor_partitioned(tmp_path, monkeypatch, vendor) -> None:
    monkeypatch.setattr("infra.storage.parquet._PARQUET_AVAILABLE", False)
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
    assert result["total_files"] == 2

    emitted_paths = [Path(entry["path"]) for entry in result["files"]]
    for path in emitted_paths:
        rel_path = path.relative_to(base_path)
        _assert_equity_layout(rel_path, vendor)


def _assert_equity_layout(rel_path: Path, vendor: str) -> None:
    parts = rel_path.parts
    assert parts[0] == vendor
    assert parts[1] == "equity_ohlcv"
    assert len(parts) == 7, f"unexpected layout segments: {parts}"
    symbol = parts[2]
    assert symbol
    assert parts[3] == "daily"
    year, month = parts[4], parts[5]
    day_part = parts[6]
    assert year.isdigit() and len(year) == 4
    assert month.isdigit() and len(month) == 2
    assert day_part.endswith(".parquet")
    day = day_part.replace(".parquet", "")
    assert day.isdigit() and len(day) == 2
