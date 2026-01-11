from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from infra.storage import parquet


pytestmark = pytest.mark.skipif(not parquet._PARQUET_AVAILABLE, reason="pyarrow is required for parquet writer tests")


def test_write_parquet_atomic_verifies_output(tmp_path, monkeypatch):
    destination = tmp_path / "data.parquet"
    records = [
        {
            "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "symbol": "AAPL",
            "close": 1.0,
        }
    ]

    def _boom(path: Path, *args, **kwargs):
        raise RuntimeError("corrupted shard")

    monkeypatch.setattr(parquet.pq, "read_table", _boom)

    with pytest.raises(RuntimeError, match="Parquet verification failed"):
        parquet.write_parquet_atomic(records, destination)

    assert not destination.exists()
