from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from research.datasets.canonical_equity_loader import load_canonical_equity

UTC = timezone.utc


def _write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _sample_rows(days: list[int]) -> list[dict]:
    rows = []
    for day in days:
        ts = datetime(2023, 1, day, 16, tzinfo=UTC)
        rows.append(
            {
                "symbol": "MSFT",
                "timestamp": ts.isoformat(),
                "close": float(100 + day),
                "open": float(90 + day),
                "high": float(110 + day),
                "low": float(80 + day),
                "volume": float(1000 + day),
            }
        )
    return rows


def test_loader_filters_yearly_daily(tmp_path):
    root = tmp_path / "quanto_data"
    yearly_path = root / "canonical" / "equity_ohlcv" / "MSFT" / "daily" / "2023.parquet"
    _write_json(yearly_path, _sample_rows([2, 3, 4]))

    slices, file_hashes = load_canonical_equity(
        ["MSFT"],
        start_date="2023-01-02",
        end_date="2023-01-04",
        data_root=root,
    )

    slice_data = slices["MSFT"]
    assert slice_data.symbol == "MSFT"
    assert list(slice_data.frame.columns) == ["symbol", "open", "high", "low", "close", "volume"]
    assert slice_data.frame.index.min().date().isoformat() == "2023-01-02"
    assert slice_data.frame.index.max().date().isoformat() == "2023-01-04"
    assert slice_data.frame.shape[0] == 3
    assert slice_data.closes == [102.0, 103.0, 104.0]
    assert slice_data.timestamps[0].tzinfo is UTC
    assert list(file_hashes.keys()) == ["canonical/equity_ohlcv/MSFT/daily/2023.parquet"]
    assert slice_data.rows[0]["timestamp"].tzinfo is UTC


def test_loader_missing_yearly_shard(tmp_path):
    root = tmp_path / "quanto_data"
    base = root / "canonical" / "equity_ohlcv" / "MSFT" / "daily"
    _write_json(base / "2022.parquet", _sample_rows([2]))

    with pytest.raises(FileNotFoundError) as excinfo:
        load_canonical_equity(
            ["MSFT"],
            start_date="2023-01-02",
            end_date="2023-01-03",
            data_root=root,
        )
    message = str(excinfo.value)
    assert "yearly shard missing" in message.lower()
    assert "2023.parquet" in message
