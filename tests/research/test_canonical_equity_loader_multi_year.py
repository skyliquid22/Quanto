from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from research.datasets.canonical_equity_loader import load_canonical_equity

UTC = timezone.utc


def _write_yearly(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _build_rows(symbol: str, timestamps: list[datetime], base: float) -> list[dict]:
    rows: list[dict] = []
    for idx, ts in enumerate(timestamps):
        close = base + idx * 0.5
        rows.append(
            {
                "symbol": symbol,
                "timestamp": ts.isoformat(),
                "open": close - 1.0,
                "high": close + 1.0,
                "low": close - 2.0,
                "close": close,
                "volume": 1000 + idx,
            }
        )
    return rows


def _days(year: int, month: int, first: int, last: int) -> list[datetime]:
    return [datetime(year, month, day, 16, tzinfo=UTC) for day in range(first, last + 1)]


def test_loader_handles_multi_year_ranges(tmp_path: Path):
    root = tmp_path / "quanto_data"
    base = root / "canonical" / "equity_ohlcv" / "MSFT" / "daily"
    dec_days = _days(2022, 12, 20, 30)
    jan_days = _days(2023, 1, 2, 13)
    _write_yearly(base / "2022.parquet", _build_rows("MSFT", dec_days, 98.0))
    _write_yearly(base / "2023.parquet", _build_rows("MSFT", jan_days, 105.0))

    start = date(2022, 12, 22)
    end = date(2023, 1, 10)
    slices, hashes = load_canonical_equity(
        ["MSFT"],
        start_date=start,
        end_date=end,
        data_root=root,
    )

    slice_data = slices["MSFT"]
    frame = slice_data.frame
    assert frame.index[0].date() == date(2022, 12, 22)
    assert frame.index[-1].date() == date(2023, 1, 10)
    assert frame.index.is_monotonic_increasing
    assert frame.index.is_unique
    assert frame.index.tz == UTC
    assert list(frame.columns) == ["symbol", "open", "high", "low", "close", "volume"]
    expected_dates = [
        ts for ts in dec_days if ts.date() >= start
    ] + [ts for ts in jan_days if ts.date() <= end]
    assert list(frame.index.to_pydatetime()) == expected_dates
    assert slice_data.timestamps == expected_dates
    assert len(slice_data.timestamps) == len(slice_data.closes)
    assert slice_data.rows[0]["timestamp"].tzinfo is UTC
    assert len(hashes) == 2
    assert all("canonical/equity_ohlcv/MSFT/daily" in key for key in hashes)


def test_loader_errors_when_intermediate_year_missing(tmp_path: Path):
    root = tmp_path / "quanto_data"
    base = root / "canonical" / "equity_ohlcv" / "MSFT" / "daily"
    _write_yearly(base / "2023.parquet", _build_rows("MSFT", _days(2023, 1, 2, 5), 110.0))

    with pytest.raises(FileNotFoundError) as excinfo:
        load_canonical_equity(
            ["MSFT"],
            start_date=date(2023, 12, 28),
            end_date=date(2024, 1, 5),
            data_root=root,
        )
    message = str(excinfo.value)
    assert "2024.parquet" in message
    assert "equity_ohlcv" in message
    assert "regenerate" in message.lower()
