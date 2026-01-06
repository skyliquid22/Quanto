"""Canonical equity OHLCV loader for research workflows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from infra.normalization.lineage import compute_file_hash
from infra.paths import get_data_root

try:  # pragma: no cover - pyarrow is optional
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover - canonicals have json fallback
    pq = None

UTC = timezone.utc


@dataclass(frozen=True)
class CanonicalEquitySlice:
    """In-memory canonical OHLCV bars for a single symbol."""

    symbol: str
    rows: List[Dict[str, Any]]
    file_paths: List[Path]

    @property
    def timestamps(self) -> List[datetime]:
        return [row["timestamp"] for row in self.rows]

    @property
    def closes(self) -> List[float]:
        return [float(row["close"]) for row in self.rows]


def load_canonical_equity(
    symbols: Sequence[str],
    start_date: date | str,
    end_date: date | str,
    *,
    data_root: Path | None = None,
) -> Tuple[Dict[str, CanonicalEquitySlice], Dict[str, str]]:
    """Load canonical OHLCV bars for the requested symbols and dates.

    Returns a mapping of symbol → slice plus the SHA hashes of every file that
    participated in the load (keyed by data-root relative paths).
    """

    resolved_root = Path(data_root) if data_root else get_data_root()
    start = _coerce_date(start_date)
    end = _coerce_date(end_date)
    if end < start:
        raise ValueError("end_date must be greater than or equal to start_date")

    discovered: Dict[str, CanonicalEquitySlice] = {}
    file_hashes: Dict[str, str] = {}
    for symbol in _ordered_dedup(symbols):
        rows, files = _load_symbol(symbol, start, end, resolved_root)
        discovered[symbol] = CanonicalEquitySlice(symbol=symbol, rows=rows, file_paths=files)
        for path in files:
            rel = _relative_to(path, resolved_root)
            file_hashes[rel] = compute_file_hash(path)
    return discovered, file_hashes


def _ordered_dedup(symbols: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for symbol in symbols:
        clean = str(symbol).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        ordered.append(clean)
    return ordered


def _load_symbol(symbol: str, start: date, end: date, data_root: Path) -> Tuple[List[Dict[str, Any]], List[Path]]:
    base = data_root / "canonical" / "equity_ohlcv" / symbol / "daily"
    rows: List[Dict[str, Any]] = []
    touched_files: List[Path] = []
    for day in _iter_days(start, end):
        path = base / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.day:02d}.parquet"
        if not path.exists():
            continue
        touched_files.append(path)
        rows.extend(_normalize_records(symbol, path, start, end))
    rows.sort(key=lambda row: (row["timestamp"], row.get("symbol", symbol)))
    return rows, touched_files


def _normalize_records(symbol: str, path: Path, start: date, end: date) -> Iterable[Dict[str, Any]]:
    payload = _read_records(path)
    for entry in payload:
        entry_symbol = str(entry.get("symbol") or symbol)
        if entry_symbol != symbol:
            continue
        timestamp = _coerce_timestamp(entry.get("timestamp"))
        if timestamp.date() < start or timestamp.date() > end:
            continue
        clean = dict(entry)
        clean["symbol"] = entry_symbol
        clean["timestamp"] = timestamp
        clean["close"] = float(entry.get("close", 0.0))
        yield clean


def _iter_days(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _coerce_date(value: date | str) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _coerce_timestamp(value: Any) -> datetime:
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()  # type: ignore[assignment]
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    raise ValueError("record missing timestamp")


def _read_records(path: Path) -> List[Mapping[str, Any]]:
    if pq is not None:
        try:  # pragma: no branch - prefer pyarrow when available
            table = pq.read_table(path)
            return table.to_pylist()
        except Exception:
            pass
    text = path.read_text(encoding="utf-8")
    return json.loads(text) if text.strip() else []


def _relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


__all__ = ["CanonicalEquitySlice", "load_canonical_equity"]
