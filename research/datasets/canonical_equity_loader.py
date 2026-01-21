"""Canonical equity OHLCV loader for research workflows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from infra.normalization.lineage import compute_file_hash
from infra.paths import get_data_root
from research.regime.universe import PRIMARY_REGIME_UNIVERSE

try:  # pragma: no cover - pandas optional in some environments
    import pandas as pd  # type: ignore
except Exception as exc:  # pragma: no cover
    pd = None
    _PANDAS_ERROR: Exception | None = exc
else:  # pragma: no cover
    _PANDAS_ERROR = None

try:  # pragma: no cover - pyarrow is optional
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover - canonicals have json fallback
    pq = None

UTC = timezone.utc
CANONICAL_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class CanonicalEquitySlice:
    """In-memory canonical OHLCV bars for a single symbol."""

    symbol: str
    frame: "pd.DataFrame"
    file_paths: List[Path]

    @property
    def timestamps(self) -> List[datetime]:
        return list(self.frame.index.to_pydatetime())

    @property
    def closes(self) -> List[float]:
        return self.frame["close"].astype(float).tolist()

    @property
    def rows(self) -> List[Dict[str, Any]]:
        """Legacy compatibility for code still consuming dict rows."""

        if self.frame.empty:
            return []
        return self.frame.reset_index().to_dict("records")


def load_canonical_equity(
    symbols: Sequence[str],
    start_date: date | str,
    end_date: date | str,
    *,
    data_root: Path | None = None,
    interval: str = "daily",
) -> Tuple[Dict[str, CanonicalEquitySlice], Dict[str, str]]:
    """Load canonical OHLCV bars for the requested symbols and dates.

    Returns a mapping of symbol → slice plus the SHA hashes of every file that
    participated in the load (keyed by data-root relative paths).
    """

    _ensure_pandas_available()
    resolved_root = Path(data_root) if data_root else get_data_root()
    normalized_interval = str(interval).strip().lower() or "daily"
    if normalized_interval != "daily":
        raise ValueError(f"Unsupported interval '{interval}'; only daily data is available in v1.")
    start = _coerce_date(start_date)
    end = _coerce_date(end_date)
    if end < start:
        raise ValueError("end_date must be greater than or equal to start_date")

    discovered: Dict[str, CanonicalEquitySlice] = {}
    file_hashes: Dict[str, str] = {}
    for symbol in _ordered_dedup(symbols):
        frame, files = _load_symbol(symbol, start, end, resolved_root, normalized_interval)
        discovered[symbol] = CanonicalEquitySlice(symbol=symbol, frame=frame, file_paths=files)
        for path in files:
            rel = _relative_to(path, resolved_root)
            file_hashes[rel] = compute_file_hash(path)
    return discovered, file_hashes


def load_primary_regime_universe(
    start_date: date | str,
    end_date: date | str,
    *,
    data_root: Path | None = None,
    interval: str = "daily",
) -> Tuple[Dict[str, CanonicalEquitySlice], Dict[str, str]]:
    """Load canonical OHLCV bars for the fixed primary regime universe."""

    slices, hashes = load_canonical_equity(
        PRIMARY_REGIME_UNIVERSE,
        start_date,
        end_date,
        data_root=data_root,
        interval=interval,
    )
    missing = [symbol for symbol in PRIMARY_REGIME_UNIVERSE if symbol not in slices]
    empty = [symbol for symbol, slice_data in slices.items() if slice_data.frame.empty]
    if missing or empty:
        missing = missing + [symbol for symbol in empty if symbol not in missing]
        raise ValueError(f"Primary regime universe missing canonical data for symbols: {missing}")
    return slices, hashes


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


def verify_coverage(symbols: Sequence[str], start_date: str, end_date: str, *, data_root: Path) -> Dict[str, Any]:
    """Verify yearly-daily canonical coverage for the provided symbols."""

    start = _coerce_date(start_date)
    end = _coerce_date(end_date)
    if end < start:
        raise ValueError("end_date must be greater than or equal to start_date")
    ordered_symbols = _ordered_dedup(symbols)
    years = list(range(start.year, end.year + 1))
    resolved_root = Path(data_root).expanduser()
    missing_by_symbol: Dict[str, List[int]] = {}
    missing_pairs: List[Tuple[str, int]] = []
    base = resolved_root / "canonical" / "equity_ohlcv"
    for symbol in ordered_symbols:
        symbol_missing: List[int] = []
        for year in years:
            shard = base / symbol / "daily" / f"{year}.parquet"
            exists = shard.exists()
            valid = False
            if exists:
                try:
                    valid = shard.stat().st_size > 0
                except OSError:
                    valid = False
            if not exists or not valid:
                symbol_missing.append(year)
                missing_pairs.append((symbol, year))
        missing_by_symbol[symbol] = symbol_missing
    return {
        "symbols": ordered_symbols,
        "years": years,
        "missing_by_symbol": missing_by_symbol,
        "missing_pairs": missing_pairs,
    }


def _load_symbol(
    symbol: str,
    start: date,
    end: date,
    data_root: Path,
    interval: str,
) -> Tuple["pd.DataFrame", List[Path]]:
    base = data_root / "canonical" / "equity_ohlcv" / symbol / interval
    rows: List[Dict[str, Any]] = []
    touched_files: List[Path] = []
    years = range(start.year, end.year + 1)
    for year in years:
        path = base / f"{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Canonical yearly shard missing for {symbol} {year}. "
                f"Expected file: {path}. Regenerate via "
                f"`python -m scripts.build_canonical_datasets "
                f"--domains equity_ohlcv --start-date {year}-01-01 --end-date {year}-12-31`."
            )
        touched_files.append(path)
        rows.extend(_normalize_records(symbol, path, start, end))
    rows.sort(key=lambda row: (row["timestamp"], row.get("symbol", symbol)))
    frame = _rows_to_frame(rows, symbol)
    return frame, touched_files


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
        for column in CANONICAL_COLUMNS:
            clean[column] = float(entry.get(column, 0.0))
        yield clean


def _rows_to_frame(rows: List[Dict[str, Any]], symbol: str) -> "pd.DataFrame":
    _ensure_pandas_available()
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=["timestamp", "symbol", *CANONICAL_COLUMNS])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"])
    frame.sort_values("timestamp", inplace=True, kind="mergesort")
    frame.reset_index(drop=True, inplace=True)
    frame["symbol"] = symbol
    for column in CANONICAL_COLUMNS:
        frame[column] = frame[column].astype(float)
    frame.set_index("timestamp", inplace=True)
    frame.index = pd.DatetimeIndex(frame.index, tz=UTC, name="timestamp")
    ordered_columns = ["symbol", *CANONICAL_COLUMNS]
    for column in ordered_columns:
        if column not in frame.columns:
            frame[column] = symbol if column == "symbol" else 0.0
    frame = frame[ordered_columns]
    return frame


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


def build_union_calendar(
    slices: Mapping[str, CanonicalEquitySlice],
    *,
    start_date: date | None = None,
    end_date: date | None = None,
) -> "pd.DatetimeIndex":
    """Construct a deterministic union calendar covering all provided slices."""

    _ensure_pandas_available()
    calendar: "pd.DatetimeIndex | None" = None
    start_ts = pd.Timestamp(start_date, tz=UTC) if start_date else None
    end_ts = pd.Timestamp(end_date, tz=UTC) if end_date else None
    for slice_data in slices.values():
        frame = slice_data.frame
        if frame.empty:
            continue
        index = frame.index
        if start_ts is not None:
            index = index[index >= start_ts]
        if end_ts is not None:
            index = index[index <= end_ts]
        if calendar is None:
            calendar = index
        else:
            calendar = calendar.union(index)
    if calendar is None or calendar.empty:
        return pd.DatetimeIndex([], tz=UTC, name="timestamp")
    return pd.DatetimeIndex(calendar.sort_values(), tz=UTC, name="timestamp")


def align_ohlcv_panel(
    slices: Mapping[str, CanonicalEquitySlice],
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    forward_fill_limit: int = 3,
    drop_missing_closes: bool = True,
) -> "pd.DataFrame":
    """Align canonical slices onto a shared calendar with explicit fill policy."""

    _ensure_pandas_available()
    calendar = build_union_calendar(slices, start_date=start_date, end_date=end_date)
    if calendar.empty:
        return pd.DataFrame(index=calendar)
    per_symbol: Dict[str, "pd.DataFrame"] = {}
    for symbol, slice_data in slices.items():
        frame = slice_data.frame.copy()
        if frame.empty:
            continue
        trimmed = frame.loc[:, CANONICAL_COLUMNS]
        aligned = trimmed.reindex(calendar)
        aligned = aligned.ffill(limit=max(0, int(forward_fill_limit)))
        per_symbol[symbol] = aligned
    if not per_symbol:
        return pd.DataFrame(index=calendar)
    combined = pd.concat(per_symbol, axis=1).sort_index()
    if drop_missing_closes:
        invalid = pd.Series(False, index=combined.index)
        for symbol in per_symbol:
            invalid |= combined[(symbol, "close")].isna()
        combined = combined[~invalid]
    # Deterministic fill for any remaining NaNs (e.g., volume)
    combined = combined.fillna(0.0)
    return combined


def _relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _ensure_pandas_available() -> None:
    if pd is None:
        raise RuntimeError("pandas is required to load canonical equity data") from _PANDAS_ERROR


__all__ = [
    "CanonicalEquitySlice",
    "load_canonical_equity",
    "load_primary_regime_universe",
    "build_union_calendar",
    "align_ohlcv_panel",
]
