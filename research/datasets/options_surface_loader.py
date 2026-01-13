"""Dataset loader for derived options surface storage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from infra.normalization.lineage import compute_file_hash
from infra.paths import get_data_root
from infra.normalization.options_surface import (
    NORMALIZED_OPTIONS_SURFACE_COLUMNS,
    OPT_COVERAGE_COLUMNS,
)

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
except Exception as exc:  # pragma: no cover
    pd = None
    _PANDAS_ERROR: Exception | None = exc
else:  # pragma: no cover
    _PANDAS_ERROR = None


@dataclass(frozen=True)
class OptionsSurfaceSlice:
    """Per-symbol derived surface rows."""

    symbol: str
    frame: "pd.DataFrame"
    file_paths: List[Path]


def load_options_surface(
    symbols: Sequence[str],
    start_date: date | str,
    end_date: date | str,
    *,
    data_root: Path | None = None,
) -> Tuple[Dict[str, OptionsSurfaceSlice], Dict[str, str]]:
    """Load derived options surface files for the requested symbols/dates."""

    _ensure_pandas_available()
    root = Path(data_root) if data_root else get_data_root()
    start = _coerce_date(start_date)
    end = _coerce_date(end_date)
    ordered_symbols = _ordered_symbols(symbols)
    slices: Dict[str, OptionsSurfaceSlice] = {}
    file_hashes: Dict[str, str] = {}

    for symbol in ordered_symbols:
        frame, files = _load_symbol(symbol, start, end, root)
        slices[symbol] = OptionsSurfaceSlice(symbol=symbol, frame=frame, file_paths=files)
        for path in files:
            rel = _relative_to(path, root)
            file_hashes[rel] = compute_file_hash(path)
    return slices, file_hashes


def _ordered_symbols(symbols: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for symbol in symbols:
        clean = str(symbol).strip().upper()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        ordered.append(clean)
    return ordered


def _load_symbol(symbol: str, start: date, end: date, root: Path) -> Tuple["pd.DataFrame", List[Path]]:
    base = root / "derived" / "options_surface_v1" / symbol / "daily"
    years = range(start.year, end.year + 1)
    frames: List["pd.DataFrame"] = []
    files: List[Path] = []
    for year in years:
        shard = base / f"{year}.parquet"
        if not shard.exists():
            continue
        files.append(shard)
        frame = _read_parquet(shard)
        if frame is None or frame.empty:
            continue
        frames.append(frame)

    if not frames:
        empty = pd.DataFrame(columns=NORMALIZED_OPTIONS_SURFACE_COLUMNS)
        return empty, files

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], utc=True)
    combined.sort_values("date", inplace=True, kind="mergesort")
    combined = combined.drop_duplicates(subset="date", keep="last")
    mask = (combined["date"].dt.date >= start) & (combined["date"].dt.date <= end)
    filtered = combined.loc[mask].copy()
    filtered["symbol"] = symbol
    for column in NORMALIZED_OPTIONS_SURFACE_COLUMNS:
        if column not in filtered.columns:
            if column in OPT_COVERAGE_COLUMNS:
                filtered[column] = False
            else:
                filtered[column] = pd.Series(dtype="float64")
    filtered = filtered[list(NORMALIZED_OPTIONS_SURFACE_COLUMNS)]
    filtered.reset_index(drop=True, inplace=True)
    return filtered, files


def _read_parquet(path: Path) -> "pd.DataFrame | None":
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _coerce_date(value: date | str) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()
    return datetime.fromisoformat(str(value)).date()


def _relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _ensure_pandas_available() -> None:
    if pd is None:
        raise RuntimeError("pandas is required for options surface loaders") from _PANDAS_ERROR


__all__ = ["OptionsSurfaceSlice", "load_options_surface"]
