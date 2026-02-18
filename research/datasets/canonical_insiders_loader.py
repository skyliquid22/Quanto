"""Canonical insider trades loader for research feature building."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from infra.normalization.lineage import compute_file_hash
from infra.paths import get_data_root

try:  # pragma: no cover - pandas optional in some environments
    import pandas as pd  # type: ignore
except Exception as exc:  # pragma: no cover
    pd = None
    _PANDAS_ERROR: Exception | None = exc
else:  # pragma: no cover
    _PANDAS_ERROR = None


INSIDERS_CANONICAL_COLUMNS = (
    "filing_date",
    "transaction_date",
    "transaction_value",
    "transaction_shares",
    "shares_owned_before_transaction",
    "shares_owned_after_transaction",
    "security_title",
    "title",
    "is_board_director",
    "name",
    "issuer",
)


@dataclass(frozen=True)
class CanonicalInsidersSlice:
    """Per-symbol canonical insider rows."""

    symbol: str
    frame: "pd.DataFrame"
    file_paths: List[Path]


def load_canonical_insiders(
    symbols: Sequence[str],
    start_date: date | str,
    end_date: date | str,
    *,
    data_root: Path | None = None,
    lookback_days: int = 820,
) -> Tuple[Dict[str, CanonicalInsidersSlice], Dict[str, str]]:
    """Load canonical insider trades for the requested symbols and dates."""

    _ensure_pandas_available()
    root = Path(data_root) if data_root else get_data_root()
    start = _coerce_date(start_date)
    end = _coerce_date(end_date)
    if end < start:
        raise ValueError("end_date must be greater than or equal to start_date")
    lookback = max(0, int(lookback_days))
    lookback_start = start - timedelta(days=lookback)
    slices: Dict[str, CanonicalInsidersSlice] = {}
    file_hashes: Dict[str, str] = {}

    for symbol in _ordered_symbols(symbols):
        frame, files = _load_symbol(symbol, lookback_start, end, root)
        slices[symbol] = CanonicalInsidersSlice(symbol=symbol, frame=frame, file_paths=files)
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
    base = root / "canonical" / "insiders" / symbol
    frames: List["pd.DataFrame"] = []
    files: List[Path] = []
    years = range(start.year, end.year + 1)
    for year in years:
        year_dir = base / f"{year}"
        if not year_dir.exists():
            continue
        for path in year_dir.rglob("*.parquet"):
            files.append(path)
            frame = _read_parquet(path)
            if frame is None or frame.empty:
                continue
            frames.append(frame)

    if not frames:
        empty = pd.DataFrame(columns=INSIDERS_CANONICAL_COLUMNS)
        return empty, files

    combined = pd.concat(frames, ignore_index=True)
    combined = _normalize_dates(combined)
    combined = _normalize_types(combined)

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    mask = (combined["filing_date"] >= start_ts) & (combined["filing_date"] <= end_ts)
    filtered = combined.loc[mask].copy()

    for column in INSIDERS_CANONICAL_COLUMNS:
        if column not in filtered.columns:
            filtered[column] = pd.Series(dtype="float64")
    filtered = filtered[list(INSIDERS_CANONICAL_COLUMNS)]
    filtered.reset_index(drop=True, inplace=True)
    return filtered, files


def _normalize_dates(frame: "pd.DataFrame") -> "pd.DataFrame":
    data = frame.copy()
    data["filing_date"] = _coerce_series(data.get("filing_date"))
    data["transaction_date"] = _coerce_series(data.get("transaction_date"))
    data.dropna(subset=["filing_date"], inplace=True)
    return data


def _normalize_types(frame: "pd.DataFrame") -> "pd.DataFrame":
    data = frame.copy()
    for column in (
        "transaction_value",
        "transaction_shares",
        "shares_owned_before_transaction",
        "shares_owned_after_transaction",
    ):
        data[column] = pd.to_numeric(data.get(column), errors="coerce")
    data["security_title"] = data.get("security_title").astype(str) if "security_title" in data else ""
    data["title"] = data.get("title").astype(str) if "title" in data else ""
    data["name"] = data.get("name").astype(str) if "name" in data else ""
    data["issuer"] = data.get("issuer").astype(str) if "issuer" in data else ""
    if "is_board_director" in data:
        data["is_board_director"] = data["is_board_director"].fillna(False).astype(bool)
    else:
        data["is_board_director"] = False
    return data


def _read_parquet(path: Path) -> "pd.DataFrame | None":
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _coerce_series(series: object) -> "pd.Series":
    if series is None:
        return pd.Series(dtype="datetime64[ns, UTC]")
    data = pd.to_datetime(series, utc=True, errors="coerce")
    return data.dt.normalize()


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
        raise RuntimeError("pandas is required for canonical insiders loaders") from _PANDAS_ERROR


__all__ = ["CanonicalInsidersSlice", "load_canonical_insiders", "INSIDERS_CANONICAL_COLUMNS"]
