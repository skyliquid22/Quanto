"""Canonical fundamentals loader for research feature building."""

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


FUNDAMENTALS_CANONICAL_COLUMNS = (
    "report_date",
    "period",
    "fiscal_period",
    "filing_date",
    "revenue",
    "net_income",
    "eps",
    "operating_income",
    "free_cash_flow",
    "total_assets",
    "total_liabilities",
    "shareholder_equity",
    "shares_outstanding",
)


@dataclass(frozen=True)
class CanonicalFundamentalsSlice:
    """Per-symbol canonical fundamentals rows."""

    symbol: str
    frame: "pd.DataFrame"
    file_paths: List[Path]


def load_canonical_fundamentals(
    symbols: Sequence[str],
    start_date: date | str,
    end_date: date | str,
    *,
    data_root: Path | None = None,
    lookback_days: int = 400,
) -> Tuple[Dict[str, CanonicalFundamentalsSlice], Dict[str, str]]:
    """Load canonical fundamentals for the requested symbols and dates."""

    _ensure_pandas_available()
    root = Path(data_root) if data_root else get_data_root()
    start = _coerce_date(start_date)
    end = _coerce_date(end_date)
    if end < start:
        raise ValueError("end_date must be greater than or equal to start_date")
    lookback = max(0, int(lookback_days))
    lookback_start = start - timedelta(days=lookback)
    slices: Dict[str, CanonicalFundamentalsSlice] = {}
    file_hashes: Dict[str, str] = {}

    for symbol in _ordered_symbols(symbols):
        frame, files = _load_symbol(symbol, lookback_start, end, root)
        slices[symbol] = CanonicalFundamentalsSlice(symbol=symbol, frame=frame, file_paths=files)
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
    base = root / "canonical" / "fundamentals" / symbol
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
        empty = pd.DataFrame(columns=FUNDAMENTALS_CANONICAL_COLUMNS)
        return empty, files

    combined = pd.concat(frames, ignore_index=True)
    combined["symbol"] = symbol
    combined = _normalize_dates(combined)
    combined = _normalize_periods(combined)
    combined = _filter_statement_type(combined)
    combined.sort_values(["report_date", "filing_date"], inplace=True, kind="mergesort")
    combined.drop_duplicates(subset=["report_date", "period"], keep="last", inplace=True)

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    mask = (combined["report_date"] >= start_ts) & (combined["report_date"] <= end_ts)
    filtered = combined.loc[mask].copy()
    for column in FUNDAMENTALS_CANONICAL_COLUMNS:
        if column not in filtered.columns:
            filtered[column] = pd.Series(dtype="float64")
    numeric = [
        "revenue",
        "net_income",
        "eps",
        "operating_income",
        "free_cash_flow",
        "total_assets",
        "total_liabilities",
        "shareholder_equity",
        "shares_outstanding",
    ]
    for column in numeric:
        filtered[column] = pd.to_numeric(filtered[column], errors="coerce")
    filtered = filtered[FUNDAMENTALS_CANONICAL_COLUMNS]
    filtered.reset_index(drop=True, inplace=True)
    return filtered, files


def _normalize_dates(frame: "pd.DataFrame") -> "pd.DataFrame":
    report_col = "report_date" if "report_date" in frame else "report_period"
    report = _coerce_series(frame.get(report_col))
    frame["report_date"] = report
    frame["filing_date"] = _coerce_series(frame.get("filing_date"))
    frame.dropna(subset=["report_date"], inplace=True)
    return frame


def _normalize_periods(frame: "pd.DataFrame") -> "pd.DataFrame":
    period = frame.get("period")
    fiscal = frame.get("fiscal_period")
    normalized = []
    if period is None:
        period = [""] * len(frame)
    if fiscal is None:
        fiscal = [""] * len(frame)
    for raw, fiscal_raw in zip(period, fiscal):
        normalized.append(_normalize_period(raw, fiscal_raw))
    frame["period"] = normalized
    return frame


def _filter_statement_type(frame: "pd.DataFrame") -> "pd.DataFrame":
    if "statement_type" not in frame:
        return frame
    values = frame["statement_type"].astype(str).str.lower()
    if (values == "all").any():
        return frame.loc[values == "all"].copy()
    return frame


def _normalize_period(raw: object, fiscal_raw: object) -> str:
    text = str(raw or "").strip().lower()
    if text in {"quarter", "quarterly", "q", "qtr"}:
        return "quarterly"
    if text in {"annual", "fy", "year", "yearly"}:
        return "annual"
    fiscal = str(fiscal_raw or "").strip().upper()
    if fiscal.startswith("Q"):
        return "quarterly"
    if fiscal.startswith("FY") or fiscal.startswith("A"):
        return "annual"
    return text


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
        raise RuntimeError("pandas is required for canonical fundamentals loaders") from _PANDAS_ERROR


__all__ = ["CanonicalFundamentalsSlice", "load_canonical_fundamentals", "FUNDAMENTALS_CANONICAL_COLUMNS"]
