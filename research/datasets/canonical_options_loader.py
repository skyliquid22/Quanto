"""Canonical options dataset loader for downstream feature engineering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from infra.timestamps import coerce_timestamp as _coerce_timestamp
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from infra.normalization.lineage import compute_file_hash
from infra.paths import get_data_root

try:  # pragma: no cover - pandas is optional in some environments
    import pandas as pd  # type: ignore
except Exception as exc:  # pragma: no cover - surface a clear error later
    pd = None
    _PANDAS_ERROR: Exception | None = exc
else:  # pragma: no cover - tracking import status for type checkers
    _PANDAS_ERROR = None

UTC = timezone.utc
FREQUENCY = "daily"


@dataclass(frozen=True)
class CanonicalOptionData:
    """Container holding canonical option contract datasets for one underlying."""

    reference: "pd.DataFrame"
    ohlcv: "pd.DataFrame"
    open_interest: "pd.DataFrame"
    file_paths: List[Path]


def load_canonical_options(
    underlying_symbol: str,
    start_date: date | str,
    end_date: date | str,
    *,
    data_root: Path | None = None,
) -> Tuple[CanonicalOptionData, Dict[str, str]]:
    """Load canonical option reference/ohlcv/open-interest files for an underlying."""

    _ensure_pandas_available()
    resolved = Path(data_root) if data_root else get_data_root()
    start = _coerce_date(start_date)
    end = _coerce_date(end_date)
    if end < start:
        raise ValueError("end_date must be greater than or equal to start_date")
    symbol = str(underlying_symbol).strip().upper()
    if not symbol:
        raise ValueError("underlying_symbol must be provided")

    reference_rows, reference_files = _load_domain_rows("option_contract_reference", symbol, start, end, resolved)
    oi_rows, oi_files = _load_domain_rows("option_open_interest", symbol, start, end, resolved)
    ohlcv_rows, ohlcv_files = _load_domain_rows("option_contract_ohlcv", symbol, start, end, resolved)

    reference_df = _build_reference_frame(reference_rows, symbol)
    oi_df = _build_open_interest_frame(oi_rows, symbol)
    ohlcv_df = _build_ohlcv_frame(ohlcv_rows, symbol)

    touched_files = reference_files + oi_files + ohlcv_files
    file_hashes = {
        _relative_to(path, resolved): compute_file_hash(path)
        for path in touched_files
        if path.exists()
    }
    payload = CanonicalOptionData(
        reference=reference_df,
        ohlcv=ohlcv_df,
        open_interest=oi_df,
        file_paths=touched_files,
    )
    return payload, file_hashes


def _build_reference_frame(rows: List[Mapping[str, Any]], underlying: str) -> "pd.DataFrame":
    columns = ["option_symbol", "underlying_symbol", "expiration_date", "strike", "option_type", "multiplier"]
    if not rows:
        return pd.DataFrame(columns=columns)

    normalized: List[Dict[str, Any]] = []
    for entry in rows:
        option_symbol = str(entry.get("option_symbol") or "").strip().upper()
        if not option_symbol:
            continue
        entry_underlying = str(entry.get("underlying_symbol") or "").strip().upper()
        if entry_underlying and entry_underlying != underlying:
            continue
        option_type = _normalize_option_type(entry.get("option_type"))
        normalized.append(
            {
                "option_symbol": option_symbol,
                "underlying_symbol": underlying,
                "expiration_date": _coerce_date(entry.get("expiration_date")) if entry.get("expiration_date") else None,
                "strike": float(entry.get("strike", 0.0)),
                "option_type": option_type,
                "multiplier": float(entry.get("multiplier", 100.0)),
            }
        )
    if not normalized:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(normalized, columns=columns)
    frame.sort_values(["option_symbol", "expiration_date"], inplace=True, kind="mergesort")
    frame = frame.drop_duplicates(subset="option_symbol", keep="last")
    frame.reset_index(drop=True, inplace=True)
    return frame


def _build_open_interest_frame(rows: List[Mapping[str, Any]], underlying: str) -> "pd.DataFrame":
    columns = ["timestamp", "option_symbol", "open_interest", "underlying_symbol"]
    if not rows:
        return pd.DataFrame(columns=columns)
    normalized: List[Dict[str, Any]] = []
    for entry in rows:
        option_symbol = str(entry.get("option_symbol") or "").strip().upper()
        if not option_symbol:
            continue
        timestamp = entry.get("timestamp")
        if timestamp is None:
            continue
        normalized.append(
            {
                "timestamp": _coerce_timestamp(timestamp),
                "option_symbol": option_symbol,
                "open_interest": float(entry.get("open_interest", 0.0)),
                "underlying_symbol": str(entry.get("underlying_symbol") or underlying).strip().upper(),
            }
        )
    if not normalized:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(normalized, columns=columns)
    frame.sort_values(["timestamp", "option_symbol"], inplace=True, kind="mergesort")
    frame.reset_index(drop=True, inplace=True)
    return frame


def _build_ohlcv_frame(rows: List[Mapping[str, Any]], underlying: str) -> "pd.DataFrame":
    columns = ["timestamp", "option_symbol", "volume", "underlying_symbol"]
    if not rows:
        return pd.DataFrame(columns=columns)
    normalized: List[Dict[str, Any]] = []
    for entry in rows:
        option_symbol = str(entry.get("option_symbol") or "").strip().upper()
        if not option_symbol:
            continue
        timestamp = entry.get("timestamp")
        if timestamp is None:
            continue
        normalized.append(
            {
                "timestamp": _coerce_timestamp(timestamp),
                "option_symbol": option_symbol,
                "volume": float(entry.get("volume", 0.0)),
                "underlying_symbol": str(entry.get("underlying_symbol") or underlying).strip().upper(),
            }
        )
    if not normalized:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(normalized, columns=columns)
    frame.sort_values(["timestamp", "option_symbol"], inplace=True, kind="mergesort")
    frame.reset_index(drop=True, inplace=True)
    return frame


def _load_domain_rows(
    domain: str,
    underlying_symbol: str,
    start: date,
    end: date,
    data_root: Path,
) -> Tuple[List[Mapping[str, Any]], List[Path]]:
    base = data_root / "canonical" / domain / underlying_symbol / FREQUENCY
    rows: List[Mapping[str, Any]] = []
    files: List[Path] = []
    for day in _iter_days(start, end):
        path = base / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.day:02d}.parquet"
        if not path.exists():
            continue
        files.append(path)
        try:
            payload = _read_records(path)
        except Exception:
            continue
        rows.extend(payload)
    return rows, files


def _iter_days(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _coerce_date(value: date | str | None) -> date:
    if value is None:
        raise ValueError("date value is required")
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))



# _coerce_timestamp is imported from infra.timestamps.


def _read_records(path: Path) -> List[Mapping[str, Any]]:
    try:  # pragma: no branch - prefer pyarrow when available
        import pyarrow.parquet as pq  # type: ignore

        table = pq.read_table(path)
        return table.to_pylist()
    except Exception:
        text = path.read_text(encoding="utf-8")
        return json.loads(text) if text.strip() else []


def _relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _normalize_option_type(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"call", "c"}:
        return "call"
    if text in {"put", "p"}:
        return "put"
    return None


def _ensure_pandas_available() -> None:
    if pd is None:
        raise RuntimeError("pandas is required to load canonical options data") from _PANDAS_ERROR


__all__ = ["CanonicalOptionData", "load_canonical_options"]
