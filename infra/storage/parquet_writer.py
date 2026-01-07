"""Canonical Parquet writer helpers ensuring deterministic outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None

try:  # pragma: no cover - optional dependency
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover
    pq = None

from .parquet import write_parquet_atomic


def merge_write_parquet(
    path: Path | str,
    records: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    *,
    dedup_cols: Sequence[str],
    sort_key: str = "timestamp",
) -> dict[str, object]:
    destination = Path(path)
    combined: list[Mapping[str, object]] = []
    if destination.exists():
        combined.extend(_read_existing(destination))
    combined.extend(records)
    df = _records_to_dataframe(combined)
    if dedup_cols:
        df = df.drop_duplicates(subset=list(dedup_cols), keep="last")
    result = _write_normalized_dataframe(destination, df, sort_keys=[sort_key])
    return result


def write_normalized_parquet(
    path: Path | str,
    records: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    *,
    sort_keys: Sequence[str],
) -> dict[str, object]:
    destination = Path(path)
    df = _records_to_dataframe(records)
    return _write_normalized_dataframe(destination, df, sort_keys=sort_keys)


def _read_existing(path: Path) -> list[dict[str, object]]:
    if pq is not None:
        try:
            table = pq.read_table(path)
            return table.to_pylist()
        except Exception:  # pragma: no cover - fall back to json
            pass
    text = path.read_text(encoding="utf-8")
    return json.loads(text) if text.strip() else []


def _records_to_dataframe(records: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]]):
    if pd is None:
        raise RuntimeError("pandas is required for canonical parquet operations")
    if isinstance(records, pd.DataFrame):
        return records.copy()
    return pd.DataFrame.from_records(list(records))


def normalize_for_write(df: "pd.DataFrame", *, sort_keys: Sequence[str]) -> "pd.DataFrame":
    if pd is None:
        raise RuntimeError("pandas is required for canonical parquet operations")
    frame = df.copy()
    if "timestamp" not in frame.columns and "t" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["t"], unit="ms", utc=True)
    elif "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    if "timestamp" in frame.columns:
        frame["timestamp"] = frame["timestamp"].dt.tz_convert("UTC")
    sort_cols = [col for col in sort_keys if col in frame.columns]
    if sort_cols:
        frame = frame.sort_values(sort_cols, kind="mergesort")
    frame = frame.reset_index(drop=True)
    ordered_cols = sorted(frame.columns)
    frame = frame[ordered_cols]
    return frame


def compute_content_hash(
    df: "pd.DataFrame",
    *,
    sort_keys: Sequence[str],
    normalized: bool = False,
) -> str:
    if pd is None:
        raise RuntimeError("pandas is required for canonical parquet operations")
    frame = df if normalized else normalize_for_write(df, sort_keys=sort_keys)
    csv_bytes = frame.to_csv(index=False).encode("utf-8")
    return f"sha256:{hashlib.sha256(csv_bytes).hexdigest()}"


def _write_normalized_dataframe(path: Path, df: "pd.DataFrame", *, sort_keys: Sequence[str]) -> dict[str, object]:
    normalized = normalize_for_write(df, sort_keys=sort_keys)
    content_hash = compute_content_hash(normalized, sort_keys=sort_keys, normalized=True)
    records = normalized.to_dict("records")
    result = write_parquet_atomic(records, path)
    result["content_hash"] = content_hash
    result["records"] = len(records)
    return result


__all__ = ["write_parquet_atomic", "merge_write_parquet", "write_normalized_parquet", "normalize_for_write", "compute_content_hash"]
