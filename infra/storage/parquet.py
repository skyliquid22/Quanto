"""Single-surface deterministic Parquet writing utilities."""

from __future__ import annotations

from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # pragma: no cover - optional dependency
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore

    _PARQUET_AVAILABLE = True
except Exception:  # pragma: no cover - fallback for local dev without pyarrow
    pa = None
    pq = None
    _PARQUET_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore

    _PANDAS_AVAILABLE = True
except Exception:  # pragma: no cover - pandas is optional
    pd = None
    _PANDAS_AVAILABLE = False


def write_parquet_atomic(
    records: Sequence[Mapping[str, Any]]
    | Iterable[Mapping[str, Any]]
    | "pd.DataFrame",  # type: ignore[name-defined]
    path: Path | str,
    *,
    schema: Any | None = None,
    compression: str = "zstd",
    use_pyarrow: bool = True,
    filesystem: Any | None = None,
) -> dict[str, Any]:
    """Write records atomically to a Parquet file, returning deterministic metadata."""

    if filesystem is not None:
        raise NotImplementedError("Custom filesystem backends are not supported in this writer")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_name(destination.name + ".tmp")

    materialized, column_order = _materialize_records(records)
    column_order = _resolve_column_order(column_order, schema)

    if use_pyarrow and _PARQUET_AVAILABLE:
        _write_with_pyarrow(materialized, column_order, tmp_path, schema=schema, compression=compression)
    else:
        _write_json_fallback(materialized, tmp_path)

    os.replace(tmp_path, destination)
    bytes_written = destination.stat().st_size
    file_hash = _sha256_file(destination)
    return {"path": str(destination), "file_hash": file_hash, "bytes_written": bytes_written}


def _materialize_records(
    records: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]] | "pd.DataFrame",  # type: ignore[name-defined]
) -> tuple[list[dict[str, Any]], list[str]]:
    if _PANDAS_AVAILABLE and isinstance(records, pd.DataFrame):  # type: ignore[arg-type]
        return records.to_dict("records"), [str(column) for column in records.columns]

    materialized: list[dict[str, Any]] = []
    ordered_keys: list[str] = []
    for record in records:
        coerced = dict(record)
        materialized.append(coerced)
        for key in coerced.keys():
            if key not in ordered_keys:
                ordered_keys.append(str(key))

    if not ordered_keys:
        ordered_keys = []
    else:
        ordered_keys = sorted(set(ordered_keys))

    return materialized, ordered_keys


def _resolve_column_order(column_order: Sequence[str], schema: Any | None) -> list[str]:
    if schema is not None and hasattr(schema, "names"):
        return list(schema.names)
    return list(column_order)


def _write_with_pyarrow(
    records: Sequence[Mapping[str, Any]],
    column_order: Sequence[str],
    tmp_path: Path,
    *,
    schema: Any | None,
    compression: str,
) -> None:
    if not _PARQUET_AVAILABLE or pa is None or pq is None:  # pragma: no cover - guard
        raise RuntimeError("pyarrow is not available")

    table = pa.Table.from_pylist(list(records), schema=schema)
    if column_order:
        existing = [name for name in table.schema.names if name in column_order]
        extras = [name for name in table.schema.names if name not in column_order]
        desired = [name for name in column_order if name in table.schema.names] + extras
        if desired != table.schema.names:
            table = table.select(desired)
    pq.write_table(
        table,
        tmp_path,
        compression=compression,
        use_dictionary=True,
        coerce_timestamps="ms",
        flavor="spark",
        data_page_version="1.0",
    )


def _write_json_fallback(records: Sequence[Mapping[str, Any]], tmp_path: Path) -> None:
    payload = json.dumps(records, default=_json_default, sort_keys=True, separators=(",", ":"))
    tmp_path.write_text(payload, encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"Unsupported type {type(value)} during serialization")


__all__ = ["write_parquet_atomic", "_PARQUET_AVAILABLE"]
