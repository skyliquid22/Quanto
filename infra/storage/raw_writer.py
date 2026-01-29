"""Raw storage helpers for canonical parquet layout."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

try:  # pragma: no cover - optional dependency
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover - keep optional dependency soft
    pq = None

from infra.paths import raw_root
from infra.storage.parquet import write_parquet_atomic


class RawEquityOHLCVWriter:
    """Writes validated equity OHLCV bars into canonical raw storage."""

    def __init__(
        self,
        base_path: Path | str | None = None,
        *,
        shard_yearly_daily: bool | None = None,
    ) -> None:
        resolved = base_path if base_path is not None else raw_root()
        self.base_path = Path(resolved)
        env_flag = _env_shard_flag()
        if env_flag is not None:
            self.shard_yearly_daily = env_flag
        elif shard_yearly_daily is not None:
            self.shard_yearly_daily = shard_yearly_daily
        else:
            self.shard_yearly_daily = True

    def write_records(
        self,
        vendor: str,
        records: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    ) -> MutableMapping[str, object]:
        """Persist records and return manifest metadata describing the writes."""

        if self.shard_yearly_daily:
            return self._write_sharded_records(vendor, records)

        grouped: MutableMapping[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
        for index, record in enumerate(records):
            symbol = str(record["symbol"])
            timestamp = _coerce_datetime(record["timestamp"], index)
            day_key = timestamp.date().isoformat()
            grouped[(symbol, day_key)].append(record)

        file_details = []
        for (symbol, day_key), items in grouped.items():
            sorted_items = sorted(items, key=lambda rec: _coerce_datetime(rec["timestamp"]))
            path = self._resolve_path(vendor, symbol, day_key)
            payload = self._prepare_parquet_records(sorted_items)
            result = write_parquet_atomic(payload, path)
            file_details.append({"path": str(path), "hash": result["file_hash"], "records": len(sorted_items)})

        return {"files": file_details, "total_files": len(file_details)}

    def _write_sharded_records(
        self,
        vendor: str,
        records: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    ) -> MutableMapping[str, object]:
        shards: MutableMapping[Path, list[dict[str, object]]] = defaultdict(list)
        for index, record in enumerate(records):
            normalized = dict(record)
            timestamp = _coerce_datetime(normalized["timestamp"], index)
            normalized["timestamp"] = timestamp
            symbol = str(normalized["symbol"])
            interval = _normalize_interval(normalized.get("interval"))
            base_dir = self.base_path / vendor / "equity_ohlcv" / symbol / interval
            shard_path = _shard_path(base_dir, timestamp, interval)
            shards[shard_path].append(normalized)

        file_details = []
        for path, items in shards.items():
            merged = _merge_with_existing(path, items)
            result = write_parquet_atomic(merged, path)
            file_details.append({"path": str(path), "hash": result["file_hash"], "records": len(merged)})
        return {"files": file_details, "total_files": len(file_details)}

    def _resolve_path(self, vendor: str, symbol: str, iso_date: str) -> Path:
        year, month, day = iso_date.split("-")
        return (
            self.base_path
            / vendor
            / "equity_ohlcv"
            / symbol
            / "daily"
            / year
            / month
            / f"{day}.parquet"
        )

    def _prepare_parquet_records(
        self, records: Sequence[Mapping[str, object]]
    ) -> Sequence[MutableMapping[str, object]]:
        materialized = []
        for record in records:
            normalized = dict(record)
            normalized["timestamp"] = _coerce_datetime(normalized["timestamp"])
            materialized.append(normalized)
        return materialized


class RawOptionsWriter:
    """Writes option reference, OHLCV, and open interest records."""

    def __init__(self, base_path: Path | str | None = None) -> None:
        resolved = base_path if base_path is not None else raw_root()
        self.base_path = Path(resolved)

    def write_contract_reference(
        self,
        vendor: str,
        records: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
        *,
        snapshot_date: date,
    ) -> MutableMapping[str, object]:
        iso_date = _coerce_date(snapshot_date).isoformat()
        grouped: MutableMapping[str, list[Mapping[str, object]]] = defaultdict(list)
        for record in records:
            underlying = str(record["underlying_symbol"])
            grouped[underlying].append(record)

        file_details = []
        for underlying, items in grouped.items():
            sorted_items = sorted(items, key=lambda rec: str(rec["option_symbol"]))
            path = self._resolve_reference_path(vendor, underlying, iso_date)
            result = write_parquet_atomic(sorted_items, path)
            file_details.append({"path": str(path), "hash": result["file_hash"], "records": len(sorted_items)})

        return {"files": file_details, "total_files": len(file_details)}

    def write_option_ohlcv(
        self,
        vendor: str,
        records: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    ) -> MutableMapping[str, object]:
        return self._write_timeseries(
            vendor,
            records,
            domain_dir="option_contract_ohlcv",
            symbol_field="option_symbol",
        )

    def write_option_open_interest(
        self,
        vendor: str,
        records: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    ) -> MutableMapping[str, object]:
        return self._write_timeseries(
            vendor,
            records,
            domain_dir="option_open_interest",
            symbol_field="option_symbol",
        )

    def _write_timeseries(
        self,
        vendor: str,
        records: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
        *,
        domain_dir: str,
        symbol_field: str,
    ) -> MutableMapping[str, object]:
        grouped: MutableMapping[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
        for index, record in enumerate(records):
            symbol = str(record[symbol_field])
            timestamp = _coerce_datetime(record["timestamp"], index)
            day_key = timestamp.date().isoformat()
            grouped[(symbol, day_key)].append(record)

        file_details = []
        for (symbol, day_key), items in grouped.items():
            sorted_items = sorted(items, key=lambda rec: _coerce_datetime(rec["timestamp"]))
            path = self._resolve_timeseries_path(vendor, domain_dir, symbol, day_key)
            prepared = [dict(item, timestamp=_coerce_datetime(item["timestamp"])) for item in sorted_items]
            result = write_parquet_atomic(prepared, path)
            file_details.append({"path": str(path), "hash": result["file_hash"], "records": len(sorted_items)})

        return {"files": file_details, "total_files": len(file_details)}

    def _resolve_reference_path(self, vendor: str, underlying: str, iso_date: str) -> Path:
        year, month, day = iso_date.split("-")
        return self.base_path / vendor / "option_contract_reference" / underlying / year / month / f"{day}.parquet"

    def _resolve_timeseries_path(self, vendor: str, domain: str, symbol: str, iso_date: str) -> Path:
        year, month, day = iso_date.split("-")
        return (
            self.base_path
            / vendor
            / domain
            / symbol
            / "daily"
            / year
            / month
            / f"{day}.parquet"
        )


def _coerce_datetime(value: object, index: int | None = None) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:  # pragma: no cover - defensive
            position = f" at index {index}" if index is not None else ""
            raise TypeError(f"timestamp must be datetime{position}") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    position = f" at index {index}" if index is not None else ""
    raise TypeError(f"timestamp must be datetime{position}")


def _coerce_date(value: object, index: int | None = None) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return date.fromisoformat(text)
        except ValueError:
            try:
                return datetime.fromisoformat(text).date()
            except ValueError as exc:
                position = f" at index {index}" if index is not None else ""
                raise TypeError(f"snapshot_date must be a date{position}") from exc
    position = f" at index {index}" if index is not None else ""
    raise TypeError(f"snapshot_date must be a date{position}")


def _resolve_key(
    record: Mapping[str, object],
    keys: Sequence[str],
    *,
    label: str,
    index: int | None = None,
) -> str:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    position = f" at index {index}" if index is not None else ""
    raise ValueError(f"Missing {label} field{position}; expected one of {list(keys)}")


class RawFundamentalsWriter:
    """Writes fundamentals statements partitioned by report date."""

    def __init__(self, base_path: Path | str | None = None) -> None:
        resolved = base_path if base_path is not None else raw_root()
        self.base_path = Path(resolved)

    def write_records(
        self,
        vendor: str,
        records: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    ) -> MutableMapping[str, object]:
        grouped: MutableMapping[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
        for index, record in enumerate(records):
            symbol = str(record["symbol"])
            iso_date = _coerce_date(record.get("report_date"), index).isoformat()
            grouped[(symbol, iso_date)].append(record)

        file_details = []
        for (symbol, iso_date), items in grouped.items():
            prepared = []
            deduped: dict[tuple[str, str, str, str], dict[str, object]] = {}
            for entry in items:
                normalized = dict(entry)
                normalized_symbol = str(normalized["symbol"])
                normalized["symbol"] = normalized_symbol
                normalized_date = _coerce_date(normalized["report_date"]).isoformat()
                normalized["report_date"] = normalized_date
                fiscal_period = str(normalized.get("fiscal_period") or "")
                normalized["fiscal_period"] = fiscal_period
                vendor_field = str(normalized.get("source_vendor") or vendor)
                normalized["source_vendor"] = vendor_field
                key = (normalized_symbol, normalized_date, fiscal_period, vendor_field)
                existing = deduped.get(key)
                if existing:
                    if existing != normalized:
                        raise ValueError(
                            f"Conflicting fundamentals record for {key}"
                        )
                    continue
                deduped[key] = normalized

            ordered = sorted(
                deduped.values(),
                key=lambda rec: (
                    rec["fiscal_period"],
                    str(rec.get("filing_id") or ""),
                    rec["source_vendor"],
                ),
            )
            prepared.extend(ordered)
            path = self._resolve_path(vendor, symbol, iso_date)
            result = write_parquet_atomic(prepared, path)
            file_details.append({"path": str(path), "hash": result["file_hash"], "records": len(prepared)})

        return {"files": file_details, "total_files": len(file_details)}

    def _resolve_path(self, vendor: str, symbol: str, iso_date: str) -> Path:
        year, month, day = iso_date.split("-")
        return self.base_path / vendor / "fundamentals" / symbol / year / month / f"{day}.parquet"


class RawFinancialDatasetsWriter:
    """Writes Financial Datasets raw domains into canonical raw storage."""

    def __init__(self, base_path: Path | str | None = None) -> None:
        resolved = base_path if base_path is not None else raw_root()
        self.base_path = Path(resolved)

    def write_company_facts(
        self,
        vendor: str,
        records: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    ) -> MutableMapping[str, object]:
        return self._write_symbol_date(
            vendor,
            records,
            domain_dir="company_facts",
            symbol_keys=("symbol", "ticker"),
            date_keys=("as_of_date", "snapshot_date", "ingest_date", "date"),
        )

    def write_financial_metrics(
        self,
        vendor: str,
        records: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    ) -> MutableMapping[str, object]:
        return self._write_symbol_date(
            vendor,
            records,
            domain_dir="financial_metrics",
            symbol_keys=("symbol", "ticker"),
            date_keys=("report_period", "as_of_date", "ingest_date", "date"),
        )

    def write_financial_metrics_snapshot(
        self,
        vendor: str,
        records: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    ) -> MutableMapping[str, object]:
        return self._write_symbol_date(
            vendor,
            records,
            domain_dir="financial_metrics_snapshot",
            symbol_keys=("symbol", "ticker"),
            date_keys=("as_of_date", "ingest_date", "date"),
        )

    def write_financial_statements(
        self,
        vendor: str,
        records: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    ) -> MutableMapping[str, object]:
        return self._write_symbol_date(
            vendor,
            records,
            domain_dir="financial_statements",
            symbol_keys=("symbol", "ticker"),
            date_keys=("report_date", "report_period"),
        )

    def write_insider_trades(
        self,
        vendor: str,
        records: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    ) -> MutableMapping[str, object]:
        return self._write_symbol_date(
            vendor,
            records,
            domain_dir="insider_trades",
            symbol_keys=("symbol", "ticker"),
            date_keys=("transaction_date", "filing_date"),
        )

    def write_institutional_ownership(
        self,
        vendor: str,
        records: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    ) -> MutableMapping[str, object]:
        return self._write_symbol_date(
            vendor,
            records,
            domain_dir="institutional_ownership",
            symbol_keys=("symbol", "ticker", "investor"),
            date_keys=("report_period",),
        )

    def write_news(
        self,
        vendor: str,
        records: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    ) -> MutableMapping[str, object]:
        return self._write_symbol_date(
            vendor,
            records,
            domain_dir="news",
            symbol_keys=("symbol", "ticker"),
            date_keys=("date", "published_at", "published_date"),
        )

    def _write_symbol_date(
        self,
        vendor: str,
        records: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
        *,
        domain_dir: str,
        symbol_keys: Sequence[str],
        date_keys: Sequence[str],
    ) -> MutableMapping[str, object]:
        grouped: MutableMapping[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
        for index, record in enumerate(records):
            symbol = _resolve_key(record, symbol_keys, label="symbol", index=index)
            raw_date = _resolve_key(record, date_keys, label="date", index=index)
            iso_date = _coerce_date(raw_date, index).isoformat()
            grouped[(symbol, iso_date)].append(record)

        file_details = []
        for (symbol, iso_date), items in grouped.items():
            ordered = sorted(items, key=lambda rec: str(rec.get("ingest_ts") or ""))
            path = self._resolve_path(vendor, domain_dir, symbol, iso_date)
            result = write_parquet_atomic(list(ordered), path)
            file_details.append({"path": str(path), "hash": result["file_hash"], "records": len(items)})

        return {"files": file_details, "total_files": len(file_details)}

    def _resolve_path(self, vendor: str, domain_dir: str, symbol: str, iso_date: str) -> Path:
        year, month, day = iso_date.split("-")
        return self.base_path / vendor / domain_dir / symbol / year / month / f"{day}.parquet"


def _merge_with_existing(path: Path, new_records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    combined: list[dict[str, object]] = []
    if path.exists():
        combined.extend(_read_existing_records(path))
    combined.extend(dict(record) for record in new_records)
    return _dedup_records(combined)


def _read_existing_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    if pq is not None:
        try:
            table = pq.read_table(path)
            return table.to_pylist()
        except Exception:  # pragma: no cover - fallback when parquet reader unavailable
            pass
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            f"Cannot merge existing parquet shard at {path} without pyarrow installed"
        ) from exc
    return json.loads(text) if text.strip() else []


def _dedup_records(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    dedup: dict[tuple[object, str, str, object], dict[str, object]] = {}
    for record in records:
        normalized = dict(record)
        symbol = str(normalized.get("symbol"))
        timestamp = _coerce_datetime(normalized["timestamp"])
        raw_interval = normalized.get("interval")
        interval = _normalize_interval(raw_interval) if raw_interval is not None else None
        normalized["symbol"] = symbol
        normalized["timestamp"] = timestamp
        if raw_interval is not None:
            normalized["interval"] = interval
        source_vendor = normalized.get("source_vendor")
        key = (symbol, timestamp.isoformat(), interval, source_vendor)
        dedup[key] = normalized
    ordered = sorted(dedup.values(), key=lambda rec: rec["timestamp"])
    return ordered


def _normalize_interval(value: object | None) -> str:
    if not value:
        return "daily"
    return str(value).strip().lower()


def _shard_mode(interval: str) -> str:
    return "year" if interval == "daily" else "month"


def _shard_path(base_dir: Path, timestamp: datetime, interval: str) -> Path:
    if _shard_mode(interval) == "year":
        return base_dir / f"{timestamp.year}.parquet"
    return base_dir / f"{timestamp.year:04d}" / f"{timestamp.month:02d}.parquet"


def _env_shard_flag() -> bool | None:
    value = os.environ.get("QUANTO_RAW_YEARLY_DAILY")
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"", "0", "false", "no"}:
        return False
    return True


__all__ = ["RawEquityOHLCVWriter", "RawFundamentalsWriter", "RawOptionsWriter"]
