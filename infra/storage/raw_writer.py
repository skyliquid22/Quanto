"""Raw storage helpers for canonical parquet layout."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

from infra.paths import raw_root
from infra.storage.parquet import write_parquet_atomic


class RawEquityOHLCVWriter:
    """Writes validated equity OHLCV bars into canonical raw storage."""

    def __init__(self, base_path: Path | str | None = None) -> None:
        resolved = base_path if base_path is not None else raw_root()
        self.base_path = Path(resolved)

    def write_records(
        self,
        vendor: str,
        records: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    ) -> MutableMapping[str, object]:
        """Persist records and return manifest metadata describing the writes."""

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
    position = f" at index {index}" if index is not None else ""
    raise TypeError(f"timestamp must be datetime{position}")


def _coerce_date(value: object, index: int | None = None) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        return date.fromisoformat(value)
    position = f" at index {index}" if index is not None else ""
    raise TypeError(f"snapshot_date must be a date{position}")


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


__all__ = ["RawEquityOHLCVWriter", "RawFundamentalsWriter", "RawOptionsWriter"]
