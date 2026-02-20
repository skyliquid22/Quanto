"""Raw storage helpers for canonical parquet layout."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timezone

from infra.timestamps import coerce_timestamp
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence

try:  # pragma: no cover - optional dependency
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover - keep optional dependency soft
    pq = None

try:  # pragma: no cover - pandas optional
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None

from infra.paths import raw_root
from infra.storage.parquet import write_parquet_atomic
from infra.storage.financialdatasets_policy import DomainPolicy, load_financialdatasets_policies

LOGGER = logging.getLogger(__name__)


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
            path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = path.with_suffix(".lock")
            _with_lock(lock_path)
            try:
                merged = _merge_with_existing(path, items)
                result = write_parquet_atomic(merged, path)
            finally:
                _release_lock(lock_path)
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
    return coerce_timestamp(value, index=index)


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
    """Writes Financial Datasets raw domains into policy-driven CSV layouts."""

    def __init__(
        self,
        base_path: Path | str | None = None,
        *,
        policy_map: Mapping[str, DomainPolicy] | None = None,
        config_path: Path | str | None = None,
        allow_old_layout: bool = False,
        vendor_prefix: bool = True,
    ) -> None:
        resolved = base_path if base_path is not None else raw_root()
        self.base_path = Path(resolved)
        if policy_map is not None:
            normalized: Dict[str, DomainPolicy] = {}
            for domain, raw in policy_map.items():
                if isinstance(raw, DomainPolicy):
                    normalized[domain] = raw
                elif isinstance(raw, Mapping):
                    normalized[domain] = DomainPolicy(
                        policy=str(raw.get("policy", "")),
                        date_priority=tuple(raw.get("date_priority", []) or []),
                        date_kind=str(raw.get("date_kind", "date") or "date"),
                        dedup_keys=tuple(raw.get("dedup_keys", []) or []),
                        expected_columns=tuple(raw.get("expected_columns", []) or []),
                    )
            self.policy_map = normalized
        else:
            self.policy_map = load_financialdatasets_policies(config_path)
        self.allow_old_layout = allow_old_layout
        self.vendor_prefix = vendor_prefix

    def write_company_facts(
        self,
        vendor: str,
        records: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    ) -> MutableMapping[str, object]:
        return self._write_domain(vendor, "company_facts", records)

    def write_financial_metrics(
        self,
        vendor: str,
        records: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    ) -> MutableMapping[str, object]:
        return self._write_domain(vendor, "financial_metrics", records)

    def write_financial_metrics_snapshot(
        self,
        vendor: str,
        records: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    ) -> MutableMapping[str, object]:
        return self._write_domain(vendor, "financial_metrics_snapshot", records)

    def write_financial_statements(
        self,
        vendor: str,
        records: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    ) -> MutableMapping[str, object]:
        return self._write_domain(vendor, "financial_statements", records)

    def write_insider_trades(
        self,
        vendor: str,
        records: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    ) -> MutableMapping[str, object]:
        return self._write_domain(vendor, "insider_trades", records)

    def write_institutional_ownership(
        self,
        vendor: str,
        records: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    ) -> MutableMapping[str, object]:
        return self._write_domain(vendor, "institutional_ownership", records)

    def write_news(
        self,
        vendor: str,
        records: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    ) -> MutableMapping[str, object]:
        return self._write_domain(vendor, "news", records)

    def _write_domain(
        self,
        vendor: str,
        domain: str,
        records: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    ) -> MutableMapping[str, object]:
        if pd is None:
            raise RuntimeError("pandas is required for FinancialDatasets raw CSV writes")
        policy = self._resolve_policy(domain)
        vendor_root = self._vendor_root(vendor)
        if not self.allow_old_layout:
            self._assert_no_old_layout(vendor_root, domain)
        if not records:
            return {"files": [], "total_files": 0}

        if policy.policy == "snapshot_single_csv":
            return self._write_snapshot_single(vendor_root, domain, policy, records)
        if policy.policy == "snapshot_table_csv":
            return self._write_snapshot_table(vendor_root, domain, policy, records)
        if policy.policy == "timeseries_csv_unsharded":
            return self._write_timeseries_unsharded(vendor_root, domain, policy, records)
        if policy.policy == "timeseries_csv_yearly":
            return self._write_timeseries_yearly(vendor_root, domain, policy, records)
        if policy.policy == "timeseries_parquet_sharded":
            raise ValueError("timeseries_parquet_sharded policy is reserved for future domains")
        raise ValueError(f"Unsupported raw storage policy '{policy.policy}' for {domain}")

    def _vendor_root(self, vendor: str) -> Path:
        if self.vendor_prefix:
            return self.base_path / vendor
        return self.base_path

    def _resolve_policy(self, domain: str) -> DomainPolicy:
        policy = self.policy_map.get(domain)
        if policy is None:
            raise ValueError(f"No FinancialDatasets raw storage policy configured for '{domain}'")
        return policy

    def _assert_no_old_layout(self, vendor_root: Path, domain: str) -> None:
        legacy_root = vendor_root / domain
        if not legacy_root.exists():
            return
        for path in legacy_root.rglob("*.parquet"):
            if ".migration_staging" in path.parts:
                continue
            raise RuntimeError(
                f"Legacy FinancialDatasets layout detected at {path}. "
                "Run scripts/migrate_financialdatasets_raw_layout.py before ingesting new data."
            )
        for path in legacy_root.rglob("*.csv"):
            if ".migration_staging" in path.parts:
                continue
            try:
                rel = path.relative_to(legacy_root)
            except ValueError:
                continue
            if len(rel.parts) < 4:
                continue
            raise RuntimeError(
                f"Legacy FinancialDatasets layout detected at {path}. "
                "Run scripts/migrate_financialdatasets_raw_layout.py before ingesting new data."
            )

    def _write_snapshot_single(
        self,
        vendor_root: Path,
        domain: str,
        policy: DomainPolicy,
        records: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    ) -> MutableMapping[str, object]:
        df = self._records_to_frame(records)
        df = self._ensure_ticker(df)
        df = self._normalize_date_columns(df, policy)
        path = vendor_root / domain / "Facts.csv"
        return self._upsert_single_csv(path, df, policy)

    def _write_snapshot_table(
        self,
        vendor_root: Path,
        domain: str,
        policy: DomainPolicy,
        records: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    ) -> MutableMapping[str, object]:
        df = self._records_to_frame(records)
        df = self._ensure_ticker(df)
        df = self._normalize_date_columns(df, policy)
        df = self._filter_missing_date(df, policy, domain)
        if df.empty:
            return {"files": [], "total_files": 0}
        file_details = []
        for ticker, group in df.groupby("ticker"):
            path = vendor_root / domain / f"{ticker}.csv"
            result = self._append_csv(path, group, policy, domain)
            file_details.append(result)
        return {"files": file_details, "total_files": len(file_details)}

    def _write_timeseries_unsharded(
        self,
        vendor_root: Path,
        domain: str,
        policy: DomainPolicy,
        records: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    ) -> MutableMapping[str, object]:
        df = self._records_to_frame(records)
        df = self._ensure_ticker(df)
        df = self._normalize_date_columns(df, policy)
        df = self._filter_missing_date(df, policy, domain)
        if df.empty:
            return {"files": [], "total_files": 0}
        file_details = []
        for ticker, group in df.groupby("ticker"):
            path = vendor_root / domain / f"{ticker}.csv"
            result = self._append_csv(path, group, policy, domain)
            file_details.append(result)
        return {"files": file_details, "total_files": len(file_details)}

    def _write_timeseries_yearly(
        self,
        vendor_root: Path,
        domain: str,
        policy: DomainPolicy,
        records: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    ) -> MutableMapping[str, object]:
        df = self._records_to_frame(records)
        df = self._ensure_ticker(df)
        df = self._normalize_date_columns(df, policy)
        df = self._filter_missing_date(df, policy, domain)
        if df.empty:
            return {"files": [], "total_files": 0}
        df["__year"] = df["__date_key"].str.slice(0, 4)
        file_details = []
        for (ticker, year), group in df.groupby(["ticker", "__year"]):
            path = vendor_root / domain / ticker / f"{year}.csv"
            result = self._append_csv(path, group, policy, domain)
            file_details.append(result)
        return {"files": file_details, "total_files": len(file_details)}

    def _records_to_frame(self, records: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]]) -> "pd.DataFrame":
        return pd.DataFrame([dict(record) for record in records])

    def _ensure_ticker(self, df: "pd.DataFrame") -> "pd.DataFrame":
        if "ticker" not in df.columns:
            if "symbol" in df.columns:
                df["ticker"] = df["symbol"].astype(str)
            else:
                raise ValueError("FinancialDatasets records must include ticker or symbol")
        df["ticker"] = df["ticker"].astype(str)
        return df

    def _normalize_date_columns(self, df: "pd.DataFrame", policy: DomainPolicy) -> "pd.DataFrame":
        for column in policy.date_priority:
            if column not in df.columns:
                continue
            if policy.date_kind == "datetime":
                df[column] = _format_datetime_series(df[column])
            else:
                df[column] = _format_date_series(df[column])
        return df

    def _filter_missing_date(self, df: "pd.DataFrame", policy: DomainPolicy, domain: str) -> "pd.DataFrame":
        if not policy.date_priority:
            return df
        date_key = pd.Series([None] * len(df), index=df.index, dtype="object")
        for column in policy.date_priority:
            if column not in df.columns:
                continue
            date_key = date_key.fillna(df[column].where(df[column].astype(str).str.len() > 0))
        df["__date_key"] = date_key
        missing = df["__date_key"].isna() | (df["__date_key"].astype(str).str.len() == 0)
        if missing.any():
            LOGGER.warning(
                "FinancialDatasets %s rows missing date key (%s); dropping %d rows",
                domain,
                ", ".join(policy.date_priority),
                int(missing.sum()),
            )
        return df.loc[~missing].copy()

    def _append_csv(
        self,
        path: Path,
        new_rows: "pd.DataFrame",
        policy: DomainPolicy,
        domain: str,
        *,
        dedupe: bool = True,
    ) -> Dict[str, object]:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(".lock")
        _with_lock(lock_path)
        try:
            existing = _read_csv(path)
            expected = policy.expected_columns or tuple(existing.columns)
            if expected:
                new_cols = set(new_rows.columns) - set(expected)
                if new_cols:
                    LOGGER.warning(
                        "FinancialDatasets %s encountered new columns: %s",
                        domain,
                        ", ".join(sorted(new_cols)),
                    )
            combined = _union_frames(existing, new_rows)
            if dedupe:
                combined = self._dedup_frame(combined, policy, domain)
            combined = _order_columns(combined, expected)
            _write_csv(path, combined)
            file_hash = _compute_file_hash(path)
        finally:
            _release_lock(lock_path)
        return {"path": str(path), "hash": file_hash, "records": int(len(combined))}

    def _upsert_single_csv(
        self,
        path: Path,
        new_rows: "pd.DataFrame",
        policy: DomainPolicy,
    ) -> Dict[str, object]:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(".lock")
        _with_lock(lock_path)
        try:
            existing = _read_csv(path)
            expected = policy.expected_columns or tuple(existing.columns)
            if expected:
                new_cols = set(new_rows.columns) - set(expected)
                if new_cols:
                    LOGGER.warning(
                        "FinancialDatasets company_facts encountered new columns: %s",
                        ", ".join(sorted(new_cols)),
                    )
            combined = _union_frames(existing, new_rows)
            combined = combined.drop_duplicates(subset=["ticker"], keep="last")
            combined = combined.sort_values("ticker", kind="mergesort")
            combined = _order_columns(combined, expected)
            _write_csv(path, combined)
            file_hash = _compute_file_hash(path)
        finally:
            _release_lock(lock_path)
        return {"path": str(path), "hash": file_hash, "records": int(len(combined))}

    def _dedup_frame(self, df: "pd.DataFrame", policy: DomainPolicy, domain: str) -> "pd.DataFrame":
        if domain == "news":
            title = df.get("title")
            url = df.get("url")
            if title is None:
                title = pd.Series([""] * len(df))
            if url is None:
                url = pd.Series([""] * len(df))
            title = title.fillna("").astype(str)
            url = url.fillna("").astype(str)
            df["__title_or_url"] = title.where(title.str.len() > 0, url)
            subset = ["ticker", "__date_key", "__title_or_url"]
        elif domain == "insider_trades":
            for col in (
                "filing_date",
                "name",
                "transaction_date",
                "transaction_value",
                "transaction_shares",
                "security_title",
            ):
                if col not in df.columns:
                    df[col] = ""
            subset = ["ticker", "filing_date", "name", "transaction_date", "transaction_value", "transaction_shares", "security_title"]
        else:
            subset = ["ticker"]
            if policy.date_priority:
                subset.append("__date_key")
            for key in policy.dedup_keys:
                if key in {"ticker", "__date_key"}:
                    continue
                if key not in df.columns:
                    df[key] = ""
                subset.append(key)
        df = df.drop_duplicates(subset=subset, keep="last")
        if "__date_key" in df.columns:
            df = df.sort_values(["ticker", "__date_key"], kind="mergesort")
        return df


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


def _read_csv(path: Path) -> "pd.DataFrame":
    if pd is None or not path.exists():
        return pd.DataFrame() if pd is not None else None  # type: ignore[return-value]
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _write_csv(path: Path, frame: "pd.DataFrame") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def _union_frames(existing: "pd.DataFrame", incoming: "pd.DataFrame") -> "pd.DataFrame":
    if existing is None or existing.empty:
        return incoming.copy()
    if incoming is None or incoming.empty:
        return existing.copy()
    all_cols = sorted(set(existing.columns).union(incoming.columns))
    left = existing.reindex(columns=all_cols)
    right = incoming.reindex(columns=all_cols)
    return pd.concat([left, right], ignore_index=True)


def _order_columns(frame: "pd.DataFrame", expected: Sequence[str] | tuple[str, ...]) -> "pd.DataFrame":
    if not expected:
        return frame
    ordered = [col for col in expected if col in frame.columns]
    remainder = [col for col in frame.columns if col not in ordered]
    return frame[ordered + remainder]


def _format_date_series(series: "pd.Series") -> "pd.Series":
    data = pd.to_datetime(series, utc=True, errors="coerce").dt.date
    return data.map(lambda value: value.isoformat() if pd.notna(value) else "")


def _format_datetime_series(series: "pd.Series") -> "pd.Series":
    data = pd.to_datetime(series, utc=True, errors="coerce")
    return data.map(lambda value: value.isoformat() if pd.notna(value) else "")


def _compute_file_hash(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


_LOCK_HANDLES: dict[Path, object] = {}


def _with_lock(path: Path) -> None:
    try:
        import fcntl  # type: ignore
    except Exception as exc:  # pragma: no cover - Windows fallback
        raise RuntimeError(
            "company_facts writes require Unix file locking (fcntl not available). "
            "Run single-threaded ingestion or use a Unix environment."
        ) from exc
    handle = path.open("w")
    fcntl.flock(handle, fcntl.LOCK_EX)
    _LOCK_HANDLES[path] = handle


def _release_lock(path: Path) -> None:
    handle = _LOCK_HANDLES.pop(path, None)
    if handle is None:
        return
    try:
        import fcntl  # type: ignore
    except Exception:  # pragma: no cover - platform guard
        handle.close()
        return
    try:
        fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        handle.close()


def _env_shard_flag() -> bool | None:
    value = os.environ.get("QUANTO_RAW_YEARLY_DAILY")
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"", "0", "false", "no"}:
        return False
    return True


__all__ = [
    "RawEquityOHLCVWriter",
    "RawFundamentalsWriter",
    "RawOptionsWriter",
    "RawFinancialDatasetsWriter",
]
