"""iVolatility vendor adapter for equity OHLCV ingestion."""

from __future__ import annotations

import csv
import io
import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence

from infra.validation import validate_records

from ..ivolatility_client import IvolatilityClient
from .polygon_equity import EquityIngestionRequest

UTC = timezone.utc
LOGGER = logging.getLogger(__name__)


class IvolatilityEquityAdapter:
    """Adapter that maps iVolatility equity payloads into canonical records."""

    BULK_ENDPOINT = "equities/stock-market-data"
    STOCK_PRICES_ENDPOINT = "equities/eod/stock-prices"

    def __init__(
        self,
        client: IvolatilityClient,
        *,
        vendor: str = "ivolatility",
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.client = client
        self.vendor = vendor
        cfg = dict(config or {})
        self.stock_group = cfg.get("stock_group")
        self.stock_ids = tuple(str(item).strip() for item in cfg.get("stock_ids", []) if str(item).strip())
        self.bulk_columns = tuple(str(item).strip() for item in cfg.get("columns", []) if str(item).strip())
        self.bulk_symbols_override = tuple(
            str(item).strip() for item in cfg.get("symbols", []) if str(item).strip()
        )
        self.use_bulk = bool(cfg.get("use_bulk", True))
        self.bulk_endpoint = cfg.get("bulk_endpoint", self.BULK_ENDPOINT)
        self.per_symbol_endpoint = cfg.get("stock_prices_endpoint", self.STOCK_PRICES_ENDPOINT)
        self.flat_file_resolver = self._unsupported_flat_file_resolver
        self.ingestion_stats: dict[str, Any] = {}

    def fetch_raw(
        self,
        request: EquityIngestionRequest,
        *,
        endpoint: str | None = None,
    ) -> List[Mapping[str, Any]]:
        if self.use_bulk:
            return self._fetch_bulk_records(request, endpoint or self.bulk_endpoint)
        return self._fetch_per_symbol_records(request, endpoint or self.per_symbol_endpoint)

    def normalize_records(self, raw_records: Iterable[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
        records = list(raw_records)
        normalized: List[Mapping[str, Any]] = []
        seen_keys: set[tuple[str, datetime]] = set()
        dropped = 0
        for raw in records:
            try:
                record = self._normalize_record(raw)
            except ValueError as exc:
                dropped += 1
                LOGGER.warning("Skipping invalid iVolatility equity record: %s", exc)
                continue
            key = (record["symbol"], record["timestamp"])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            normalized.append(record)
        self.ingestion_stats = {
            "raw_records": len(records),
            "dropped_records": dropped,
        }
        if dropped:
            LOGGER.info("Dropped %d invalid iVolatility equity records", dropped)
        normalized.sort(key=lambda rec: (rec["symbol"], rec["timestamp"]))
        return normalized

    def validate(
        self,
        records: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]],
        *,
        run_id: str,
        config: Mapping[str, Any] | None = None,
    ):
        return validate_records(
            "equity_ohlcv",
            records,
            source_vendor=self.vendor,
            run_id=run_id,
            config=config,
        )

    async def fetch_equity_ohlcv_rest(self, request: EquityIngestionRequest) -> List[Mapping[str, Any]]:
        raw_records = self.fetch_raw(request)
        return self.normalize_records(raw_records)

    def stream_flat_file_equity_bars(self, request: EquityIngestionRequest) -> Iterable[Mapping[str, Any]]:
        raise RuntimeError("Flat-file ingestion is not supported for iVolatility equity data")

    def _fetch_bulk_records(
        self,
        request: EquityIngestionRequest,
        endpoint: str,
    ) -> List[Mapping[str, Any]]:
        params = {
            **IvolatilityClient.map_date_params(start_date=request.start_date, end_date=request.end_date),
        }
        if self.bulk_columns:
            params["columns"] = ",".join(self.bulk_columns)

        identifier_applied = False
        if self.stock_group:
            params["stockGroup"] = self.stock_group
            identifier_applied = True
        elif self.stock_ids:
            params["stockID"] = ",".join(self.stock_ids)
            identifier_applied = True
        else:
            symbols = self.bulk_symbols_override or tuple(request.symbols)
            if symbols:
                params["symbols"] = ",".join(symbols)
                identifier_applied = True

        if not identifier_applied:
            raise ValueError(
                "iVolatility bulk ingestion requires stock_group, stock_ids, or request symbols to be provided."
            )

        payload = self.client.fetch_async_dataset(endpoint, params)
        records = self._coerce_bulk_payload(payload)
        if not records:
            records = self._resolve_remote_payload(payload)
        symbol_filter = {symbol.upper() for symbol in request.symbols}
        filtered: List[Mapping[str, Any]] = []
        for entry in records:
            symbol = self._extract_symbol(entry)
            if symbol_filter and symbol.upper() not in symbol_filter:
                continue
            payload = dict(entry)
            payload.setdefault("symbol", symbol)
            filtered.append(payload)
        return filtered

    def _fetch_per_symbol_records(
        self,
        request: EquityIngestionRequest,
        endpoint: str,
    ) -> List[Mapping[str, Any]]:
        records: List[Mapping[str, Any]] = []
        for single_date in _iter_dates(request.start_date, request.end_date):
            for symbol in sorted(request.symbols):
                params = {
                    "symbol": symbol,
                    "date": single_date.isoformat(),
                }
                rows = self.client.fetch(endpoint, params)
                for entry in rows:
                    payload = dict(entry)
                    payload.setdefault("symbol", symbol)
                    payload.setdefault("date", single_date.isoformat())
                    records.append(payload)
        return records

    def _normalize_record(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        symbol = self._extract_symbol(record)
        timestamp_value = self._first_present(
            record,
            (
                "timestamp",
                "date",
                "as_of",
                "pricingDate",
                "price_date",
                "trade_date",
                "tradeDate",
            ),
        )
        if timestamp_value is None:
            raise ValueError("equity record missing timestamp/date field")
        normalized = {
            "timestamp": _coerce_timestamp(timestamp_value),
            "symbol": symbol,
            "open": self._extract_number(
                record,
                "open",
                ("open", "open_price", "openPrice", "OpenPrice", "Open", "OPEN_PRICE"),
            ),
            "high": self._extract_number(
                record,
                "high",
                ("high", "high_price", "highPrice", "HighPrice", "High", "HIGH"),
            ),
            "low": self._extract_number(
                record,
                "low",
                ("low", "low_price", "lowPrice", "LowPrice", "Low", "LOW"),
            ),
            "close": self._extract_number(
                record,
                "close",
                ("close", "close_price", "closePrice", "ClosePrice", "Close", "CLOSE_PRICE"),
            ),
            "volume": self._extract_number(
                record,
                "volume",
                (
                    "volume",
                    "Volume",
                    "totalVolume",
                    "volumeDaily",
                    "vol",
                    "stockVolume",
                    "STOCK_VOLUME",
                    "StockVolume",
                ),
            ),
            "source_vendor": self.vendor,
        }
        return normalized

    def _coerce_bulk_payload(self, payload: Any) -> List[Mapping[str, Any]]:
        if isinstance(payload, (bytes, bytearray)):
            text = payload.decode("utf-8", errors="ignore")
            return self._parse_csv_text(text)
        if isinstance(payload, str):
            stripped = payload.strip()
            if not stripped:
                return []
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    return self._parse_csv_text(stripped)
                return self._coerce_bulk_payload(parsed)
            return self._parse_csv_text(stripped)
        if isinstance(payload, Mapping):
            data = payload.get("data")
            if isinstance(data, Sequence):
                rows = [self._normalize_record_keys(entry) for entry in data if isinstance(entry, Mapping)]
                if rows and any(row.get("symbol") for row in rows):
                    return rows
            if payload.get("symbol"):
                return [self._normalize_record_keys(payload)]
        if isinstance(payload, Sequence):
            normalized: List[Mapping[str, Any]] = []
            for entry in payload:
                if isinstance(entry, Mapping):
                    normalized.append(self._normalize_record_keys(entry))
            if normalized:
                symbols_present = any(entry.get("symbol") for entry in normalized)
                if symbols_present:
                    return normalized
        return []

    def _parse_csv_text(self, text: str) -> List[Mapping[str, Any]]:
        buffer = io.StringIO(text)
        reader = csv.DictReader(buffer)
        return [self._normalize_record_keys(row) for row in reader]

    def _normalize_record_keys(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in record.items():
            normalized_key = str(key).strip() if key is not None else ""
            if not normalized_key:
                continue
            if isinstance(value, str):
                normalized[normalized_key] = value.strip()
            else:
                normalized[normalized_key] = value
        return normalized

    def _resolve_remote_payload(self, payload: Any) -> List[Mapping[str, Any]]:
        url: str | None = None
        text: str | None = None
        if isinstance(payload, (bytes, bytearray)):
            text = payload.decode("utf-8", errors="ignore")
        elif isinstance(payload, str):
            text = payload
        if text:
            url = self.client._extract_download_url_from_text(text)
        if not url and isinstance(payload, (Mapping, Sequence)):
            url = self.client._extract_download_url(payload)
        if not url:
            return []
        csv_bytes = self.client._download_and_normalize_file(url)
        try:
            csv_text = csv_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return []
        return self._parse_csv_text(csv_text)

    def _extract_symbol(self, record: Mapping[str, Any]) -> str:
        symbol = self._first_present(record, ("symbol", "ticker", "stockSymbol", "security")) or ""
        symbol = str(symbol).strip()
        if not symbol:
            raise ValueError("equity record missing symbol field")
        return symbol

    @staticmethod
    def _first_present(record: Mapping[str, Any], fields: Sequence[str]) -> Any | None:
        for field in fields:
            for candidate in {field, field.lower(), field.upper()}:
                if candidate in record and record[candidate] not in (None, ""):
                    return record[candidate]
        return None

    def _extract_number(self, record: Mapping[str, Any], label: str, candidates: Sequence[str]) -> float:
        for field in candidates:
            for candidate in {field, field.lower(), field.upper()}:
                if candidate in record and record[candidate] not in (None, ""):
                    try:
                        return float(record[candidate])
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"equity field '{candidate}' must be numeric") from exc
        raise ValueError(f"equity record missing required field '{label}'")

    def _unsupported_flat_file_resolver(self, uri: str) -> Path:
        raise RuntimeError(f"Flat-file ingestion is not supported for iVolatility URIs: {uri}")


def _iter_dates(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _coerce_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=UTC)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("timestamp string cannot be empty")
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"Invalid timestamp '{value}'") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    raise ValueError("timestamp field missing or unparseable for equity record")


__all__ = ["IvolatilityEquityAdapter"]
