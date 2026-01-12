"""iVolatility vendor adapter for equity OHLCV ingestion."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence

from infra.validation import validate_records

from ..ivolatility_client import IvolatilityClient
from .polygon_equity import EquityIngestionRequest

UTC = timezone.utc


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
        normalized: List[Mapping[str, Any]] = []
        seen_keys: set[tuple[str, datetime]] = set()
        for raw in raw_records:
            record = self._normalize_record(raw)
            key = (record["symbol"], record["timestamp"])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            normalized.append(record)
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

        records = self.client.fetch_async_dataset(endpoint, params)
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
                ("open", "open_price", "openPrice", "Open", "OPEN_PRICE"),
            ),
            "high": self._extract_number(
                record,
                "high",
                ("high", "high_price", "highPrice", "High", "HIGH"),
            ),
            "low": self._extract_number(
                record,
                "low",
                ("low", "low_price", "lowPrice", "Low", "LOW"),
            ),
            "close": self._extract_number(
                record,
                "close",
                ("close", "close_price", "closePrice", "Close", "CLOSE_PRICE"),
            ),
            "volume": self._extract_number(
                record,
                "volume",
                ("volume", "Volume", "totalVolume", "volumeDaily", "vol", "stockVolume", "STOCK_VOLUME"),
            ),
            "source_vendor": self.vendor,
        }
        return normalized

    def _extract_symbol(self, record: Mapping[str, Any]) -> str:
        symbol = self._first_present(record, ("symbol", "ticker", "stockSymbol", "security")) or ""
        symbol = str(symbol).strip()
        if not symbol:
            raise ValueError("equity record missing symbol field")
        return symbol

    @staticmethod
    def _first_present(record: Mapping[str, Any], fields: Sequence[str]) -> Any | None:
        for field in fields:
            if field in record and record[field] not in (None, ""):
                return record[field]
        return None

    def _extract_number(self, record: Mapping[str, Any], label: str, candidates: Sequence[str]) -> float:
        for field in candidates:
            if field in record and record[field] not in (None, ""):
                try:
                    return float(record[field])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"equity field '{field}' must be numeric") from exc
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
