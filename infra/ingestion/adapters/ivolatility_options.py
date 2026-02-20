"""iVolatility adapter for option contract reference, OHLCV, and open interest."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence

from infra.validation import validate_records

from ..ivolatility_client import IvolatilityClient
from .ivolatility_equity import _coerce_timestamp  # delegates to infra.timestamps(epoch_unit="s")
from .polygon_options import OptionReferenceIngestionRequest, OptionTimeseriesIngestionRequest

UTC = timezone.utc


class IvolatilityOptionsAdapter:
    """Adapter coordinating the three options-related canonical domains for iVolatility."""

    SERIES_ENDPOINT = "equities/eod/option-series-on-date"
    SINGLE_CONTRACT_ENDPOINT = "equities/eod/single-stock-option"

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
        self.supports_open_interest = cfg.get("supports_open_interest", True)
        self.series_endpoint = cfg.get("series_endpoint", self.SERIES_ENDPOINT)
        self.single_contract_endpoint = cfg.get("single_contract_endpoint", self.SINGLE_CONTRACT_ENDPOINT)
        self.flat_file_resolver = self._unsupported_flat_file_resolver

    def fetch_contract_reference(
        self,
        request: OptionReferenceIngestionRequest,
        *,
        endpoint: str | None = None,
    ) -> List[Mapping[str, Any]]:
        if not isinstance(request, OptionReferenceIngestionRequest):
            raise TypeError("request must be an OptionReferenceIngestionRequest instance")
        target_endpoint = endpoint or self.series_endpoint
        filters = self._series_filters(request.options)
        records: List[Mapping[str, Any]] = []
        for underlying in sorted(request.underlying_symbols):
            params = {
                "symbol": underlying,
                "date": request.as_of_date.isoformat(),
                **filters,
            }
            for entry in self.client.fetch_async_dataset(target_endpoint, params):
                payload = dict(entry)
                payload.setdefault("underlying_symbol", underlying)
                payload.setdefault("as_of", request.as_of_date.isoformat())
                records.append(payload)
        return records

    def normalize_contract_reference(
        self, raw_records: Iterable[Mapping[str, Any]]
    ) -> List[Mapping[str, Any]]:
        normalized: List[Mapping[str, Any]] = []
        for raw in raw_records:
            normalized.append(self._normalize_reference_record(raw))
        normalized.sort(key=lambda rec: (rec["underlying_symbol"], rec["option_symbol"]))
        return normalized

    def fetch_option_ohlcv(
        self,
        request: OptionTimeseriesIngestionRequest,
        *,
        endpoint: str | None = None,
    ) -> List[Mapping[str, Any]]:
        if request.domain != "option_contract_ohlcv":
            raise ValueError("request domain must be option_contract_ohlcv")
        return self._fetch_contract_timeseries(request, endpoint or self.single_contract_endpoint)

    def normalize_option_ohlcv(
        self,
        raw_records: Iterable[Mapping[str, Any]],
    ) -> List[Mapping[str, Any]]:
        normalized: List[Mapping[str, Any]] = []
        for raw in raw_records:
            normalized.append(self._normalize_ohlcv_record(raw))
        normalized.sort(key=lambda rec: (rec["option_symbol"], rec["timestamp"]))
        return normalized

    def fetch_option_open_interest(
        self,
        request: OptionTimeseriesIngestionRequest,
        *,
        endpoint: str | None = None,
    ) -> List[Mapping[str, Any]]:
        if not self.supports_open_interest:
            raise RuntimeError(
                "iVolatility subscription does not include open interest. "
                "Set supports_open_interest=True once the API plan supports it."
            )
        if request.domain != "option_open_interest":
            raise ValueError("request domain must be option_open_interest")
        return self._fetch_contract_timeseries(request, endpoint or self.single_contract_endpoint)

    def normalize_option_open_interest(
        self,
        raw_records: Iterable[Mapping[str, Any]],
    ) -> List[Mapping[str, Any]]:
        if not self.supports_open_interest:
            raise RuntimeError("Open interest ingestion is disabled for this adapter.")
        normalized: List[Mapping[str, Any]] = []
        for raw in raw_records:
            normalized.append(self._normalize_open_interest_record(raw))
        normalized.sort(key=lambda rec: (rec["option_symbol"], rec["timestamp"]))
        return normalized

    def validate(
        self,
        domain: str,
        records: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]],
        *,
        run_id: str,
        config: Mapping[str, Any] | None = None,
    ):
        return validate_records(domain, records, source_vendor=self.vendor, run_id=run_id, config=config)

    def _fetch_contract_timeseries(
        self,
        request: OptionTimeseriesIngestionRequest,
        endpoint: str,
    ) -> List[Mapping[str, Any]]:
        params = IvolatilityClient.map_date_params(
            start_date=request.start_date,
            end_date=request.end_date,
        )
        records: List[Mapping[str, Any]] = []
        for option_symbol in sorted(request.option_symbols):
            symbol_params = {"symbol": option_symbol, **params}
            rows = self.client.fetch_async_dataset(endpoint, symbol_params)
            for entry in rows:
                payload = dict(entry)
                payload.setdefault("option_symbol", option_symbol)
                records.append(payload)
        return records

    def _series_filters(self, options: Mapping[str, Any] | None) -> Mapping[str, str]:
        if not options:
            return {}
        filters: dict[str, str] = {}
        if options.get("exp_from"):
            filters["expFrom"] = _normalize_date_option(options["exp_from"]).isoformat()
        if options.get("exp_to"):
            filters["expTo"] = _normalize_date_option(options["exp_to"]).isoformat()
        if options.get("strike_from") is not None:
            filters["strikeFrom"] = str(options["strike_from"])
        if options.get("strike_to") is not None:
            filters["strikeTo"] = str(options["strike_to"])
        call_put = options.get("call_put")
        if call_put and str(call_put).strip().upper() in {"C", "P"}:
            filters["callPut"] = str(call_put).strip().upper()
        return filters

    def _normalize_reference_record(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        option_symbol = _resolve_option_symbol(record)
        underlying = record.get("underlying_symbol") or record.get("underlying") or ""
        if not underlying:
            raise ValueError("option contract record missing underlying symbol")
        expiration_value = record.get("expiration_date") or record.get("expiration") or record.get("expDate")
        expiration = _coerce_date(expiration_value, "expiration_date")
        option_type = _normalize_option_type(record.get("option_type") or record.get("optionType") or record.get("callPut"))
        strike = _extract_number(record, "strike", ("strike", "strike_price", "strikePrice"))
        multiplier = _extract_number(record, "multiplier", ("multiplier",))

        normalized = {
            "option_symbol": option_symbol,
            "underlying_symbol": str(underlying),
            "expiration_date": expiration,
            "strike": strike,
            "option_type": option_type,
            "multiplier": multiplier,
            "source_vendor": self.vendor,
        }
        return normalized

    def _normalize_ohlcv_record(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        option_symbol = _resolve_option_symbol(record)
        timestamp_value = record.get("timestamp") or record.get("date") or record.get("pricingDate")
        if timestamp_value is None:
            raise ValueError("option OHLCV record missing timestamp/date field")
        normalized = {
            "timestamp": _coerce_timestamp(timestamp_value),
            "option_symbol": option_symbol,
            "open": _extract_number(record, "open", ("open", "openPrice", "o")),
            "high": _extract_number(record, "high", ("high", "highPrice", "h")),
            "low": _extract_number(record, "low", ("low", "lowPrice", "l")),
            "close": _extract_number(record, "close", ("close", "closePrice", "c")),
            "volume": _extract_number(record, "volume", ("volume", "Volume", "vol")),
            "source_vendor": self.vendor,
        }
        return normalized

    def _normalize_open_interest_record(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        option_symbol = _resolve_option_symbol(record)
        timestamp_value = record.get("timestamp") or record.get("date") or record.get("pricingDate")
        if timestamp_value is None:
            raise ValueError("open interest record missing timestamp/date field")
        normalized = {
            "timestamp": _coerce_timestamp(timestamp_value),
            "option_symbol": option_symbol,
            "open_interest": _extract_number(record, "open_interest", ("open_interest", "openInterest", "oi")),
            "source_vendor": self.vendor,
        }
        return normalized

    def _unsupported_flat_file_resolver(self, uri: str) -> Path:
        raise RuntimeError(f"Flat-file ingestion is not supported for iVolatility URIs: {uri}")


def _resolve_option_symbol(record: Mapping[str, Any]) -> str:
    symbol = (
        record.get("option_symbol")
        or record.get("optionSymbol")
        or record.get("symbol")
        or record.get("contract")
    )
    if not symbol:
        raise ValueError("option record missing option symbol")
    return str(symbol).strip()


def _normalize_option_type(value: Any) -> str:
    if value is None:
        raise ValueError("option record missing option type")
    text = str(value).strip().lower()
    if text in {"call", "c"}:
        return "call"
    if text in {"put", "p"}:
        return "put"
    raise ValueError(f"Unknown option type '{value}'")


def _coerce_date(value: Any, field: str) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{field} string cannot be empty")
        return date.fromisoformat(text)
    raise ValueError(f"{field} must be an ISO date string")


def _normalize_date_option(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value.strip())
    raise ValueError("Date filters must be ISO date strings")


def _extract_number(record: Mapping[str, Any], label: str, candidates: Sequence[str]) -> float:
    for field in candidates:
        if field in record and record[field] not in (None, ""):
            try:
                return float(record[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"option field '{field}' must be numeric") from exc
    raise ValueError(f"option record missing required field '{label}'")


__all__ = ["IvolatilityOptionsAdapter"]
