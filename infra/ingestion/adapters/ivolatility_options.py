"""iVolatility adapter for option contract reference, OHLCV, and open interest."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Iterable, List, Mapping, Sequence

from infra.validation import validate_records

from ..ivolatility_client import IvolatilityClient
from .ivolatility_equity import _coerce_timestamp, _require_number
from .polygon_options import OptionReferenceIngestionRequest, OptionTimeseriesIngestionRequest

UTC = timezone.utc


class IvolatilityOptionsAdapter:
    """Adapter coordinating the three options-related canonical domains for iVolatility."""

    CONTRACT_ENDPOINT = "options/contracts"
    OHLCV_ENDPOINT = "options/ohlcv"
    OPEN_INTEREST_ENDPOINT = "options/open-interest"

    def __init__(
        self,
        client: IvolatilityClient,
        *,
        vendor: str = "ivolatility",
        supports_open_interest: bool = True,
    ) -> None:
        self.client = client
        self.vendor = vendor
        self.supports_open_interest = supports_open_interest

    def fetch_contract_reference(
        self,
        request: OptionReferenceIngestionRequest,
        *,
        endpoint: str | None = None,
    ) -> List[Mapping[str, Any]]:
        if not isinstance(request, OptionReferenceIngestionRequest):
            raise TypeError("request must be an OptionReferenceIngestionRequest")
        records: List[Mapping[str, Any]] = []
        target_endpoint = endpoint or self.CONTRACT_ENDPOINT
        for symbol in sorted(request.underlying_symbols):
            params = {
                "underlying_symbol": symbol,
                "as_of": request.as_of_date.isoformat(),
            }
            for entry in self.client.fetch(target_endpoint, params):
                payload = dict(entry)
                payload.setdefault("underlying_symbol", symbol)
                payload.setdefault("as_of", request.as_of_date.isoformat())
                records.append(payload)
        return records

    def fetch_option_ohlcv(
        self,
        request: OptionTimeseriesIngestionRequest,
        *,
        endpoint: str | None = None,
    ) -> List[Mapping[str, Any]]:
        if request.domain != "option_contract_ohlcv":
            raise ValueError("request domain must be option_contract_ohlcv")
        records: List[Mapping[str, Any]] = []
        target_endpoint = endpoint or self.OHLCV_ENDPOINT
        for option_symbol in sorted(request.option_symbols):
            params = {
                "option_symbol": option_symbol,
                "start_date": request.start_date.isoformat(),
                "end_date": request.end_date.isoformat(),
                "interval": "daily",
            }
            for entry in self.client.fetch(target_endpoint, params):
                payload = dict(entry)
                payload.setdefault("option_symbol", option_symbol)
                records.append(payload)
        return records

    def fetch_option_open_interest(
        self,
        request: OptionTimeseriesIngestionRequest,
        *,
        endpoint: str | None = None,
    ) -> List[Mapping[str, Any]]:
        if not self.supports_open_interest:
            raise RuntimeError(
                "iVolatility subscription does not include open interest. "
                "Upgrade to Backtest API Plus or disable option_open_interest ingestion."
            )
        if request.domain != "option_open_interest":
            raise ValueError("request domain must be option_open_interest")

        records: List[Mapping[str, Any]] = []
        target_endpoint = endpoint or self.OPEN_INTEREST_ENDPOINT
        for option_symbol in sorted(request.option_symbols):
            params = {
                "option_symbol": option_symbol,
                "start_date": request.start_date.isoformat(),
                "end_date": request.end_date.isoformat(),
                "interval": "daily",
            }
            for entry in self.client.fetch(target_endpoint, params):
                payload = dict(entry)
                payload.setdefault("option_symbol", option_symbol)
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

    def normalize_option_ohlcv(
        self,
        raw_records: Iterable[Mapping[str, Any]],
    ) -> List[Mapping[str, Any]]:
        normalized: List[Mapping[str, Any]] = []
        for raw in raw_records:
            normalized.append(self._normalize_ohlcv_record(raw))
        normalized.sort(key=lambda rec: (rec["option_symbol"], rec["timestamp"]))
        return normalized

    def normalize_option_open_interest(
        self,
        raw_records: Iterable[Mapping[str, Any]],
    ) -> List[Mapping[str, Any]]:
        if not self.supports_open_interest:
            raise RuntimeError(
                "Open interest ingestion is disabled for this adapter. "
                "Set supports_open_interest=True to enable normalization."
            )
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

    def _normalize_reference_record(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        option_symbol = _resolve_option_symbol(record)
        underlying = record.get("underlying_symbol") or record.get("underlying")
        if not underlying:
            raise ValueError("option contract record missing underlying symbol")
        expiration_value = record.get("expiration_date") or record.get("expiration")
        expiration = _coerce_date(expiration_value, "expiration_date")
        option_type = _normalize_option_type(record.get("option_type") or record.get("right"))
        strike = _require_number(record, "strike")
        multiplier = record.get("multiplier", 100.0)
        try:
            multiplier_value = float(multiplier)
        except (TypeError, ValueError) as exc:
            raise ValueError("option contract multiplier must be numeric") from exc

        normalized = {
            "option_symbol": option_symbol,
            "underlying_symbol": str(underlying),
            "expiration_date": expiration,
            "strike": strike,
            "option_type": option_type,
            "multiplier": multiplier_value,
            "source_vendor": self.vendor,
        }
        return normalized

    def _normalize_ohlcv_record(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        option_symbol = _resolve_option_symbol(record)
        normalized = {
            "timestamp": _coerce_timestamp(
                record.get("timestamp")
                or record.get("price_date")
                or record.get("as_of")
                or record.get("date")
            ),
            "option_symbol": option_symbol,
            "open": _require_number(record, "open"),
            "high": _require_number(record, "high"),
            "low": _require_number(record, "low"),
            "close": _require_number(record, "close"),
            "volume": _require_number(record, "volume"),
            "source_vendor": self.vendor,
        }
        return normalized

    def _normalize_open_interest_record(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        option_symbol = _resolve_option_symbol(record)
        timestamp = _coerce_timestamp(
            record.get("timestamp")
            or record.get("as_of")
            or record.get("date")
            or record.get("price_date")
        )
        open_interest = _require_number(record, "open_interest")
        if open_interest < 0:
            raise ValueError("open_interest must be non-negative")
        return {
            "timestamp": timestamp,
            "option_symbol": option_symbol,
            "open_interest": open_interest,
            "source_vendor": self.vendor,
        }


def _resolve_option_symbol(record: Mapping[str, Any]) -> str:
    for key in ("option_symbol", "optionSymbol", "symbol", "contract_symbol", "contract"):
        value = record.get(key)
        if value:
            return str(value)

    underlying = record.get("underlying_symbol") or record.get("underlying")
    if not underlying:
        raise ValueError("option record missing both option_symbol and underlying_symbol")
    expiration = _coerce_date(
        record.get("expiration_date") or record.get("expiration") or record.get("expiry"),
        "expiration_date",
    )
    option_type = _normalize_option_type(record.get("option_type") or record.get("right"))
    strike = _require_number(record, "strike")
    strike_text = _format_strike(strike)
    constructed = f"{underlying}-{expiration.strftime('%Y%m%d')}-{option_type.upper()}-{strike_text}"
    return constructed


def _coerce_date(value: Any, field_name: str) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.astimezone(UTC).date()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{field_name} string cannot be empty")
        if "T" in text:
            text = text.split("T", 1)[0]
        try:
            return date.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be ISO formatted YYYY-MM-DD") from exc
    raise ValueError(f"{field_name} must be a date-like value")


def _normalize_option_type(value: Any) -> str:
    if not value:
        raise ValueError("option_type is required")
    text = str(value).strip().lower()
    if text not in {"call", "put"}:
        raise ValueError("option_type must be either call or put")
    return text


def _format_strike(value: float) -> str:
    text = f"{value:.4f}"
    return text.rstrip("0").rstrip(".")


__all__ = ["IvolatilityOptionsAdapter"]
