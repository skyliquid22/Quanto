"""iVolatility vendor adapter for equity OHLCV ingestion."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Iterable, List, Mapping, Sequence

from infra.validation import validate_records

from ..ivolatility_client import IvolatilityClient
from .polygon_equity import EquityIngestionRequest

UTC = timezone.utc


class IvolatilityEquityAdapter:
    """Adapter that maps iVolatility equity payloads into canonical records."""

    EQUITY_ENDPOINT = "equities/ohlcv"

    def __init__(self, client: IvolatilityClient, *, vendor: str = "ivolatility") -> None:
        self.client = client
        self.vendor = vendor

    def fetch_raw(
        self,
        request: EquityIngestionRequest,
        *,
        endpoint: str | None = None,
    ) -> List[Mapping[str, Any]]:
        """Collect raw payloads across all requested symbols."""

        if not isinstance(request, EquityIngestionRequest):
            raise TypeError("request must be an EquityIngestionRequest instance")
        records: List[Mapping[str, Any]] = []
        target_endpoint = endpoint or self.EQUITY_ENDPOINT
        for symbol in sorted(request.symbols):
            params = {
                "symbol": symbol,
                "start_date": request.start_date.isoformat(),
                "end_date": request.end_date.isoformat(),
                "interval": request.frequency or "daily",
            }
            for entry in self.client.fetch(target_endpoint, params):
                payload = dict(entry)
                payload.setdefault("symbol", symbol)
                records.append(payload)
        return records

    def normalize_records(self, raw_records: Iterable[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
        """Convert raw vendor payloads into canonical equity_ohlcv records."""

        normalized: List[Mapping[str, Any]] = []
        for raw in raw_records:
            normalized.append(self._normalize_record(raw))
        normalized.sort(key=lambda rec: (rec["symbol"], rec["timestamp"]))
        return normalized

    def validate(
        self,
        records: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]],
        *,
        run_id: str,
        config: Mapping[str, Any] | None = None,
    ):
        """Validate records against the canonical schema using the shared validator."""

        return validate_records(
            "equity_ohlcv",
            records,
            source_vendor=self.vendor,
            run_id=run_id,
            config=config,
        )

    def _normalize_record(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        symbol = (
            record.get("symbol")
            or record.get("ticker")
            or record.get("asset")
            or record.get("security")
        )
        if not symbol:
            raise ValueError("equity record missing symbol field")
        timestamp = _coerce_timestamp(
            record.get("timestamp")
            or record.get("price_date")
            or record.get("trade_date")
            or record.get("date")
        )
        normalized = {
            "timestamp": timestamp,
            "symbol": str(symbol),
            "open": _require_number(record, "open"),
            "high": _require_number(record, "high"),
            "low": _require_number(record, "low"),
            "close": _require_number(record, "close"),
            "volume": _require_number(record, "volume"),
            "source_vendor": self.vendor,
        }
        return normalized


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
        except ValueError as exc:  # pragma: no cover - defensive path
            raise ValueError(f"Invalid timestamp '{value}'") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    raise ValueError("timestamp field missing or unparseable for equity record")


def _require_number(record: Mapping[str, Any], field: str) -> float:
    try:
        value = record[field]
    except KeyError as exc:
        raise ValueError(f"equity record missing required field '{field}'") from exc
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"equity field '{field}' must be numeric") from exc


__all__ = ["IvolatilityEquityAdapter"]
