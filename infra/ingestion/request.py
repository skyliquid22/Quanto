"""Normalized ingestion request container shared across CLI + router layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Mapping, Sequence


def _maybe_date(value: Any | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value).strip()
    if not text:
        return None
    return date.fromisoformat(text)


def _normalize_sequence(value: Any | None) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else tuple()
    if isinstance(value, Sequence):
        normalized = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                normalized.append(text)
        return tuple(dict.fromkeys(normalized))
    raise TypeError("symbols configuration must be a sequence of strings")


def _coerce_mapping(value: Any | None) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("vendor-specific configuration blocks must be mappings")
    return dict(value)


@dataclass(frozen=True)
class IngestionRequest:
    """Normalized configuration shared across all ingestion domains."""

    domain: str
    vendor: str
    mode: str
    symbols: tuple[str, ...] = field(default_factory=tuple)
    start_date: date | None = None
    end_date: date | None = None
    interval: str | None = None
    as_of_date: date | None = None
    flat_file_uris: tuple[str, ...] = field(default_factory=tuple)
    vendor_params: Mapping[str, Any] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, domain: str, payload: Mapping[str, Any]) -> "IngestionRequest":
        if not isinstance(payload, Mapping):
            raise TypeError("ingestion config must be a mapping")
        normalized_domain = str(domain).strip()
        if not normalized_domain:
            raise ValueError("domain must be provided")

        vendor = str(payload.get("vendor", "polygon")).strip().lower()
        if not vendor:
            raise ValueError("vendor must be provided in config")

        mode = str(payload.get("mode", "") or "").strip().lower() or "auto"
        interval = payload.get("interval")
        start_date = _maybe_date(payload.get("start_date"))
        end_date = _maybe_date(payload.get("end_date"))
        as_of_date = _maybe_date(payload.get("as_of_date"))
        flat_file_uris = _normalize_sequence(payload.get("flat_file_uris"))

        if normalized_domain == "option_contract_reference" and as_of_date:
            start_date = start_date or as_of_date
            end_date = end_date or as_of_date

        symbols = _resolve_domain_symbols(normalized_domain, payload)
        vendor_params = _coerce_mapping(payload.get(vendor))
        options = _coerce_mapping(payload.get("options"))

        return cls(
            domain=normalized_domain,
            vendor=vendor,
            mode=mode,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            interval=str(interval).strip().lower() if interval else None,
            as_of_date=as_of_date,
            flat_file_uris=flat_file_uris,
            vendor_params=vendor_params,
            options=options,
        )


def _resolve_domain_symbols(domain: str, payload: Mapping[str, Any]) -> tuple[str, ...]:
    if domain == "option_contract_reference":
        preferred_keys = ("underlying_symbols", "symbols")
    elif domain in {"option_contract_ohlcv", "option_open_interest"}:
        preferred_keys = ("option_symbols", "symbols")
    else:
        preferred_keys = ("symbols",)

    for key in preferred_keys:
        if key in payload:
            return _normalize_sequence(payload.get(key))
    return tuple()


__all__ = ["IngestionRequest"]
