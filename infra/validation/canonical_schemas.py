"""Canonical schema definitions for ingestion validation.

The schemas implemented here mirror DATA_SPEC.md §4 and are versioned so
ingestion jobs can reason about compatibility guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import numbers
from typing import Dict, Mapping, Tuple


DATA_SPEC_VERSION = "1.2"


@dataclass(frozen=True)
class FieldSpec:
    """Definition for a single canonical field."""

    dtype: Tuple[type, ...]
    required: bool = True
    is_timestamp: bool = False
    is_date: bool = False
    numeric_constraint: str | None = None  # positive, non_negative, finite


@dataclass(frozen=True)
class CanonicalSchema:
    """Container for all schema metadata for a domain."""

    domain: str
    schema_version: str
    field_specs: Mapping[str, FieldSpec]
    uniqueness_key: Tuple[str, ...]

    @property
    def required_fields(self) -> Tuple[str, ...]:
        return tuple(name for name, spec in self.field_specs.items() if spec.required)


def _schema(domain: str, fields: Mapping[str, FieldSpec], key: Tuple[str, ...]) -> CanonicalSchema:
    return CanonicalSchema(
        domain=domain,
        schema_version=DATA_SPEC_VERSION,
        field_specs=dict(fields),
        uniqueness_key=key,
    )


CANONICAL_SCHEMAS: Dict[str, CanonicalSchema] = {
    "equity_ohlcv": _schema(
        "equity_ohlcv",
        {
            "timestamp": FieldSpec((datetime,), is_timestamp=True),
            "symbol": FieldSpec((str,)),
            "open": FieldSpec((numbers.Real,), numeric_constraint="positive"),
            "high": FieldSpec((numbers.Real,), numeric_constraint="positive"),
            "low": FieldSpec((numbers.Real,), numeric_constraint="positive"),
            "close": FieldSpec((numbers.Real,), numeric_constraint="positive"),
            "volume": FieldSpec((numbers.Real,), numeric_constraint="non_negative"),
            "source_vendor": FieldSpec((str,)),
        },
        ("symbol", "timestamp", "source_vendor"),
    ),
    "option_contract_reference": _schema(
        "option_contract_reference",
        {
            "option_symbol": FieldSpec((str,)),
            "underlying_symbol": FieldSpec((str,)),
            "expiration_date": FieldSpec((date,), is_date=True),
            "strike": FieldSpec((numbers.Real,), numeric_constraint="positive"),
            "option_type": FieldSpec((str,)),
            "multiplier": FieldSpec((numbers.Real,), numeric_constraint="positive"),
            "source_vendor": FieldSpec((str,)),
        },
        ("option_symbol", "source_vendor"),
    ),
    "option_contract_ohlcv": _schema(
        "option_contract_ohlcv",
        {
            "timestamp": FieldSpec((datetime,), is_timestamp=True),
            "option_symbol": FieldSpec((str,)),
            "open": FieldSpec((numbers.Real,), numeric_constraint="positive"),
            "high": FieldSpec((numbers.Real,), numeric_constraint="positive"),
            "low": FieldSpec((numbers.Real,), numeric_constraint="positive"),
            "close": FieldSpec((numbers.Real,), numeric_constraint="positive"),
            "volume": FieldSpec((numbers.Real,), numeric_constraint="non_negative"),
            "source_vendor": FieldSpec((str,)),
        },
        ("option_symbol", "timestamp", "source_vendor"),
    ),
    "option_open_interest": _schema(
        "option_open_interest",
        {
            "timestamp": FieldSpec((datetime,), is_timestamp=True),
            "option_symbol": FieldSpec((str,)),
            "open_interest": FieldSpec((numbers.Real,), numeric_constraint="non_negative"),
            "source_vendor": FieldSpec((str,)),
        },
        ("option_symbol", "timestamp", "source_vendor"),
    ),
    "fundamentals": _schema(
        "fundamentals",
        {
            "symbol": FieldSpec((str,)),
            "report_date": FieldSpec((date,), is_date=True),
            "fiscal_period": FieldSpec((str,)),
            "revenue": FieldSpec((numbers.Real,), numeric_constraint="finite"),
            "net_income": FieldSpec((numbers.Real,), numeric_constraint="finite"),
            "eps": FieldSpec((numbers.Real,), numeric_constraint="finite"),
            "total_assets": FieldSpec((numbers.Real,), numeric_constraint="finite"),
            "total_liabilities": FieldSpec((numbers.Real,), numeric_constraint="finite"),
            "shareholder_equity": FieldSpec((numbers.Real,), numeric_constraint="finite"),
            "operating_income": FieldSpec((numbers.Real,), numeric_constraint="finite"),
            "free_cash_flow": FieldSpec((numbers.Real,), numeric_constraint="finite"),
            "shares_outstanding": FieldSpec((numbers.Real,), numeric_constraint="finite"),
            "source_vendor": FieldSpec((str,)),
        },
        ("symbol", "report_date", "fiscal_period", "source_vendor"),
    ),
}


def get_canonical_schema(domain: str) -> CanonicalSchema:
    """Return the canonical schema for a domain or raise KeyError."""

    try:
        return CANONICAL_SCHEMAS[domain]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise KeyError(f"Unsupported domain '{domain}'") from exc


__all__ = ["CanonicalSchema", "FieldSpec", "CANONICAL_SCHEMAS", "get_canonical_schema", "DATA_SPEC_VERSION"]
