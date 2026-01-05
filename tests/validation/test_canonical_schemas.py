import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import guard
    sys.path.insert(0, str(PROJECT_ROOT))

from infra.validation.canonical_schemas import (
    CANONICAL_SCHEMAS,
    DATA_SPEC_VERSION,
    get_canonical_schema,
)


def test_schema_registry_contains_expected_domains():
    expected_domains = {
        "equity_ohlcv",
        "option_contract_reference",
        "option_contract_ohlcv",
        "option_open_interest",
        "fundamentals",
    }
    assert set(CANONICAL_SCHEMAS.keys()) == expected_domains
    for domain in expected_domains:
        schema = get_canonical_schema(domain)
        assert schema.schema_version == DATA_SPEC_VERSION


def test_equity_schema_fields_and_keys_align_with_spec():
    schema = get_canonical_schema("equity_ohlcv")
    assert tuple(schema.field_specs.keys()) == (
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source_vendor",
    )
    assert schema.uniqueness_key == ("symbol", "timestamp", "source_vendor")
    for price_field in ("open", "high", "low", "close"):
        assert schema.field_specs[price_field].numeric_constraint == "positive"
    assert schema.field_specs["volume"].numeric_constraint == "non_negative"


def test_fundamentals_schema_enforces_lineage_key():
    schema = get_canonical_schema("fundamentals")
    assert schema.uniqueness_key == (
        "symbol",
        "report_date",
        "fiscal_period",
        "source_vendor",
    )
    assert all(
        schema.field_specs[field].numeric_constraint == "finite"
        for field in (
            "revenue",
            "net_income",
            "eps",
            "total_assets",
            "total_liabilities",
            "shareholder_equity",
            "operating_income",
            "free_cash_flow",
            "shares_outstanding",
        )
    )
