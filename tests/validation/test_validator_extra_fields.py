from __future__ import annotations

from infra.validation import validate_records


def test_validator_allows_extra_fields_when_configured():
    record = {
        "symbol": "AAPL",
        "report_date": "2023-01-01",
        "fiscal_period": "FY23",
        "revenue": 100.0,
        "net_income": 10.0,
        "eps": 1.0,
        "total_assets": 50.0,
        "total_liabilities": 20.0,
        "shareholder_equity": 30.0,
        "operating_income": 12.0,
        "free_cash_flow": 8.0,
        "shares_outstanding": 1000.0,
        "source_vendor": "financialdatasets",
        "extra_field": "keep_me",
    }

    validated, _manifest = validate_records(
        "fundamentals",
        [record],
        source_vendor="financialdatasets",
        run_id="extra",
        config={"allow_extra_fields": True},
    )

    assert validated[0]["extra_field"] == "keep_me"
