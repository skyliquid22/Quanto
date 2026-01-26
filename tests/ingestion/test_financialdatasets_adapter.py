from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import guard
    sys.path.insert(0, str(PROJECT_ROOT))

from infra.ingestion.adapters import FinancialDatasetsAdapter


class _FakeRESTClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    async def get(self, path, params=None):
        self.calls.append({"path": path, "params": dict(params or {})})
        if not self.payloads:
            return {}
        return self.payloads.pop(0)


def _financials_payload(*, missing_balance: bool = False) -> dict:
    balance_sheet = {
        "report_period": "2023-12-31",
        "period": "annual",
        "fiscal_period": "FY23",
        "currency": "USD",
        "total_assets": 50.0,
        "total_liabilities": 20.0,
        "shareholders_equity": 30.0,
        "outstanding_shares": 1000.0,
    }
    if missing_balance:
        balance_sheet.pop("total_assets")
    return {
        "financials": {
            "income_statements": [
                {
                    "report_period": "2023-12-31",
                    "period": "annual",
                    "fiscal_period": "FY23",
                    "currency": "USD",
                    "revenue": 100.0,
                    "net_income": 10.0,
                    "operating_income": 12.0,
                    "earnings_per_share": 1.0,
                }
            ],
            "balance_sheets": [balance_sheet],
            "cash_flow_statements": [
                {
                    "report_period": "2023-12-31",
                    "period": "annual",
                    "fiscal_period": "FY23",
                    "currency": "USD",
                    "free_cash_flow": 8.0,
                }
            ],
        }
    }


def test_financial_statements_merge_and_required_fields():
    client = _FakeRESTClient([_financials_payload()])
    adapter = FinancialDatasetsAdapter(rest_client=client)

    result = asyncio.run(adapter.fetch_financial_statements_rest(symbols=("AAPL",), period="annual", limit=4))

    assert len(result.records) == 1
    record = result.records[0]
    assert record["symbol"] == "AAPL"
    assert record["report_date"] == "2023-12-31"
    assert record["revenue"] == 100.0
    assert record["total_assets"] == 50.0
    assert record["free_cash_flow"] == 8.0
    assert record["income_revenue"] == 100.0
    assert record["balance_total_assets"] == 50.0
    assert record["cashflow_free_cash_flow"] == 8.0


def test_financial_statements_missing_required_fields_raises():
    client = _FakeRESTClient([_financials_payload(missing_balance=True)])
    adapter = FinancialDatasetsAdapter(rest_client=client)

    with pytest.raises(ValueError):
        asyncio.run(adapter.fetch_financial_statements_rest(symbols=("AAPL",), period="annual", limit=4))


def test_company_facts_includes_as_of_date():
    payload = {"company_facts": {"ticker": "AAPL", "name": "Apple"}}
    client = _FakeRESTClient([payload])
    adapter = FinancialDatasetsAdapter(rest_client=client)

    result = asyncio.run(adapter.fetch_company_facts_rest(symbols=("AAPL",), as_of_date=date(2024, 1, 5)))

    record = result.records[0]
    assert record["symbol"] == "AAPL"
    assert record["as_of_date"] == "2024-01-05"
