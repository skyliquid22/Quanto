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
from infra.ingestion.adapters.polygon_equity import RateLimitError


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


def test_insider_trades_supports_filing_date_filters():
    payload = {"insider_trades": [{"ticker": "AAPL", "transaction_date": "2023-12-25"}]}
    client = _FakeRESTClient([payload])
    adapter = FinancialDatasetsAdapter(rest_client=client)

    result = asyncio.run(
        adapter.fetch_insider_trades_rest(
            symbols=("AAPL",),
            limit=50,
            filing_date_filters={
                "filing_date_gte": date(2023, 1, 1),
                "filing_date_lt": "2024-01-01",
            },
        )
    )

    assert result.records
    call = client.calls[0]
    assert call["path"] == "/insider-trades"
    assert call["params"]["ticker"] == "AAPL"
    assert call["params"]["limit"] == "50"
    assert call["params"]["filing_date_gte"] == "2023-01-01"
    assert call["params"]["filing_date_lt"] == "2024-01-01"


class _RateLimitRESTClient:
    """REST client that raises RateLimitError a configurable number of times before succeeding."""

    def __init__(self, fail_count: int, payload: dict, *, retry_after: float | None = None):
        self._fail_count = fail_count
        self._payload = payload
        self._retry_after = retry_after
        self.attempt_count = 0

    async def get(self, path, params=None):
        self.attempt_count += 1
        if self.attempt_count <= self._fail_count:
            raise RateLimitError("rate limited", retry_after=self._retry_after)
        return self._payload


def test_guarded_fetch_retries_on_rate_limit(monkeypatch):
    """_guarded_fetch retries after RateLimitError and eventually succeeds."""

    async def noop_sleep(_):
        pass

    monkeypatch.setattr(asyncio, "sleep", noop_sleep)

    payload = {"company_facts": {"ticker": "AAPL", "name": "Apple"}}
    client = _RateLimitRESTClient(fail_count=2, payload=payload, retry_after=0)
    adapter = FinancialDatasetsAdapter(rest_client=client)

    result = asyncio.run(adapter.fetch_company_facts_rest(symbols=("AAPL",), as_of_date=date(2024, 1, 5)))

    assert len(result.records) == 1
    assert result.records[0]["symbol"] == "AAPL"
    assert client.attempt_count == 3  # 2 failures + 1 success


def test_guarded_fetch_backoff_escalates(monkeypatch):
    """Backoff delay doubles on each retry, capped at backoff_max."""
    delays = []

    async def fake_sleep(d):
        delays.append(d)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    payload = {"company_facts": {"ticker": "AAPL", "name": "Apple"}}
    client = _RateLimitRESTClient(fail_count=4, payload=payload)
    adapter = FinancialDatasetsAdapter(
        rest_client=client,
        rest_config={"concurrency": 1, "backoff_initial": 1.0, "backoff_multiplier": 2.0, "backoff_max": 5.0},
    )

    result = asyncio.run(adapter.fetch_company_facts_rest(symbols=("AAPL",), as_of_date=date(2024, 1, 5)))

    assert len(result.records) == 1
    assert client.attempt_count == 5  # 4 failures + 1 success
    # Backoff sequence: 1.0, 2.0, 4.0, 5.0 (capped at max)
    assert delays == [1.0, 2.0, 4.0, 5.0]
