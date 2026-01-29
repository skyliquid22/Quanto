"""Financial Datasets adapter handling fundamentals and related market metadata."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import numbers
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .polygon_equity import RateLimitError


DEFAULT_REST_CONFIG: Dict[str, Any] = {
    "concurrency": 4,
    "timeout": 30.0,
}

_MAX_LIMITS = {
    "financial_statements": 4,
    "financial_metrics": None,
    "financial_metrics_snapshot": None,
    "insider_trades": 1000,
    "institutional_ownership": 1000,
    "news": 100,
}


@dataclass(frozen=True)
class FinancialDatasetsAdapterResult:
    """Container for normalized records and source payload lineage."""

    records: List[Dict[str, Any]]
    source_payloads: List[Dict[str, Any]]
    filings: List[Dict[str, Any]] = field(default_factory=list)


class FinancialDatasetsRESTClient:
    """Minimal httpx-backed REST client for Financial Datasets endpoints."""

    BASE_URL = "https://api.financialdatasets.ai"

    def __init__(self, api_key: str, *, timeout: float = 30.0) -> None:
        try:
            import httpx  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError("httpx must be installed to use FinancialDatasetsRESTClient") from exc
        self._httpx = httpx
        self._client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)
        self.api_key = api_key

    async def get(self, path: str, params: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        headers = {"X-API-KEY": self.api_key}
        response = await self._client.get(f"{self.BASE_URL}{path}", params=params, headers=headers)
        if response.status_code == 429:
            retry = float(response.headers.get("Retry-After", "1"))
            raise RateLimitError("Financial Datasets rate limit exceeded", retry_after=retry)
        response.raise_for_status()
        return response.json()

    async def post(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        headers = {"X-API-KEY": self.api_key}
        response = await self._client.post(f"{self.BASE_URL}{path}", json=payload, headers=headers)
        if response.status_code == 429:
            retry = float(response.headers.get("Retry-After", "1"))
            raise RateLimitError("Financial Datasets rate limit exceeded", retry_after=retry)
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        await self._client.aclose()


class FinancialDatasetsAdapter:
    """Adapter encapsulating Financial Datasets REST endpoints."""

    def __init__(
        self,
        *,
        rest_client: FinancialDatasetsRESTClient | None = None,
        rest_config: Mapping[str, Any] | None = None,
        vendor: str = "financialdatasets",
    ) -> None:
        self.vendor = vendor
        self.rest_client = rest_client
        self.rest_config = {**DEFAULT_REST_CONFIG, **(rest_config or {})}

    async def fetch_company_facts_rest(
        self,
        symbols: Sequence[str],
        *,
        as_of_date: date | None = None,
    ) -> FinancialDatasetsAdapterResult:
        return await self._fetch_per_symbol(
            symbols,
            lambda symbol: self._fetch_company_facts(symbol, as_of_date=as_of_date),
        )

    async def fetch_financial_metrics_rest(
        self,
        symbols: Sequence[str],
        *,
        period: str,
        limit: int | None,
        as_of_date: date | None = None,
    ) -> FinancialDatasetsAdapterResult:
        return await self._fetch_per_symbol(
            symbols,
            lambda symbol: self._fetch_financial_metrics(symbol, period=period, limit=limit, as_of_date=as_of_date),
        )

    async def fetch_financial_metrics_snapshot_rest(
        self,
        symbols: Sequence[str],
        *,
        as_of_date: date | None = None,
    ) -> FinancialDatasetsAdapterResult:
        return await self._fetch_per_symbol(
            symbols,
            lambda symbol: self._fetch_financial_metrics_snapshot(symbol, as_of_date=as_of_date),
        )

    async def fetch_financial_statements_rest(
        self,
        symbols: Sequence[str],
        *,
        period: str,
        limit: int | None,
    ) -> FinancialDatasetsAdapterResult:
        return await self._fetch_per_symbol(
            symbols,
            lambda symbol: self._fetch_financial_statements(symbol, period=period, limit=limit),
        )

    async def fetch_insider_trades_rest(
        self,
        symbols: Sequence[str],
        *,
        limit: int | None,
    ) -> FinancialDatasetsAdapterResult:
        return await self._fetch_per_symbol(
            symbols,
            lambda symbol: self._fetch_insider_trades(symbol, limit=limit),
        )

    async def fetch_institutional_ownership_rest(
        self,
        symbols: Sequence[str],
        *,
        start_date: date | None,
        end_date: date | None,
        limit: int | None,
    ) -> FinancialDatasetsAdapterResult:
        return await self._fetch_per_symbol(
            symbols,
            lambda symbol: self._fetch_institutional_ownership(
                symbol, start_date=start_date, end_date=end_date, limit=limit
            ),
        )

    async def fetch_news_rest(
        self,
        symbols: Sequence[str],
        *,
        start_date: date | None,
        end_date: date | None,
        limit: int | None,
    ) -> FinancialDatasetsAdapterResult:
        return await self._fetch_per_symbol(
            symbols,
            lambda symbol: self._fetch_news(symbol, start_date=start_date, end_date=end_date, limit=limit),
        )

    async def search_financials_screener(
        self,
        *,
        filters: Sequence[Mapping[str, Any]],
        limit: int | None = None,
    ) -> Mapping[str, Any]:
        if not self.rest_client:
            raise RuntimeError("REST ingestion requested but no rest_client configured")
        payload: Dict[str, Any] = {"filters": list(filters)}
        if limit is not None:
            payload["limit"] = int(limit)
        return await self.rest_client.post("/financials/search/screener", payload)

    async def aclose(self) -> None:
        if self.rest_client and hasattr(self.rest_client, "aclose"):
            await self.rest_client.aclose()

    async def _fetch_per_symbol(
        self,
        symbols: Sequence[str],
        handler,
    ) -> FinancialDatasetsAdapterResult:
        if not self.rest_client:
            raise RuntimeError("REST ingestion requested but no rest_client configured")
        semaphore = asyncio.Semaphore(max(1, int(self.rest_config["concurrency"])))
        tasks = [
            asyncio.create_task(self._guarded_fetch(symbol, handler, semaphore))
            for symbol in sorted(symbols)
        ]
        aggregated = FinancialDatasetsAdapterResult(records=[], source_payloads=[], filings=[])
        for result in await asyncio.gather(*tasks):
            aggregated.records.extend(result.records)
            aggregated.source_payloads.extend(result.source_payloads)
            aggregated.filings.extend(result.filings)
        return aggregated

    async def _guarded_fetch(self, symbol: str, handler, semaphore: asyncio.Semaphore) -> FinancialDatasetsAdapterResult:
        async with semaphore:
            return await handler(symbol)

    async def _fetch_company_facts(
        self,
        symbol: str,
        *,
        as_of_date: date | None,
    ) -> FinancialDatasetsAdapterResult:
        payload = await self.rest_client.get("/company/facts", params={"ticker": symbol})
        record = _normalize_company_facts(symbol, payload, as_of_date, vendor=self.vendor)
        return _result_from_payload("company_facts", payload, [record])

    async def _fetch_financial_metrics(
        self,
        symbol: str,
        *,
        period: str,
        limit: int | None,
        as_of_date: date | None,
    ) -> FinancialDatasetsAdapterResult:
        params = {"ticker": symbol, "period": period}
        capped = _cap_limit(limit, _MAX_LIMITS["financial_metrics"])
        if capped:
            params["limit"] = str(capped)
        payload = await self.rest_client.get("/financial-metrics", params=params)
        record = _normalize_financial_metrics(symbol, payload, period=period, as_of_date=as_of_date, vendor=self.vendor)
        return _result_from_payload("financial_metrics", payload, [record])

    async def _fetch_financial_metrics_snapshot(
        self,
        symbol: str,
        *,
        as_of_date: date | None,
    ) -> FinancialDatasetsAdapterResult:
        payload = await self.rest_client.get("/financial-metrics/snapshot", params={"ticker": symbol})
        record = _normalize_financial_metrics_snapshot(symbol, payload, as_of_date=as_of_date, vendor=self.vendor)
        return _result_from_payload("financial_metrics_snapshot", payload, [record])

    async def _fetch_financial_statements(
        self,
        symbol: str,
        *,
        period: str,
        limit: int | None,
    ) -> FinancialDatasetsAdapterResult:
        params = {"ticker": symbol, "period": period}
        capped = _cap_limit(limit, _MAX_LIMITS["financial_statements"])
        if capped:
            params["limit"] = str(capped)
        payload = await self.rest_client.get("/financials", params=params)
        records = _normalize_financial_statements(symbol, payload, period, vendor=self.vendor)
        return _result_from_payload("financial_statements", payload, records)

    async def _fetch_insider_trades(
        self,
        symbol: str,
        *,
        limit: int | None,
    ) -> FinancialDatasetsAdapterResult:
        params = {"ticker": symbol}
        capped = _cap_limit(limit, _MAX_LIMITS["institutional_ownership"])
        if capped:
            params["limit"] = str(capped)
        payload = await self.rest_client.get("/insider-trades", params=params)
        records = _normalize_list_payload(symbol, payload, "insider_trades", vendor=self.vendor)
        return _result_from_payload("insider_trades", payload, records)

    async def _fetch_institutional_ownership(
        self,
        symbol: str,
        *,
        start_date: date | None,
        end_date: date | None,
        limit: int | None,
    ) -> FinancialDatasetsAdapterResult:
        params: Dict[str, Any] = {"ticker": symbol}
        if start_date:
            params["report_period_gte"] = start_date.isoformat()
        if end_date:
            params["report_period_lte"] = end_date.isoformat()
        capped = _cap_limit(limit, _MAX_LIMITS["insider_trades"])
        if capped:
            params["limit"] = str(capped)
        payload = await self.rest_client.get("/institutional-ownership", params=params)
        records = _normalize_list_payload(symbol, payload, "institutional_ownership", vendor=self.vendor)
        return _result_from_payload("institutional_ownership", payload, records)

    async def _fetch_news(
        self,
        symbol: str,
        *,
        start_date: date | None,
        end_date: date | None,
        limit: int | None,
    ) -> FinancialDatasetsAdapterResult:
        params: Dict[str, Any] = {"ticker": symbol}
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()
        capped = _cap_limit(limit, _MAX_LIMITS["news"])
        if capped:
            params["limit"] = str(capped)
        payload = await self.rest_client.get("/news", params=params)
        records = _normalize_list_payload(symbol, payload, "news", vendor=self.vendor)
        return _result_from_payload("news", payload, records)


def _normalize_company_facts(
    symbol: str,
    payload: Mapping[str, Any],
    as_of_date: date | None,
    *,
    vendor: str,
) -> Dict[str, Any]:
    record = dict(payload.get("company_facts") or payload)
    record["symbol"] = record.get("ticker") or symbol
    record["as_of_date"] = _resolve_as_of_date(as_of_date).isoformat()
    record["source_vendor"] = vendor
    record["ingest_ts"] = _ingest_ts()
    return record


def _normalize_financial_metrics(
    symbol: str,
    payload: Mapping[str, Any],
    *,
    period: str,
    as_of_date: date | None,
    vendor: str,
) -> Dict[str, Any]:
    record = dict(payload)
    record["symbol"] = record.get("ticker") or symbol
    record["period"] = period
    record["as_of_date"] = _resolve_as_of_date(as_of_date).isoformat()
    record["source_vendor"] = vendor
    record["ingest_ts"] = _ingest_ts()
    return record


def _normalize_financial_metrics_snapshot(
    symbol: str,
    payload: Mapping[str, Any],
    *,
    as_of_date: date | None,
    vendor: str,
) -> Dict[str, Any]:
    record = dict(payload)
    record["symbol"] = record.get("ticker") or symbol
    record["as_of_date"] = _resolve_as_of_date(as_of_date).isoformat()
    record["source_vendor"] = vendor
    record["ingest_ts"] = _ingest_ts()
    return record


def _normalize_financial_statements(
    symbol: str,
    payload: Mapping[str, Any],
    period: str,
    *,
    vendor: str,
) -> List[Dict[str, Any]]:
    financials = payload.get("financials") if isinstance(payload, Mapping) else None
    if not isinstance(financials, Mapping):
        financials = payload
    income = list(financials.get("income_statements") or [])
    balance = list(financials.get("balance_sheets") or [])
    cashflow = list(financials.get("cash_flow_statements") or [])

    income_map = _index_statements(income)
    balance_map = _index_statements(balance)
    cashflow_map = _index_statements(cashflow)

    all_keys = set(income_map) | set(balance_map) | set(cashflow_map)
    records: List[Dict[str, Any]] = []
    for key in sorted(all_keys):
        income_row = income_map.get(key)
        balance_row = balance_map.get(key)
        cashflow_row = cashflow_map.get(key)
        report_period, statement_period, fiscal_period = key
        record: Dict[str, Any] = {
            "symbol": symbol,
            "report_date": report_period,
            "report_period": report_period,
            "fiscal_period": fiscal_period or "",
            "period": statement_period or period,
            "statement_type": "all",
            "currency": _first_value(income_row, balance_row, cashflow_row, key="currency"),
            "source_vendor": vendor,
            "ingest_ts": _ingest_ts(),
        }
        _merge_statement_fields(record, "income", income_row)
        _merge_statement_fields(record, "balance", balance_row)
        _merge_statement_fields(record, "cashflow", cashflow_row)

        record["revenue"] = _extract_required(income_row, "revenue")
        record["net_income"] = _extract_required(income_row, "net_income")
        record["eps"] = _extract_required(income_row, "earnings_per_share", alt="earnings_per_share_diluted")
        record["operating_income"] = _extract_required(income_row, "operating_income")
        record["total_assets"] = _extract_required(balance_row, "total_assets")
        record["total_liabilities"] = _extract_required(balance_row, "total_liabilities")
        record["shareholder_equity"] = _extract_required(balance_row, "shareholders_equity")
        record["shares_outstanding"] = _extract_required(balance_row, "outstanding_shares", alt="weighted_average_shares")
        record["free_cash_flow"] = _extract_required(cashflow_row, "free_cash_flow")

        _ensure_required_fields(symbol, record)
        records.append(record)
    return records


def _index_statements(rows: Iterable[Mapping[str, Any]]) -> Dict[tuple[str, str, str], Mapping[str, Any]]:
    indexed: Dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        report = _coerce_report_period(row.get("report_period"))
        period = str(row.get("period") or "")
        fiscal = str(row.get("fiscal_period") or "")
        key = (report, period, fiscal)
        indexed[key] = row
    return indexed


def _coerce_report_period(value: Any) -> str:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.date().isoformat()
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if "T" in text:
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            return text
    return text


def _merge_statement_fields(record: Dict[str, Any], prefix: str, row: Mapping[str, Any] | None) -> None:
    if not row:
        return
    for key, value in row.items():
        record[f"{prefix}_{key}"] = value


def _first_value(*rows: Mapping[str, Any] | None, key: str) -> Any:
    for row in rows:
        if row and row.get(key) is not None:
            return row.get(key)
    return None


def _extract_required(row: Mapping[str, Any] | None, field: str, *, alt: str | None = None) -> Any:
    if not row:
        return None
    value = row.get(field)
    if value is None and alt:
        value = row.get(alt)
    return value


def _ensure_required_fields(symbol: str, record: Mapping[str, Any]) -> None:
    required = (
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
    missing = [field for field in required if record.get(field) is None]
    if missing:
        raise ValueError(f"Missing required fields {missing} for symbol {symbol}")
    for field in required:
        value = record[field]
        if not isinstance(value, numbers.Real) or isinstance(value, bool):
            raise TypeError(f"'{field}' must be numeric for symbol {symbol}")


def _normalize_list_payload(
    symbol: str,
    payload: Mapping[str, Any],
    key: str,
    *,
    vendor: str,
) -> List[Dict[str, Any]]:
    rows = payload.get(key)
    if rows is None:
        if "_" in key:
            rows = payload.get(key.replace("_", "-"))
        else:
            rows = payload.get(key.replace("-", "_"))
    if rows is None:
        rows = []
    records: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        record = dict(row)
        record["symbol"] = record.get("ticker") or symbol
        record["source_vendor"] = vendor
        record["ingest_ts"] = _ingest_ts()
        records.append(record)
    return records


def _resolve_as_of_date(candidate: date | None) -> date:
    if candidate is not None:
        return candidate
    return datetime.now(timezone.utc).date()


def _cap_limit(limit: int | None, cap: int | None) -> int | None:
    if limit is None:
        return cap
    if cap is None:
        return limit
    return min(limit, cap)


def _result_from_payload(
    kind: str,
    payload: Mapping[str, Any],
    records: List[Dict[str, Any]],
) -> FinancialDatasetsAdapterResult:
    payload_entry = {"kind": kind, "count": len(records)}
    return FinancialDatasetsAdapterResult(records=records, source_payloads=[payload_entry], filings=[])


def _ingest_ts() -> str:
    return datetime.now(timezone.utc).isoformat()
