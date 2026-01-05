"""Polygon fundamentals adapter handling REST and flat-file ingestion."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime
import gzip
import json
import numbers
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Protocol, Sequence, Tuple

from .polygon_equity import RateLimitError


StatementType = str
_ALLOWED_STATEMENT_TYPES: Tuple[StatementType, ...] = ("quarterly", "annual")


@dataclass(frozen=True)
class FundamentalsIngestionRequest:
    """Configuration describing a fundamentals ingest window."""

    symbols: Sequence[str]
    start_date: date
    end_date: date
    statement_types: Sequence[StatementType] = field(default_factory=lambda: _ALLOWED_STATEMENT_TYPES)
    flat_file_uris: Sequence[str] = field(default_factory=tuple)
    vendor: str = "polygon"
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.symbols:
            raise ValueError("symbols must be provided")
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        normalized = tuple(dict.fromkeys(self.statement_types)) or _ALLOWED_STATEMENT_TYPES
        for statement in normalized:
            if statement not in _ALLOWED_STATEMENT_TYPES:
                raise ValueError(f"Unsupported statement type '{statement}'")
        object.__setattr__(self, "statement_types", normalized)

    @property
    def total_symbols(self) -> int:
        return len(self.symbols)

    def for_symbols(self, symbols: Sequence[str]) -> "FundamentalsIngestionRequest":
        return FundamentalsIngestionRequest(
            symbols=tuple(symbols),
            start_date=self.start_date,
            end_date=self.end_date,
            statement_types=self.statement_types,
            flat_file_uris=self.flat_file_uris,
            vendor=self.vendor,
            options=self.options,
        )


@dataclass(frozen=True)
class FundamentalsAdapterResult:
    """Container for normalized fundamentals records and lineage metadata."""

    records: List[Dict[str, Any]]
    filings: List[Dict[str, Any]]
    source_payloads: List[Dict[str, Any]]


class PolygonFundamentalsRESTClientProtocol(Protocol):
    """Protocol describing the REST dependency for fundamentals."""

    async def fetch_fundamentals(
        self,
        symbol: str,
        *,
        statement_type: StatementType,
        page_url: str | None,
        limit: int,
        start_date: date,
        end_date: date,
    ) -> Mapping[str, Any]:
        ...


class PolygonFundamentalsRESTClient:
    """Minimal httpx-backed REST client for Polygon fundamentals endpoints."""

    BASE_URL = "https://api.polygon.io/vX/reference/financials"

    def __init__(self, api_key: str, *, timeout: float = 30.0) -> None:
        try:
            import httpx  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError("httpx must be installed to use PolygonFundamentalsRESTClient") from exc
        self._httpx = httpx
        self._client = httpx.AsyncClient(timeout=timeout)
        self.api_key = api_key

    async def fetch_fundamentals(
        self,
        symbol: str,
        *,
        statement_type: StatementType,
        page_url: str | None = None,
        limit: int = 50,
        start_date: date,
        end_date: date,
    ) -> Mapping[str, Any]:
        params = {
            "ticker": symbol,
            "timeframe": statement_type,
            "limit": str(limit),
            "apiKey": self.api_key,
            "period_of_report_date.gte": start_date.isoformat(),
            "period_of_report_date.lte": end_date.isoformat(),
        }
        url = page_url or self.BASE_URL
        if page_url:
            params = None
        response = await self._client.get(url, params=params)
        if response.status_code == 429:
            retry = float(response.headers.get("Retry-After", "1"))
            raise RateLimitError("Polygon fundamentals rate limit exceeded", retry_after=retry)
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        await self._client.aclose()


DEFAULT_REST_CONFIG: Dict[str, Any] = {
    "concurrency": 4,
    "page_size": 50,
    "backoff_initial": 1.0,
    "backoff_multiplier": 2.0,
    "backoff_max": 30.0,
}

DEFAULT_FLAT_FILE_CONFIG: Dict[str, Any] = {
    "decompression_workers": 2,
}


class PolygonFundamentalsAdapter:
    """Adapter encapsulating Polygon fundamentals semantics."""

    def __init__(
        self,
        *,
        rest_client: PolygonFundamentalsRESTClientProtocol | None = None,
        rest_config: Mapping[str, Any] | None = None,
        flat_file_config: Mapping[str, Any] | None = None,
        flat_file_resolver: Callable[[str], Path] | None = None,
        vendor: str = "polygon",
    ) -> None:
        self.vendor = vendor
        self.rest_client = rest_client
        self.rest_config = {**DEFAULT_REST_CONFIG, **(rest_config or {})}
        self.flat_file_config = {**DEFAULT_FLAT_FILE_CONFIG, **(flat_file_config or {})}
        self.flat_file_resolver = flat_file_resolver or (lambda uri: Path(uri).expanduser())

    async def fetch_fundamentals_rest(self, request: FundamentalsIngestionRequest) -> FundamentalsAdapterResult:
        if not self.rest_client:
            raise RuntimeError("REST ingestion requested but no rest_client configured")

        semaphore = asyncio.Semaphore(max(1, int(self.rest_config["concurrency"])))
        tasks = [
            asyncio.create_task(self._fetch_symbol_rest(symbol, request, semaphore))
            for symbol in sorted(request.symbols)
        ]
        aggregated = FundamentalsAdapterResult(records=[], filings=[], source_payloads=[])
        for result in await asyncio.gather(*tasks):
            aggregated.records.extend(result.records)
            aggregated.filings.extend(result.filings)
            aggregated.source_payloads.extend(result.source_payloads)
        return aggregated

    def load_flat_file_fundamentals(self, request: FundamentalsIngestionRequest) -> FundamentalsAdapterResult:
        if not request.flat_file_uris:
            raise ValueError("flat_file_uris must be populated for flat-file ingestion")

        symbol_filter = set(request.symbols)
        statement_filter = set(request.statement_types)
        records: List[Dict[str, Any]] = []
        filings: List[Dict[str, Any]] = []
        payloads: List[Dict[str, Any]] = []
        for uri in request.flat_file_uris:
            path = self.flat_file_resolver(uri)
            payloads.append({"kind": "flat_file", "uri": uri, "hash": _hash_file(path)})
            for entry in self._iter_flat_file_payload(path):
                symbol = entry.get("symbol")
                statement_type = (entry.get("statement_type") or "").lower() or "quarterly"
                if symbol not in symbol_filter or statement_type not in statement_filter:
                    continue
                record_date = self._parse_report_date(entry)
                if record_date < request.start_date or record_date > request.end_date:
                    continue
                record, lineage = self._normalize_entry(symbol, statement_type, entry)
                records.append(record)
                filings.append(lineage)
        return FundamentalsAdapterResult(records=records, filings=filings, source_payloads=payloads)

    async def _fetch_symbol_rest(
        self,
        symbol: str,
        request: FundamentalsIngestionRequest,
        semaphore: asyncio.Semaphore,
    ) -> FundamentalsAdapterResult:
        records: List[Dict[str, Any]] = []
        filings: List[Dict[str, Any]] = []
        payloads: List[Dict[str, Any]] = []
        for statement_type in request.statement_types:
            page_url: str | None = None
            backoff = float(self.rest_config["backoff_initial"])
            while True:
                try:
                    async with semaphore:
                        payload = await self.rest_client.fetch_fundamentals(
                            symbol,
                            statement_type=statement_type,
                            page_url=page_url,
                            limit=int(self.rest_config["page_size"]),
                            start_date=request.start_date,
                            end_date=request.end_date,
                        )
                except RateLimitError as exc:
                    delay = min(backoff, float(self.rest_config["backoff_max"]))
                    await asyncio.sleep(delay)
                    backoff = min(
                        float(self.rest_config["backoff_max"]),
                        backoff * float(self.rest_config["backoff_multiplier"]),
                    )
                    continue
                backoff = float(self.rest_config["backoff_initial"])
                payloads.append(
                    {
                        "kind": "rest_page",
                        "symbol": symbol,
                        "statement_type": statement_type,
                        "hash": _hash_payload(payload),
                        "count": len(payload.get("results", []) or []),
                    }
                )
                for entry in payload.get("results", []) or []:
                    report_date = self._parse_report_date(entry)
                    if report_date < request.start_date or report_date > request.end_date:
                        continue
                    record, lineage = self._normalize_entry(symbol, statement_type, entry)
                    records.append(record)
                    filings.append(lineage)
                page_url = payload.get("next_url")
                if not page_url:
                    break
        return FundamentalsAdapterResult(records=records, filings=filings, source_payloads=payloads)

    def _normalize_entry(
        self,
        symbol: str,
        statement_type: StatementType,
        entry: Mapping[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        report_date = self._parse_report_date(entry)
        fiscal_period = entry.get("fiscal_period") or entry.get("fiscal_period_label")
        if not fiscal_period:
            fiscal_year = entry.get("fiscal_year")
            prefix = "FY" if statement_type == "annual" else "FQ"
            if fiscal_year:
                fiscal_period = f"{prefix}{fiscal_year}"
            else:
                fiscal_period = f"{prefix}{report_date.year}"

        def metric(field: str) -> float:
            value = self._extract_metric(entry, field)
            if value is None:
                raise ValueError(f"'{field}' missing for symbol {symbol}")
            if not isinstance(value, numbers.Real) or isinstance(value, bool):
                raise TypeError(f"'{field}' must be numeric for symbol {symbol}")
            return float(value)

        report_date_iso = report_date.isoformat()
        record = {
            "symbol": symbol,
            "report_date": report_date_iso,
            "fiscal_period": str(fiscal_period),
            "revenue": metric("revenue"),
            "net_income": metric("net_income"),
            "eps": metric("eps"),
            "total_assets": metric("total_assets"),
            "total_liabilities": metric("total_liabilities"),
            "shareholder_equity": metric("shareholder_equity"),
            "operating_income": metric("operating_income"),
            "free_cash_flow": metric("free_cash_flow"),
            "shares_outstanding": metric("shares_outstanding"),
            "source_vendor": self.vendor,
        }

        lineage = {
            "symbol": symbol,
            "statement_type": statement_type,
            "filing_id": str(entry.get("filing_id") or entry.get("filed_under_filing_id") or f"{symbol}-{report_date}"),
            "filing_date": _iso_date(entry.get("filing_date") or entry.get("filed_date")),
            "report_date": report_date_iso,
            "restated": bool(entry.get("restated")),
            "supersedes": entry.get("restatement_of") or entry.get("restated_from"),
        }
        if lineage["supersedes"]:
            lineage["restatement_note"] = f"Supersedes {lineage['supersedes']}"
        return record, lineage

    def _iter_flat_file_payload(self, path: Path) -> Iterable[Mapping[str, Any]]:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            text = handle.read()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                yield json.loads(stripped)
            return
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, Mapping):
                    yield item
        elif isinstance(payload, Mapping):
            results = payload.get("results")
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, Mapping):
                        yield item
            else:
                yield payload

    def _extract_metric(self, entry: Mapping[str, Any], field: str) -> Any:
        if field in entry:
            return entry[field]
        metrics = entry.get("metrics")
        if isinstance(metrics, Mapping) and field in metrics:
            return metrics[field]
        financials = entry.get("financials")
        if isinstance(financials, Mapping):
            for section in financials.values():
                if not isinstance(section, Mapping):
                    continue
                candidate = section.get(field)
                if isinstance(candidate, Mapping):
                    candidate = candidate.get("value")
                if candidate is not None:
                    return candidate
        return None

    def _parse_report_date(self, entry: Mapping[str, Any]) -> date:
        candidate = entry.get("report_date") or entry.get("report_period") or entry.get("period_of_report_date")
        if isinstance(candidate, date) and not isinstance(candidate, datetime):
            return candidate
        if isinstance(candidate, datetime):
            return candidate.date()
        if isinstance(candidate, str):
            return date.fromisoformat(candidate.split("T")[0])
        raise ValueError("report_date/report_period must be provided")


def _hash_payload(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    import hashlib

    return f"sha256:{hashlib.sha256(serialized.encode('utf-8')).hexdigest()}"


def _hash_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _iso_date(value: Any | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, str):
        return value.split("T")[0]
    return str(value)


__all__ = [
    "FundamentalsAdapterResult",
    "FundamentalsIngestionRequest",
    "PolygonFundamentalsAdapter",
    "PolygonFundamentalsRESTClient",
    "PolygonFundamentalsRESTClientProtocol",
]
