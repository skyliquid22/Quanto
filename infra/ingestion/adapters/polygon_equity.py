"""Polygon vendor adapter for equity OHLCV ingestion."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import csv
import gzip
import io
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Protocol, Sequence

try:  # pragma: no cover - optional dependency for parquet parsing
    import pyarrow.parquet as pq  # type: ignore

    _PARQUET_ENABLED = True
except Exception:  # pragma: no cover - fallback when pyarrow is missing
    pq = None
    _PARQUET_ENABLED = False


class RateLimitError(RuntimeError):
    """Raised when Polygon responds with HTTP 429."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class PolygonRESTClientProtocol(Protocol):
    """Protocol describing the REST client dependency used by the adapter."""

    async def fetch_aggregates(
        self,
        symbol: str,
        start: date,
        end: date,
        *,
        page_url: str | None,
        page_size: int,
    ) -> Mapping[str, Any]:
        ...


class PolygonRESTClient:
    """Minimal REST client built on httpx for production usage."""

    BASE_URL = "https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"

    def __init__(self, api_key: str, *, timeout: float = 30.0) -> None:
        try:
            import httpx  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised only when dependency missing
            raise RuntimeError("httpx must be installed to use PolygonRESTClient") from exc
        self._httpx = httpx
        self._client = httpx.AsyncClient(timeout=timeout)
        self.api_key = api_key

    async def fetch_aggregates(
        self,
        symbol: str,
        start: date,
        end: date,
        *,
        page_url: str | None = None,
        page_size: int = 5000,
    ) -> Mapping[str, Any]:
        params = {"adjusted": "true", "sort": "asc", "limit": str(page_size), "apiKey": self.api_key}
        if page_url:
            url = page_url
            params = None
        else:
            url = self.BASE_URL.format(symbol=symbol, start=start.isoformat(), end=end.isoformat())
        response = await self._client.get(url, params=params)
        if response.status_code == 429:
            retry = float(response.headers.get("Retry-After", "1"))
            raise RateLimitError("Polygon rate limit exceeded", retry_after=retry)
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        await self._client.aclose()


@dataclass(frozen=True)
class EquityIngestionRequest:
    """Configuration describing a single ingestion run."""

    symbols: Sequence[str]
    start_date: date
    end_date: date
    frequency: str = "daily"
    flat_file_uris: Sequence[str] = field(default_factory=tuple)
    vendor: str = "polygon"
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.symbols:
            raise ValueError("symbols must be provided")
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")

    @property
    def total_days(self) -> int:
        return (self.end_date - self.start_date).days + 1

    @property
    def total_symbols(self) -> int:
        return len(self.symbols)


DEFAULT_REST_CONFIG: Dict[str, Any] = {
    "concurrency": 4,
    "page_size": 5000,
    "backoff_initial": 1.0,
    "backoff_multiplier": 2.0,
    "backoff_max": 30.0,
}

DEFAULT_FLAT_FILE_CONFIG: Dict[str, Any] = {
    "decompression_workers": 4,
    "queue_size": 1000,
}


class PolygonEquityAdapter:
    """Adapter encapsulating Polygon-specific ingestion logic."""

    def __init__(
        self,
        *,
        rest_client: PolygonRESTClientProtocol | None = None,
        flat_file_resolver: Callable[[str], Path] | None = None,
        rest_config: Mapping[str, Any] | None = None,
        flat_file_config: Mapping[str, Any] | None = None,
        vendor: str = "polygon",
    ) -> None:
        self.vendor = vendor
        self.rest_client = rest_client
        self.rest_config = {**DEFAULT_REST_CONFIG, **(rest_config or {})}
        self.flat_file_config = {**DEFAULT_FLAT_FILE_CONFIG, **(flat_file_config or {})}
        self.flat_file_resolver = flat_file_resolver or self._default_flat_file_resolver

    async def fetch_equity_ohlcv_rest(self, request: EquityIngestionRequest) -> List[Dict[str, Any]]:
        """Fetch bars over REST using async concurrency with pagination and rate limiting."""

        if not self.rest_client:
            raise RuntimeError("REST ingestion requested but no rest_client provided")

        semaphore = asyncio.Semaphore(max(1, int(self.rest_config["concurrency"])))
        tasks = [
            asyncio.create_task(self._fetch_symbol_rest(symbol, request, semaphore))
            for symbol in sorted(request.symbols)
        ]
        records: List[Dict[str, Any]] = []
        for result in await asyncio.gather(*tasks):
            records.extend(result)
        return records

    def stream_flat_file_equity_bars(self, request: EquityIngestionRequest) -> Iterable[Dict[str, Any]]:
        """Yield bars from compressed flat-files using multithreaded decompression."""

        if not request.flat_file_uris:
            raise ValueError("flat_file_uris must be populated for flat-file ingestion")

        for uri in request.flat_file_uris:
            path = self.flat_file_resolver(uri)
            for record in self._iter_flat_file_records(path):
                yield self._normalize_flat_file_record(record)

    async def _fetch_symbol_rest(
        self, symbol: str, request: EquityIngestionRequest, semaphore: asyncio.Semaphore
    ) -> List[Dict[str, Any]]:
        backoff = float(self.rest_config["backoff_initial"])
        multiplier = float(self.rest_config["backoff_multiplier"])
        max_backoff = float(self.rest_config["backoff_max"])
        page_url = None
        collected: List[Dict[str, Any]] = []

        while True:
            try:
                async with semaphore:
                    payload = await self.rest_client.fetch_aggregates(
                        symbol=symbol,
                        start=request.start_date,
                        end=request.end_date,
                        page_url=page_url,
                        page_size=int(self.rest_config["page_size"]),
                    )
            except RateLimitError as err:
                delay = err.retry_after or backoff
                await asyncio.sleep(delay)
                backoff = min(backoff * multiplier, max_backoff)
                continue

            results = payload.get("results") or []
            for raw in results:
                collected.append(self._normalize_rest_record(symbol, raw))

            page_url = payload.get("next_url")
            if not page_url:
                break

        return collected

    def _iter_flat_file_records(self, path: Path) -> Iterable[Dict[str, Any]]:
        suffixes = path.suffixes
        is_gzip = suffixes and suffixes[-1] == ".gz"
        base_suffix = suffixes[-2] if is_gzip and len(suffixes) >= 2 else suffixes[-1] if suffixes else ""

        if base_suffix == ".csv":
            opener: Callable[..., Any]
            if is_gzip:
                opener = gzip.open
            else:
                opener = open  # type: ignore
            with opener(path, mode="rt", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    yield dict(row)
        elif base_suffix == ".parquet":
            if not _PARQUET_ENABLED:
                raise RuntimeError("pyarrow is required to consume parquet flat files")
            raw_bytes: io.BufferedReader | gzip.GzipFile
            if is_gzip:
                raw_bytes = gzip.open(path, "rb")
            else:
                raw_bytes = open(path, "rb")  # type: ignore
            with raw_bytes as handle:
                file_obj = io.BytesIO(handle.read())
            pq_file = pq.ParquetFile(file_obj)
            for batch in pq_file.iter_batches():
                for record in batch.to_pylist():
                    yield record
        else:  # pragma: no cover - defensive guard
            raise ValueError(f"Unsupported flat file format for {path}")

    def _normalize_rest_record(self, symbol: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        timestamp_value = payload.get("timestamp") or payload.get("t")
        if timestamp_value is None:
            raise ValueError("REST payload missing timestamp field")

        return {
            "timestamp": _coerce_timestamp(timestamp_value),
            "symbol": symbol,
            "open": float(payload.get("open", payload.get("o"))),
            "high": float(payload.get("high", payload.get("h"))),
            "low": float(payload.get("low", payload.get("l"))),
            "close": float(payload.get("close", payload.get("c"))),
            "volume": float(payload.get("volume", payload.get("v", 0.0))),
            "source_vendor": self.vendor,
        }

    def _normalize_flat_file_record(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        symbol = payload.get("symbol") or payload.get("ticker")
        if not symbol:
            raise ValueError("Flat-file row missing symbol field")
        timestamp_value = payload.get("timestamp") or payload.get("t") or payload.get("date")
        if timestamp_value is None:
            raise ValueError("Flat-file row missing timestamp field")

        return {
            "timestamp": _coerce_timestamp(timestamp_value),
            "symbol": str(symbol),
            "open": float(payload.get("open", payload.get("o"))),
            "high": float(payload.get("high", payload.get("h"))),
            "low": float(payload.get("low", payload.get("l"))),
            "close": float(payload.get("close", payload.get("c"))),
            "volume": float(payload.get("volume", payload.get("v", 0.0))),
            "source_vendor": self.vendor,
        }

    def _default_flat_file_resolver(self, uri: str) -> Path:
        if uri.startswith("file://"):
            return Path(uri.removeprefix("file://"))
        if uri.startswith("s3://"):
            # TODO: integrate with S3 client when credentials plumbing is available.
            raise ValueError("s3:// URIs require a custom resolver")
        return Path(uri)


def _coerce_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        # Treat numeric values as epoch milliseconds
        seconds = float(value) / 1000 if value > 1e12 else float(value)
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    if isinstance(value, str):
        clean_value = value.strip()
        if clean_value.endswith("Z"):
            clean_value = clean_value[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(clean_value)
        except ValueError:
            # interpret YYYY-MM-DD as date
            if len(clean_value) == 10:
                dt = datetime.fromisoformat(clean_value + "T00:00:00+00:00")
            else:
                raise
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    raise TypeError(f"Unsupported timestamp type {type(value)}")


__all__ = [
    "EquityIngestionRequest",
    "PolygonEquityAdapter",
    "PolygonRESTClient",
    "RateLimitError",
]
