"""Polygon adapter for option contract reference, OHLCV, and open interest."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from infra.timestamps import coerce_timestamp as _coerce_timestamp
import csv
import gzip
import io
import json
import queue
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Protocol, Sequence
import zipfile

try:  # pragma: no cover - optional dependency for parquet parsing
    import pyarrow.parquet as pq  # type: ignore

    _PARQUET_ENABLED = True
except Exception:  # pragma: no cover - fallback when pyarrow is missing
    pq = None
    _PARQUET_ENABLED = False

from .polygon_equity import RateLimitError

OptionDomain = str


@dataclass(frozen=True)
class OptionReferenceIngestionRequest:
    """Configuration for option contract reference loads."""

    underlying_symbols: Sequence[str]
    as_of_date: date
    flat_file_uris: Sequence[str] = field(default_factory=tuple)
    vendor: str = "polygon"
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.underlying_symbols:
            raise ValueError("underlying_symbols must be provided")

    @property
    def domain(self) -> OptionDomain:
        return "option_contract_reference"

    @property
    def total_underlyings(self) -> int:
        return len(self.underlying_symbols)

    @property
    def symbols(self) -> Sequence[str]:
        return tuple(self.underlying_symbols)


@dataclass(frozen=True)
class OptionTimeseriesIngestionRequest:
    """Configuration shared by option OHLCV and open interest loads."""

    domain: OptionDomain
    option_symbols: Sequence[str]
    start_date: date
    end_date: date
    flat_file_uris: Sequence[str] = field(default_factory=tuple)
    vendor: str = "polygon"
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.domain not in {"option_contract_ohlcv", "option_open_interest"}:
            raise ValueError("domain must be option_contract_ohlcv or option_open_interest")
        if not self.option_symbols:
            raise ValueError("option_symbols must be provided")
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")

    @property
    def total_days(self) -> int:
        return (self.end_date - self.start_date).days + 1

    @property
    def total_symbols(self) -> int:
        return len(self.option_symbols)

    @property
    def symbols(self) -> Sequence[str]:
        return tuple(self.option_symbols)


class PolygonOptionsRESTClientProtocol(Protocol):
    """Protocol describing the REST client dependency."""

    async def fetch_option_contracts(
        self,
        underlying: str,
        as_of: date,
        *,
        page_url: str | None,
        page_size: int,
    ) -> Mapping[str, Any]:
        ...

    async def fetch_option_ohlcv(
        self,
        option_symbol: str,
        start: date,
        end: date,
        *,
        page_url: str | None,
        page_size: int,
    ) -> Mapping[str, Any]:
        ...

    async def fetch_option_open_interest(
        self,
        option_symbol: str,
        start: date,
        end: date,
        *,
        page_url: str | None,
        page_size: int,
    ) -> Mapping[str, Any]:
        ...

    async def aclose(self) -> None:
        ...


class PolygonOptionsRESTClient:
    """Minimal httpx-based REST client for Polygon option endpoints."""

    CONTRACTS_URL = "https://api.polygon.io/v3/reference/options/contracts"
    AGGS_URL = "https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
    OPEN_INTEREST_URL = "https://api.polygon.io/v3/reference/options/open-interest/{symbol}"

    def __init__(self, api_key: str, *, timeout: float = 30.0) -> None:
        try:
            import httpx  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("httpx must be installed to use PolygonOptionsRESTClient") from exc
        self._httpx = httpx
        self._client = httpx.AsyncClient(timeout=timeout)
        self.api_key = api_key

    async def fetch_option_contracts(
        self,
        underlying: str,
        as_of: date,
        *,
        page_url: str | None = None,
        page_size: int = 1000,
    ) -> Mapping[str, Any]:
        params = {
            "underlying_ticker": underlying,
            "as_of": as_of.isoformat(),
            "limit": str(page_size),
            "apiKey": self.api_key,
        }
        url = page_url or self.CONTRACTS_URL
        if page_url:
            params = None
        response = await self._client.get(url, params=params)
        return self._handle_response(response)

    async def fetch_option_ohlcv(
        self,
        option_symbol: str,
        start: date,
        end: date,
        *,
        page_url: str | None = None,
        page_size: int = 5000,
    ) -> Mapping[str, Any]:
        params = {"adjusted": "true", "sort": "asc", "limit": str(page_size), "apiKey": self.api_key}
        url = page_url or self.AGGS_URL.format(symbol=option_symbol, start=start.isoformat(), end=end.isoformat())
        if page_url:
            params = None
        response = await self._client.get(url, params=params)
        return self._handle_response(response)

    async def fetch_option_open_interest(
        self,
        option_symbol: str,
        start: date,
        end: date,
        *,
        page_url: str | None = None,
        page_size: int = 5000,
    ) -> Mapping[str, Any]:
        params = {
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
            "limit": str(page_size),
            "apiKey": self.api_key,
        }
        url = page_url or self.OPEN_INTEREST_URL.format(symbol=option_symbol)
        if page_url:
            params = None
        response = await self._client.get(url, params=params)
        return self._handle_response(response)

    def _handle_response(self, response: Any) -> Mapping[str, Any]:
        if response.status_code == 429:
            retry = float(response.headers.get("Retry-After", "1"))
            raise RateLimitError("Polygon rate limit exceeded", retry_after=retry)
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        await self._client.aclose()


DEFAULT_REST_CONFIG: Dict[str, Any] = {
    "concurrency": 4,
    "page_size": 1000,
    "backoff_initial": 1.0,
    "backoff_multiplier": 2.0,
    "backoff_max": 30.0,
}

DEFAULT_FLAT_FILE_CONFIG: Dict[str, Any] = {
    "decompression_workers": 4,
    "queue_size": 1000,
}


class PolygonOptionsAdapter:
    """Adapter encapsulating Polygon-specific option ingestion logic."""

    def __init__(
        self,
        *,
        rest_client: PolygonOptionsRESTClientProtocol | None = None,
        flat_file_resolver: Callable[[str], Path] | None = None,
        rest_config: Mapping[str, Any] | None = None,
        flat_file_config: Mapping[str, Any] | None = None,
        vendor: str = "polygon",
    ) -> None:
        self.rest_client = rest_client
        self.vendor = vendor
        self.rest_config = {**DEFAULT_REST_CONFIG, **(rest_config or {})}
        self.flat_file_config = {**DEFAULT_FLAT_FILE_CONFIG, **(flat_file_config or {})}
        self.flat_file_resolver = flat_file_resolver or self._default_flat_file_resolver

    async def fetch_contract_reference_rest(self, request: OptionReferenceIngestionRequest) -> List[Dict[str, Any]]:
        if not self.rest_client:
            raise RuntimeError("REST ingestion requested but no rest_client provided")

        semaphore = asyncio.Semaphore(max(1, int(self.rest_config["concurrency"])))
        tasks = [
            asyncio.create_task(self._fetch_reference_for_underlying(symbol, request, semaphore))
            for symbol in sorted(request.underlying_symbols)
        ]
        records: List[Dict[str, Any]] = []
        for payload in await asyncio.gather(*tasks):
            records.extend(payload)
        return records

    async def fetch_option_ohlcv_rest(self, request: OptionTimeseriesIngestionRequest) -> List[Dict[str, Any]]:
        if not self.rest_client:
            raise RuntimeError("REST ingestion requested but no rest_client provided")

        semaphore = asyncio.Semaphore(max(1, int(self.rest_config["concurrency"])))
        tasks = [
            asyncio.create_task(self._fetch_option_timeseries(symbol, request, semaphore, "ohlcv"))
            for symbol in sorted(request.option_symbols)
        ]
        records: List[Dict[str, Any]] = []
        for payload in await asyncio.gather(*tasks):
            records.extend(payload)
        return records

    async def fetch_option_open_interest_rest(self, request: OptionTimeseriesIngestionRequest) -> List[Dict[str, Any]]:
        if not self.rest_client:
            raise RuntimeError("REST ingestion requested but no rest_client provided")

        semaphore = asyncio.Semaphore(max(1, int(self.rest_config["concurrency"])))
        tasks = [
            asyncio.create_task(self._fetch_option_timeseries(symbol, request, semaphore, "open_interest"))
            for symbol in sorted(request.option_symbols)
        ]
        records: List[Dict[str, Any]] = []
        for payload in await asyncio.gather(*tasks):
            records.extend(payload)
        return records

    def stream_reference_flat_files(self, request: OptionReferenceIngestionRequest) -> Iterable[Dict[str, Any]]:
        return self._stream_flat_files(request.flat_file_uris, self._normalize_reference_record)

    def stream_option_ohlcv_flat_files(self, request: OptionTimeseriesIngestionRequest) -> Iterable[Dict[str, Any]]:
        return self._stream_flat_files(request.flat_file_uris, self._normalize_ohlcv_record)

    def stream_option_open_interest_flat_files(self, request: OptionTimeseriesIngestionRequest) -> Iterable[Dict[str, Any]]:
        return self._stream_flat_files(request.flat_file_uris, self._normalize_open_interest_record)

    async def _fetch_reference_for_underlying(
        self,
        underlying: str,
        request: OptionReferenceIngestionRequest,
        semaphore: asyncio.Semaphore,
    ) -> List[Dict[str, Any]]:
        return await self._fetch_paginated(
            lambda **kwargs: self.rest_client.fetch_option_contracts(underlying, request.as_of_date, **kwargs),  # type: ignore[arg-type]
            self._normalize_reference_record,
            semaphore,
        )

    async def _fetch_option_timeseries(
        self,
        option_symbol: str,
        request: OptionTimeseriesIngestionRequest,
        semaphore: asyncio.Semaphore,
        mode: str,
    ) -> List[Dict[str, Any]]:
        if mode == "ohlcv":
            fetcher = lambda **kwargs: self.rest_client.fetch_option_ohlcv(  # type: ignore[call-arg]
                option_symbol,
                request.start_date,
                request.end_date,
                **kwargs,
            )
            normalizer = lambda payload: self._normalize_ohlcv_record({**payload, "option_symbol": option_symbol})
        else:
            fetcher = lambda **kwargs: self.rest_client.fetch_option_open_interest(  # type: ignore[call-arg]
                option_symbol,
                request.start_date,
                request.end_date,
                **kwargs,
            )
            normalizer = lambda payload: self._normalize_open_interest_record({**payload, "option_symbol": option_symbol})
        return await self._fetch_paginated(fetcher, normalizer, semaphore)

    async def _fetch_paginated(
        self,
        fetcher: Callable[..., Any],
        normalizer: Callable[[Mapping[str, Any]], Dict[str, Any]],
        semaphore: asyncio.Semaphore,
    ) -> List[Dict[str, Any]]:
        backoff = float(self.rest_config["backoff_initial"])
        multiplier = float(self.rest_config["backoff_multiplier"])
        max_backoff = float(self.rest_config["backoff_max"])
        page_url = None
        collected: List[Dict[str, Any]] = []

        while True:
            try:
                async with semaphore:
                    payload = await fetcher(page_url=page_url, page_size=int(self.rest_config["page_size"]))
            except RateLimitError as err:
                delay = err.retry_after or backoff
                await asyncio.sleep(delay)
                backoff = min(backoff * multiplier, max_backoff)
                continue

            results = payload.get("results") or payload.get("ticks") or []
            for raw in results:
                collected.append(normalizer(raw))

            page_url = payload.get("next_url") or payload.get("next_uri")
            if not page_url:
                break

        return collected

    def _stream_flat_files(
        self,
        uris: Sequence[str],
        normalizer: Callable[[Mapping[str, Any]], Dict[str, Any]],
    ) -> Iterable[Dict[str, Any]]:
        if not uris:
            raise ValueError("flat_file_uris must be provided for flat-file ingestion")

        queue_size = max(10, int(self.flat_file_config["queue_size"]))
        record_queue: queue.Queue[tuple[int, int, Any]] = queue.Queue(maxsize=queue_size)
        from concurrent.futures import ThreadPoolExecutor

        executor_workers = max(1, int(self.flat_file_config["decompression_workers"]))
        executor = ThreadPoolExecutor(max_workers=executor_workers)
        uri_order = [str(uri) for uri in uris]

        def worker(uri_index: int, uri: str) -> None:
            path = self.flat_file_resolver(uri)
            seq = 0
            try:
                for record in self._iter_flat_file_records(path):
                    normalized = normalizer(record)
                    record_queue.put((uri_index, seq, normalized))
                    seq += 1
            except Exception as exc:  # pragma: no cover - bubbled up to caller
                record_queue.put((uri_index, -2, exc))
            finally:
                record_queue.put((uri_index, -1, None))

        for idx, uri in enumerate(uri_order):
            executor.submit(worker, idx, uri)

        active_files = len(uri_order)
        buffers: Dict[int, Dict[int, Dict[str, Any]]] = {idx: {} for idx in range(active_files)}
        expected_index: Dict[int, int] = {idx: 0 for idx in range(active_files)}
        completed: set[int] = set()
        current_uri_index = 0

        try:
            while len(completed) < active_files:
                uri_idx, seq_idx, payload = record_queue.get()
                if seq_idx == -2 and isinstance(payload, Exception):
                    raise payload
                if seq_idx == -1 and payload is None:
                    completed.add(uri_idx)
                    continue
                buffers[uri_idx][seq_idx] = payload
                while current_uri_index < active_files:
                    next_seq = expected_index[current_uri_index]
                    buffer = buffers[current_uri_index]
                    if next_seq in buffer:
                        yield buffer.pop(next_seq)
                        expected_index[current_uri_index] += 1
                    else:
                        if current_uri_index in completed and not buffer:
                            current_uri_index += 1
                            continue
                        break
        finally:
            executor.shutdown(wait=True)

    def _iter_flat_file_records(self, path: Path) -> Iterable[Dict[str, Any]]:
        suffixes = [suffix.lower() for suffix in path.suffixes]
        if suffixes and suffixes[-1] == ".zip":
            with zipfile.ZipFile(path, "r") as archive:
                for name in sorted(archive.namelist()):
                    if name.endswith("/"):
                        continue
                    member_suffixes = [suffix.lower() for suffix in Path(name).suffixes]
                    raw = archive.read(name)
                    yield from self._iter_from_bytes(raw, member_suffixes)
            return

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
        elif base_suffix in {".json", ".jsonl"}:
            text: str
            if is_gzip:
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    text = handle.read()
            else:
                text = path.read_text(encoding="utf-8")
            yield from self._parse_json_payload(text, base_suffix == ".jsonl")
        elif base_suffix == ".parquet":
            if not _PARQUET_ENABLED:
                raise RuntimeError("pyarrow is required to consume parquet flat files")
            raw_bytes: io.BufferedReader | gzip.GzipFile
            if is_gzip:
                raw_bytes = gzip.open(path, "rb")
            else:
                raw_bytes = open(path, "rb")  # type: ignore
            with raw_bytes as handle:
                buffer = io.BytesIO(handle.read())
            pq_file = pq.ParquetFile(buffer)
            for batch in pq_file.iter_batches():
                for record in batch.to_pylist():
                    yield record
        else:  # pragma: no cover - defensive guard
            raise ValueError(f"Unsupported flat file format for {path}")

    def _iter_from_bytes(self, raw: bytes, suffixes: Sequence[str]) -> Iterable[Dict[str, Any]]:
        is_gzip = suffixes and suffixes[-1] == ".gz"
        base_suffix = suffixes[-2] if is_gzip and len(suffixes) >= 2 else suffixes[-1] if suffixes else ""

        if base_suffix == ".csv":
            data = raw if not is_gzip else gzip.decompress(raw)
            text = data.decode("utf-8")
            reader = csv.DictReader(io.StringIO(text))
            for row in reader:
                yield dict(row)
        elif base_suffix in {".json", ".jsonl"}:
            data = raw if not is_gzip else gzip.decompress(raw)
            text = data.decode("utf-8")
            yield from self._parse_json_payload(text, base_suffix == ".jsonl")
        elif base_suffix == ".parquet":
            if not _PARQUET_ENABLED:
                raise RuntimeError("pyarrow is required to consume parquet flat files")
            data = raw if not is_gzip else gzip.decompress(raw)
            pq_file = pq.ParquetFile(io.BytesIO(data))
            for batch in pq_file.iter_batches():
                for record in batch.to_pylist():
                    yield record
        else:  # pragma: no cover - defensive guard
            raise ValueError("Unsupported archive member format")

    def _parse_json_payload(self, text: str, jsonl: bool) -> Iterable[Dict[str, Any]]:
        if jsonl:
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        else:
            payload = json.loads(text)
            if isinstance(payload, list):
                for item in payload:
                    yield item
            elif isinstance(payload, Mapping):
                yield dict(payload)
            else:  # pragma: no cover - guard
                raise ValueError("JSON payload must be list or object")

    def _normalize_reference_record(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        option_symbol = payload.get("option_symbol") or payload.get("symbol") or payload.get("ticker") or payload.get("contract_symbol")
        if not option_symbol:
            raise ValueError("Reference payload missing option_symbol")
        underlying = payload.get("underlying_symbol") or payload.get("underlying_ticker")
        if not underlying:
            raise ValueError("Reference payload missing underlying_symbol")
        expiration = payload.get("expiration_date") or payload.get("expiration")
        strike = payload.get("strike") or payload.get("strike_price")
        multiplier = payload.get("multiplier", 100)
        option_type = payload.get("option_type") or payload.get("type")
        if option_type:
            normalized_type = str(option_type).lower()
            if normalized_type in {"call", "c"}:
                option_type = "call"
            elif normalized_type in {"put", "p"}:
                option_type = "put"
            else:
                option_type = normalized_type
        if strike is None or multiplier is None or not expiration:
            raise ValueError("Reference payload missing required numeric fields")

        return {
            "option_symbol": str(option_symbol),
            "underlying_symbol": str(underlying),
            "expiration_date": _coerce_date(expiration),
            "strike": float(strike),
            "option_type": str(option_type or "").lower() or "call",
            "multiplier": float(multiplier),
            "source_vendor": self.vendor,
        }

    def _normalize_ohlcv_record(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        symbol = payload.get("option_symbol") or payload.get("symbol") or payload.get("ticker")
        if not symbol:
            raise ValueError("OHLCV payload missing option_symbol")
        timestamp_value = payload.get("timestamp") or payload.get("t") or payload.get("time")
        if timestamp_value is None:
            raise ValueError("OHLCV payload missing timestamp")
        return {
            "timestamp": _coerce_timestamp(timestamp_value),
            "option_symbol": str(symbol),
            "open": _safe_float(payload.get("open", payload.get("o")), "open"),
            "high": _safe_float(payload.get("high", payload.get("h")), "high"),
            "low": _safe_float(payload.get("low", payload.get("l")), "low"),
            "close": _safe_float(payload.get("close", payload.get("c")), "close"),
            "volume": _safe_float(payload.get("volume", payload.get("v")), "volume", default=0.0),
            "source_vendor": self.vendor,
        }

    def _normalize_open_interest_record(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        symbol = payload.get("option_symbol") or payload.get("symbol") or payload.get("ticker")
        if not symbol:
            raise ValueError("Open interest payload missing option_symbol")
        timestamp_value = payload.get("timestamp") or payload.get("t") or payload.get("time")
        if timestamp_value is None:
            raise ValueError("Open interest payload missing timestamp")
        oi_value = payload.get("open_interest") or payload.get("oi")
        if oi_value is None:
            raise ValueError("Open interest payload missing value")
        return {
            "timestamp": _coerce_timestamp(timestamp_value),
            "option_symbol": str(symbol),
            "open_interest": float(oi_value),
            "source_vendor": self.vendor,
        }

    def _default_flat_file_resolver(self, uri: str) -> Path:
        if uri.startswith("file://"):
            return Path(uri.removeprefix("file://"))
        if uri.startswith("s3://"):
            raise ValueError("s3:// URIs require a custom resolver")
        return Path(uri)


def _safe_float(value: Any, field: str, *, default: float | None = None) -> float:
    """Convert *value* to float, raising ``ValueError`` when it is ``None`` and
    no *default* is provided.  This avoids the ``TypeError`` that
    ``float(None)`` would otherwise raise when a Polygon payload is missing an
    expected key."""
    if value is None:
        if default is not None:
            return default
        raise ValueError(f"Payload missing required numeric field: {field}")
    return float(value)


# _coerce_timestamp is imported from infra.timestamps (epoch_unit="auto" by default).


def _coerce_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        return date.fromisoformat(value[:10])
    raise TypeError("Unsupported date type")


__all__ = [
    "OptionReferenceIngestionRequest",
    "OptionTimeseriesIngestionRequest",
    "PolygonOptionsAdapter",
    "PolygonOptionsRESTClient",
    "PolygonOptionsRESTClientProtocol",
]
