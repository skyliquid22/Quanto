"""Deterministic REST client for the iVolatility Backtest API Plus service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Protocol, Sequence
import time
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

Transport = Callable[
    [str, str, Mapping[str, Any] | None, Mapping[str, str], float],
    "TransportResponse",
]


class IvolatilityClientError(RuntimeError):
    """Base error for iVolatility client failures."""


class IvolatilityAuthenticationError(IvolatilityClientError):
    """Raised when credentials are missing or rejected."""


class IvolatilityRateLimitError(IvolatilityClientError):
    """Raised when the API responds with 429 Too Many Requests."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


@dataclass(frozen=True)
class TransportResponse:
    """Normalized response payload returned by transport callables."""

    status_code: int
    headers: Mapping[str, str]
    text: str


class _TransportProtocol(Protocol):
    def __call__(
        self,
        method: str,
        url: str,
        params: Mapping[str, Any] | None,
        headers: Mapping[str, str],
        timeout: float,
    ) -> TransportResponse:
        ...


class IvolatilityClient:
    """Small REST client with pagination, retries, and deterministic ordering."""

    DEFAULT_BASE_URL = "https://api.ivolatility.com/backtest"
    DEFAULT_TIMEOUT = 15.0
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_BACKOFF = 1.5
    DEFAULT_MAX_BACKOFF = 10.0
    _PAGINATION_KEYS = (
        "next_page_token",
        "nextPageToken",
        "next",
        "next_page",
        "nextCursor",
        "cursor",
    )

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        backoff_factor: float | None = None,
        max_backoff: float | None = None,
        cache_dir: str | Path | None = None,
        transport: _TransportProtocol | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("IVOLATILITY_API_KEY")
        self.api_secret = api_secret or os.environ.get("IVOLATILITY_API_SECRET")
        if not self.api_key or not self.api_secret:
            raise IvolatilityAuthenticationError(
                "IVOLATILITY_API_KEY and IVOLATILITY_API_SECRET environment variables must be set"
            )

        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout if timeout is not None else self.DEFAULT_TIMEOUT
        self.max_retries = max_retries if max_retries is not None else self.DEFAULT_MAX_RETRIES
        self.backoff_factor = backoff_factor if backoff_factor is not None else self.DEFAULT_BACKOFF
        self.max_backoff = max_backoff if max_backoff is not None else self.DEFAULT_MAX_BACKOFF
        self.transport = transport or _default_transport
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, endpoint: str, params: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """Fetch every page for an endpoint and return a deterministically ordered list."""

        resolved_endpoint = endpoint.strip("/")
        base_params = self._inject_credentials(params or {})
        collected: list[dict[str, Any]] = []
        next_token: str | None = None

        while True:
            page_params = dict(base_params)
            if next_token:
                page_params["page_token"] = next_token
            payload = self._request_with_retry(resolved_endpoint, page_params)
            records = self._extract_records(payload)
            collected.extend(records)
            next_token = self._extract_next_token(payload)
            if not next_token:
                break
        return self._normalize_records(collected)

    def fetch_one(self, endpoint: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Fetch records and enforce that exactly one payload was returned."""

        records = self.fetch(endpoint, params)
        if len(records) != 1:
            raise IvolatilityClientError(
                f"Expected a single record from {endpoint} but received {len(records)}"
            )
        return records[0]

    def _inject_credentials(self, params: Mapping[str, Any]) -> MutableMapping[str, Any]:
        merged: MutableMapping[str, Any] = dict(params)
        merged.setdefault("apiKey", self.api_key)
        merged.setdefault("apiSecret", self.api_secret)
        return merged

    def _request_with_retry(
        self,
        endpoint: str,
        params: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        cache_key = self._cache_key(endpoint, params)
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        attempt = 0
        backoff = self.backoff_factor
        url = f"{self.base_url}/{endpoint}"
        while True:
            response = self.transport("GET", url, params, self._headers(), self.timeout)
            if response.status_code == 429:
                retry_after = self._retry_after_seconds(response.headers)
                if attempt >= self.max_retries:
                    raise IvolatilityRateLimitError(
                        "iVolatility API rate limit exceeded", retry_after=retry_after
                    )
                time.sleep(min(self.max_backoff, retry_after or backoff))
                attempt += 1
                backoff = min(self.max_backoff, backoff * self.backoff_factor)
                continue

            if 500 <= response.status_code < 600:
                if attempt >= self.max_retries:
                    raise IvolatilityClientError(
                        f"iVolatility API responded with HTTP {response.status_code}"
                    )
                time.sleep(min(self.max_backoff, backoff))
                attempt += 1
                backoff = min(self.max_backoff, backoff * self.backoff_factor)
                continue

            if response.status_code < 200 or response.status_code >= 300:
                raise IvolatilityClientError(
                    f"iVolatility API responded with HTTP {response.status_code}: {response.text}"
                )

            payload = self._parse_payload(response.text)
            if self.cache_dir:
                self._write_cache(cache_key, payload)
            return payload

    def _headers(self) -> Mapping[str, str]:
        return {
            "Accept": "application/json",
            "User-Agent": "quanto-ingestion/1.0",
        }

    def _cache_key(self, endpoint: str, params: Mapping[str, Any]) -> str:
        elements = [endpoint]
        for key in sorted(params.keys()):
            value = params[key]
            if isinstance(value, (list, tuple, set)):
                serialized = ",".join(sorted(str(item) for item in value))
            else:
                serialized = str(value)
            elements.append(f"{key}={serialized}")
        digest = hashlib.sha256("|".join(elements).encode("utf-8")).hexdigest()
        return digest

    def _read_cache(self, cache_key: str) -> Mapping[str, Any] | None:
        if not self.cache_dir:
            return None
        path = self.cache_dir / f"{cache_key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _write_cache(self, cache_key: str, payload: Mapping[str, Any]) -> None:
        if not self.cache_dir:
            return
        path = self.cache_dir / f"{cache_key}.json"
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    def _extract_records(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, Mapping):
            for key in ("records", "results", "data", "items", "payload"):
                value = payload.get(key)
                if isinstance(value, list):
                    records = value
                    break
            else:
                records = []
        else:
            records = []

        normalized: list[dict[str, Any]] = []
        for entry in records:
            if not isinstance(entry, Mapping):
                raise IvolatilityClientError("API returned a non-mapping record")
            normalized.append(dict(entry))
        return normalized

    def _extract_next_token(self, payload: Mapping[str, Any]) -> str | None:
        for key in self._PAGINATION_KEYS:
            token = payload.get(key)
            if token:
                return str(token)
        meta = payload.get("meta")
        if isinstance(meta, Mapping):
            for key in self._PAGINATION_KEYS:
                token = meta.get(key)
                if token:
                    return str(token)
        return None

    def _normalize_records(self, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        normalized = [dict(record) for record in records]
        normalized.sort(key=self._sort_key)
        return normalized

    def _sort_key(self, record: Mapping[str, Any]) -> tuple[str, str, str]:
        timestamp_fields = ("timestamp", "date", "as_of", "pricingDate", "price_date", "trade_date")
        timestamp_value: Any = None
        for field in timestamp_fields:
            if field in record:
                timestamp_value = record[field]
                break
        ts_key = self._coerce_sortable_value(timestamp_value)

        symbol_fields = (
            "symbol",
            "ticker",
            "option_symbol",
            "optionSymbol",
            "underlying",
            "underlying_symbol",
            "contract",
        )
        symbol_value: Any = None
        for field in symbol_fields:
            if field in record:
                symbol_value = record[field]
                break
        symbol_key = self._coerce_sortable_value(symbol_value)

        fallback = json.dumps(record, sort_keys=True)
        return ts_key, symbol_key, fallback

    def _coerce_sortable_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (int, float)):
            return f"{float(value):020.6f}"
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, date):
            return value.isoformat()
        return str(value)

    def _retry_after_seconds(self, headers: Mapping[str, str]) -> float | None:
        for key in ("Retry-After", "retry-after", "retry_after"):
            if key in headers:
                try:
                    return float(headers[key])
                except (TypeError, ValueError):
                    continue
        return None

    def _parse_payload(self, text: str) -> Mapping[str, Any]:
        if not text.strip():
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
            raise IvolatilityClientError(f"Invalid JSON response: {exc}") from exc
        if isinstance(parsed, Mapping):
            return parsed
        if isinstance(parsed, list):
            return {"records": parsed}
        raise IvolatilityClientError("Unexpected JSON payload received from API")


def _default_transport(
    method: str,
    url: str,
    params: Mapping[str, Any] | None,
    headers: Mapping[str, str],
    timeout: float,
) -> TransportResponse:
    method = method.upper()
    query_params = dict(params or {})
    data: bytes | None = None
    request_headers = dict(headers)
    if method == "GET":
        query_string = urllib_parse.urlencode(query_params)
        if query_string:
            url = f"{url}?{query_string}"
    else:
        data = json.dumps(query_params).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")

    req = urllib_request.Request(url, data=data, method=method, headers=request_headers)
    try:
        with urllib_request.urlopen(req, timeout=timeout) as response:
            text = response.read().decode("utf-8")
            return TransportResponse(
                status_code=response.getcode(),
                headers=dict(response.headers.items()),
                text=text,
            )
    except urllib_error.HTTPError as exc:  # pragma: no cover - exercised only when real HTTP fails
        body = exc.read().decode("utf-8") if exc.fp else ""
        return TransportResponse(
            status_code=exc.code,
            headers=dict(exc.headers.items()) if exc.headers else {},
            text=body,
        )
    except urllib_error.URLError as exc:  # pragma: no cover - exercised only when transport unavailable
        raise IvolatilityClientError(f"Transport error contacting iVolatility: {exc}") from exc


__all__ = [
    "IvolatilityAuthenticationError",
    "IvolatilityClient",
    "IvolatilityClientError",
    "IvolatilityRateLimitError",
    "Transport",
    "TransportResponse",
]
