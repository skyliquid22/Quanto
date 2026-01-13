"""Deterministic REST client for the iVolatility Backtest API Plus service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping, MutableMapping, Protocol, Sequence
import time
import csv
import io
import gzip
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
    body: bytes


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

    DEFAULT_BASE_URL = "https://restapi.ivolatility.com"
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
        if not self.api_key:
            raise IvolatilityAuthenticationError(
                "IVOLATILITY_API_KEY environment variable must be set"
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

    def fetch_async_dataset(
        self,
        endpoint: str,
        params: Mapping[str, Any] | None = None,
        *,
        poll_timeout_s: int = 300,
        poll_interval_s: float = 2.0,
        completion_codes: set[str] | None = None,
        pending_codes: set[str] | None = None,
    ) -> list[dict[str, Any]] | bytes:
        """Submit an async job and poll urlForDetails until completion."""

        resolved_endpoint = endpoint.strip("/")
        params_with_credentials = self._inject_credentials(params or {})
        fingerprint = self._request_fingerprint(resolved_endpoint, params_with_credentials)
        cached_dataset = self._read_dataset_cache(fingerprint)
        if cached_dataset is not None:
            return cached_dataset

        completion_set = {
            str(code).upper()
            for code in (
                completion_codes
                or {"COMPLETED", "COMPLETE", "SUCCESS", "DONE"}
            )
        }
        pending_set = {str(code).upper() for code in (pending_codes or {"PENDING", "RUNNING", "PROCESSING"})}
        payload = self._request_with_retry(
            resolved_endpoint,
            params_with_credentials,
            use_cache=False,
        )
        final_payload = self._resolve_async_payload(
            payload,
            completion_set,
            pending_set,
            poll_timeout_s,
            poll_interval_s,
            fingerprint,
        )
        dataset = self._finalize_async_dataset(final_payload, fingerprint)
        self._write_dataset_cache(fingerprint, dataset)
        return dataset

    def _inject_credentials(self, params: Mapping[str, Any]) -> MutableMapping[str, Any]:
        merged: MutableMapping[str, Any] = dict(params)
        merged.setdefault("apiKey", self.api_key)
        if self.api_secret:
            merged.setdefault("apiSecret", self.api_secret)
        return merged

    def _request_fingerprint(self, endpoint: str, params: Mapping[str, Any]) -> str:
        payload = {
            "endpoint": endpoint,
            "params": self._sanitize_fingerprint_params(params),
        }
        digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        return digest[:16]

    def _sanitize_fingerprint_params(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        sanitized: dict[str, Any] = {}
        for key, value in params.items():
            lowered = str(key).lower()
            if lowered in {"apikey", "apisecret"}:
                continue
            sanitized[str(key)] = value
        return sanitized

    def _read_dataset_cache(self, fingerprint: str) -> list[dict[str, Any]] | bytes | None:
        if not self.cache_dir:
            return None
        json_path = self.cache_dir / f"async-{fingerprint}.json"
        if json_path.exists():
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None
            payload = self._maybe_decode_nested_json(payload)
            if isinstance(payload, list):
                normalized: list[dict[str, Any]] = []
                for entry in payload:
                    if isinstance(entry, Mapping):
                        normalized.append(dict(entry))
                return self._normalize_records(normalized)
        bin_path = self.cache_dir / f"async-{fingerprint}.bin"
        if bin_path.exists():
            data = bin_path.read_bytes()
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                return data
            decoded = self._decode_json_text(text)
            if decoded is not None:
                result = self._finalize_async_dataset(decoded, fingerprint)
                self._write_dataset_cache(fingerprint, result)
                return result
            url = self._extract_download_url_from_text(text)
            if url:
                dataset = self._download_and_normalize_file(url)
                self._write_dataset_cache(fingerprint, dataset)
                return dataset
            return data
        return None

    def _write_dataset_cache(self, fingerprint: str, dataset: list[dict[str, Any]] | bytes) -> None:
        if not self.cache_dir:
            return
        json_path = self.cache_dir / f"async-{fingerprint}.json"
        bin_path = self.cache_dir / f"async-{fingerprint}.bin"
        if isinstance(dataset, bytes):
            if json_path.exists():
                json_path.unlink()
            bin_path.write_bytes(dataset)
        else:
            if bin_path.exists():
                bin_path.unlink()
            serialized = json.dumps(dataset, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            json_path.write_text(serialized, encoding="utf-8")

    @staticmethod
    def map_date_params(
        *,
        start_date: date | None = None,
        end_date: date | None = None,
        as_of_date: date | None = None,
    ) -> dict[str, str]:
        """Translate canonical date fields into iVolatility query parameters."""

        params: dict[str, str] = {}
        if start_date:
            params["from"] = start_date.isoformat()
        if end_date:
            params["to"] = end_date.isoformat()
        if as_of_date:
            params["date"] = as_of_date.isoformat()
        return params

    def _request_with_retry(
        self,
        endpoint_or_url: str,
        params: Mapping[str, Any],
        *,
        use_cache: bool = True,
        parse_json: bool = True,
    ) -> Mapping[str, Any] | TransportResponse:
        cache_key: str | None = None
        cached: Mapping[str, Any] | None = None
        if use_cache and parse_json:
            cache_key = self._cache_key(endpoint_or_url, params)
            cached = self._read_cache(cache_key)
            if cached is not None:
                return cached

        attempt = 0
        backoff = self.backoff_factor
        url = self._resolve_url(endpoint_or_url)
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

            if not parse_json:
                return response

            payload = self._parse_payload(response)
            if use_cache and cache_key:
                self._write_cache(cache_key, payload)
            return payload

    def _headers(self) -> Mapping[str, str]:
        return {
            "Accept": "application/json",
            "User-Agent": "quanto-ingestion/1.0",
        }

    def _resolve_url(self, endpoint_or_url: str) -> str:
        text = (endpoint_or_url or "").strip()
        if text.startswith("http://") or text.startswith("https://"):
            return text
        return f"{self.base_url}/{text.lstrip('/')}"

    def _resolve_async_payload(
        self,
        payload: Mapping[str, Any],
        completion_codes: set[str],
        pending_codes: set[str],
        poll_timeout_s: int,
        poll_interval_s: float,
        fingerprint: str,
    ) -> Mapping[str, Any]:
        status = self._extract_status(payload)
        if not status:
            return payload
        code = self._normalize_status_code(status.get("code"))
        if not code or code in completion_codes:
            return payload
        if code in pending_codes:
            detail_url = (
                status.get("urlForDetails")
                or status.get("url")
                or payload.get("urlForDetails")
                or payload.get("pollUrl")
            )
            if detail_url:
                return self._poll_for_completion(
                    str(detail_url),
                    completion_codes,
                    pending_codes,
                    poll_timeout_s,
                    poll_interval_s,
                    fingerprint,
                )
            return payload

        description = status.get("description") or status.get("message") or ""
        raise IvolatilityClientError(
            f"Async job failed with status {code} (fingerprint={fingerprint}) {description}"
        )

    def _extract_status(self, payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
        status = payload.get("status")
        if isinstance(status, Mapping):
            return status
        return None

    def _normalize_status_code(self, code: Any) -> str:
        if code is None:
            return ""
        return str(code).strip().upper()

    def _poll_for_completion(
        self,
        detail_url: str,
        completion_codes: set[str],
        pending_codes: set[str],
        poll_timeout_s: int,
        poll_interval_s: float,
        fingerprint: str,
    ) -> Mapping[str, Any]:
        deadline = time.time() + max(0, poll_timeout_s)
        interval = max(0.0, poll_interval_s)
        params = self._credential_params_for_url(detail_url) if detail_url else {}
        while True:
            now = time.time()
            if now >= deadline:
                raise IvolatilityClientError(
                    f"Async job timed out after {poll_timeout_s}s (fingerprint={fingerprint})"
                )
            wait_time = min(interval, max(0.0, deadline - now))
            if wait_time > 0:
                time.sleep(wait_time)
            payload = self._request_with_retry(detail_url, params, use_cache=False) if detail_url else {}
            status = self._extract_status(payload)
            if not status:
                if detail_url:
                    return payload
                raise IvolatilityClientError(
                    f"Async job pending but missing status payload (fingerprint={fingerprint})"
                )
            code = self._normalize_status_code(status.get("code"))
            if not code or code in completion_codes:
                return payload
            if code in pending_codes:
                interval = self._next_poll_interval(interval)
                continue
            description = status.get("description") or status.get("message") or ""
            raise IvolatilityClientError(
                f"Async job failed with status {code} (fingerprint={fingerprint}) {description}"
            )

    def _next_poll_interval(self, current: float) -> float:
        if current <= 0:
            return 0.0
        growth = max(current * 2.5, current + 1.0)
        return min(10.0, growth)

    def _credential_params_for_url(self, url: str) -> Mapping[str, Any]:
        parsed = urllib_parse.urlparse(url)
        query_keys = {key.lower() for key, _ in urllib_parse.parse_qsl(parsed.query, keep_blank_values=True)}
        params: dict[str, Any] = {}
        if "apikey" not in query_keys:
            params["apiKey"] = self.api_key
        if self.api_secret and "apisecret" not in query_keys:
            params["apiSecret"] = self.api_secret
        return params

    def _finalize_async_dataset(
        self,
        payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        fingerprint: str,
    ) -> list[dict[str, Any]] | bytes:
        records: list[dict[str, Any]] = []
        if isinstance(payload, Mapping):
            records = self._extract_records(payload)
            if records:
                for record in records:
                    download_url = self._extract_download_url(record)
                    if download_url:
                        return self._download_and_normalize_file(download_url)
                return self._normalize_records(records)
        download_url = self._extract_download_url(payload)
        if download_url:
            return self._download_and_normalize_file(download_url)
        if isinstance(payload, Sequence) and not isinstance(payload, (Mapping, str, bytes)):
            normalized_entries = [dict(entry) for entry in payload if isinstance(entry, Mapping)]
            for entry in normalized_entries:
                download_url = self._extract_download_url(entry)
                if download_url:
                    return self._download_and_normalize_file(download_url)
            if normalized_entries:
                return self._normalize_records(normalized_entries)
            return []
        if isinstance(payload, (str, bytes)):
            text = payload.decode("utf-8", errors="ignore") if isinstance(payload, bytes) else payload
            url = self._extract_download_url_from_text(text)
            if url:
                return self._download_and_normalize_file(url)
        raise IvolatilityClientError(
            f"Async job completed without data payload (fingerprint={fingerprint})"
        )

    def _extract_download_url(self, payload: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> str | None:
        if isinstance(payload, Mapping):
            direct = payload.get("__download_url")
            if isinstance(direct, str) and direct.strip():
                return direct.strip()
            raw_text = payload.get("__raw_text")
            if isinstance(raw_text, str):
                candidate = self._extract_download_url_from_text(raw_text)
                if candidate:
                    return candidate
            candidates = (
                "urlForDownload",
                "downloadUrl",
                "downloadURL",
                "urlForFile",
                "fileUrl",
                "fileURL",
                "urlForResult",
                "resultUrl",
                "urlForDetails",
            )
            for key in candidates:
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            status = self._extract_status(payload)
            if status:
                for key in candidates:
                    value = status.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
            data = payload.get("data")
            if isinstance(data, Mapping):
                for key in candidates:
                    value = data.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
            if isinstance(data, Sequence):
                for entry in data:
                    url = self._extract_download_url(entry)
                    if url:
                        return url
            file_name = payload.get("fileName") or payload.get("filename")
            if isinstance(file_name, str) and file_name.strip():
                return f"{self.base_url.rstrip('/')}/data/download/{file_name.strip()}"
        if isinstance(payload, Sequence):
            for entry in payload:
                url = self._extract_download_url(entry)
                if url:
                    return url
        return None

    def _extract_download_url_from_text(self, text: str) -> str | None:
        marker = "urlForDownload"
        idx = text.find(marker)
        if idx != -1:
            idx = text.find("http", idx)
            if idx != -1:
                end = idx
                while end < len(text) and text[end] not in {'"', "'", "\\", " ", "\n", "\r", ","}:
                    end += 1
                candidate = text[idx:end]
                if candidate.startswith("http"):
                    return candidate
        match = re.search(r'fileName"+"?\s*:\s*"+([^"\s]+)', text)
        if match:
            file_name = match.group(1)
            if file_name:
                return f"{self.base_url.rstrip('/')}/data/download/{file_name}"
        return None

    def _decode_json_text(self, text: str) -> Any | None:
        stripped = text.strip()
        if not stripped:
            return None
        candidates = [stripped]
        if stripped.startswith('"') and stripped.endswith('"'):
            collapsed = stripped[1:-1].replace('""', '"')
            candidates.append(collapsed)
        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate:
                continue
            if candidate.startswith("{") or candidate.startswith("["):
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue
        return None

    def _download_and_normalize_file(self, url: str) -> bytes:
        params = self._credential_params_for_url(url)
        delay = max(1.0, self.backoff_factor)
        deadline = time.time() + max(30, self.max_backoff * 3)
        while True:
            response = self._request_with_retry(
                url,
                params,
                use_cache=False,
                parse_json=False,
            )
            if not isinstance(response, TransportResponse):  # pragma: no cover - defensive
                raise IvolatilityClientError("Transport returned unexpected payload for download")
            raw_bytes = response.body
            text = response.text or ""
            lower_text = text.lower()
            if 'status:"pending"' in lower_text or 'status":"pending"' in lower_text:
                if time.time() >= deadline:
                    raise IvolatilityClientError("Download URL remained pending after retries")
                time.sleep(min(delay, self.max_backoff))
                delay = min(delay * 1.5, self.max_backoff)
                continue
            content_type = (response.headers.get("Content-Type") or "").lower()
            if "gzip" in content_type or raw_bytes.startswith(b"\x1f\x8b"):
                try:
                    text = gzip.decompress(raw_bytes).decode("utf-8")
                except Exception:
                    pass
                else:
                    return self._canonicalize_csv_bytes(text)
            if "csv" in content_type or self._looks_like_csv(text):
                return self._canonicalize_csv_bytes(text)
            return raw_bytes or text.encode("utf-8")

    def _canonicalize_csv_bytes(self, text: str) -> bytes:
        reader = csv.DictReader(io.StringIO(text), delimiter=",")
        rows: list[dict[str, Any]] = [dict(row) for row in reader]
        normalized = self._normalize_records(rows)
        fieldnames = list(reader.fieldnames or [])
        observed_keys: set[str] = set()
        for record in normalized:
            observed_keys.update(record.keys())
        if not fieldnames:
            fieldnames = sorted(observed_keys)
        else:
            missing = sorted(observed_keys.difference(fieldnames))
            fieldnames.extend(missing)
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for record in normalized:
            writer.writerow({name: self._format_csv_value(record.get(name)) for name in fieldnames})
        return output.getvalue().encode("utf-8")

    def _format_csv_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, date):
            return value.isoformat()
        return str(value)

    def _looks_like_csv(self, text: str) -> bool:
        if not text:
            return False
        first_line = text.splitlines()[0]
        return "," in first_line

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
            if isinstance(entry, Mapping):
                normalized.append(dict(entry))
                continue
            if isinstance(entry, str):
                decoded = self._decode_json_text(entry)
                if decoded is None:
                    url = self._extract_download_url_from_text(entry)
                    if url:
                        normalized.append({"__download_url": url})
                    else:
                        normalized.append({"__raw_text": entry})
                    continue
                if isinstance(decoded, Mapping):
                    normalized.append(dict(decoded))
                elif isinstance(decoded, list):
                    for item in decoded:
                        if isinstance(item, Mapping):
                            normalized.append(dict(item))
                continue
            raise IvolatilityClientError("API returned a non-mapping record")
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

    def _parse_payload(self, response: TransportResponse) -> Mapping[str, Any]:
        text = response.text or ""
        decoded = self._decode_json_text(text)
        stripped = text.lstrip()
        content_type = (response.headers.get("Content-Type") or "").lower()
        if decoded is not None:
            if isinstance(decoded, Mapping):
                return decoded
            if isinstance(decoded, list):
                return {"records": decoded}
        if not stripped:
            return {}
        # CSV payloads
        first_line = stripped.splitlines()[0] if stripped else ""
        if "csv" in content_type or "," in first_line:
            reader = csv.DictReader(io.StringIO(text))
            return {"records": [dict(row) for row in reader]}
        raise IvolatilityClientError(
            f"Unsupported response content-type '{response.headers.get('Content-Type')}'"
        )

    def _maybe_decode_nested_json(self, value: Any) -> Any:
        if isinstance(value, str):
            decoded = self._decode_json_text(value)
            if decoded is not None:
                return decoded
        return value


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
            raw = response.read()
            encoding = (response.headers.get("Content-Encoding") or "").lower()
            if "gzip" in encoding:
                try:
                    raw = gzip.decompress(raw)
                except OSError:
                    raw = raw
            text = raw.decode("utf-8", errors="replace")
            return TransportResponse(
                status_code=response.getcode(),
                headers=dict(response.headers.items()),
                text=text,
                body=raw,
            )
    except urllib_error.HTTPError as exc:  # pragma: no cover - exercised only when real HTTP fails
        raw = exc.read() if exc.fp else b""
        text = raw.decode("utf-8", errors="replace")
        return TransportResponse(
            status_code=exc.code,
            headers=dict(exc.headers.items()) if exc.headers else {},
            text=text,
            body=raw,
        )
    except urllib_error.URLError as exc:  # pragma: no cover - exercised only when transport unavailable
        raise IvolatilityClientError(f"Transport error contacting iVolatility: {exc}") from exc


def _canonical_json(value: Any) -> str:
    return json.dumps(_canonicalize_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonicalize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonicalize_value(value[key]) for key in sorted(value.keys(), key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize_value(item) for item in value]
    if isinstance(value, set):
        return [_canonicalize_value(item) for item in sorted(value, key=str)]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "IvolatilityAuthenticationError",
    "IvolatilityClient",
    "IvolatilityClientError",
    "IvolatilityRateLimitError",
    "Transport",
    "TransportResponse",
]
