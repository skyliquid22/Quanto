"""iVolatility adapter for bulk options surface ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import csv
import io
import json
from pathlib import Path
from typing import Any, Iterable, List, Mapping, MutableMapping, Sequence

from infra.normalization.lineage import compute_file_hash
from infra.paths import get_data_root
from infra.storage.parquet import write_parquet_atomic

from ..ivolatility_client import IvolatilityClient
from infra.normalization.options_surface import NORMALIZED_OPTIONS_SURFACE_COLUMNS, normalize_ivol_surface


@dataclass(frozen=True)
class OptionsSurfaceIngestionRequest:
    """Normalized configuration for bulk surface ingestion."""

    symbols: tuple[str, ...]
    start_date: date
    end_date: date
    vendor: str
    options: Mapping[str, Any]
    vendor_params: Mapping[str, Any]

    @property
    def total_symbols(self) -> int:
        return len(self.symbols)

    @property
    def total_days(self) -> int:
        return (self.end_date - self.start_date).days + 1


class OptionsSurfaceStorage:
    """Handles deterministic storage + manifest emission for derived options surfaces."""

    def __init__(
        self,
        *,
        base_dir: Path | str | None = None,
        manifest_dir: Path | str | None = None,
    ) -> None:
        resolved_base = Path(base_dir) if base_dir else get_data_root() / "derived" / "options_surface_v1"
        self.base_dir = resolved_base
        self.manifest_dir = Path(manifest_dir) if manifest_dir else resolved_base / "manifests"

    def manifest_path(self, run_id: str) -> Path:
        return self.manifest_dir / f"{run_id}.json"

    def persist(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        request: OptionsSurfaceIngestionRequest,
        run_id: str,
        endpoint: str,
        params: Mapping[str, Any],
        created_at: str | None = None,
    ) -> Mapping[str, Any]:
        files = self._write_rows(rows)
        manifest = self._build_manifest(
            rows=rows,
            files=files,
            request=request,
            run_id=run_id,
            endpoint=endpoint,
            params=params,
            created_at=created_at,
        )
        manifest_path = self._write_manifest(run_id, manifest)
        manifest["manifest_path"] = str(manifest_path)
        return manifest

    def _write_rows(self, rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
        grouped: MutableMapping[Path, List[Mapping[str, Any]]] = {}
        for row in rows:
            symbol = str(row.get("symbol", "")).strip().upper()
            if not symbol:
                continue
            date_str = str(row.get("date") or "").strip()
            if not date_str or len(date_str) < 4:
                continue
            year = date_str[:4]
            path = self.base_dir / symbol / "daily" / f"{year}.parquet"
            grouped.setdefault(path, []).append(dict(row))

        files_written: List[Mapping[str, Any]] = []
        for path, new_rows in grouped.items():
            existing = self._read_existing_rows(path)
            merged = self._merge_rows(existing, new_rows)
            write_parquet_atomic(merged, path)
            file_hash = compute_file_hash(path)
            files_written.append(
                {
                    "path": str(path),
                    "records": len(merged),
                    "sha256": file_hash,
                }
            )
        files_written.sort(key=lambda item: item["path"])
        return files_written

    def _read_existing_rows(self, path: Path) -> List[Mapping[str, Any]]:
        if not path.exists():
            return []
        try:
            import pyarrow.parquet as pq  # type: ignore

            table = pq.read_table(path)
            return [dict(entry) for entry in table.to_pylist()]
        except Exception:
            return []

    def _merge_rows(
        self,
        existing: Sequence[Mapping[str, Any]],
        new_rows: Sequence[Mapping[str, Any]],
    ) -> List[Mapping[str, Any]]:
        merged: dict[tuple[str, str], Mapping[str, Any]] = {}
        for payload in existing:
            key = (str(payload.get("symbol") or "").upper(), str(payload.get("date") or ""))
            merged[key] = dict(payload)
        for payload in new_rows:
            key = (str(payload.get("symbol") or "").upper(), str(payload.get("date") or ""))
            merged[key] = dict(payload)
        ordered = sorted(
            merged.values(),
            key=lambda item: (str(item.get("symbol") or ""), str(item.get("date") or "")),
        )
        return [dict(entry) for entry in ordered]

    def _build_manifest(
        self,
        *,
        rows: Sequence[Mapping[str, Any]],
        files: Sequence[Mapping[str, Any]],
        request: OptionsSurfaceIngestionRequest,
        run_id: str,
        endpoint: str,
        params: Mapping[str, Any],
        created_at: str | None,
    ) -> Mapping[str, Any]:
        timestamp = created_at or datetime.now(timezone.utc).isoformat()
        manifest = {
            "run_id": run_id,
            "domain": "options_surface_v1",
            "vendor": request.vendor,
            "status": "succeeded",
            "symbols": sorted(request.symbols),
            "start_date": request.start_date.isoformat(),
            "end_date": request.end_date.isoformat(),
            "row_count": len(rows),
            "files_written": [dict(item) for item in files],
            "endpoint": endpoint,
            "params": dict(params),
            "columns": list(NORMALIZED_OPTIONS_SURFACE_COLUMNS),
            "created_at": timestamp,
        }
        return manifest

    def _write_manifest(self, run_id: str, payload: Mapping[str, Any]) -> Path:
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.manifest_path(run_id)
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        return manifest_path


class IvolatilityOptionsSurfaceAdapter:
    """Adapter coordinating vendor fetch + normalization for surface ingestion."""

    # iVolatility bulk surface metrics live on the stock-market-data endpoint.
    DEFAULT_ENDPOINT = "equities/stock-market-data"
    DEFAULT_SYMBOLS_PER_REQUEST = 3

    def __init__(
        self,
        client: IvolatilityClient,
        *,
        vendor: str = "ivolatility",
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.client = client
        self.vendor = vendor
        cfg = dict(config or {})
        self.endpoint = cfg.get("endpoint", self.DEFAULT_ENDPOINT)
        self.extra_params = dict(cfg.get("params") or {})
        self.vol_unit = str(cfg.get("vol_unit", "percent"))
        self.symbols_per_request = max(1, int(cfg.get("symbols_per_request", self.DEFAULT_SYMBOLS_PER_REQUEST)))

    def fetch_surface(
        self,
        request: OptionsSurfaceIngestionRequest,
    ) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
        sorted_symbols = tuple(sorted(request.symbols))
        if not sorted_symbols:
            raise ValueError("options_surface_v1 ingestion requires at least one symbol")
        all_rows: list[dict[str, Any]] = []
        manifest_chunks: list[Mapping[str, Any]] = []
        for window_start, window_end in self._iter_year_windows(request.start_date, request.end_date):
            for batch in self._chunk_symbols(sorted_symbols):
                params = self._build_params(batch, window_start, window_end)
                payload = self.client.fetch_async_dataset(self.endpoint, params)
                records = self._coerce_dataset(payload)
                if not records:
                    resolved_records = self._resolve_remote_payload(payload)
                    if resolved_records:
                        records = resolved_records
                rows = normalize_ivol_surface(
                    records,
                    allowed_symbols=batch,
                    start_date=window_start,
                    end_date=window_end,
                    vol_unit=self.vol_unit,
                )
                all_rows.extend(rows)
                manifest_chunks.append(self._sanitize_manifest_params(params, batch))
        sanitized = {"requests": manifest_chunks}
        return all_rows, sanitized

    def _build_params(
        self,
        symbols: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> dict[str, Any]:
        base = IvolatilityClient.map_date_params(start_date=start_date, end_date=end_date)
        base["symbols"] = ",".join(symbols)
        for key, value in self.extra_params.items():
            base[key] = value
        return base

    def _coerce_dataset(self, payload: List[Mapping[str, Any]] | bytes | str | Any) -> List[Mapping[str, Any]]:
        if isinstance(payload, list):
            return [dict(entry) for entry in payload if isinstance(entry, Mapping)]
        if isinstance(payload, bytes):
            text = payload.decode("utf-8", errors="ignore")
            return self._parse_csv(text)
        if isinstance(payload, str):
            text = payload.strip()
            if not text:
                return []
            if text.startswith("{") or text.startswith("["):
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    return []
                if isinstance(parsed, list):
                    return [dict(entry) for entry in parsed if isinstance(entry, Mapping)]
                if isinstance(parsed, Mapping):
                    return [dict(parsed)]
            return self._parse_csv(text)
        return []

    def _parse_csv(self, text: str) -> List[Mapping[str, Any]]:
        reader = csv.DictReader(io.StringIO(text))
        return [dict(row) for row in reader]

    def _resolve_remote_payload(self, payload: Any) -> List[Mapping[str, Any]]:
        text: str | None = None
        if isinstance(payload, bytes):
            text = payload.decode("utf-8", errors="ignore")
        elif isinstance(payload, str):
            text = payload
        if not text:
            return []
        url = self.client._extract_download_url_from_text(text)
        if not url:
            return []
        csv_bytes = self.client._download_and_normalize_file(url)
        try:
            csv_text = csv_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return []
        return self._parse_csv(csv_text)

    def _sanitize_manifest_params(
        self,
        params: Mapping[str, Any],
        sorted_symbols: Sequence[str],
    ) -> Mapping[str, Any]:
        sanitized: dict[str, Any] = {"symbols": list(sorted_symbols)}
        for key, value in params.items():
            lowered = str(key).lower()
            if lowered in {"apikey", "api_key", "apisecret", "api_secret"}:
                continue
            if key == "symbols":
                continue
            sanitized[str(key)] = value
        return sanitized

    def _iter_year_windows(self, start_date: date, end_date: date) -> Iterable[tuple[date, date]]:
        start_year = start_date.year
        end_year = end_date.year
        for year in range(start_year, end_year + 1):
            window_start = date(year, 1, 1)
            window_end = date(year, 12, 31)
            yield max(start_date, window_start), min(end_date, window_end)

    def _chunk_symbols(self, symbols: Sequence[str]) -> Iterable[tuple[str, ...]]:
        chunk_size = self.symbols_per_request
        for idx in range(0, len(symbols), chunk_size):
            yield tuple(symbols[idx : idx + chunk_size])


__all__ = [
    "IvolatilityOptionsSurfaceAdapter",
    "OptionsSurfaceIngestionRequest",
    "OptionsSurfaceStorage",
]
