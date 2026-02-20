"""Dedicated ingestion pipeline for insider trades."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from infra.paths import raw_root
from infra.storage.raw_writer import RawFinancialDatasetsWriter

from .adapters import FinancialDatasetsAdapter, FinancialDatasetsAdapterResult
from .request import IngestionRequest
from .router import Mode


def _extract_filing_date_filters(options: Mapping[str, Any]) -> Dict[str, Any]:
    filters: Dict[str, Any] = {}
    for key in ("filing_date_lte", "filing_date_lt", "filing_date_gte", "filing_date_gt"):
        if key in options and options[key] is not None:
            filters[key] = options[key]
    return filters


class InsiderTradesIngestionPipeline:
    """Coordinates Financial Datasets insider-trades ingestion."""

    SUPPORTED_DOMAIN = "insider_trades"

    def __init__(
        self,
        *,
        adapter: FinancialDatasetsAdapter,
        raw_writer: RawFinancialDatasetsWriter | None = None,
        manifest_base_dir: Path | str | None = None,
    ) -> None:
        self.adapter = adapter
        self.raw_writer = raw_writer or RawFinancialDatasetsWriter()
        base = Path(manifest_base_dir) if manifest_base_dir is not None else raw_root()
        self.manifest_base_dir = base

    def run(
        self,
        request: IngestionRequest,
        *,
        run_id: str,
        forced_mode: Mode | None = None,
    ) -> Dict[str, Any]:
        if request.domain != self.SUPPORTED_DOMAIN:
            raise ValueError(f"Unsupported insider trades domain '{request.domain}'")
        if not run_id:
            raise ValueError("run_id must be provided")

        mode = forced_mode or "rest"
        if mode != "rest":
            raise ValueError("Insider trades ingestion only supports REST mode")

        adapter_result = self._collect_records(request)
        creation_ts = self._resolve_creation_timestamp(run_id)
        write_result = self.raw_writer.write_insider_trades(request.vendor, adapter_result.records)
        files_written = sorted(write_result.get("files", []), key=lambda item: item["path"])
        manifest = self._build_manifest(
            request=request,
            run_id=run_id,
            mode=mode,
            records=adapter_result.records,
            files_written=files_written,
            source_payloads=adapter_result.source_payloads,
            status="succeeded",
            error=None,
            created_at=creation_ts,
        )
        manifest_path = self._persist_manifest(request.vendor, request.domain, run_id, manifest)
        manifest["manifest_path"] = str(manifest_path)
        return manifest

    def _collect_records(self, request: IngestionRequest) -> FinancialDatasetsAdapterResult:
        return asyncio.run(self._collect_records_async(request))

    async def _collect_records_async(self, request: IngestionRequest) -> FinancialDatasetsAdapterResult:
        options = request.options or {}
        limit = options.get("limit")
        if limit is not None:
            limit = int(limit)
        filing_filters = _extract_filing_date_filters(options)
        result = await self.adapter.fetch_insider_trades_rest(
            request.symbols,
            limit=limit,
            filing_date_filters=filing_filters,
        )
        await self.adapter.aclose()
        return result

    def _build_manifest(
        self,
        *,
        request: IngestionRequest,
        run_id: str,
        mode: Mode,
        records: Sequence[Mapping[str, Any]],
        files_written: Sequence[Mapping[str, Any]],
        source_payloads: Sequence[Mapping[str, Any]],
        status: str,
        error: str | None,
        created_at: str,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "run_id": run_id,
            "vendor": request.vendor,
            "domain": request.domain,
            "mode": mode,
            "status": status,
            "symbols": list(request.symbols),
            "record_counts": {"requested": len(records), "validated": 0},
            "files_written": list(files_written),
            "source_payloads": list(source_payloads),
            "created_at": created_at,
        }
        if error:
            payload["error"] = error
        return payload

    def _manifest_dir(self, vendor: str, domain: str) -> Path:
        return self.manifest_base_dir / vendor / domain / "manifests"

    def _persist_manifest(self, vendor: str, domain: str, run_id: str, manifest: Mapping[str, Any]) -> Path:
        directory = self._manifest_dir(vendor, domain)
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path = directory / f"{run_id}.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        return manifest_path

    def _resolve_creation_timestamp(self, seed: str) -> str:
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        seconds = int(digest[:8], 16) % (365 * 24 * 60 * 60)
        base = datetime(2020, 1, 1, tzinfo=timezone.utc)
        return (base + timedelta(seconds=seconds)).isoformat()


__all__ = ["InsiderTradesIngestionPipeline"]
