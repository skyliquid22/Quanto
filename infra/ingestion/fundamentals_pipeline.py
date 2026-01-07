"""Config-driven fundamentals ingestion pipeline."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping

from infra.paths import raw_root
from infra.storage.raw_writer import RawFundamentalsWriter
from infra.validation import ValidationError, validate_records

from .adapters import (
    FundamentalsAdapterResult,
    FundamentalsIngestionRequest,
    PolygonFundamentalsAdapter,
)
from .router import IngestionRouter, Mode


class FundamentalsIngestionPipeline:
    """Coordinates fundamentals ingestion, validation, checkpointing, and storage."""

    def __init__(
        self,
        *,
        adapter: PolygonFundamentalsAdapter,
        router: IngestionRouter | None = None,
        raw_writer: RawFundamentalsWriter | None = None,
        manifest_base_dir: Path | str | None = None,
        checkpoint_dir: Path | str | None = None,
    ) -> None:
        self.adapter = adapter
        self.router = router or IngestionRouter()
        self.raw_writer = raw_writer or RawFundamentalsWriter()
        base = Path(manifest_base_dir) if manifest_base_dir is not None else raw_root()
        self.manifest_base_dir = base
        default_checkpoint = base / "checkpoints" / "fundamentals"
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else default_checkpoint

    def run(
        self,
        request: FundamentalsIngestionRequest,
        *,
        run_id: str,
        forced_mode: Mode | None = None,
    ) -> Dict[str, Any]:
        if not run_id:
            raise ValueError("run_id must be provided")
        mode = forced_mode or self.router.route_fundamentals(request)
        checkpoint = self._load_checkpoint(request.vendor, run_id)
        checkpoint["mode"] = mode
        stored_symbols: MutableMapping[str, Dict[str, Any]] = checkpoint.setdefault("symbols", {})

        for symbol in sorted(request.symbols):
            existing = stored_symbols.get(symbol)
            if existing and existing.get("status") == "completed":
                continue

            symbol_request = request.for_symbols((symbol,))
            adapter_result = self._collect_records(symbol_request, mode)
            creation_ts = self._resolve_creation_timestamp(f"{run_id}:{symbol}")
            validation_config = {
                "manifest_base_path": self._manifest_dir(request.vendor) / "validation",
                "input_file_hashes": [item["hash"] for item in adapter_result.source_payloads if "hash" in item],
                "creation_timestamp": creation_ts,
            }

            try:
                validated, validation_manifest = validate_records(
                    "fundamentals",
                    adapter_result.records,
                    source_vendor=request.vendor,
                    run_id=f"{run_id}:{symbol}",
                    config=validation_config,
                )
            except ValidationError as exc:
                failure_entry = self._build_symbol_failure(symbol, mode, adapter_result, exc, creation_ts)
                manifest = self._build_manifest(
                    request=request,
                    run_id=run_id,
                    mode=mode,
                    stored_symbols=stored_symbols,
                    failure=failure_entry,
                    status="failed",
                    error=str(exc),
                )
                manifest_path = self._persist_manifest(request.vendor, run_id, manifest)
                manifest["manifest_path"] = str(manifest_path)
                self._persist_checkpoint(request.vendor, run_id, checkpoint)
                raise

            write_result = self.raw_writer.write_records(request.vendor, validated)
            files_written = sorted(write_result.get("files", []), key=lambda item: item["path"])
            stored_symbols[symbol] = {
                "symbol": symbol,
                "status": "completed",
                "mode": mode,
                "record_counts": {"requested": len(adapter_result.records), "validated": len(validated)},
                "files_written": files_written,
                "filings": adapter_result.filings,
                "source_payloads": adapter_result.source_payloads,
                "validation_manifest": validation_manifest["manifest_path"],
                "created_at": creation_ts,
            }
            self._persist_checkpoint(request.vendor, run_id, checkpoint)

        manifest = self._build_manifest(
            request=request,
            run_id=run_id,
            mode=mode,
            stored_symbols=stored_symbols,
            failure=None,
            status="succeeded",
            error=None,
        )
        manifest_path = self._persist_manifest(request.vendor, run_id, manifest)
        manifest["manifest_path"] = str(manifest_path)
        return manifest

    def _collect_records(
        self,
        request: FundamentalsIngestionRequest,
        mode: Mode,
    ) -> FundamentalsAdapterResult:
        if mode == "rest":
            return asyncio.run(self.adapter.fetch_fundamentals_rest(request))
        return self.adapter.load_flat_file_fundamentals(request)

    def _build_manifest(
        self,
        *,
        request: FundamentalsIngestionRequest,
        run_id: str,
        mode: Mode,
        stored_symbols: Mapping[str, Mapping[str, Any]],
        failure: Mapping[str, Any] | None,
        status: str,
        error: str | None,
    ) -> Dict[str, Any]:
        entries = self._ordered_symbol_entries(stored_symbols)
        files_written: List[Mapping[str, Any]] = []
        validation_manifests: List[str] = []
        payloads: List[Mapping[str, Any]] = []
        filings: List[Mapping[str, Any]] = []
        total_requested = 0
        total_validated = 0
        for entry in entries:
            files_written.extend(entry.get("files_written", []))
            if entry.get("validation_manifest"):
                validation_manifests.append(entry["validation_manifest"])
            payloads.extend(entry.get("source_payloads", []))
            filings.extend(entry.get("filings", []))
            counts = entry.get("record_counts") or {}
            total_requested += counts.get("requested", 0)
            total_validated += counts.get("validated", 0)
        manifest = {
            "run_id": run_id,
            "vendor": request.vendor,
            "mode": mode,
            "status": status,
            "symbols": sorted(request.symbols),
            "statement_types": list(request.statement_types),
            "record_counts": {"requested": total_requested, "validated": total_validated},
            "files_written": sorted(files_written, key=lambda item: item["path"]),
            "validation_manifests": validation_manifests,
            "source_payloads": payloads,
            "filing_lineage": filings,
            "restatements": [
                {k: v for k, v in filing.items() if k in {"symbol", "filing_id", "supersedes", "report_date", "restatement_note"}}
                for filing in filings
                if filing.get("supersedes")
            ],
            "created_at": self._resolve_creation_timestamp(run_id),
            "failures": [],
        }
        if failure:
            manifest["failures"].append(dict(failure))
        if error:
            manifest["error"] = error
        return manifest

    def _build_symbol_failure(
        self,
        symbol: str,
        mode: Mode,
        adapter_result: FundamentalsAdapterResult,
        exc: ValidationError,
        created_at: str,
    ) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "mode": mode,
            "status": "failed",
            "record_counts": {"requested": len(adapter_result.records), "validated": len(exc.validated_records)},
            "validation_manifest": exc.manifest.get("manifest_path"),
            "errors": exc.errors,
            "created_at": created_at,
            "source_payloads": adapter_result.source_payloads,
        }

    def _ordered_symbol_entries(self, stored_symbols: Mapping[str, Mapping[str, Any]]) -> List[Mapping[str, Any]]:
        return [stored_symbols[symbol] for symbol in sorted(stored_symbols.keys()) if stored_symbols.get(symbol)]

    def _manifest_dir(self, vendor: str) -> Path:
        return self.manifest_base_dir / vendor / "fundamentals" / "manifests"

    def _persist_manifest(self, vendor: str, run_id: str, manifest: Mapping[str, Any]) -> Path:
        directory = self._manifest_dir(vendor)
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path = directory / f"{run_id}.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        return manifest_path

    def _checkpoint_path(self, vendor: str, run_id: str) -> Path:
        return self.checkpoint_dir / vendor / f"{run_id}.json"

    def _load_checkpoint(self, vendor: str, run_id: str) -> Dict[str, Any]:
        path = self._checkpoint_path(vendor, run_id)
        if path.exists():
            return json.loads(path.read_text())
        return {"run_id": run_id, "vendor": vendor, "symbols": {}}

    def _persist_checkpoint(self, vendor: str, run_id: str, payload: Mapping[str, Any]) -> None:
        path = self._checkpoint_path(vendor, run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def _resolve_creation_timestamp(self, seed: str) -> str:
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        seconds = int(digest[:8], 16) % (365 * 24 * 60 * 60)
        base = datetime(2020, 1, 1, tzinfo=timezone.utc)
        return (base + timedelta(seconds=seconds)).isoformat()


__all__ = ["FundamentalsIngestionPipeline"]
