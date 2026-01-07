"""Config-driven orchestration for equity OHLCV ingestion."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from infra.paths import raw_root
from infra.storage.raw_writer import RawEquityOHLCVWriter
from infra.validation import ValidationError, validate_records

from .adapters import EquityIngestionRequest, PolygonEquityAdapter
from .router import IngestionRouter, Mode

def default_equity_manifest_dir(vendor: str = "polygon") -> Path:
    return raw_root(vendor) / "equity_ohlcv" / "manifests"


RAW_MANIFEST_DIR = default_equity_manifest_dir()


class EquityIngestionPipeline:
    """Coordinates adapter routing, validation, storage, and manifest emission."""

    def __init__(
        self,
        *,
        adapter: PolygonEquityAdapter,
        router: IngestionRouter | None = None,
        raw_writer: RawEquityOHLCVWriter | None = None,
        manifest_dir: Path | str | None = None,
        raw_shard_yearly_daily: bool | None = None,
    ) -> None:
        self.adapter = adapter
        self.router = router or IngestionRouter()
        if raw_writer:
            self.raw_writer = raw_writer
        else:
            self.raw_writer = RawEquityOHLCVWriter(shard_yearly_daily=raw_shard_yearly_daily)
        manifest_base = manifest_dir if manifest_dir is not None else default_equity_manifest_dir()
        self.manifest_dir = Path(manifest_base)

    def run(
        self,
        request: EquityIngestionRequest,
        *,
        run_id: str,
        forced_mode: Mode | None = None,
    ) -> Dict[str, Any]:
        """Execute an ingestion run and persist a raw-layer manifest."""

        if not run_id:
            raise ValueError("run_id must be provided for deterministic runs")

        mode = forced_mode or self.router.route_equity_ohlcv(request)
        source_files = self._source_file_metadata(request) if mode == "flat_file" else []
        records = self._collect_records(request, mode)
        resolved_timestamp = self._resolve_creation_timestamp(request, run_id)
        validation_config = {
            "manifest_base_path": self.manifest_dir / "validation",
            "input_file_hashes": [item["hash"] for item in source_files],
            "creation_timestamp": resolved_timestamp,
        }

        try:
            validated, validation_manifest = validate_records(
                "equity_ohlcv",
                records,
                source_vendor=request.vendor,
                run_id=run_id,
                config=validation_config,
            )
        except ValidationError as exc:
            manifest = self._build_manifest(
                request=request,
                mode=mode,
                run_id=run_id,
                status="failed",
                validation_manifest_path=exc.manifest.get("manifest_path"),
                source_files=source_files,
                files_written=[],
                total_records=len(records),
                validated_records=len(exc.validated_records),
                error=str(exc),
                created_at=resolved_timestamp,
            )
            self._persist_manifest(manifest)
            raise

        sorted_records = sorted(validated, key=lambda item: (item["symbol"], item["timestamp"]))
        write_result = self.raw_writer.write_records(request.vendor, sorted_records)
        files_written = sorted(write_result.get("files", []), key=lambda item: item["path"])

        manifest = self._build_manifest(
            request=request,
            mode=mode,
            run_id=run_id,
            status="succeeded",
            validation_manifest_path=validation_manifest["manifest_path"],
            source_files=source_files,
            files_written=files_written,
            total_records=len(records),
            validated_records=len(sorted_records),
            created_at=resolved_timestamp,
        )
        self._persist_manifest(manifest)
        return manifest

    def _collect_records(self, request: EquityIngestionRequest, mode: Mode) -> List[Dict[str, Any]]:
        if mode == "rest":
            return asyncio.run(self.adapter.fetch_equity_ohlcv_rest(request))
        return list(self.adapter.stream_flat_file_equity_bars(request))

    def _source_file_metadata(self, request: EquityIngestionRequest) -> List[Dict[str, Any]]:
        metadata: List[Dict[str, Any]] = []
        for uri in request.flat_file_uris:
            path = self.adapter.flat_file_resolver(uri)  # type: ignore[attr-defined]
            metadata.append({"uri": uri, "hash": _hash_file(path)})
        return metadata

    def _build_manifest(
        self,
        *,
        request: EquityIngestionRequest,
        mode: Mode,
        run_id: str,
        status: str,
        validation_manifest_path: str | None,
        source_files: Sequence[Mapping[str, Any]],
        files_written: Sequence[Mapping[str, Any]],
        total_records: int,
        validated_records: int,
        error: str | None = None,
        created_at: str | None = None,
    ) -> Dict[str, Any]:
        failures: List[Mapping[str, Any]] = []
        if error:
            failures.append({"message": error})

        manifest: Dict[str, Any] = {
            "run_id": run_id,
            "vendor": request.vendor,
            "mode": mode,
            "status": status,
            "symbols": sorted(request.symbols),
            "start_date": request.start_date.isoformat(),
            "end_date": request.end_date.isoformat(),
            "record_counts": {
                "requested": total_records,
                "validated": validated_records,
            },
            "source_files": list(source_files),
            "files_written": list(files_written),
            "validation_manifest": validation_manifest_path,
            "created_at": created_at,
            "failures": failures,
        }
        if error:
            manifest["error"] = error
        return manifest

    def _persist_manifest(self, manifest: Mapping[str, Any]) -> Path:
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.manifest_dir / f"{manifest['run_id']}.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        return manifest_path

    def _resolve_creation_timestamp(self, request: EquityIngestionRequest, run_id: str) -> str:
        if request.options and request.options.get("creation_timestamp"):
            return str(request.options["creation_timestamp"])
        digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
        seconds = int(digest[:8], 16) % (365 * 24 * 60 * 60)
        base = datetime(2020, 1, 1, tzinfo=timezone.utc)
        return (base + timedelta(seconds=seconds)).isoformat()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


__all__ = ["EquityIngestionPipeline", "RAW_MANIFEST_DIR"]
