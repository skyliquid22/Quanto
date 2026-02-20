"""Option ingestion pipeline coordinating reference, OHLCV, and open interest."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from infra.paths import raw_root
from infra.storage.raw_writer import RawOptionsWriter
from infra.validation import ValidationError, validate_records

from .adapters import OptionReferenceIngestionRequest, OptionTimeseriesIngestionRequest, PolygonOptionsAdapter
from .router import IngestionRouter, Mode

OptionDomain = str


@dataclass(frozen=True)
class OptionPartition:
    """Single unit of work for a particular option domain."""

    partition_id: str
    request: OptionReferenceIngestionRequest | OptionTimeseriesIngestionRequest
    mode: Mode | None = None

    def __post_init__(self) -> None:
        if not self.partition_id:
            raise ValueError("partition_id must be provided")

    @property
    def domain(self) -> OptionDomain:
        return self.request.domain

    @property
    def symbols(self) -> Sequence[str]:
        return self.request.symbols

    @classmethod
    def from_mapping(
        cls,
        domain: OptionDomain,
        payload: Mapping[str, Any],
        *,
        vendor: str,
    ) -> "OptionPartition":
        partition_id = str(payload["partition_id"])
        mode = payload.get("mode")
        if domain == "option_contract_reference":
            if "as_of_date" not in payload:
                raise ValueError("reference partition requires as_of_date")
            request = OptionReferenceIngestionRequest(
                underlying_symbols=tuple(payload.get("underlying_symbols", [])),
                as_of_date=date.fromisoformat(payload["as_of_date"]),
                flat_file_uris=tuple(payload.get("flat_file_uris", [])),
                vendor=vendor,
                options=payload.get("options", {}),
            )
        else:
            request = OptionTimeseriesIngestionRequest(
                domain=domain,
                option_symbols=tuple(payload.get("option_symbols", [])),
                start_date=date.fromisoformat(payload["start_date"]),
                end_date=date.fromisoformat(payload["end_date"]),
                flat_file_uris=tuple(payload.get("flat_file_uris", [])),
                vendor=vendor,
                options=payload.get("options", {}),
            )
        return cls(partition_id=partition_id, request=request, mode=mode)


@dataclass(frozen=True)
class OptionsIngestionPlan:
    """Set of partitions per domain for a single run."""

    vendor: str = "polygon"
    reference: Sequence[OptionPartition] = field(default_factory=tuple)
    ohlcv: Sequence[OptionPartition] = field(default_factory=tuple)
    open_interest: Sequence[OptionPartition] = field(default_factory=tuple)

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any], *, default_vendor: str = "polygon") -> "OptionsIngestionPlan":
        vendor = str(config.get("vendor", default_vendor))
        builder = _PartitionBuilder(vendor=vendor)
        reference = builder.build_partitions("option_contract_reference", config.get("reference"))
        ohlcv = builder.build_partitions("option_contract_ohlcv", config.get("ohlcv"))
        open_interest = builder.build_partitions("option_open_interest", config.get("open_interest"))
        return cls(vendor=vendor, reference=reference, ohlcv=ohlcv, open_interest=open_interest)


@dataclass
class _PartitionBuilder:
    vendor: str

    def build_partitions(
        self,
        domain: OptionDomain,
        section: Mapping[str, Any] | None,
    ) -> Sequence[OptionPartition]:
        if not section:
            return tuple()
        partitions = section.get("partitions")
        if not partitions:
            return tuple()
        built = [OptionPartition.from_mapping(domain, mapping, vendor=self.vendor) for mapping in partitions]
        ids = [part.partition_id for part in built]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate partition ids detected for domain {domain}")
        return tuple(built)


class OptionsIngestionPipeline:
    """Coordinates adapter routing, validation, checkpointing, and storage for options."""

    def __init__(
        self,
        *,
        adapter: PolygonOptionsAdapter,
        router: IngestionRouter | None = None,
        raw_writer: RawOptionsWriter | None = None,
        manifest_base_dir: Path | str | None = None,
        checkpoint_dir: Path | str | None = None,
    ) -> None:
        self.adapter = adapter
        self.router = router or IngestionRouter()
        self.raw_writer = raw_writer or RawOptionsWriter()
        manifest_base = manifest_base_dir if manifest_base_dir is not None else raw_root()
        self.manifest_base_dir = Path(manifest_base)
        default_checkpoint = self.manifest_base_dir / "checkpoints" / "options"
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else default_checkpoint

    def run(self, plan: OptionsIngestionPlan, *, run_id: str) -> Dict[str, Any]:
        if not run_id:
            raise ValueError("run_id must be provided")
        results: Dict[str, Any] = {}
        statuses: List[str] = []
        for domain, partitions in (
            ("option_contract_reference", plan.reference),
            ("option_contract_ohlcv", plan.ohlcv),
            ("option_open_interest", plan.open_interest),
        ):
            if not partitions:
                continue
            manifest = self._run_domain(domain, partitions, vendor=plan.vendor, run_id=run_id)
            results[domain] = manifest
            statuses.append(manifest["status"])
        overall_status = "succeeded" if all(status == "succeeded" for status in statuses or [True]) else "failed"
        return {"run_id": run_id, "vendor": plan.vendor, "status": overall_status, "domains": results}

    def _run_domain(
        self,
        domain: OptionDomain,
        partitions: Sequence[OptionPartition],
        *,
        vendor: str,
        run_id: str,
    ) -> Dict[str, Any]:
        checkpoint = self._load_checkpoint(domain, vendor, run_id)
        stored_partitions: MutableMapping[str, Dict[str, Any]] = checkpoint.setdefault("partitions", {})
        partition_results: List[Dict[str, Any]] = []
        for partition in partitions:
            existing = stored_partitions.get(partition.partition_id)
            if existing and existing.get("status") == "completed":
                partition_results.append(dict(existing))
                continue

            mode = partition.mode or self._route_partition(domain, partition.request)
            source_files = self._source_file_metadata(partition.request)
            records = self._collect_records(domain, partition.request, mode)
            creation_ts = self._resolve_creation_timestamp(f"{run_id}:{partition.partition_id}")
            validation_config = {
                "manifest_base_path": self._manifest_dir(domain, vendor) / "validation",
                "input_file_hashes": [item["hash"] for item in source_files],
                "creation_timestamp": creation_ts,
            }
            try:
                validated, validation_manifest = validate_records(
                    domain,
                    records,
                    source_vendor=partition.request.vendor,
                    run_id=f"{run_id}:{partition.partition_id}",
                    config=validation_config,
                )
            except ValidationError as exc:
                failure_entry = self._build_partition_failure(
                    partition,
                    mode,
                    source_files,
                    records,
                    exc,
                    creation_ts,
                )
                manifest = self._build_domain_manifest(
                    domain=domain,
                    vendor=vendor,
                    run_id=run_id,
                    partitions=partition_results + [failure_entry],
                    status="failed",
                    failure_message=str(exc),
                )
                manifest_path = self._persist_manifest(domain, vendor, run_id, manifest)
                manifest["manifest_path"] = str(manifest_path)
                raise

            write_result = self._write_records(domain, vendor, validated, partition.request)
            files_written = sorted(write_result.get("files", []), key=lambda item: item["path"])
            partition_entry = {
                "partition_id": partition.partition_id,
                "domain": domain,
                "mode": mode,
                "status": "completed",
                "symbols": sorted(partition.symbols),
                "source_files": source_files,
                "files_written": files_written,
                "record_counts": {"requested": len(records), "validated": len(validated)},
                "validation_manifest": validation_manifest["manifest_path"],
                "created_at": creation_ts,
            }
            partition_results.append(partition_entry)
            stored_partitions[partition.partition_id] = dict(partition_entry)
            self._persist_checkpoint(domain, vendor, run_id, checkpoint)

        manifest = self._build_domain_manifest(
            domain=domain,
            vendor=vendor,
            run_id=run_id,
            partitions=partition_results,
            status="succeeded",
        )
        manifest_path = self._persist_manifest(domain, vendor, run_id, manifest)
        manifest["manifest_path"] = str(manifest_path)
        return manifest

    def _build_partition_failure(
        self,
        partition: OptionPartition,
        mode: Mode,
        source_files: Sequence[Mapping[str, Any]],
        records: Sequence[Mapping[str, Any]],
        exc: ValidationError,
        creation_ts: str,
    ) -> Dict[str, Any]:
        return {
            "partition_id": partition.partition_id,
            "domain": partition.domain,
            "mode": mode,
            "status": "failed",
            "symbols": sorted(partition.symbols),
            "source_files": list(source_files),
            "files_written": [],
            "record_counts": {"requested": len(records), "validated": len(exc.validated_records)},
            "validation_manifest": exc.manifest.get("manifest_path"),
            "created_at": creation_ts,
            "failures": exc.errors,
        }

    def _build_domain_manifest(
        self,
        *,
        domain: OptionDomain,
        vendor: str,
        run_id: str,
        partitions: Sequence[Mapping[str, Any]],
        status: str,
        failure_message: str | None = None,
    ) -> Dict[str, Any]:
        ordered_partitions = sorted(partitions, key=lambda item: item["partition_id"])
        files_written: List[Mapping[str, Any]] = []
        source_files: List[Mapping[str, Any]] = []
        validation_manifests: List[str] = []
        total_requested = 0
        total_validated = 0
        for entry in ordered_partitions:
            files_written.extend(entry.get("files_written", []))
            source_files.extend(entry.get("source_files", []))
            if entry.get("validation_manifest"):
                validation_manifests.append(entry["validation_manifest"])
            counts = entry.get("record_counts") or {}
            total_requested += counts.get("requested", 0)
            total_validated += counts.get("validated", 0)
        manifest = {
            "run_id": run_id,
            "vendor": vendor,
            "domain": domain,
            "status": status,
            "partitions": ordered_partitions,
            "files_written": sorted(files_written, key=lambda item: item["path"]),
            "source_files": source_files,
            "record_counts": {"requested": total_requested, "validated": total_validated},
            "validation_manifests": validation_manifests,
            "created_at": self._resolve_creation_timestamp(run_id),
            "failures": [],
        }
        if failure_message:
            manifest["error"] = failure_message
            manifest["failures"].append({"message": failure_message})
        return manifest

    def _route_partition(self, domain: OptionDomain, request: Any) -> Mode:
        if domain == "option_contract_reference":
            return self.router.route_option_contract_reference(request)
        if domain == "option_contract_ohlcv":
            return self.router.route_option_contract_ohlcv(request)
        return self.router.route_option_open_interest(request)

    def _collect_records(
        self,
        domain: OptionDomain,
        request: OptionReferenceIngestionRequest | OptionTimeseriesIngestionRequest,
        mode: Mode,
    ) -> List[Dict[str, Any]]:
        if mode == "rest":
            if domain == "option_contract_reference":
                return asyncio.run(self.adapter.fetch_contract_reference_rest(request))
            if domain == "option_contract_ohlcv":
                return asyncio.run(self.adapter.fetch_option_ohlcv_rest(request))
            return asyncio.run(self.adapter.fetch_option_open_interest_rest(request))
        if domain == "option_contract_reference":
            return list(self.adapter.stream_reference_flat_files(request))
        if domain == "option_contract_ohlcv":
            return list(self.adapter.stream_option_ohlcv_flat_files(request))
        return list(self.adapter.stream_option_open_interest_flat_files(request))

    def _source_file_metadata(
        self,
        request: OptionReferenceIngestionRequest | OptionTimeseriesIngestionRequest,
    ) -> List[Dict[str, Any]]:
        metadata: List[Dict[str, Any]] = []
        for uri in request.flat_file_uris:
            path = self.adapter.flat_file_resolver(uri)  # type: ignore[attr-defined]
            metadata.append({"uri": uri, "hash": _hash_file(path)})
        return metadata

    def _write_records(
        self,
        domain: OptionDomain,
        vendor: str,
        records: Sequence[Mapping[str, Any]],
        request: OptionReferenceIngestionRequest | OptionTimeseriesIngestionRequest,
    ) -> MutableMapping[str, Any]:
        if domain == "option_contract_reference":
            return self.raw_writer.write_contract_reference(vendor, records, snapshot_date=request.as_of_date)
        if domain == "option_contract_ohlcv":
            return self.raw_writer.write_option_ohlcv(vendor, records)
        return self.raw_writer.write_option_open_interest(vendor, records)

    def _manifest_dir(self, domain: OptionDomain, vendor: str) -> Path:
        return self.manifest_base_dir / vendor / domain / "manifests"

    def _persist_manifest(
        self,
        domain: OptionDomain,
        vendor: str,
        run_id: str,
        manifest: Mapping[str, Any],
    ) -> Path:
        directory = self._manifest_dir(domain, vendor)
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path = directory / f"{run_id}.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        return manifest_path

    def _checkpoint_path(self, domain: OptionDomain, vendor: str, run_id: str) -> Path:
        return self.checkpoint_dir / vendor / domain / f"{run_id}.json"

    def _load_checkpoint(self, domain: OptionDomain, vendor: str, run_id: str) -> Dict[str, Any]:
        path = self._checkpoint_path(domain, vendor, run_id)
        if path.exists():
            return json.loads(path.read_text())
        return {"run_id": run_id, "domain": domain, "vendor": vendor, "partitions": {}}

    def _persist_checkpoint(
        self,
        domain: OptionDomain,
        vendor: str,
        run_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        path = self._checkpoint_path(domain, vendor, run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def _resolve_creation_timestamp(self, seed: str) -> str:
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        seconds = int(digest[:8], 16) % (365 * 24 * 60 * 60)
        base = datetime(2020, 1, 1, tzinfo=timezone.utc)
        return (base + timedelta(seconds=seconds)).isoformat()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


__all__ = [
    "OptionPartition",
    "OptionsIngestionPlan",
    "OptionsIngestionPipeline",
]
