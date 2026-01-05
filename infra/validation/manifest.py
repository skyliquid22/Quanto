"""Helpers for persisting validation manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence


DEFAULT_MANIFEST_DIR = Path("data/validation/manifests")


def build_manifest(
    *,
    domain: str,
    schema_version: str,
    source_vendor: str,
    run_id: str,
    input_file_hashes: Sequence[str],
    total_records: int,
    valid_records: int,
    invalid_records: int,
    validation_status: str,
    creation_timestamp: str,
    errors: Iterable[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Compose the manifest payload that gets persisted alongside validation runs."""

    manifest: Dict[str, Any] = {
        "domain": domain,
        "schema_version": schema_version,
        "source_vendor": source_vendor,
        "run_id": run_id,
        "input_file_hashes": list(input_file_hashes),
        "total_records": total_records,
        "valid_records": valid_records,
        "invalid_records": invalid_records,
        "validation_status": validation_status,
        "creation_timestamp": creation_timestamp,
    }
    if errors:
        manifest["errors"] = list(errors)
    return manifest


def persist_manifest(manifest: Dict[str, Any], base_dir: Path | str = DEFAULT_MANIFEST_DIR) -> Path:
    """Write the manifest to disk and return the resolved path."""

    base_path = Path(base_dir)
    manifest_dir = base_path / manifest["domain"]
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{manifest['run_id']}.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    return manifest_path


__all__ = ["build_manifest", "persist_manifest", "DEFAULT_MANIFEST_DIR"]
