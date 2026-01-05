"""Lineage helpers used by canonical reconciliation builder."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class LineageInput:
    """Represents a single raw input file participating in reconciliation."""

    path: str
    vendor: str
    domain: str
    file_hash: str
    record_count: int | None = None


@dataclass(frozen=True)
class LineageOutput:
    """Describes a canonical artifact emitted by the builder."""

    path: str
    file_hash: str
    record_count: int | None = None


def compute_file_hash(path: Path | str) -> str:
    """Return a deterministic SHA-256 hash for a file on disk."""

    resolved = Path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def build_lineage_payload(
    *,
    domain: str,
    run_id: str,
    inputs: Sequence[LineageInput] | Iterable[LineageInput],
    outputs: Sequence[LineageOutput] | Iterable[LineageOutput],
    metadata: Mapping[str, object] | None = None,
) -> dict:
    """Compose canonical lineage payload wiring inputs to outputs."""

    payload = {
        "domain": domain,
        "run_id": run_id,
        "inputs": [asdict(item) for item in inputs],
        "outputs": [asdict(item) for item in outputs],
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    return payload


__all__ = ["LineageInput", "LineageOutput", "build_lineage_payload", "compute_file_hash"]
