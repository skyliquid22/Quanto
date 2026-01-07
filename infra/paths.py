"""Shared helpers for resolving deterministic runtime storage roots."""

from __future__ import annotations

import os
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]


def get_repo_root() -> Path:
    """Return the repository root for callers that need absolute resolution."""

    return _REPO_ROOT


def get_data_root() -> Path:
    """Resolve the runtime data root honoring the QUANTO_DATA_ROOT override."""

    override = os.environ.get("QUANTO_DATA_ROOT")
    if override:
        return Path(override).expanduser()
    return get_repo_root() / ".quanto_data"


def raw_root(vendor: str | None = None) -> Path:
    """Base directory for raw-layer files, optionally scoped to a vendor."""

    base = get_data_root() / "raw"
    if vendor:
        return base / vendor
    return base


def processed_root(*segments: str) -> Path:
    """Base directory for processed-layer storage."""

    base = get_data_root() / "processed"
    return base.joinpath(*segments) if segments else base


def features_root(*segments: str) -> Path:
    """Base directory for derived feature storage."""

    base = get_data_root() / "features"
    return base.joinpath(*segments) if segments else base


def validation_manifest_root() -> Path:
    """Directory used for validation manifest persistence."""

    return get_data_root() / "validation" / "manifests"


__all__ = [
    "features_root",
    "get_data_root",
    "get_repo_root",
    "processed_root",
    "raw_root",
    "validation_manifest_root",
]
