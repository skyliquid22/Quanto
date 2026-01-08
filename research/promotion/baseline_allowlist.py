"""Filesystem-backed baseline execution allowlist."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

from infra.paths import get_data_root

ALLOWLIST_DIRNAME = "baseline_allowlist"


@dataclass(frozen=True)
class AllowlistEntry:
    """Structured representation of an allowlist record."""

    experiment_id: str
    path: Path
    payload: Mapping[str, object]


def allowlist_root(root: Path | None = None) -> Path:
    """Resolve the baseline allowlist root directory."""

    return Path(root) if root else get_data_root() / ALLOWLIST_DIRNAME


def entry_path(experiment_id: str, *, root: Path | None = None) -> Path:
    """Return the canonical path for the allowlist entry."""

    clean_id = experiment_id.strip()
    if not clean_id:
        raise ValueError("experiment_id must be provided for allowlist operations.")
    return allowlist_root(root) / f"{clean_id}.json"


def is_allowlisted(experiment_id: str, *, root: Path | None = None) -> bool:
    """Return True when experiment_id is recorded in the allowlist."""

    return entry_path(experiment_id, root=root).exists()


def load_entry(
    experiment_id: str,
    *,
    root: Path | None = None,
) -> AllowlistEntry | None:
    """Load a specific allowlist entry if it exists."""

    path = entry_path(experiment_id, root=root)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        payload = {}
    return AllowlistEntry(experiment_id=experiment_id, path=path, payload=payload)


def add(
    experiment_id: str,
    *,
    reason: str,
    notes: str | None = None,
    root: Path | None = None,
) -> Path:
    """Add an experiment_id to the baseline allowlist."""

    if not reason:
        raise ValueError("reason must be provided when adding to the allowlist.")
    path = entry_path(experiment_id, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, object] = {
        "experiment_id": experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "created_by": "cli",
    }
    if notes:
        payload["notes"] = notes
    path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return path


def remove(experiment_id: str, *, root: Path | None = None) -> bool:
    """Remove an experiment_id from the allowlist."""

    path = entry_path(experiment_id, root=root)
    if not path.exists():
        return False
    path.unlink()
    return True


def list_entries(*, root: Path | None = None) -> List[AllowlistEntry]:
    """List allowlist entries ordered by creation time, then path."""

    base = allowlist_root(root)
    if not base.exists():
        return []
    entries: List[Tuple[str, AllowlistEntry]] = []
    for candidate in base.glob("*.json"):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        experiment_id = payload.get("experiment_id") or candidate.stem
        created = str(payload.get("created_at") or "")
        entries.append(
            (
                created,
                AllowlistEntry(
                    experiment_id=str(experiment_id),
                    path=candidate,
                    payload=payload,
                ),
            )
        )
    entries.sort(key=lambda entry: (entry[0], entry[1].path.as_posix()))
    return [entry for _, entry in entries]


__all__ = [
    "AllowlistEntry",
    "add",
    "allowlist_root",
    "entry_path",
    "is_allowlisted",
    "list_entries",
    "load_entry",
    "remove",
]
