"""Utility helpers for Phase 1 validation harness notebooks."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess command and optionally raise when it fails."""

    result = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        env=dict(env) if env else None,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        message = f"Command {' '.join(cmd)} failed with code {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        raise RuntimeError(message)
    return result


def _hash_bytes(payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def hash_json_normalized(path: Path) -> str:
    """Compute a stable hash for the JSON payload at path."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return _hash_bytes(normalized)


def hash_jsonl_normalized(path: Path) -> str:
    """Compute a stable hash for JSON Lines content irrespective of whitespace."""

    normalized_lines: List[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        normalized_lines.append(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    canonical = "\n".join(normalized_lines).encode("utf-8")
    return _hash_bytes(canonical)


def find_first_existing(base_dir: Path, rel_candidates: Sequence[str]) -> Path:
    """Return the first existing path resolved against base_dir."""

    for rel in rel_candidates:
        candidate = (base_dir / rel).expanduser()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No candidates found under {base_dir}: {rel_candidates}")


def discover_run_artifacts(run_dir: Path) -> Dict[str, Path]:
    """Locate common run artifacts, accounting for slight path drift."""

    steps = find_first_existing(run_dir, ["logs/steps.jsonl", "steps.jsonl"])
    metrics = None
    execution_metrics = None
    summary = None
    state = None
    try:
        metrics = find_first_existing(run_dir, ["metrics.json"])
    except FileNotFoundError:
        pass
    try:
        execution_metrics = find_first_existing(run_dir, ["execution_metrics.json", "logs/execution_metrics.json"])
    except FileNotFoundError:
        pass
    try:
        summary = find_first_existing(run_dir, ["summary.json", "logs/summary.json"])
    except FileNotFoundError:
        pass
    try:
        state = find_first_existing(run_dir, ["state.json", "state/state.json"])
    except FileNotFoundError:
        pass
    artifacts: Dict[str, Path] = {"steps": steps}
    if metrics:
        artifacts["metrics"] = metrics
    if execution_metrics:
        artifacts["execution_metrics"] = execution_metrics
    if summary:
        artifacts["summary"] = summary
    if state:
        artifacts["state"] = state
    return artifacts


__all__ = [
    "discover_run_artifacts",
    "find_first_existing",
    "hash_json_normalized",
    "hash_jsonl_normalized",
    "run_cmd",
]
