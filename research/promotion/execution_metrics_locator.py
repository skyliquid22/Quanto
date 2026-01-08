"""Utilities for locating execution metrics across registry and shadow runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ExecutionMetricsLocatorResult:
    """Structured result describing execution metrics discovery."""

    experiment_id: str
    found: bool
    source: str | None
    metrics_path: Path | None
    execution_metrics_path: Path | None
    attempted: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "found": bool(self.found),
            "source": self.source,
            "metrics_path": str(self.metrics_path) if self.metrics_path else None,
            "execution_metrics_path": str(self.execution_metrics_path) if self.execution_metrics_path else None,
            "attempted": list(self.attempted),
        }


def locate_execution_metrics(
    experiment_id: str,
    *,
    registry_root: Path,
    shadow_root: Path,
) -> ExecutionMetricsLocatorResult:
    """Attempt to locate execution metrics for the provided experiment id."""

    attempted: List[str] = []
    registry_dir = registry_root / experiment_id
    if registry_dir.exists():
        result = _search_registry(experiment_id, registry_dir, attempted)
        if result:
            return result
    shadow_dir = shadow_root / experiment_id
    if shadow_dir.exists():
        result = _search_shadow(experiment_id, shadow_dir, attempted)
        if result:
            return result
    return ExecutionMetricsLocatorResult(
        experiment_id=experiment_id,
        found=False,
        source=None,
        metrics_path=None,
        execution_metrics_path=None,
        attempted=tuple(attempted),
    )


def _search_registry(
    experiment_id: str,
    base_dir: Path,
    attempted: List[str],
) -> ExecutionMetricsLocatorResult | None:
    for metrics_path in _ordered_registry_metrics(base_dir):
        result = _inspect_metrics_pair(
            experiment_id,
            metrics_path,
            sibling_candidates=[metrics_path.with_name("execution_metrics.json")],
            attempted=attempted,
            source="embedded",
        )
        if result:
            return result
    return None


def _search_shadow(
    experiment_id: str,
    shadow_dir: Path,
    attempted: List[str],
) -> ExecutionMetricsLocatorResult | None:
    run_dirs = _ordered_shadow_runs(shadow_dir)
    for run_dir in run_dirs:
        candidates = [
            run_dir / "execution_metrics.json",
            run_dir / "logs" / "execution_metrics.json",
        ]
        metrics_files = [
            run_dir / "metrics.json",
            run_dir / "logs" / "metrics.json",
        ]
        for metrics_path in metrics_files:
            result = _inspect_metrics_pair(
                experiment_id,
                metrics_path,
                sibling_candidates=candidates,
                attempted=attempted,
                source="shadow",
            )
            if result:
                return result
    return None


def _inspect_metrics_pair(
    experiment_id: str,
    metrics_path: Path,
    *,
    sibling_candidates: Sequence[Path],
    attempted: List[str],
    source: str,
) -> ExecutionMetricsLocatorResult | None:
    """Check a metrics file for embedded execution or sibling execution metrics."""

    attempted.append(str(metrics_path))
    if metrics_path.exists():
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and payload.get("execution"):
            return ExecutionMetricsLocatorResult(
                experiment_id=experiment_id,
                found=True,
                source="embedded" if source == "embedded" else source,
                metrics_path=metrics_path,
                execution_metrics_path=metrics_path,
                attempted=tuple(attempted),
            )
    for sibling in sibling_candidates:
        attempted.append(str(sibling))
        if sibling.exists():
            return ExecutionMetricsLocatorResult(
                experiment_id=experiment_id,
                found=True,
                source=source,
                metrics_path=metrics_path if metrics_path.exists() else None,
                execution_metrics_path=sibling,
                attempted=tuple(attempted),
            )
    return None


def _ordered_registry_metrics(base_dir: Path) -> Iterable[Path]:
    explicit = [
        base_dir / "evaluation" / "metrics.json",
        base_dir / "metrics.json",
    ]
    for path in explicit:
        if path.exists():
            yield path
    runs_dir = base_dir / "runs"
    if not runs_dir.exists():
        return
    run_metrics: List[Tuple[float, Path]] = []
    for run in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        metrics_path = run / "metrics.json"
        if metrics_path.exists():
            run_metrics.append((metrics_path.stat().st_mtime, metrics_path))
        logs_metrics = run / "logs" / "metrics.json"
        if logs_metrics.exists():
            run_metrics.append((logs_metrics.stat().st_mtime, logs_metrics))
    for _, path in sorted(run_metrics, key=lambda entry: (-entry[0], entry[1].as_posix())):
        yield path


def _ordered_shadow_runs(shadow_dir: Path) -> List[Path]:
    runs = []
    for run in shadow_dir.glob("*"):
        if not run.is_dir():
            continue
        steps = _resolve_steps_file(run)
        if not steps or not steps.exists():
            continue
        try:
            mtime = steps.stat().st_mtime
        except OSError:
            mtime = 0.0
        runs.append((mtime, run))
    runs.sort(key=lambda entry: (-entry[0], entry[1].as_posix()))
    return [entry[1] for entry in runs]


def _resolve_steps_file(run_dir: Path) -> Optional[Path]:
    candidates = [
        run_dir / "logs" / "steps.jsonl",
        run_dir / "steps.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


__all__ = ["ExecutionMetricsLocatorResult", "locate_execution_metrics"]
