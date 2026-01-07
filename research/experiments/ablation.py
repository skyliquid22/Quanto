"""Data structures and artifact helpers for experiment sweeps."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

from research.experiments.spec import ExperimentSpec
from research.experiments.sweep import SweepSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SWEEP_ROOT = PROJECT_ROOT / ".quanto_data" / "sweeps"


@dataclass(frozen=True)
class SweepExperiment:
    """Sweep member metadata captured for aggregation and artifacts."""

    experiment_id: str
    spec: ExperimentSpec
    dimensions: Mapping[str, Any]
    status: str  # "completed" or "skipped"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "status": self.status,
            "dimensions": {key: self.dimensions[key] for key in self.dimensions},
            "spec": self.spec.to_dict(),
        }


@dataclass(frozen=True)
class SweepResult:
    """Structured result returned by run_sweep."""

    sweep_spec: SweepSpec
    experiments: tuple[SweepExperiment, ...]

    @property
    def sweep_name(self) -> str:
        return self.sweep_spec.sweep_name

    @property
    def dimension_names(self) -> tuple[str, ...]:
        return self.sweep_spec.dimension_names

    @property
    def total_requested(self) -> int:
        return len(self.experiments)

    @property
    def completed(self) -> int:
        return sum(1 for entry in self.experiments if entry.status == "completed")

    @property
    def skipped(self) -> int:
        return sum(1 for entry in self.experiments if entry.status == "skipped")


def ensure_sweep_artifact_dir(sweep_name: str, root: Path | None = None) -> Path:
    base = Path(root) if root else DEFAULT_SWEEP_ROOT
    target = base / sweep_name
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_sweep_spec_artifact(sweep_spec: SweepSpec, directory: Path) -> Path:
    path = directory / "sweep_spec.json"
    payload = sweep_spec.to_dict()
    path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return path


def write_sweep_experiments_artifact(result: SweepResult, directory: Path) -> Path:
    path = directory / "experiments.json"
    experiments_payload: List[Dict[str, Any]] = [entry.to_dict() for entry in result.experiments]
    payload = {
        "sweep_name": result.sweep_name,
        "dimension_names": list(result.dimension_names),
        "total_experiments": result.total_requested,
        "completed": result.completed,
        "skipped": result.skipped,
        "experiments": experiments_payload,
    }
    path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return path


def sweep_experiment_lookup(result: SweepResult) -> Dict[str, SweepExperiment]:
    return {entry.experiment_id: entry for entry in result.experiments}


__all__ = [
    "DEFAULT_SWEEP_ROOT",
    "SweepExperiment",
    "SweepResult",
    "ensure_sweep_artifact_dir",
    "write_sweep_spec_artifact",
    "write_sweep_experiments_artifact",
    "sweep_experiment_lookup",
]
