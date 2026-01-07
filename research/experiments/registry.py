"""Experiment registry rooted under .quanto_data/experiments."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from research.experiments.spec import ExperimentSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_ROOT = PROJECT_ROOT / ".quanto_data" / "experiments"


class ExperimentAlreadyExistsError(RuntimeError):
    """Raised when attempting to reuse an occupied registry slot."""


@dataclass(frozen=True)
class ExperimentPaths:
    """Resolved registry directories for a single experiment."""

    root: Path
    spec_dir: Path
    runs_dir: Path
    evaluation_dir: Path
    logs_dir: Path
    promotion_dir: Path


@dataclass(frozen=True)
class ExperimentRecord:
    """Resolved paths for an existing experiment with completed artifacts."""

    experiment_id: str
    root: Path
    spec_path: Path
    evaluation_dir: Path
    logs_dir: Path
    metrics_path: Path
    promotion_dir: Path


class ExperimentRegistry:
    """Filesystem-backed experiment registry."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else DEFAULT_REGISTRY_ROOT

    def has_completed(self, experiment_id: str) -> bool:
        """Return True when the registry already contains completed artifacts."""

        try:
            self.get(experiment_id)
        except FileNotFoundError:
            return False
        else:
            return True

    def prepare(self, experiment_id: str, *, force: bool = False) -> ExperimentPaths:
        base = self.root / experiment_id
        if base.exists():
            if not force:
                raise ExperimentAlreadyExistsError(
                    f"Experiment '{experiment_id}' already exists. Use --force to re-run."
                )
            shutil.rmtree(base)
        spec_dir = base / "spec"
        runs_dir = base / "runs"
        evaluation_dir = base / "evaluation"
        promotion_dir = base / "promotion"
        logs_dir = base / "logs"
        for path in (spec_dir, runs_dir, evaluation_dir, logs_dir, promotion_dir):
            path.mkdir(parents=True, exist_ok=True)
        return ExperimentPaths(
            root=base,
            spec_dir=spec_dir,
            runs_dir=runs_dir,
            evaluation_dir=evaluation_dir,
            logs_dir=logs_dir,
            promotion_dir=promotion_dir,
        )

    def write_spec(self, spec: ExperimentSpec, paths: ExperimentPaths) -> Path:
        spec_path = paths.spec_dir / "experiment_spec.json"
        spec_path.write_text(spec.canonical_json + "\n", encoding="utf-8")
        return spec_path

    def write_run_summary(self, paths: ExperimentPaths, payload: Dict[str, Any]) -> Path:
        logs_path = paths.logs_dir / "run_summary.json"
        ordered = dict(sorted(payload.items()))
        ordered["recorded_at"] = datetime.now(timezone.utc).isoformat()
        logs_path.write_text(json.dumps(ordered, sort_keys=True, indent=2), encoding="utf-8")
        return logs_path

    def get(self, experiment_id: str) -> ExperimentRecord:
        """Return paths for an existing experiment, ensuring metrics exist."""

        experiment_id = experiment_id.strip()
        if not experiment_id:
            raise ValueError("experiment_id must be provided.")
        base = self.root / experiment_id
        if not base.exists():
            raise FileNotFoundError(f"Experiment '{experiment_id}' not found in registry: {self.root}")
        spec_path = base / "spec" / "experiment_spec.json"
        evaluation_dir = base / "evaluation"
        metrics_path = evaluation_dir / "metrics.json"
        logs_dir = base / "logs"
        promotion_dir = base / "promotion"
        if not spec_path.exists():
            raise FileNotFoundError(f"Spec not found for experiment '{experiment_id}': {spec_path}")
        if not metrics_path.exists():
            raise FileNotFoundError(f"metrics.json not found for experiment '{experiment_id}'.")
        return ExperimentRecord(
            experiment_id=experiment_id,
            root=base,
            spec_path=spec_path,
            evaluation_dir=evaluation_dir,
            logs_dir=logs_dir,
            metrics_path=metrics_path,
            promotion_dir=promotion_dir,
        )

    def resolve_identifier(self, identifier: str) -> ExperimentRecord:
        """Resolve a baseline reference to a concrete experiment record."""

        if not identifier:
            raise ValueError("Baseline identifier must be provided.")
        token = identifier.strip()
        if token.startswith("latest:"):
            experiment_name = token.split(":", 1)[1].strip()
            return self.latest_by_name(experiment_name)
        candidate = self.root / token
        if candidate.is_dir():
            return self.get(token)
        return self.latest_by_name(token)

    def latest_by_name(self, experiment_name: str) -> ExperimentRecord:
        """Return the most recent experiment for the provided experiment_name."""

        if not experiment_name:
            raise ValueError("experiment_name must be provided when resolving latest baseline.")
        matches = list(self._iter_experiments_by_name(experiment_name.strip()))
        if not matches:
            raise FileNotFoundError(
                f"No experiments found for name '{experiment_name}' under registry {self.root}."
            )
        matches.sort(key=lambda entry: (entry[0], entry[1]))
        latest_id = matches[-1][1]
        return self.get(latest_id)

    def _iter_experiments_by_name(self, experiment_name: str) -> Iterable[Tuple[datetime, str]]:
        for entry in sorted(self.root.glob("*")):
            if not entry.is_dir():
                continue
            spec_path = entry / "spec" / "experiment_spec.json"
            if not spec_path.exists():
                continue
            try:
                spec_payload = json.loads(spec_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if spec_payload.get("experiment_name") != experiment_name:
                continue
            recorded_at = self._recorded_at(entry / "logs" / "run_summary.json")
            yield recorded_at, entry.name

    @staticmethod
    def _recorded_at(path: Path) -> datetime:
        if not path.exists():
            return datetime.min.replace(tzinfo=timezone.utc)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return datetime.min.replace(tzinfo=timezone.utc)
        recorded = payload.get("recorded_at")
        if not recorded:
            return datetime.min.replace(tzinfo=timezone.utc)
        try:
            return datetime.fromisoformat(recorded)
        except ValueError:
            return datetime.min.replace(tzinfo=timezone.utc)

    def resolve_with_spec(self, experiment_id: str) -> Tuple[ExperimentRecord, ExperimentSpec]:
        """Return the experiment record alongside its parsed specification."""

        record = self.get(experiment_id)
        spec = ExperimentSpec.from_file(record.spec_path)
        return record, spec


__all__ = ["ExperimentAlreadyExistsError", "ExperimentPaths", "ExperimentRecord", "ExperimentRegistry"]
