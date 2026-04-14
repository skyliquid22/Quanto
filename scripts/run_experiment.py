#!/usr/bin/env python3
"""CLI: Run a deterministic Quanto experiment from a spec file."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(PROJECT_ROOT))

from monitoring.trace_emitter import emit_trace
from research.experiments import (
    ExperimentRegistry,
    ExperimentSpec,
    ExperimentAlreadyExistsError,
    run_experiment,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a deterministic Quanto experiment from a spec file.")
    parser.add_argument("--spec", required=True, help="Path to the experiment spec (YAML).")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if the experiment ID already exists (registry directory is reset).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        spec = ExperimentSpec.from_file(args.spec)
    except Exception as exc:  # pragma: no cover - argument errors exercised elsewhere
        print(f"Failed to load experiment spec: {exc}", file=sys.stderr)
        return 1
    registry = ExperimentRegistry()
    t0 = time.monotonic()
    try:
        result = run_experiment(spec, registry=registry, force=args.force)
    except ExperimentAlreadyExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - runtime errors depend on environment
        emit_trace(
            initializer="run_experiment_cli",
            task_title=f"experiment {spec.experiment_name}",
            run_id=spec.experiment_id,
            status="failed",
            error=str(exc),
            duration_s=time.monotonic() - t0,
        )
        print(f"Experiment failed: {exc}", file=sys.stderr)
        return 3
    emit_trace(
        initializer="run_experiment_cli",
        task_title=f"experiment {spec.experiment_name}",
        run_id=result.experiment_id,
        status="succeeded",
        duration_s=time.monotonic() - t0,
        artifacts=[str(result.metrics_path), str(result.rollout_artifact)],
    )
    payload = {
        "experiment_id": result.experiment_id,
        "registry_path": str(result.registry_paths.root),
        "metrics_path": str(result.metrics_path),
        "rollout_artifact": str(result.rollout_artifact),
        "training_artifacts": {key: str(path) for key, path in result.training_artifacts.items()},
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
