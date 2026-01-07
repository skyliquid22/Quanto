#!/usr/bin/env python3
"""CLI for executing deterministic experiment sweeps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import path wiring
    sys.path.insert(0, str(PROJECT_ROOT))

from research.experiments import (  # noqa: E402
    ExperimentRegistry,
    SweepSpec,
    aggregate_sweep,
    run_sweep,
)
from research.experiments.ablation import (  # noqa: E402
    ensure_sweep_artifact_dir,
    write_sweep_experiments_artifact,
    write_sweep_spec_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Quanto experiment sweep declared in YAML/JSON.")
    parser.add_argument("--sweep", required=True, help="Path to the sweep spec file.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run experiments even when the registry already has completed results.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        sweep_spec = SweepSpec.from_file(args.sweep)
    except Exception as exc:  # pragma: no cover - validated in unit tests
        print(f"Failed to load sweep spec: {exc}", file=sys.stderr)
        return 1

    registry = ExperimentRegistry()
    try:
        sweep_result = run_sweep(sweep_spec, registry=registry, force=args.force)
    except Exception as exc:  # pragma: no cover - runtime errors depend on env
        print(f"Sweep execution failed: {exc}", file=sys.stderr)
        return 2

    artifact_dir = ensure_sweep_artifact_dir(sweep_spec.sweep_name)
    write_sweep_spec_artifact(sweep_spec, artifact_dir)
    write_sweep_experiments_artifact(sweep_result, artifact_dir)

    aggregation = aggregate_sweep(sweep_result, registry=registry)
    metrics_path = artifact_dir / "aggregate_metrics.json"
    regression_path = artifact_dir / "regression_summary.json"
    metrics_path.write_text(
        json.dumps(aggregation.metrics_summary, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    regression_path.write_text(
        json.dumps(aggregation.regression_summary, sort_keys=True, indent=2),
        encoding="utf-8",
    )

    _print_summary(sweep_result, aggregation.metrics_summary, aggregation.regression_summary)
    return 0


def _print_summary(result, metrics_summary, regression_summary) -> None:
    print(  # noqa: T201 - CLI feedback
        f"Sweep '{result.sweep_name}' finished: {result.completed} completed, {result.skipped} skipped."
    )
    print(  # noqa: T201 - CLI feedback
        f"Regression gates → passes: {regression_summary['passes']}, fails: {regression_summary['fails']}, "
        f"missing reports: {len(regression_summary['missing_reports'])}."
    )
    extrema = metrics_summary.get("extrema", {})
    for metric_id in ("performance.sharpe", "performance.total_return", "performance.max_drawdown"):
        metric_extrema = extrema.get(metric_id)
        if not metric_extrema:
            continue
        best = metric_extrema.get("best")
        worst = metric_extrema.get("worst")
        if not best or not worst:
            continue
        delta = best["value"] - worst["value"]
        print(  # noqa: T201 - CLI feedback
            f"{metric_id}: best={best['value']:.6f} ({best['experiment_id']}), "
            f"worst={worst['value']:.6f} ({worst['experiment_id']}), Δ={delta:.6f}"
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
