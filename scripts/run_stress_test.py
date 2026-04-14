#!/usr/bin/env python3
"""CLI: Run a stress test suite against an experiment spec and evaluate gates.

Loads a StressTestConfig from YAML, resolves the agent's ExperimentSpec, runs
every scenario × seed combination through the simulation engine, evaluates
pass/fail gates, and prints a JSON summary. Exits with code 1 if any scenario
fails a hard gate.

Usage:
    python scripts/run_stress_test.py \\
        --stress-config configs/stress/example_stress_test.yml \\
        [--out-dir runs/stress/]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(PROJECT_ROOT))

from research.experiments.registry import ExperimentRegistry
from research.experiments.spec import ExperimentSpec
from research.stress.config import StressTestConfig
from research.stress.runner import StressTestRunner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a stress test suite against an experiment spec and evaluate gates."
    )
    parser.add_argument(
        "--stress-config",
        required=True,
        help="Path to the stress test configuration YAML.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Root directory for run artifacts (default: runs/stress/<config_name>).",
    )
    return parser.parse_args(argv)


def _print_summary(result) -> None:
    """Print a human-readable table to stderr, JSON payload to stdout."""
    counts = result.to_dict()["counts"]
    print(
        f"\nStress test: {result.config_name}  "
        f"[pass={counts['pass']} warn={counts['warn']} fail={counts['fail']} error={counts['error']}]",
        file=sys.stderr,
    )
    col_w = 40
    print(f"{'Scenario':<{col_w}}  {'Seed':>6}  {'Status'}", file=sys.stderr)
    print("-" * (col_w + 16), file=sys.stderr)
    for run in result.scenario_results:
        marker = {"pass": "OK", "warn": "WARN", "fail": "FAIL", "error": "ERR"}.get(run.status, run.status)
        print(f"{run.scenario_name:<{col_w}}  {run.seed:>6}  {marker}", file=sys.stderr)
        for gate in run.gate_results:
            if gate.status in {"fail", "warn"}:
                print(f"    [{gate.gate_name}] {gate.status.upper()}: {gate.reason}", file=sys.stderr)
        if run.error:
            print(f"    ERROR: {run.error}", file=sys.stderr)
    print(file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        config = StressTestConfig.from_file(args.stress_config)
    except Exception as exc:
        print(f"Failed to load stress config: {exc}", file=sys.stderr)
        return 1

    if not config.experiment_file:
        print("stress config must specify experiment_file.", file=sys.stderr)
        return 1

    try:
        spec = ExperimentSpec.from_file(config.experiment_file)
    except Exception as exc:
        print(f"Failed to load experiment spec from '{config.experiment_file}': {exc}", file=sys.stderr)
        return 1

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in config.name)
        out_dir = PROJECT_ROOT / "runs" / "stress" / safe_name

    registry = ExperimentRegistry()
    runner = StressTestRunner(config, spec, out_dir=out_dir, registry=registry)

    try:
        result = runner.run()
    except Exception as exc:
        print(f"Stress test runner failed unexpectedly: {exc}", file=sys.stderr)
        return 3

    _print_summary(result)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))

    return 0 if result.overall_status == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
