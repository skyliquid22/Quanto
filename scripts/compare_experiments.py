"""CLI to compare experiment results and enforce regression gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(PROJECT_ROOT))

from research.experiments.comparator import compare_experiments
from research.experiments.registry import ExperimentRegistry
from research.experiments.regression import evaluate_gates, load_gate_rules


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Quanto experiments against a baseline.")
    parser.add_argument("--candidate", required=True, help="Candidate experiment_id to evaluate.")
    parser.add_argument(
        "--baseline",
        required=True,
        help="Baseline reference (experiment_id, experiment_name, or latest:<experiment_name>).",
    )
    parser.add_argument(
        "--gates",
        required=False,
        help="Optional path to a regression gate configuration (YAML/JSON). Defaults to built-ins when omitted.",
    )
    parser.add_argument(
        "--registry-root",
        required=False,
        help="Override the experiment registry root (defaults to .quanto_data/experiments).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    registry = ExperimentRegistry(root=Path(args.registry_root)) if args.registry_root else ExperimentRegistry()
    candidate_record = registry.get(args.candidate)
    baseline_record = registry.resolve_identifier(args.baseline)

    comparison = compare_experiments(
        candidate_record.experiment_id,
        baseline_record.experiment_id,
        registry=registry,
    )
    gate_rules = load_gate_rules(args.gates)
    gate_report = evaluate_gates(comparison, gate_rules)
    artifact_dir = candidate_record.root / "comparison"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _write_json(artifact_dir / "comparison.json", comparison.to_dict())
    _write_json(artifact_dir / "gate_report.json", gate_report.to_dict())

    if gate_report.overall_status == "fail":
        print(  # noqa: T201 - CLI feedback
            f"HARD gate failure for {comparison.candidate_experiment_id} vs baseline {comparison.baseline_experiment_id}."
        )
        return 1
    if gate_report.soft_warnings:
        print(  # noqa: T201 - CLI feedback
            f"Completed with {gate_report.soft_warnings} soft gate warning(s)."
        )
    else:
        print(  # noqa: T201 - CLI feedback
            f"Comparison succeeded for {comparison.candidate_experiment_id} vs {comparison.baseline_experiment_id}."
        )
    return 0


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
