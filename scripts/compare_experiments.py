"""CLI: Compare Quanto experiments against a baseline."""

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
from research.monitoring.experiment_report import format_table


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

    # Print gate details summary table
    _print_gate_summary(gate_report)

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


def _print_gate_summary(gate_report: Any) -> None:
    """Print a formatted table summarizing gate evaluation results."""
    print("\nGate Details Summary")
    rows = []
    for eval_entry in gate_report.evaluations:
        status_symbol = "FAIL" if eval_entry.status == "fail" else ("WARN" if eval_entry.status == "warn" else "PASS")
        row = {
            "gate": eval_entry.gate_id,
            "metric": eval_entry.metric_id,
            "type": eval_entry.gate_type,
            "status": status_symbol,
            "candidate": _format_metric_value(eval_entry.observed.get("candidate")),
            "baseline": _format_metric_value(eval_entry.observed.get("baseline")),
            "delta": _format_delta(eval_entry.observed.get("delta")),
            "delta_pct": _format_delta_pct(eval_entry.observed.get("delta_pct")),
            "message": eval_entry.message,
        }
        rows.append(row)
    print(format_table(rows))


def _format_metric_value(value: float | None) -> str:
    """Format a metric value for display."""
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def _format_delta(value: float | None) -> str:
    """Format a delta value for display."""
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def _format_delta_pct(value: float | None) -> str:
    """Format a delta percentage value for display."""
    if value is None:
        return "n/a"
    return f"{value:.2f}%"


if __name__ == "__main__":
    raise SystemExit(main())
