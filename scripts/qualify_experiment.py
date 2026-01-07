#!/usr/bin/env python3
"""CLI to evaluate experiment promotion readiness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - CLI import path
    sys.path.insert(0, str(PROJECT_ROOT))

from research.experiments.registry import ExperimentRegistry
from research.experiments.regression import load_gate_rules
from research.promotion.criteria import QualificationCriteria, RegimeDiagnosticsCriteria, SweepRobustnessCriteria
from research.promotion.qualify import run_qualification


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qualify an experiment for promotion gating.")
    parser.add_argument("--experiment-id", required=True, help="Experiment identifier to qualify.")
    parser.add_argument(
        "--baseline",
        required=True,
        help="Baseline reference (experiment id, experiment name, or latest:<experiment_name>).",
    )
    parser.add_argument("--sweep-name", help="Optional sweep name providing robustness context.")
    parser.add_argument("--registry-root", help="Override the experiment registry root.")
    parser.add_argument("--gate-config", help="Optional regression gate configuration (JSON/YAML).")
    parser.add_argument("--sweep-root", help="Override the sweep artifact root (.quanto_data/sweeps).")
    parser.add_argument("--max-drawdown", type=float, default=1.0, help="Maximum allowed drawdown ratio (default=1.0).")
    parser.add_argument(
        "--max-turnover",
        type=float,
        default=1.0,
        help="Maximum allowed turnover according to turnover_metric (default=1.0).",
    )
    parser.add_argument(
        "--turnover-metric",
        default="trading.turnover_1d_mean",
        help="Metric id used for turnover sanity enforcement.",
    )
    parser.add_argument(
        "--sweep-min-completed",
        type=int,
        help="Optional minimum completed experiments for sweep robustness enforcement.",
    )
    parser.add_argument(
        "--sweep-max-failures",
        type=int,
        help="Optional maximum failures allowed across sweep regression summaries.",
    )
    parser.add_argument(
        "--sweep-severity",
        choices=("hard", "soft"),
        default="soft",
        help="Severity applied when sweep robustness thresholds are violated.",
    )
    parser.add_argument(
        "--min-modes",
        type=int,
        help="Optional minimum number of active modes required for regime diagnostics.",
    )
    parser.add_argument(
        "--min-mode-fraction",
        type=float,
        help="Optional minimum fraction per mode (0-1) required for regime diagnostics.",
    )
    parser.add_argument(
        "--regime-severity",
        choices=("hard", "soft"),
        default="soft",
        help="Severity applied when regime diagnostics violate thresholds.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    registry = ExperimentRegistry(root=Path(args.registry_root)) if args.registry_root else ExperimentRegistry()
    gate_rules = load_gate_rules(args.gate_config)
    sweep_criteria = None
    if args.sweep_min_completed is not None or args.sweep_max_failures is not None:
        sweep_criteria = SweepRobustnessCriteria(
            min_completed=args.sweep_min_completed,
            max_failures=args.sweep_max_failures,
            severity=args.sweep_severity,
        )
    regime_criteria = None
    if args.min_modes is not None or args.min_mode_fraction is not None:
        regime_criteria = RegimeDiagnosticsCriteria(
            min_modes=args.min_modes,
            min_mode_fraction=args.min_mode_fraction,
            severity=args.regime_severity,
        )
    criteria = QualificationCriteria(
        max_drawdown=args.max_drawdown,
        max_turnover=args.max_turnover,
        turnover_metric=args.turnover_metric,
        sweep_root=Path(args.sweep_root) if args.sweep_root else None,
        sweep_robustness=sweep_criteria,
        regime_diagnostics=regime_criteria,
    )
    result = run_qualification(
        args.experiment_id,
        args.baseline,
        registry=registry,
        gate_rules=gate_rules,
        sweep_name=args.sweep_name,
        criteria=criteria,
    )
    if result.report.passed:
        print(  # noqa: T201 - CLI feedback
            f"Qualification passed for {result.report.experiment_id}. Report: {result.report_path}"
        )
        return 0
    print(  # noqa: T201 - CLI feedback
        f"Qualification FAILED for {result.report.experiment_id}. Reasons: "
        f"{result.report.failed_hard}. Report: {result.report_path}"
    )
    if result.report.failed_soft:
        print(  # noqa: T201 - CLI feedback
            f"Soft warnings: {result.report.failed_soft}"
        )
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
