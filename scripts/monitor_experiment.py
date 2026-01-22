#!/usr/bin/env python3
"""Render a user-facing experiment metrics report with plots."""

from __future__ import annotations

import argparse
from pathlib import Path

from research.monitoring.experiment_report import (
    format_table,
    generate_experiment_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a metrics report for a single experiment.")
    parser.add_argument("--experiment-id", required=True, help="Experiment ID to summarize.")
    parser.add_argument("--output-dir", help="Optional plot output directory (CLI mode only).")
    parser.add_argument("--strict", action="store_true", help="Fail if optional artifacts are missing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else None
    report = generate_experiment_report(
        args.experiment_id,
        output_dir=output_dir,
        strict=args.strict,
    )

    metadata = report.get("metadata") or {}
    print("Experiment Summary")
    for key, value in metadata.items():
        print(f"- {key}: {value}")

    print("\nMetrics")
    print(format_table(report.get("metrics_table")))

    comparison = report.get("comparison_table")
    if comparison is not None and not comparison.empty:
        print("\nBaseline Comparison")
        print(format_table(comparison))

    figures = report.get("figures") or {}
    if figures:
        print("\nSaved plots:")
        for label, path in figures.items():
            print(f"- {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
