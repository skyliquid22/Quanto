#!/usr/bin/env python3
"""Generate evaluation timeseries artifacts and compute regime slices for an experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(PROJECT_ROOT))

from infra.paths import get_data_root
from research.eval.regime_slicing import compute_regime_slices
from research.eval.timeseries import extract_timeseries_from_rollout, load_rollout_json, write_timeseries_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute regime slices from an experiment rollout.")
    parser.add_argument("--experiment-id", required=True, help="Experiment identifier (hash).")
    parser.add_argument("--data-root", help="Override QUANTO data root.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate timeseries.json even if it already exists.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = Path(args.data_root).expanduser() if args.data_root else get_data_root()
    experiment_dir = data_root / "experiments" / args.experiment_id
    if not experiment_dir.exists():
        print(f"Experiment directory not found: {experiment_dir}", file=sys.stderr)
        return 1

    evaluation_dir = experiment_dir / "evaluation"
    timeseries_path = evaluation_dir / "timeseries.json"
    if args.force or not timeseries_path.exists():
        rollout = load_rollout_json(experiment_dir)
        timeseries = extract_timeseries_from_rollout(rollout)
        write_timeseries_json(experiment_dir, timeseries)
    else:
        timeseries = json.loads(timeseries_path.read_text(encoding="utf-8"))

    regime_payload = timeseries.get("regime")
    if not regime_payload:
        print("No regime series available; wrote timeseries.json only.", file=sys.stderr)
        return 0

    result = compute_regime_slices(
        regime_series=regime_payload.get("series"),
        feature_names=regime_payload.get("feature_names"),
        returns=timeseries.get("returns", []),
        exposures=timeseries.get("exposures", []),
        turnover_by_step=timeseries.get("turnover_by_step", []),
        annualization_days=252,
        float_precision=6,
    )
    if result is None:
        print("Unable to compute regime slices for this experiment.", file=sys.stderr)
        return 0

    slices_path = evaluation_dir / "regime_slices.json"
    slices_path.write_text(
        json.dumps(
            {
                "metadata": result.metadata,
                "performance_by_regime": result.performance_by_regime,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {slices_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
