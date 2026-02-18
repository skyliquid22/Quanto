#!/usr/bin/env python3
"""Generate evaluation timeseries artifacts and compute regime slices for an experiment."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Mapping

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
    rollout = load_rollout_json(experiment_dir)
    data_split = _extract_data_split(rollout)
    if args.force or not timeseries_path.exists():
        timeseries = extract_timeseries_from_rollout(rollout)
    else:
        timeseries = json.loads(timeseries_path.read_text(encoding="utf-8"))
    if data_split is not None:
        timeseries = _slice_timeseries(timeseries, data_split["test_start"], data_split["test_end"])
    if args.force or not timeseries_path.exists() or data_split is not None:
        write_timeseries_json(experiment_dir, timeseries)

    regime_payload = timeseries.get("regime")
    if not regime_payload:
        print("No regime series available; wrote timeseries.json only.", file=sys.stderr)
        return 0

    labeling_version, thresholds_path = _extract_regime_labeling(regime_payload)
    result = compute_regime_slices(
        regime_series=regime_payload.get("series"),
        feature_names=regime_payload.get("feature_names"),
        returns=timeseries.get("returns", []),
        exposures=timeseries.get("exposures", []),
        turnover_by_step=timeseries.get("turnover_by_step", []),
        annualization_days=252,
        float_precision=6,
        labeling_version=labeling_version,
        thresholds_path=thresholds_path,
    )
    if result is None:
        print("Unable to compute regime slices for this experiment.", file=sys.stderr)
        return 0

    slices_path = evaluation_dir / "regime_slices.json"
    metadata = dict(result.metadata)
    regime_metadata = regime_payload.get("metadata")
    if isinstance(regime_metadata, Mapping):
        metadata["regime"] = dict(regime_metadata)
    slices_path.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "performance_by_regime": result.performance_by_regime,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {slices_path}")
    return 0


def _extract_data_split(rollout: Mapping[str, object]) -> Dict[str, date] | None:
    metadata = rollout.get("metadata") if isinstance(rollout, Mapping) else None
    if not isinstance(metadata, Mapping):
        return None
    split = metadata.get("data_split")
    if not isinstance(split, Mapping):
        return None
    test_start = _parse_date(split.get("test_start"))
    test_end = _parse_date(split.get("test_end"))
    if not test_start or not test_end:
        return None
    return {"test_start": test_start, "test_end": test_end}


def _parse_date(value: object) -> date | None:
    if not value:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    return None


def _extract_regime_labeling(regime_payload: Mapping[str, object]) -> tuple[str | None, str | None]:
    metadata = regime_payload.get("metadata") if isinstance(regime_payload, Mapping) else None
    if not isinstance(metadata, Mapping):
        return None, None
    labeling = metadata.get("labeling_version") or metadata.get("labeling")
    thresholds_path = metadata.get("thresholds_file")
    return (str(labeling) if labeling else None, str(thresholds_path) if thresholds_path else None)


def _slice_timeseries(timeseries: Mapping[str, object], start: date, end: date) -> Dict[str, object]:
    timestamps = list(timeseries.get("timestamps") or [])
    indices = [
        idx
        for idx, value in enumerate(timestamps)
        if start <= _coerce_date(value) <= end
    ]
    if not indices:
        raise ValueError("No timeseries data overlaps the requested test window.")
    start_idx = indices[0]
    end_idx = indices[-1]
    sliced = dict(timeseries)
    for key in ("timestamps", "returns", "exposures", "turnover_by_step"):
        series = list(timeseries.get(key) or [])
        if series:
            sliced[key] = series[start_idx : end_idx + 1]
    regime = timeseries.get("regime")
    if isinstance(regime, Mapping):
        values = list(regime.get("series") or [])
        sliced_regime = dict(regime)
        if values:
            sliced_regime["series"] = values[start_idx : end_idx + 1]
        sliced["regime"] = sliced_regime
    return sliced


def _coerce_date(value: object) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime().date()
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    raise ValueError("Invalid timestamp in timeseries payload.")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
