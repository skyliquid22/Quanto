#!/usr/bin/env python3
"""Deterministic health report for canonical and feature data."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, Sequence

from infra.paths import get_data_root
from research.validation.data_health import (
    build_feature_frames,
    compute_canonical_health,
    compute_feature_health,
    evaluate_thresholds,
    load_canonical_equity_pandas,
    resolve_run_id,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic data health reports.")
    parser.add_argument("--domain", choices=("equity_ohlcv",), required=True)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--feature-set", help="Optional feature set to evaluate for NaN coverage.")
    parser.add_argument(
        "--calendar-mode",
        choices=("union", "intersection", "symbol"),
        default="union",
        help="Calendar mode derived from data (union/intersection/symbol).",
    )
    parser.add_argument("--run-id", help="Deterministic run identifier; auto-derived if omitted.")
    parser.add_argument("--data-root", help="Override QUANTO_DATA_ROOT for report inputs/outputs.")
    parser.add_argument("--parquet-engine", default="fastparquet", help="Parquet engine for pandas reads.")
    parser.add_argument("--strict", action="store_true", help="Fail fast if thresholds are exceeded.")
    parser.add_argument("--max-missing-ratio", type=float, help="Maximum allowed missing ratio.")
    parser.add_argument("--max-nan-ratio", type=float, help="Maximum allowed NaN ratio.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    if end < start:
        print("end_date must be greater than or equal to start_date", file=sys.stderr)
        return 2

    data_root = Path(args.data_root).expanduser() if args.data_root else get_data_root()
    load_result = load_canonical_equity_pandas(
        args.symbols,
        start,
        end,
        data_root=data_root,
        parquet_engine=args.parquet_engine,
    )
    canonical_report = compute_canonical_health(
        load_result.slices,
        start_date=start,
        end_date=end,
        calendar_mode=args.calendar_mode,
    )
    canonical_payload: Dict[str, Any] = {
        "domain": args.domain,
        "symbols": sorted(load_result.slices),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "calendar_mode": args.calendar_mode,
        "file_hashes": dict(sorted(load_result.file_hashes.items())),
        "canonical_summary": canonical_report,
    }

    feature_payload: Dict[str, Any] | None = None
    if args.feature_set:
        frames, observation_columns = build_feature_frames(
            feature_set=args.feature_set,
            slices=load_result.slices,
            start_date=start,
            end_date=end,
            data_root=data_root,
        )
        feature_report = compute_feature_health(frames, observation_columns)
        feature_payload = {
            "feature_set": args.feature_set,
            "symbols": sorted(frames),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "feature_summary": feature_report,
        }

    run_id = args.run_id or resolve_run_id(
        {
            "domain": args.domain,
            "symbols": sorted(args.symbols),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "feature_set": args.feature_set,
            "calendar_mode": args.calendar_mode,
        }
    )
    output_dir = data_root / "monitoring" / "data_health" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = output_dir / "canonical_summary.json"
    canonical_path.write_text(json.dumps(canonical_payload, indent=2, sort_keys=True), encoding="utf-8")

    feature_path = None
    if feature_payload is not None:
        feature_path = output_dir / "feature_summary.json"
        feature_path.write_text(json.dumps(feature_payload, indent=2, sort_keys=True), encoding="utf-8")

    max_missing_ratio = args.max_missing_ratio
    max_nan_ratio = args.max_nan_ratio
    if args.strict:
        if max_missing_ratio is None:
            max_missing_ratio = 0.0
        if max_nan_ratio is None and feature_payload is not None:
            max_nan_ratio = 0.0
        failures = evaluate_thresholds(
            canonical_report=canonical_report,
            feature_report=feature_payload["feature_summary"] if feature_payload else None,
            max_missing_ratio=max_missing_ratio,
            max_nan_ratio=max_nan_ratio,
        )
        if failures:
            print(f"Data health checks failed: {', '.join(failures)}", file=sys.stderr)
            return 3

    payload = {
        "run_id": run_id,
        "canonical_path": str(canonical_path),
        "feature_path": str(feature_path) if feature_path else None,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
