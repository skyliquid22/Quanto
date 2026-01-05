#!/usr/bin/env python3
"""CLI entrypoint orchestrating canonical reconciliation runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from infra.normalization import ReconciliationBuilder

UTC = timezone.utc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical datasets across vendors deterministically.")
    parser.add_argument(
        "--config",
        default="configs/data_sources.yml",
        help="Path to the data sources configuration file (YAML or JSON).",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        help="Optional list of domains to reconcile (default: all configured domains).",
    )
    parser.add_argument("--start-date", required=True, help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", required=True, help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument(
        "--run-id",
        help="Deterministic run identifier; defaults to canonical_{timestamp} if omitted.",
    )
    parser.add_argument("--raw-root", help="Override raw data root directory.")
    parser.add_argument("--canonical-root", help="Override canonical output root directory.")
    parser.add_argument("--manifest-root", help="Override validation manifest directory.")
    parser.add_argument("--metrics-root", help="Override reconciliation metrics output directory.")
    return parser.parse_args()


def build_builder(args: argparse.Namespace) -> ReconciliationBuilder:
    kwargs: Dict[str, Any] = {}
    if args.raw_root:
        kwargs["raw_data_root"] = Path(args.raw_root).expanduser()
    if args.canonical_root:
        kwargs["canonical_root"] = Path(args.canonical_root).expanduser()
    if args.manifest_root:
        kwargs["validation_manifest_root_path"] = Path(args.manifest_root).expanduser()
    if args.metrics_root:
        kwargs["metrics_root"] = Path(args.metrics_root).expanduser()
    return ReconciliationBuilder.from_config_file(args.config, **kwargs)


def main() -> int:
    args = parse_args()
    builder = build_builder(args)
    run_id = args.run_id or datetime.now(tz=UTC).strftime("canonical_%Y%m%dT%H%M%S")
    manifests = builder.run(
        domains=args.domains,
        start_date=args.start_date,
        end_date=args.end_date,
        run_id=run_id,
    )
    print(json.dumps(manifests, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
