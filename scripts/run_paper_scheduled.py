#!/usr/bin/env python3
"""Scheduled paper trading orchestrator."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - CLI import guard
    sys.path.insert(0, str(PROJECT_ROOT))

from research.ops.config import load_ops_config
from research.ops.service import PaperRunOrchestrator
from research.paper.config import load_paper_config
from scripts.run_paper import _credentials_available


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scheduled paper trading with lifecycle + alerting.")
    parser.add_argument("--config", required=True, help="Path to paper config (json/yaml).")
    parser.add_argument("--promotion-root", help="Optional override for promotion records.")
    parser.add_argument("--ops-config", help="Override path for ops.yml (defaults to configs/ops.yml).")
    parser.add_argument("--now", help="Override the wall-clock time (ISO-8601).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file {config_path} does not exist.", file=sys.stderr)  # noqa: T201
        return 2
    promotion_root = Path(args.promotion_root) if args.promotion_root else None
    try:
        paper_config = load_paper_config(config_path, promotion_root=promotion_root)
    except Exception as exc:  # pragma: no cover - CLI guard rail
        print(f"Failed to load paper config: {exc}", file=sys.stderr)  # noqa: T201
        return 3
    ops_path = Path(args.ops_config) if args.ops_config else None
    try:
        ops_config = load_ops_config(ops_path)
    except Exception as exc:  # pragma: no cover - CLI guard rail
        print(f"Failed to load ops config: {exc}", file=sys.stderr)  # noqa: T201
        return 4
    if not _credentials_available():
        print("ALPACA credentials missing; paper runs cannot start.", file=sys.stderr)  # noqa: T201
        return 5
    orchestrator = PaperRunOrchestrator(paper_config=paper_config, ops_config=ops_config)
    now = _parse_now(args.now) if args.now else None
    try:
        report = orchestrator.run(now=now)
    except Exception as exc:  # pragma: no cover - runtime guard rail
        print(f"Scheduled run failed: {exc}", file=sys.stderr)  # noqa: T201
        return 6
    if report.status == "IDLE":
        print("No scheduled paper run was due.", file=sys.stderr)  # noqa: T201
        return 0
    summary_path = report.summary_json
    print(  # noqa: T201 - CLI status
        f"Paper run {report.run_id} finished. summary={summary_path} markdown={report.summary_markdown}"
    )
    return 0


def _parse_now(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
