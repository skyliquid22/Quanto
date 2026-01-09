#!/usr/bin/env python3
"""CLI entrypoint for paper trading validation runs."""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - CLI import path
    sys.path.insert(0, str(PROJECT_ROOT))

from research.paper.config import load_paper_config
from research.paper.run import PaperRunner
from research.execution.alpaca_broker import AlpacaBrokerConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and launch Alpaca paper validation runs.")
    parser.add_argument("--config", required=True, help="Path to a paper run config (json or yaml).")
    parser.add_argument("--promotion-root", help="Optional override for promotion records.")
    parser.add_argument("--run-id", help="Optional override for the derived run identifier.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file {config_path} does not exist.", file=sys.stderr)  # noqa: T201
        return 2
    promotion_root = Path(args.promotion_root) if args.promotion_root else None
    try:
        config = load_paper_config(config_path, promotion_root=promotion_root)
    except Exception as exc:  # pragma: no cover - CLI guard rail
        print(f"Failed to load paper config: {exc}", file=sys.stderr)  # noqa: T201
        return 3
    if not _credentials_available():
        print("ALPACA_API_KEY and ALPACA_SECRET_KEY must be configured for paper runs.", file=sys.stderr)  # noqa: T201
        return 4
    runner = PaperRunner(config, run_id=args.run_id)
    print(f"Paper run prepared at {runner.run_dir}")  # noqa: T201
    return 0


def _credentials_available() -> bool:
    """Check whether Alpaca credentials are set in the environment."""

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if api_key and secret_key:
        return True
    try:
        AlpacaBrokerConfig.from_env()
        return True
    except Exception:
        return False


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
