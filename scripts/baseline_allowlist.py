#!/usr/bin/env python3
"""Manage the baseline execution allowlist."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - CLI import helper
    sys.path.insert(0, str(PROJECT_ROOT))

from research.promotion import baseline_allowlist


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage the baseline execution allowlist.")
    parser.add_argument("--root", help="Optional allowlist root override (defaults to .quanto_data/baseline_allowlist).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--add", help="Add experiment_id to the allowlist.")
    group.add_argument("--remove", help="Remove experiment_id from the allowlist.")
    group.add_argument("--list", action="store_true", help="List allowlisted experiment_ids.")
    parser.add_argument("--reason", help="Reason for allowlisting (required with --add).")
    parser.add_argument("--notes", help="Optional notes recorded with the allowlist entry.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root) if args.root else None
    if args.add:
        if not args.reason:
            print("--reason is required when adding to the allowlist.", file=sys.stderr)  # noqa: T201
            return 2
        path = baseline_allowlist.add(args.add, reason=args.reason, notes=args.notes, root=root)
        print(f"Added {args.add} to baseline allowlist at {path}")  # noqa: T201
        return 0
    if args.remove:
        removed = baseline_allowlist.remove(args.remove, root=root)
        if removed:
            print(f"Removed {args.remove} from baseline allowlist.")  # noqa: T201
            return 0
        print(f"{args.remove} not found in baseline allowlist.", file=sys.stderr)  # noqa: T201
        return 1
    entries = baseline_allowlist.list_entries(root=root)
    if not entries:
        print("Baseline allowlist is empty.")  # noqa: T201
        return 0
    for entry in entries:
        payload = dict(entry.payload)
        created = payload.get("created_at", "")
        reason = payload.get("reason", "")
        print(f"{entry.experiment_id}\t{created}\t{reason}")  # noqa: T201
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
