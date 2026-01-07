#!/usr/bin/env python3
"""CLI to record immutable experiment promotions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - CLI import path
    sys.path.insert(0, str(PROJECT_ROOT))

from research.experiments.registry import ExperimentRegistry
from research.promotion.report import PromotionRecord, write_promotion_record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record an experiment promotion decision.")
    parser.add_argument("--experiment-id", required=True, help="Experiment identifier to promote.")
    parser.add_argument(
        "--tier",
        required=True,
        choices=("research", "candidate", "production"),
        help="Promotion tier (candidate or production; research reserved for provenance).",
    )
    parser.add_argument("--reason", required=True, help="Immutable promotion justification.")
    parser.add_argument("--registry-root", help="Override the experiment registry root.")
    parser.add_argument("--promotion-root", help="Override the promotion record directory.")
    return parser.parse_args(argv)


def _load_report(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Qualification report not found: {path}") from None
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Qualification report is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Qualification report must contain an object: {path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    registry = ExperimentRegistry(root=Path(args.registry_root)) if args.registry_root else ExperimentRegistry()
    record = registry.get(args.experiment_id)
    report_path = record.promotion_dir / "qualification_report.json"
    report_payload = _load_report(report_path)
    if not report_payload.get("passed"):
        print(  # noqa: T201 - CLI feedback
            f"Cannot promote experiment {record.experiment_id}; qualification report indicates failure.",
            file=sys.stderr,
        )
        return 2
    reason = args.reason.strip()
    if not reason:
        print("Promotion reason must not be empty.", file=sys.stderr)  # noqa: T201
        return 3
    promotion_record = PromotionRecord(
        experiment_id=record.experiment_id,
        tier=args.tier.lower(),
        qualification_report_path=str(report_path),
        spec_path=str(record.spec_path),
        metrics_path=str(record.metrics_path),
        promotion_reason=reason,
    )
    try:
        output_path = write_promotion_record(promotion_record, root=Path(args.promotion_root) if args.promotion_root else None)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)  # noqa: T201
        return 4
    print(  # noqa: T201 - CLI feedback
        f"Promotion record written to {output_path}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
