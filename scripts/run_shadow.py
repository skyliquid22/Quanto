#!/usr/bin/env python3
"""CLI entrypoint for the deterministic shadow execution engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - CLI import path
    sys.path.insert(0, str(PROJECT_ROOT))

from infra.paths import get_data_root
from research.experiments.registry import ExperimentRegistry
from research.shadow.data_source import ReplayMarketDataSource
from research.shadow.engine import ShadowEngine
from research.shadow.logging import ShadowLogger
from research.shadow.state_store import StateStore


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run promoted experiments in deterministic shadow mode.")
    parser.add_argument("--experiment-id", required=True, help="Experiment identifier to execute.")
    parser.add_argument("--replay", action="store_true", help="Enable historical replay mode.")
    parser.add_argument("--live", action="store_true", help="Enable live mode (unsupported in v1).")
    parser.add_argument("--start-date", help="Inclusive start date for replay (YYYY-MM-DD).")
    parser.add_argument("--end-date", help="Inclusive end date for replay (YYYY-MM-DD).")
    parser.add_argument("--max-steps", type=int, help="Optional cap on steps processed during this invocation.")
    parser.add_argument("--reset", action="store_true", help="Delete the run directory before executing (replay only).")
    parser.add_argument("--registry-root", help="Override the experiment registry root (defaults to repo .quanto_data).")
    parser.add_argument("--promotion-root", help="Override the promotion record root (defaults to repo .quanto_data/promotions).")
    parser.add_argument("--output-dir", help="Override the shadow run output directory.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previous run directory instead of starting from scratch.",
    )
    parser.add_argument(
        "--qualification-replay",
        action="store_true",
        help="Allow replay for qualification evidence even if experiment is unpromoted.",
    )
    parser.add_argument(
        "--qualification-reason",
        help="Custom reason recorded when using --qualification-replay.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("none", "sim", "alpaca_paper"),
        help="Execution mode controlling broker/controller engagement. Defaults to 'sim' when --qualification-replay is set, otherwise 'none'.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.replay and args.live:
        print("Only one of --replay or --live may be specified.", file=sys.stderr)  # noqa: T201
        return 2
    if args.live:
        print("Live mode is not implemented; connect a real-time feed to continue.", file=sys.stderr)  # noqa: T201
        return 3
    if not args.replay:
        print("Replay mode must be specified for v1 shadow execution.", file=sys.stderr)  # noqa: T201
        return 4
    if not args.start_date or not args.end_date:
        print("--start-date and --end-date are required for replay mode.", file=sys.stderr)  # noqa: T201
        return 5
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    if end < start:
        print("end-date cannot be earlier than start-date.", file=sys.stderr)  # noqa: T201
        return 6
    max_steps = args.max_steps
    if max_steps is not None and max_steps <= 0:
        print("--max-steps must be positive when provided.", file=sys.stderr)  # noqa: T201
        return 7

    data_root = get_data_root()
    registry_root = Path(args.registry_root) if args.registry_root else None
    registry = ExperimentRegistry(root=registry_root) if registry_root else ExperimentRegistry()
    promotion_root = Path(args.promotion_root) if args.promotion_root else None
    record, spec = registry.resolve_with_spec(args.experiment_id)
    data_source = ReplayMarketDataSource(
        spec=spec,
        start_date=start,
        end_date=end,
        data_root=data_root,
    )
    if args.output_dir:
        run_dir = Path(args.output_dir).expanduser()
        run_id = run_dir.name
    else:
        run_id = _derive_run_id(args.experiment_id, data_source.window)
        run_dir = data_root / "shadow" / args.experiment_id / run_id

    if run_dir.exists():
        if args.reset:
            shutil.rmtree(run_dir)
        elif not args.resume:
            print(  # noqa: T201
                f"Run directory {run_dir} already exists. Pass --resume to continue or --reset to start fresh.",
                file=sys.stderr,
            )
            return 8
    else:
        if args.resume:
            print(  # noqa: T201
                f"Run directory {run_dir} does not exist; cannot resume.", file=sys.stderr
            )
            return 9

    state_store = StateStore(
        args.experiment_id,
        run_id=run_id,
        base_dir=None if args.output_dir else data_root / "shadow",
        destination=run_dir if args.output_dir else None,
    )
    logger = ShadowLogger(run_dir)
    baseline_allow_root = data_root / "baseline_allowlist"
    qualification_allow_root = data_root / "qualification_allowlist"
    execution_mode = args.execution_mode or ("sim" if args.qualification_replay else "none")
    engine = ShadowEngine(
        experiment_id=args.experiment_id,
        data_source=data_source,
        state_store=state_store,
        logger=logger,
        run_id=run_id,
        out_dir=run_dir,
        registry=registry,
        promotion_root=promotion_root,
        replay_mode=True,
        live_mode=False,
        baseline_allowlist_root=baseline_allow_root,
        qualification_allowlist_root=qualification_allow_root,
        qualification_replay_allowed=args.qualification_replay,
        qualification_allow_reason=args.qualification_reason,
        execution_mode=execution_mode,
    )
    summary = engine.run(max_steps=max_steps)
    print(  # noqa: T201 - CLI status
        f"Shadow execution finished. state={summary['state_path']} logs={summary['log_path']} summary={logger.summary_path}"
    )
    return 0


def _derive_run_id(experiment_id: str, window: tuple[str, str]) -> str:
    payload = {
        "experiment_id": experiment_id,
        "window_start": window[0],
        "window_end": window[1],
        "mode": "replay",
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"replay_{digest[:12]}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
