#!/usr/bin/env python3
"""Deterministic v1 release candidate runner (train → qualify → promote → shadow → report)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import normalization
    sys.path.insert(0, str(PROJECT_ROOT))

from infra.paths import get_data_root
from research.experiments import ExperimentRegistry, ExperimentSpec, run_experiment
from research.experiments.regression import load_gate_rules
from research.promotion.qualify import run_qualification
from research.promotion.report import PromotionRecord, write_promotion_record
from research.shadow.data_source import ReplayMarketDataSource
from research.shadow.engine import ShadowEngine
from research.shadow.logging import ShadowLogger
from research.shadow.state_store import StateStore
from research.v1.report import build_release_report, write_release_report

ReleaseSummary = Mapping[str, Any]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the deterministic v1 PPO release candidate pipeline.")
    parser.add_argument("--spec", required=True, help="Path to the experiment spec (JSON or YAML).")
    parser.add_argument("--baseline", required=True, help="Baseline identifier (experiment id, name, or latest:name).")
    parser.add_argument("--data-root", help="Override QUANTO_DATA_ROOT for this run.")
    parser.add_argument("--registry-root", help="Override the experiment registry root.")
    parser.add_argument("--promotion-root", help="Override the promotion record root.")
    parser.add_argument("--release-root", help="Override the v1 release artifacts root.")
    parser.add_argument("--gate-config", help="Optional regression gate configuration (JSON/YAML).")
    parser.add_argument(
        "--promotion-tier",
        choices=("research", "candidate", "production"),
        default="candidate",
        help="Promotion tier recorded for successful releases (default=candidate).",
    )
    parser.add_argument(
        "--promotion-reason",
        default="v1_release_candidate",
        help="Immutable promotion reason recorded when qualification passes.",
    )
    parser.add_argument("--shadow-start-date", help="Override shadow replay start date (YYYY-MM-DD).")
    parser.add_argument("--shadow-end-date", help="Override shadow replay end date (YYYY-MM-DD).")
    parser.add_argument("--shadow-max-steps", type=int, help="Optional step cap for the shadow replay.")
    parser.add_argument(
        "--force-experiment",
        action="store_true",
        help="Overwrite the registry slot if the experiment already exists.",
    )
    return parser.parse_args(argv)


def run_v1_release(args: argparse.Namespace) -> ReleaseSummary:
    spec_path = Path(args.spec)
    spec = ExperimentSpec.from_file(spec_path)
    if spec.policy != "ppo":
        raise ValueError("v1 release runner currently supports PPO experiments only.")
    baseline_identifier = str(args.baseline).strip()
    if not baseline_identifier:
        raise ValueError("Baseline identifier must be provided.")
    promotion_reason = str(args.promotion_reason or "").strip()
    if not promotion_reason:
        raise ValueError("promotion_reason must not be empty.")
    shadow_window = _resolve_shadow_window(spec, args.shadow_start_date, args.shadow_end_date)
    max_steps = args.shadow_max_steps
    if max_steps is not None and max_steps <= 0:
        raise ValueError("--shadow-max-steps must be positive when provided.")

    data_root = Path(args.data_root) if args.data_root else get_data_root()
    os.environ["QUANTO_DATA_ROOT"] = str(data_root)
    registry_root = Path(args.registry_root) if args.registry_root else data_root / "experiments"
    promotion_root = Path(args.promotion_root) if args.promotion_root else data_root / "promotions"
    release_root = Path(args.release_root) if args.release_root else data_root / "v1_release"
    release_root.mkdir(parents=True, exist_ok=True)
    gate_rules = load_gate_rules(args.gate_config)

    run_id = _derive_run_id(
        spec,
        baseline_identifier=baseline_identifier,
        window=shadow_window,
        promotion_tier=str(args.promotion_tier or "candidate").lower(),
    )
    release_dir = release_root / run_id
    if release_dir.exists():
        shutil.rmtree(release_dir)
    release_dir.mkdir(parents=True, exist_ok=True)
    registry = ExperimentRegistry(root=registry_root)

    experiment_result = run_experiment(spec, registry=registry, force=bool(args.force_experiment), data_root=data_root)
    _copy_tree(experiment_result.registry_paths.root, release_dir / "experiment")

    qualification_result = run_qualification(
        experiment_result.experiment_id,
        baseline_identifier,
        registry=registry,
        gate_rules=gate_rules,
    )
    report_dir = release_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    qualification_report_path = report_dir / "qualification_report.json"
    shutil.copy2(qualification_result.report_path, qualification_report_path)

    promotion_copy_path: Path | None = None
    shadow_summary: Mapping[str, Any] | None = None
    tier = str(args.promotion_tier or "candidate").lower()
    if qualification_result.report.passed:
        promotion_record = PromotionRecord(
            experiment_id=experiment_result.experiment_id,
            tier=tier,
            qualification_report_path=str(qualification_result.report_path),
            spec_path=str(experiment_result.registry_paths.spec_dir / "experiment_spec.json"),
            metrics_path=str(experiment_result.metrics_path),
            promotion_reason=promotion_reason,
        )
        promotion_path = write_promotion_record(promotion_record, root=promotion_root)
        promotion_copy_path = report_dir / "promotion_record.json"
        shutil.copy2(promotion_path, promotion_copy_path)
        shadow_run_id = f"{run_id}_shadow"
        shadow_summary = _run_shadow_replay(
            spec=spec,
            registry=registry,
            data_root=data_root,
            shadow_dir=release_dir / "shadow",
            run_id=shadow_run_id,
            promotion_root=promotion_root,
            start_date=shadow_window[0],
            end_date=shadow_window[1],
            max_steps=max_steps,
        )
    release_payload = build_release_report(
        run_id=run_id,
        spec=spec,
        baseline_id=qualification_result.report.baseline_experiment_id,
        qualification=qualification_result.report,
        qualification_report_path=qualification_report_path,
        shadow_summary=shadow_summary,
        promotion_record_path=promotion_copy_path,
        promotion_tier=tier if promotion_copy_path else None,
        release_dir=release_dir,
    )
    report_path = write_release_report(report_dir, release_payload)
    summary = {
        "run_id": run_id,
        "release_dir": str(release_dir),
        "report_path": str(report_path),
        "experiment_id": experiment_result.experiment_id,
        "baseline_id": qualification_result.report.baseline_experiment_id,
        "v1_ready": bool(release_payload["v1_ready"]),
    }
    return summary


def _resolve_shadow_window(
    spec: ExperimentSpec,
    start_override: str | None,
    end_override: str | None,
) -> tuple[date, date]:
    start = date.fromisoformat(start_override) if start_override else spec.start_date
    end = date.fromisoformat(end_override) if end_override else spec.end_date
    if end < start:
        raise ValueError("shadow end date cannot be earlier than start date.")
    return start, end


def _derive_run_id(
    spec: ExperimentSpec,
    *,
    baseline_identifier: str,
    window: tuple[date, date],
    promotion_tier: str,
) -> str:
    payload = {
        "experiment_id": spec.experiment_id,
        "baseline": baseline_identifier,
        "shadow_start": window[0].isoformat(),
        "shadow_end": window[1].isoformat(),
        "promotion_tier": promotion_tier,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return f"v1_{digest[:12]}"


def _run_shadow_replay(
    *,
    spec: ExperimentSpec,
    registry: ExperimentRegistry,
    data_root: Path,
    shadow_dir: Path,
    run_id: str,
    promotion_root: Path,
    start_date: date,
    end_date: date,
    max_steps: int | None,
) -> Mapping[str, Any]:
    if shadow_dir.exists():
        shutil.rmtree(shadow_dir)
    shadow_dir.mkdir(parents=True, exist_ok=True)
    data_source = ReplayMarketDataSource(
        spec=spec,
        start_date=start_date,
        end_date=end_date,
        data_root=data_root,
    )
    state_store = StateStore(spec.experiment_id, run_id=run_id, destination=shadow_dir)
    logger = ShadowLogger(shadow_dir)
    baseline_allow_root = data_root / "baseline_allowlist"
    qualification_allow_root = data_root / "qualification_allowlist"
    engine = ShadowEngine(
        experiment_id=spec.experiment_id,
        data_source=data_source,
        state_store=state_store,
        logger=logger,
        run_id=run_id,
        out_dir=shadow_dir,
        registry=registry,
        promotion_root=promotion_root,
        replay_mode=True,
        live_mode=False,
        baseline_allowlist_root=baseline_allow_root,
        qualification_allowlist_root=qualification_allow_root,
        execution_mode="sim",
    )
    return engine.run(max_steps=max_steps)


def _copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run_v1_release(args)
    except Exception as exc:  # pragma: no cover - exercised in CLI integration tests
        print(f"v1 release runner failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
