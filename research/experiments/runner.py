"""Deterministic experiment runner building on existing Quanto CLIs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

from infra.paths import get_data_root as default_data_root
from research.experiments.ablation import SweepExperiment, SweepResult
from research.experiments.registry import ExperimentPaths, ExperimentRegistry
from research.experiments.spec import ExperimentSpec
from research.experiments.sweep import SweepSpec, expand_sweep_entries
from scripts.evaluate_agent import run_evaluation as evaluate_cli
from scripts.train_ppo_weight_agent import run_training as train_ppo_cli


@dataclass(frozen=True)
class ExperimentResult:
    """Summary returned by run_experiment."""

    experiment_id: str
    registry_paths: ExperimentPaths
    metrics_path: Path
    evaluation_payload: Mapping[str, Any]
    evaluation_summary: Mapping[str, Any]
    training_artifacts: Mapping[str, Path]
    rollout_artifact: Path


def run_experiment(
    spec: ExperimentSpec,
    *,
    registry: ExperimentRegistry | None = None,
    force: bool = False,
    data_root: Path | None = None,
) -> ExperimentResult:
    """Run the experiment defined by spec and persist artifacts in the registry."""

    registry = registry or ExperimentRegistry()
    experiment_id = spec.experiment_id
    paths = registry.prepare(experiment_id, force=force)
    registry.write_spec(spec, paths)
    resolved_data_root = _resolve_data_root(data_root)
    os.environ.setdefault("QUANTO_DATA_ROOT", str(resolved_data_root))

    training_artifacts: Dict[str, Path] = {}
    checkpoint_path: Path | None = None

    if spec.policy == "ppo":
        training_summary = _run_training(spec, resolved_data_root)
        checkpoint_path = _extract_checkpoint(training_summary, resolved_data_root)
        if checkpoint_path is None or not checkpoint_path.exists():
            raise RuntimeError("PPO training did not produce a checkpoint; cannot continue.")
        training_artifacts = _materialize_training_artifacts(training_summary, resolved_data_root, paths.runs_dir)

    evaluation_summary, evaluation_payload, metrics_src = _run_evaluation(
        spec, resolved_data_root, paths.evaluation_dir, checkpoint_path
    )
    metrics_path = _finalize_metrics(metrics_src, paths.evaluation_dir)
    rollout_artifact = _write_rollout_artifact(spec, evaluation_payload, paths.runs_dir)
    registry.write_run_summary(
        paths,
        {
            "experiment_id": experiment_id,
            "policy": spec.policy,
            "regime_feature_set": spec.regime_feature_set,
            "metrics_path": str(metrics_path),
            "runs_dir": str(paths.runs_dir),
            "evaluation_dir": str(paths.evaluation_dir),
        },
    )
    return ExperimentResult(
        experiment_id=experiment_id,
        registry_paths=paths,
        metrics_path=metrics_path,
        evaluation_payload=evaluation_payload,
        evaluation_summary=evaluation_summary,
        training_artifacts=training_artifacts,
        rollout_artifact=rollout_artifact,
    )


def run_sweep(
    sweep_spec: SweepSpec,
    *,
    registry: ExperimentRegistry | None = None,
    force: bool = False,
    data_root: Path | None = None,
) -> SweepResult:
    """Expand and execute a sweep, skipping completed experiments when possible."""

    registry = registry or ExperimentRegistry()
    expansions = expand_sweep_entries(sweep_spec)
    total = len(expansions)
    print(  # noqa: T201 - CLI feedback required for sweeps
        f"Sweep '{sweep_spec.sweep_name}' expanding into {total} experiment(s)."
    )
    experiments: list[SweepExperiment] = []
    for entry in expansions:
        spec = entry.spec
        experiment_id = spec.experiment_id
        if not force and registry.has_completed(experiment_id):
            experiments.append(
                SweepExperiment(
                    experiment_id=experiment_id,
                    spec=spec,
                    dimensions=entry.dimension_values,
                    status="skipped",
                )
            )
            continue
        result = run_experiment(spec, registry=registry, force=force, data_root=data_root)
        experiments.append(
            SweepExperiment(
                experiment_id=result.experiment_id,
                spec=spec,
                dimensions=entry.dimension_values,
                status="completed",
            )
        )
    return SweepResult(sweep_spec=sweep_spec, experiments=tuple(experiments))


def _run_training(spec: ExperimentSpec, data_root: Path) -> Mapping[str, Any]:
    params = spec.policy_params
    fast_window = _require_int(params, "fast_window", spec.policy)
    slow_window = _require_int(params, "slow_window", spec.policy)
    if fast_window >= slow_window:
        raise ValueError("fast_window must be less than slow_window for PPO.")
    timesteps = _require_int(params, "timesteps", spec.policy)
    learning_rate = float(params.get("learning_rate", 3e-4))
    gamma = float(params.get("gamma", 0.99))
    policy_id = str(params.get("policy", "MlpPolicy"))

    risk_cfg = spec.risk_config
    args = argparse.Namespace(
        symbol=spec.symbols[0],
        symbols=list(spec.symbols) if len(spec.symbols) > 1 else None,
        start_date=spec.start_date.isoformat(),
        end_date=spec.end_date.isoformat(),
        interval=spec.interval,
        feature_set=spec.feature_set,
        fast_window=fast_window,
        slow_window=slow_window,
        transaction_cost_bp=spec.cost_config.transaction_cost_bp,
        timesteps=timesteps,
        seed=spec.seed,
        learning_rate=learning_rate,
        gamma=gamma,
        policy=policy_id,
        run_id=spec.experiment_id,
        data_root=str(data_root),
        regime_feature_set=spec.regime_feature_set,
        vendor="polygon",
        live=False,
        ingest_mode="rest",
        canonical_domain="equity_ohlcv",
        force_ingest=False,
        force_canonical_build=False,
        max_weight=risk_cfg.max_weight,
        exposure_cap=risk_cfg.exposure_cap,
        min_cash=risk_cfg.min_cash,
        max_turnover_1d=risk_cfg.max_turnover_1d,
        allow_short=not risk_cfg.long_only,
    )
    try:
        return train_ppo_cli(args)
    except SystemExit as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"PPO training failed: {exc}") from exc


def _extract_checkpoint(summary: Mapping[str, Any], data_root: Path) -> Path | None:
    artifacts = summary.get("artifacts") or {}
    checkpoint_rel = artifacts.get("model")
    if not checkpoint_rel:
        return None
    checkpoint_path = Path(checkpoint_rel)
    return checkpoint_path if checkpoint_path.is_absolute() else data_root / checkpoint_path


def _materialize_training_artifacts(
    summary: Mapping[str, Any],
    data_root: Path,
    runs_dir: Path,
) -> Dict[str, Path]:
    training_dir = runs_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    artifacts = summary.get("artifacts") or {}
    resolved: Dict[str, Path] = {}
    for name, relative in artifacts.items():
        if not relative:
            continue
        source = Path(relative)
        source_path = source if source.is_absolute() else data_root / source
        if not source_path.exists():
            continue
        destination = training_dir / source_path.name
        shutil.copy2(source_path, destination)
        resolved[name] = destination
    summary_path = training_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8")
    resolved["summary"] = summary_path
    return resolved


def _run_evaluation(
    spec: ExperimentSpec,
    data_root: Path,
    evaluation_dir: Path,
    checkpoint_path: Path | None,
) -> tuple[Mapping[str, Any], Dict[str, Any], Path]:
    policy = spec.policy
    params = dict(spec.policy_params)
    fast_window: int
    slow_window: int
    policy_mode = str(params.get("policy_mode", "hard"))
    sigmoid_scale = float(params.get("sigmoid_scale", 5.0))
    transaction_cost = spec.cost_config.transaction_cost_bp

    DEFAULT_FAST = 20
    DEFAULT_SLOW = 50
    risk_cfg = spec.risk_config

    if policy == "sma":
        fast_window = _optional_int(params, "fast_window") or DEFAULT_FAST
        slow_window = _optional_int(params, "slow_window") or DEFAULT_SLOW
        if fast_window >= slow_window:
            raise ValueError("fast_window must be less than slow_window for SMA policy.")
        params["fast_window"] = fast_window
        params["slow_window"] = slow_window
    elif policy == "equal_weight":
        fast_window = DEFAULT_FAST
        slow_window = DEFAULT_SLOW
        policy_mode = "hard"
    elif policy == "ppo":
        if not checkpoint_path:
            raise ValueError("PPO policy requires a checkpoint.")
        fast_window = DEFAULT_FAST
        slow_window = DEFAULT_SLOW
    else:
        raise ValueError(f"Unsupported policy '{policy}'.")

    args = argparse.Namespace(
        symbol=spec.symbols[0],
        symbols=list(spec.symbols) if len(spec.symbols) > 1 else None,
        start_date=spec.start_date.isoformat(),
        end_date=spec.end_date.isoformat(),
        interval=spec.interval,
        feature_set=spec.feature_set,
        regime_feature_set=spec.regime_feature_set,
        policy=policy,
        fast_window=fast_window,
        slow_window=slow_window,
        policy_mode=policy_mode,
        sigmoid_scale=sigmoid_scale,
        transaction_cost_bp=transaction_cost,
        checkpoint=str(checkpoint_path) if checkpoint_path else None,
        data_root=str(data_root),
        out_dir=str(evaluation_dir),
        run_id=spec.experiment_id,
        max_weight=risk_cfg.max_weight,
        exposure_cap=risk_cfg.exposure_cap,
        min_cash=risk_cfg.min_cash,
        max_turnover_1d=risk_cfg.max_turnover_1d,
        allow_short=not risk_cfg.long_only,
    )
    try:
        summary = evaluate_cli(args)
    except SystemExit as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Evaluation failed: {exc}") from exc
    metrics_path = Path(summary["metrics_path"])
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    return summary, payload, metrics_path


def _finalize_metrics(metrics_src: Path, evaluation_dir: Path) -> Path:
    destination = evaluation_dir / "metrics.json"
    if metrics_src.resolve() == destination.resolve():
        return destination
    if destination.exists():
        destination.unlink()
    metrics_src.replace(destination)
    return destination


def _write_rollout_artifact(
    spec: ExperimentSpec,
    evaluation_payload: Mapping[str, Any],
    runs_dir: Path,
) -> Path:
    rollout_path = runs_dir / "rollout.json"
    payload = {
        "experiment_id": spec.experiment_id,
        "metadata": evaluation_payload.get("metadata"),
        "series": evaluation_payload.get("series"),
        "inputs_used": evaluation_payload.get("inputs_used"),
        "performance": evaluation_payload.get("performance"),
        "trading": evaluation_payload.get("trading"),
        "safety": evaluation_payload.get("safety"),
    }
    rollout_path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return rollout_path


def _resolve_data_root(data_root: Path | None) -> Path:
    if data_root:
        return Path(data_root).expanduser()
    override = os.environ.get("QUANTO_DATA_ROOT")
    if override:
        return Path(override).expanduser()
    return default_data_root()


def _require_int(params: Mapping[str, Any], key: str, policy: str) -> int:
    if key not in params:
        raise ValueError(f"{key} must be provided in policy_params for policy '{policy}'.")
    try:
        return int(params[key])
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"{key} must be an integer for policy '{policy}'.") from exc


def _optional_int(params: Mapping[str, Any], key: str) -> int | None:
    if key not in params:
        return None
    try:
        return int(params[key])
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"{key} must be an integer.") from exc


__all__ = ["ExperimentResult", "run_experiment", "run_sweep", "SweepResult"]
