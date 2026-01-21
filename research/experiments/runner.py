"""Deterministic experiment runner building on existing Quanto CLIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from infra.paths import get_data_root as default_data_root
from research.datasets.canonical_equity_loader import build_union_calendar, load_canonical_equity
from research.envs.signal_weight_env import SignalWeightEnvConfig, SignalWeightTradingEnv
from research.eval.evaluate import EvaluationMetadata, evaluation_payload, from_rollout
from research.eval.metrics import MetricConfig
from research.experiments.ablation import SweepExperiment, SweepResult
from research.experiments.registry import ExperimentPaths, ExperimentRegistry
from research.experiments.spec import ExperimentSpec
from research.experiments.sweep import SweepSpec, expand_sweep_entries
from research.features.feature_eng import (
    build_sma_feature_result,
    build_universe_feature_results,
    resolve_regime_metadata,
)
from research.features.feature_registry import (
    build_universe_feature_panel,
    default_regime_for_feature_set,
    is_universe_feature_set,
    normalize_feature_set_name,
)
from research.hierarchy.allocator_registry import build_allocator
from research.hierarchy.controller import ControllerConfig, ModeController
from research.hierarchy.policy_wrapper import HierarchicalPolicy, run_hierarchical_rollout
from research.risk import RiskConfig
from research.strategies.sma_crossover import SMAStrategyConfig
from scripts.evaluate_agent import run_evaluation as evaluate_cli, _derive_run_id
from scripts.train_ppo_weight_agent import run_training as train_ppo_cli

ALLOWED_TEST_WINDOWS = (1, 3, 4, 6, 12)


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


@dataclass(frozen=True)
class DataSplit:
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    train_ratio: float
    test_ratio: float
    test_window_months: int | None

    def to_payload(self) -> Dict[str, Any]:
        return {
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "train_ratio": float(self.train_ratio),
            "test_ratio": float(self.test_ratio),
            "test_window_months": None if self.test_window_months is None else int(self.test_window_months),
        }


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
    data_split = _resolve_data_split(spec, resolved_data_root)

    training_artifacts: Dict[str, Path] = {}
    checkpoint_path: Path | None = None

    if spec.policy == "ppo":
        training_summary = _run_training(spec, resolved_data_root, data_split)
        checkpoint_path = _extract_checkpoint(training_summary, resolved_data_root)
        if checkpoint_path is None or not checkpoint_path.exists():
            raise RuntimeError("PPO training did not produce a checkpoint; cannot continue.")
        training_artifacts = _materialize_training_artifacts(training_summary, resolved_data_root, paths.runs_dir)

    evaluation_summary, evaluation_payload, metrics_src = _run_evaluation(
        spec, resolved_data_root, paths.evaluation_dir, checkpoint_path, data_split
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
            "hierarchy_enabled": bool(spec.hierarchy_enabled),
            "mode_timeline_path": evaluation_summary.get("mode_timeline_path"),
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


def _run_training(spec: ExperimentSpec, data_root: Path, data_split: DataSplit) -> Mapping[str, Any]:
    params = spec.policy_params
    fast_window = _require_int(params, "fast_window", spec.policy)
    slow_window = _require_int(params, "slow_window", spec.policy)
    if fast_window >= slow_window:
        raise ValueError("fast_window must be less than slow_window for PPO.")
    timesteps = _require_int(params, "timesteps", spec.policy)
    learning_rate = float(params.get("learning_rate", 3e-4))
    gamma = float(params.get("gamma", 0.99))
    policy_id = str(params.get("policy", "MlpPolicy"))
    reward_version = params.get("reward_version")
    if isinstance(reward_version, str):
        reward_version = reward_version.strip() or None

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
        reward_version=reward_version,
        run_id=spec.experiment_id,
        data_root=str(data_root),
        regime_feature_set=spec.regime_feature_set,
        vendor="polygon",
        live=False,
        ingest_mode="rest",
        canonical_domain="equity_ohlcv",
        force_ingest=False,
        force_canonical_build=False,
        train_start_date=data_split.train_start.isoformat(),
        train_end_date=data_split.train_end.isoformat(),
        test_start_date=data_split.test_start.isoformat(),
        test_end_date=data_split.test_end.isoformat(),
        train_ratio=data_split.train_ratio,
        test_ratio=data_split.test_ratio,
        test_window_months=data_split.test_window_months,
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
    data_split: DataSplit,
) -> tuple[Mapping[str, Any], Dict[str, Any], Path]:
    if spec.hierarchy_enabled:
        return _run_hierarchical_evaluation(spec, data_root, evaluation_dir, data_split)
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
        train_start_date=data_split.train_start.isoformat(),
        train_end_date=data_split.train_end.isoformat(),
        test_start_date=data_split.test_start.isoformat(),
        test_end_date=data_split.test_end.isoformat(),
        train_ratio=data_split.train_ratio,
        test_ratio=data_split.test_ratio,
        test_window_months=data_split.test_window_months,
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


def _run_hierarchical_evaluation(
    spec: ExperimentSpec,
    data_root: Path,
    evaluation_dir: Path,
    data_split: DataSplit,
) -> tuple[Mapping[str, Any], Dict[str, Any], Path]:
    symbols = list(spec.symbols)
    if len(symbols) < 2:
        raise ValueError("Hierarchical experiments require at least two symbols.")
    if not spec.controller_config or not spec.allocator_by_mode:
        raise ValueError("Hierarchical experiments require controller and allocator configs.")
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    slices, canonical_hashes = load_canonical_equity(
        symbols,
        spec.start_date,
        spec.end_date,
        data_root=data_root,
        interval=spec.interval,
    )
    (
        rows,
        observation_columns,
        feature_hashes,
        feature_set_name,
        sma_config,
    ) = _build_hierarchical_rows(spec, slices, data_root)
    normalized_feature_set = normalize_feature_set_name(spec.feature_set)
    regime_for_panel = spec.regime_feature_set or default_regime_for_feature_set(normalized_feature_set)
    regime_metadata = resolve_regime_metadata(regime_for_panel)
    rows = _slice_rows_by_date(rows, data_split.test_start, data_split.test_end)
    if len(rows) < 2:
        raise ValueError("Not enough aligned feature rows in the test window for hierarchical evaluation.")
    env_config = SignalWeightEnvConfig(
        transaction_cost_bp=spec.cost_config.transaction_cost_bp,
        risk_config=spec.risk_config,
    )
    env = SignalWeightTradingEnv(rows, config=env_config, observation_columns=observation_columns)
    controller_cfg = ControllerConfig.from_mapping(spec.controller_config)
    controller = ModeController(config=controller_cfg)
    allocators = {
        mode: build_allocator(entry, num_assets=env.num_assets)
        for mode, entry in spec.allocator_by_mode.items()
    }
    policy = HierarchicalPolicy(controller, allocators, spec.risk_config)
    rollout = run_hierarchical_rollout(env, policy, rows=rows)
    ordered_inputs = dict(sorted({**canonical_hashes, **feature_hashes}.items()))
    policy_signature = {
        "controller": controller_cfg.to_dict(),
        "allocators": spec.allocator_by_mode,
    }
    policy_id = _hierarchical_policy_id(policy_signature)
    metadata = EvaluationMetadata(
        symbols=tuple(symbols),
        start_date=data_split.test_start.isoformat(),
        end_date=data_split.test_end.isoformat(),
        interval=spec.interval,
        feature_set=feature_set_name,
        policy_id=policy_id,
        run_id=_derive_run_id(
            symbols,
            spec.start_date,
            spec.end_date,
            spec.interval,
            feature_set_name,
            policy_id,
            env_config,
            ordered_inputs,
            policy_signature,
        ),
        regime_metadata=regime_metadata,
        policy_details={
            "type": "hierarchical",
            "controller": controller_cfg.to_dict(),
            "allocators": spec.allocator_by_mode,
            "sma_config": {
                "fast_window": sma_config.fast_window,
                "slow_window": sma_config.slow_window,
            },
        },
        data_split=data_split.to_payload(),
    )
    series = from_rollout(
        timestamps=rollout.timestamps,
        account_values=rollout.account_values,
        weights=rollout.weights,
        transaction_costs=rollout.transaction_costs,
        symbols=rollout.symbols,
        rollout_metadata={"modes": rollout.modes},
        regime_features=rollout.regime_features,
        regime_feature_names=rollout.regime_feature_names,
        modes=rollout.modes,
    )
    payload = evaluation_payload(
        series,
        metadata,
        inputs_used=ordered_inputs,
        config=MetricConfig(risk_config=spec.risk_config),
    )
    metrics_path = evaluation_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    timeline_path = _write_mode_timeline(rollout.timestamps[1:], rollout.modes, evaluation_dir)
    summary = {
        "metrics_path": str(metrics_path),
        "run_id": metadata.run_id,
        "policy_id": policy_id,
        "total_return": payload["performance"].get("total_return"),
        "sharpe": payload["performance"].get("sharpe"),
        "mode_timeline_path": str(timeline_path),
    }
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


def _build_hierarchical_rows(
    spec: ExperimentSpec,
    slices,
    data_root: Path,
) -> Tuple[List[Mapping[str, object]], Tuple[str, ...], Dict[str, str], str, SMAStrategyConfig]:
    symbols = list(spec.symbols)
    normalized_feature_set = normalize_feature_set_name(spec.feature_set)
    sma_config = SMAStrategyConfig(
        fast_window=int(spec.policy_params.get("fast_window", 20)),
        slow_window=int(spec.policy_params.get("slow_window", 50)),
    )
    per_symbol_features: Dict[str, Any] = {}
    feature_hashes: Dict[str, str] = {}
    if is_universe_feature_set(normalized_feature_set):
        per_symbol_features = build_universe_feature_results(
            normalized_feature_set,
            slices,
            symbol_order=symbols,
            start_date=spec.start_date,
            end_date=spec.end_date,
            sma_config=sma_config,
            data_root=data_root,
        )
    else:
        for symbol in symbols:
            slice_data = slices[symbol]
            feature_result = build_sma_feature_result(
                slice_data,
                fast_window=sma_config.fast_window,
                slow_window=sma_config.slow_window,
                feature_set=spec.feature_set,
                start_date=spec.start_date,
                end_date=spec.end_date,
                data_root=data_root,
            )
            per_symbol_features[symbol] = feature_result
    if not per_symbol_features:
        raise RuntimeError("Failed to build feature frames for hierarchical evaluation")
    feature_set_name = next(iter(per_symbol_features.values())).feature_set
    for symbol, result in per_symbol_features.items():
        for key, value in result.inputs_used.items():
            feature_hashes[f"{symbol}:{key}"] = value
    calendar = build_union_calendar(slices, start_date=spec.start_date, end_date=spec.end_date)
    regime_for_panel = spec.regime_feature_set or default_regime_for_feature_set(normalized_feature_set)
    panel = build_universe_feature_panel(
        per_symbol_features,
        symbol_order=symbols,
        calendar=calendar,
        forward_fill_limit=3,
        regime_feature_set=regime_for_panel,
        data_root=data_root,
    )
    return panel.rows, panel.observation_columns, feature_hashes, feature_set_name, sma_config


def _hierarchical_policy_id(signature: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(signature, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"hierarchical_{digest[:12]}"


def _write_mode_timeline(timestamps: Sequence[str], modes: Sequence[str], evaluation_dir: Path) -> Path:
    timeline = [
        {"timestamp": timestamps[idx], "mode": modes[idx] if idx < len(modes) else ""}
        for idx in range(min(len(timestamps), len(modes)))
    ]
    path = evaluation_dir / "mode_timeline.json"
    path.write_text(json.dumps(timeline, sort_keys=True, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    return path


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


def _resolve_data_split(spec: ExperimentSpec, data_root: Path) -> DataSplit:
    split_config = spec.evaluation_split
    train_ratio = split_config.train_ratio if split_config else 0.8
    test_ratio = split_config.test_ratio if split_config else 0.2
    test_window_months = split_config.test_window_months if split_config else None

    slices, _ = load_canonical_equity(
        list(spec.symbols),
        spec.start_date,
        spec.end_date,
        data_root=data_root,
        interval=spec.interval,
    )
    calendar = build_union_calendar(slices, start_date=spec.start_date, end_date=spec.end_date)
    session_dates = [timestamp.date() for timestamp in calendar]
    return _resolve_split_from_sessions(
        session_dates,
        train_ratio=train_ratio,
        test_ratio=test_ratio,
        test_window_months=test_window_months,
    )


def _resolve_split_from_sessions(
    session_dates: Sequence[date],
    *,
    train_ratio: float,
    test_ratio: float,
    test_window_months: int | None,
) -> DataSplit:
    if not session_dates:
        raise ValueError("No trading sessions available to resolve evaluation split.")
    ordered = sorted(dict.fromkeys(session_dates))
    total_sessions = len(ordered)
    if total_sessions < 2:
        raise ValueError("Evaluation split requires at least two trading sessions.")
    range_start = ordered[0]
    range_end = ordered[-1]

    if _is_short_range(ordered):
        test_sessions = _fallback_test_sessions(total_sessions)
        train_sessions = max(total_sessions - test_sessions, 1)
        test_sessions = total_sessions - train_sessions
        train_end = ordered[train_sessions - 1]
        test_start = ordered[train_sessions]
        return DataSplit(
            train_start=range_start,
            train_end=train_end,
            test_start=test_start,
            test_end=range_end,
            train_ratio=train_sessions / total_sessions,
            test_ratio=test_sessions / total_sessions,
            test_window_months=None,
        )

    if test_window_months is None:
        target_sessions = total_sessions * test_ratio
        selected_months = _closest_allowed_window(ordered, target_sessions)
    else:
        selected_months = test_window_months

    test_sessions = _count_sessions_in_last_months(ordered, selected_months)
    min_month_sessions = _count_sessions_in_last_months(ordered, 1)
    if test_sessions < min_month_sessions:
        selected_months = 1
        test_sessions = min_month_sessions
    if test_sessions >= total_sessions:
        test_sessions = max(min_month_sessions, total_sessions // 2, 1)
    if test_sessions >= total_sessions:
        test_sessions = total_sessions - 1

    train_sessions = max(total_sessions - test_sessions, 1)
    test_sessions = total_sessions - train_sessions
    train_end = ordered[train_sessions - 1]
    test_start = ordered[train_sessions]
    return DataSplit(
        train_start=range_start,
        train_end=train_end,
        test_start=test_start,
        test_end=range_end,
        train_ratio=train_sessions / total_sessions,
        test_ratio=test_sessions / total_sessions,
        test_window_months=selected_months,
    )


def _closest_allowed_window(session_dates: Sequence[date], target_sessions: float) -> int:
    best_months = ALLOWED_TEST_WINDOWS[0]
    best_diff = float("inf")
    for months in ALLOWED_TEST_WINDOWS:
        count = _count_sessions_in_last_months(session_dates, months)
        diff = abs(count - target_sessions)
        if diff < best_diff or (diff == best_diff and months < best_months):
            best_diff = diff
            best_months = months
    return best_months


def _count_sessions_in_last_months(session_dates: Sequence[date], months: int) -> int:
    if not session_dates:
        return 0
    target_months = _recent_months(session_dates[-1], months)
    return sum(1 for value in session_dates if (value.year, value.month) in target_months)


def _is_short_range(session_dates: Sequence[date]) -> bool:
    months = {(value.year, value.month) for value in session_dates}
    return len(months) <= 1


def _fallback_test_sessions(total_sessions: int) -> int:
    half = max(total_sessions // 2, 1)
    minimum = min(10, max(total_sessions - 1, 1))
    return max(half, minimum)


def _subtract_months(value: date, months: int) -> date:
    import calendar

    year = value.year
    month = value.month - months
    while month <= 0:
        month += 12
        year -= 1
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(value.day, last_day))


def _recent_months(end_date: date, months: int) -> set[tuple[int, int]]:
    year = end_date.year
    month = end_date.month
    results: set[tuple[int, int]] = set()
    for _ in range(months):
        results.add((year, month))
        month -= 1
        if month <= 0:
            month = 12
            year -= 1
    return results


def _slice_rows_by_date(
    rows: Sequence[Mapping[str, object]],
    start: date,
    end: date,
) -> List[Mapping[str, object]]:
    sliced: List[Mapping[str, object]] = []
    for row in rows:
        timestamp = row.get("timestamp")
        row_date = _coerce_row_date(timestamp)
        if row_date < start or row_date > end:
            continue
        sliced.append(row)
    return sliced


def _coerce_row_date(value: object) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime().date()
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    raise ValueError("Row timestamp is missing or invalid.")


__all__ = ["ExperimentResult", "run_experiment", "run_sweep", "SweepResult"]
