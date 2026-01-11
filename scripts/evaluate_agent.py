#!/usr/bin/env python3
"""Deterministic evaluation CLI for universe rollouts and PPO checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(PROJECT_ROOT))

try:  # pragma: no cover - yaml optional
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

from infra.normalization.lineage import compute_file_hash
from infra.paths import get_data_root as default_data_root
from research.datasets.canonical_equity_loader import build_union_calendar, load_canonical_equity
from research.eval.evaluate import EvalSeries, EvaluationMetadata, MetricConfig, evaluation_payload, from_rollout
from research.envs.gym_weight_env import GymWeightTradingEnv
from research.envs.signal_weight_env import SignalWeightEnvConfig, SignalWeightTradingEnv
from research.features.feature_eng import build_sma_feature_result, build_universe_feature_results
from research.features.feature_registry import (
    FeatureSetResult,
    build_universe_feature_panel,
    default_regime_for_feature_set,
    is_universe_feature_set,
    normalize_feature_set_name,
)
from research.policies.sma_weight_policy import SMAWeightPolicy, SMAWeightPolicyConfig
from research.runners.rollout import run_rollout
from research.strategies.sma_crossover import SMAStrategyConfig
from research.training.ppo_trainer import EvaluationResult, evaluate as ppo_evaluate
from research.risk import RiskConfig
from scripts.run_sma_finrl_rollout import ensure_yearly_daily_coverage


_CONFIG_KEYS = {
    "symbol",
    "symbols",
    "start_date",
    "end_date",
    "interval",
    "feature_set",
    "policy",
    "fast_window",
    "slow_window",
    "policy_mode",
    "sigmoid_scale",
    "transaction_cost_bp",
    "checkpoint",
    "data_root",
    "out_dir",
    "run_id",
    "max_weight",
    "exposure_cap",
    "min_cash",
    "max_turnover_1d",
    "allow_short",
}


class EqualWeightPolicy:
    """Baseline policy returning uniform weights."""

    def decide(self, _: Mapping[str, object]) -> float:
        return 1.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", help="Optional eval config (YAML or JSON).")
    known, _ = config_parser.parse_known_args(argv)
    defaults = _load_config_defaults(known.config)

    parser = argparse.ArgumentParser(
        parents=[config_parser],
        description="Compute deterministic evaluation metrics for rollouts and PPO agents.",
    )
    if defaults:
        parser.set_defaults(**defaults)
    parser.add_argument("--symbol", default="AAPL", help="Single equity symbol to evaluate.")
    parser.add_argument(
        "--symbols",
        action="append",
        help="Universe symbols (comma separated, repeat flag to supply more).",
    )
    parser.add_argument("--start-date", required=True, help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", required=True, help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument("--interval", default="daily", help="Bar interval (daily only in v1).")
    parser.add_argument("--feature-set", default="sma_v1", help="Feature set used for observations.")
    parser.add_argument(
        "--policy",
        choices=["equal_weight", "sma", "ppo"],
        default="equal_weight",
        help="Policy to evaluate.",
    )
    parser.add_argument("--fast-window", type=int, default=20, help="Fast SMA window for features.")
    parser.add_argument("--slow-window", type=int, default=50, help="Slow SMA window for features.")
    parser.add_argument(
        "--policy-mode",
        choices=["hard", "sigmoid"],
        default="hard",
        help="Decision mode for SMA policy.",
    )
    parser.add_argument("--sigmoid-scale", type=float, default=5.0, help="Sigmoid scale for SMA policy.")
    parser.add_argument("--transaction-cost-bp", type=float, default=1.0, help="Transaction cost in basis points.")
    parser.add_argument("--max-weight", type=float, default=1.0, help="Per-asset weight cap enforced post-projection.")
    parser.add_argument("--exposure-cap", type=float, default=1.0, help="Total exposure cap enforced post-projection.")
    parser.add_argument("--min-cash", type=float, default=0.0, help="Minimum cash allocation (1 - max exposure).")
    parser.add_argument("--max-turnover-1d", type=float, help="Optional 1-day L1 turnover cap.")
    parser.add_argument("--allow-short", action="store_true", help="Disable the long-only projection step.")
    parser.add_argument("--checkpoint", help="Path to PPO checkpoint when --policy=ppo.")
    parser.add_argument("--data-root", help="Override QUANTO data root.")
    parser.add_argument("--out-dir", help="Destination directory for metrics.json.")
    parser.add_argument("--run-id", help="Run identifier override.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_evaluation(args)
    print(json.dumps(payload, sort_keys=True))
    return 0


def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    symbols = _resolve_symbol_list(args.symbol, getattr(args, "symbols", None))
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    if end < start:
        raise SystemExit("end-date cannot be before start-date")
    interval = str(args.interval).strip().lower()
    if interval != "daily":
        raise SystemExit("Only the daily interval is supported in v1.")
    if args.fast_window <= 0 or args.slow_window <= 0:
        raise SystemExit("SMA windows must be positive.")
    if args.fast_window >= args.slow_window:
        raise SystemExit("fast-window must be strictly less than slow-window.")
    if args.transaction_cost_bp < 0:
        raise SystemExit("transaction-cost-bp must be non-negative.")

    data_root = resolve_data_root(args.data_root)
    os.environ["QUANTO_DATA_ROOT"] = str(data_root)
    risk_config = _build_risk_config(args)

    auto_build = len(symbols) > 1
    ensure_yearly_daily_coverage(
        symbols=symbols,
        start=start,
        end=end,
        data_root=data_root,
        auto_build=auto_build,
        run_id_seed=args.run_id,
    )

    slices, canonical_hashes = load_canonical_equity(symbols, start, end, data_root=data_root, interval=interval)
    for symbol in symbols:
        slice_data = slices.get(symbol)
        if not slice_data or slice_data.frame.empty:
            raise SystemExit(f"No canonical data found for symbol {symbol}")

    (
        rows,
        observation_columns,
        feature_hashes,
        feature_set_name,
        sma_config,
    ) = _build_feature_rows(symbols, slices, args, data_root, start, end)
    if len(rows) < 2:
        raise SystemExit("Not enough aligned feature rows to evaluate.")

    combined_hashes = dict(canonical_hashes)
    combined_hashes.update(feature_hashes)

    env_config = SignalWeightEnvConfig(transaction_cost_bp=args.transaction_cost_bp, risk_config=risk_config)
    policy_id, policy_details, rollout_series, inputs_used = _run_policy_rollout(
        args=args,
        symbols=symbols,
        rows=rows,
        observation_columns=observation_columns,
        env_config=env_config,
        feature_set=feature_set_name,
        sma_config=sma_config,
        inputs_used=combined_hashes,
        start=start,
        end=end,
        interval=interval,
    )

    run_id = args.run_id or _derive_run_id(
        symbols,
        start,
        end,
        interval,
        feature_set_name,
        policy_id,
        env_config,
        inputs_used,
        policy_details,
    )

    metadata = EvaluationMetadata(
        symbols=tuple(symbols),
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        interval=interval,
        feature_set=feature_set_name,
        policy_id=policy_id,
        run_id=run_id,
        policy_details=policy_details,
    )
    payload = evaluation_payload(
        rollout_series,
        metadata,
        inputs_used=inputs_used,
        config=MetricConfig(risk_config=risk_config),
    )
    out_dir = Path(args.out_dir) if args.out_dir else data_root / "monitoring" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"metrics_{run_id}.json"
    _write_json(metrics_path, payload)
    summary = {
        "metrics_path": str(metrics_path),
        "run_id": run_id,
        "policy_id": policy_id,
        "total_return": payload["performance"].get("total_return"),
        "sharpe": payload["performance"].get("sharpe"),
    }
    return summary


def _run_policy_rollout(
    *,
    args: argparse.Namespace,
    symbols: Sequence[str],
    rows: Sequence[Mapping[str, object]],
    observation_columns: Sequence[str],
    env_config: SignalWeightEnvConfig,
    feature_set: str,
    sma_config: SMAStrategyConfig,
    inputs_used: Mapping[str, str],
    start: date,
    end: date,
    interval: str,
) -> Tuple[str, Dict[str, Any], EvalSeries, Dict[str, str]]:
    policy_name = args.policy
    rollout_metadata = _build_rollout_metadata(
        symbols,
        interval,
        start,
        end,
        env_config,
        feature_set,
        observation_columns,
        policy_name,
        args,
        sma_config,
    )
    if policy_name == "ppo":
        series, ppo_policy_id, details = _run_ppo_rollout(
            args, rows, observation_columns, env_config, rollout_metadata
        )
        return ppo_policy_id, details, series, dict(sorted(inputs_used.items()))
    env = SignalWeightTradingEnv(rows, config=env_config, observation_columns=observation_columns)
    if policy_name == "sma":
        policy_config = SMAWeightPolicyConfig(mode=args.policy_mode, sigmoid_scale=args.sigmoid_scale)
        policy = SMAWeightPolicy(policy_config)
        policy_id = f"sma_fast{args.fast_window}_slow{args.slow_window}_{policy.mode}_scale{policy.config.sigmoid_scale:g}"
        details = {
            "type": "sma_weight",
            "mode": policy.mode,
            "sigmoid_scale": policy.config.sigmoid_scale,
            "fast_window": args.fast_window,
            "slow_window": args.slow_window,
        }
    else:
        policy = EqualWeightPolicy()
        policy_id = "equal_weight"
        details = {"type": "equal_weight"}
    result = run_rollout(env, policy, inputs_used=inputs_used, metadata=rollout_metadata)
    series = from_rollout(
        timestamps=result.timestamps,
        account_values=result.account_values,
        weights=result.weights,
        transaction_costs=result.transaction_costs,
        symbols=result.symbols,
        rollout_metadata=result.metadata,
        regime_features=result.regime_features,
        regime_feature_names=result.regime_feature_names,
    )
    return policy_id, details, series, result.inputs_used


def _run_ppo_rollout(
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, object]],
    observation_columns: Sequence[str],
    env_config: SignalWeightEnvConfig,
    rollout_metadata: Mapping[str, object],
) -> Tuple[EvalSeries, str, Dict[str, Any]]:
    checkpoint = args.checkpoint
    if not checkpoint:
        raise SystemExit("--checkpoint is required when --policy=ppo")
    try:
        from stable_baselines3 import PPO as SB3PPO  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise SystemExit("stable_baselines3 is required for PPO evaluation") from exc
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    model = SB3PPO.load(str(checkpoint_path))
    env = GymWeightTradingEnv(rows, config=env_config, observation_columns=observation_columns)
    eval_result: EvaluationResult = ppo_evaluate(model, env)
    costs = [float(step.get("cost_paid", 0.0)) for step in eval_result.steps]
    series = from_rollout(
        timestamps=eval_result.timestamps,
        account_values=eval_result.account_values,
        weights=eval_result.weights,
        transaction_costs=costs,
        symbols=eval_result.symbols,
        rollout_metadata=rollout_metadata,
        regime_features=eval_result.regime_features,
        regime_feature_names=eval_result.regime_feature_names,
    )
    policy_id = f"ppo_{checkpoint_path.stem}"
    details = {
        "type": "ppo",
        "checkpoint": checkpoint_path.name,
        "checkpoint_hash": compute_file_hash(checkpoint_path),
    }
    return series, policy_id, details


def _build_feature_rows(
    symbols: Sequence[str],
    slices,
    args: argparse.Namespace,
    data_root: Path,
    start: date,
    end: date,
) -> Tuple[
    List[Mapping[str, object]],
    Tuple[str, ...],
    Dict[str, str],
    str,
    SMAStrategyConfig,
]:
    sma_config = SMAStrategyConfig(fast_window=args.fast_window, slow_window=args.slow_window)
    per_symbol_features: Dict[str, FeatureSetResult] = {}
    feature_hashes: Dict[str, str] = {}
    normalized_feature_set = normalize_feature_set_name(args.feature_set)
    multi_symbol = len(symbols) > 1
    panel_regime_feature = default_regime_for_feature_set(normalized_feature_set)
    if not multi_symbol and is_universe_feature_set(normalized_feature_set):
        raise SystemExit(f"Feature set '{args.feature_set}' requires multi-symbol mode.")
    if multi_symbol and is_universe_feature_set(normalized_feature_set):
        per_symbol_features = build_universe_feature_results(
            normalized_feature_set,
            slices,
            symbol_order=symbols,
            start_date=start,
            end_date=end,
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
                feature_set=args.feature_set,
                start_date=start,
                end_date=end,
                data_root=data_root,
            )
            per_symbol_features[symbol] = feature_result
    if not per_symbol_features:
        raise SystemExit("Failed to construct feature frames for evaluation.")
    feature_set_name = next(iter(per_symbol_features.values())).feature_set
    for symbol, feature_result in per_symbol_features.items():
        if feature_result.feature_set != feature_set_name:
            raise SystemExit("Feature set mismatch across symbols; ensure consistent registry entries.")
        if multi_symbol:
            for key, value in feature_result.inputs_used.items():
                feature_hashes[f"{symbol}:{key}"] = value
        else:
            feature_hashes.update(feature_result.inputs_used)
    if multi_symbol:
        calendar = build_union_calendar(slices, start_date=start, end_date=end)
        panel = build_universe_feature_panel(
            per_symbol_features,
            symbol_order=symbols,
            calendar=calendar,
            forward_fill_limit=3,
            regime_feature_set=panel_regime_feature,
        )
        rows = panel.rows
        base_columns = panel.observation_columns
    else:
        primary = symbols[0]
        rows = per_symbol_features[primary].frame.to_dict("records")
        base_columns = per_symbol_features[primary].observation_columns
    return rows, base_columns, feature_hashes, feature_set_name, sma_config


def _load_config_defaults(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {path}")
    text = config_path.read_text(encoding="utf-8")
    data = _parse_config_text(text)
    defaults = data.get("defaults", data) if isinstance(data, dict) else {}
    if not isinstance(defaults, Mapping):
        raise SystemExit("Eval config must be a mapping of defaults.")
    filtered: Dict[str, Any] = {}
    for key, value in defaults.items():
        attr = key.replace("-", "_")
        if attr in _CONFIG_KEYS:
            filtered[attr] = value
    return filtered


def _parse_config_text(text: str) -> Any:
    if yaml is not None:
        loaded = yaml.safe_load(text)
        if loaded is not None:
            return loaded
    return json.loads(text)


def _resolve_symbol_list(symbol: str, symbols_arg: Sequence[str] | None) -> List[str]:
    if symbols_arg:
        tokens: List[str] = []
        for entry in symbols_arg:
            for chunk in str(entry).split(","):
                clean = chunk.strip().upper()
                if clean:
                    tokens.append(clean)
        ordered = sorted(dict.fromkeys(tokens))
        if not ordered:
            raise SystemExit("At least one symbol must be provided when --symbols is set.")
        return ordered
    clean = str(symbol).upper().strip()
    if not clean:
        raise SystemExit("symbol must be provided")
    return [clean]


def resolve_data_root(arg_root: str | None) -> Path:
    if arg_root:
        return Path(arg_root).expanduser()
    override = os.environ.get("QUANTO_DATA_ROOT")
    if override:
        return Path(override).expanduser()
    return default_data_root()


def _build_risk_config(args: argparse.Namespace) -> RiskConfig:
    return RiskConfig(
        long_only=not bool(getattr(args, "allow_short", False)),
        max_weight=None if args.max_weight is None else float(args.max_weight),
        exposure_cap=None if args.exposure_cap is None else float(args.exposure_cap),
        min_cash=None if args.min_cash is None else float(args.min_cash),
        max_turnover_1d=None if args.max_turnover_1d is None else float(args.max_turnover_1d),
    )


def _derive_run_id(
    symbols: Sequence[str],
    start: date,
    end: date,
    interval: str,
    feature_set: str,
    policy_id: str,
    env_config: SignalWeightEnvConfig,
    inputs_used: Mapping[str, str],
    policy_details: Mapping[str, Any],
) -> str:
    payload = {
        "symbols": list(symbols),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "interval": interval,
        "feature_set": feature_set,
        "policy_id": policy_id,
        "transaction_cost_bp": env_config.transaction_cost_bp,
        "policy": dict(policy_details),
        "risk_config": env_config.risk_config.to_dict(),
        "inputs_used": {key: inputs_used[key] for key in sorted(inputs_used)},
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"eval_{digest[:12]}"


def _build_rollout_metadata(
    symbols: Sequence[str],
    interval: str,
    start: date,
    end: date,
    env_config: SignalWeightEnvConfig,
    feature_set: str,
    observation_columns: Sequence[str],
    policy_name: str,
    args: argparse.Namespace,
    sma_config: SMAStrategyConfig,
) -> Dict[str, Any]:
    policy_details: Dict[str, Any]
    if policy_name == "sma":
        policy_details = {
            "type": "sma_weight",
            "fast_window": args.fast_window,
            "slow_window": args.slow_window,
            "mode": args.policy_mode,
            "sigmoid_scale": args.sigmoid_scale,
        }
    elif policy_name == "ppo":
        policy_details = {
            "type": "ppo",
            "checkpoint": args.checkpoint,
        }
    else:
        policy_details = {"type": "equal_weight"}
    return {
        "symbols": list(symbols),
        "interval": interval,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "transaction_cost_bp": env_config.transaction_cost_bp,
        "risk_config": env_config.risk_config.to_dict(),
        "feature_set": feature_set,
        "observation_columns": list(observation_columns),
        "sma_config": {
            "fast_window": sma_config.fast_window,
            "slow_window": sma_config.slow_window,
        },
        "policy": policy_details,
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    path.write_text(canonical + "\n", encoding="utf-8")


__all__ = ["parse_args", "run_evaluation", "resolve_data_root"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
