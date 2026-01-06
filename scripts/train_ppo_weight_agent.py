#!/usr/bin/env python3
"""Train a PPO weight agent on canonical equity data with FinRL-style artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(PROJECT_ROOT))

from infra.normalization.lineage import compute_file_hash
from research.datasets.canonical_equity_loader import load_canonical_equity
from research.envs.gym_weight_env import GymWeightTradingEnv
from research.envs.signal_weight_env import SignalWeightEnvConfig
from research.features.feature_registry import FeatureSetResult, build_features, strategy_to_feature_frame
from research.strategies.sma_crossover import SMAStrategyConfig, run_sma_crossover
from research.training.ppo_trainer import EvaluationResult, evaluate, train_ppo
from scripts.run_sma_finrl_rollout import (  # type: ignore
    BootstrapMetadata,
    _canonical_files_exist,
    _locate_canonical_manifest,
    _rel_path,
    _render_account_weight_plot,
    _write_report,
    maybe_run_live_bootstrap,
    resolve_data_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO agent on canonical SMA-derived features.")
    parser.add_argument("--symbol", default="AAPL", help="Equity symbol to train on.")
    parser.add_argument("--start-date", required=True, help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", required=True, help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument("--interval", default="daily", help="Bar interval to use (daily only in v1).")
    parser.add_argument("--fast-window", type=int, default=20, help="Fast SMA window for features.")
    parser.add_argument("--slow-window", type=int, default=50, help="Slow SMA window for features.")
    parser.add_argument(
        "--feature-set",
        choices=["sma_v1", "options_v1", "sma_plus_options_v1"],
        default="sma_v1",
        help="Feature set used for environment observations.",
    )
    parser.add_argument("--transaction-cost-bp", type=float, default=1.0, help="Transaction cost in basis points.")
    parser.add_argument("--timesteps", type=int, default=10_000, help="Number of PPO timesteps to train.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed passed to PPO.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate for PPO.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for PPO.")
    parser.add_argument("--policy", default="MlpPolicy", help="stable-baselines3 policy class identifier.")
    parser.add_argument("--run-id", help="Optional run identifier override.")
    parser.add_argument("--data-root", help="Override automatic data root resolution.")
    parser.add_argument("--vendor", default="polygon", help="Vendor used for live ingestion when --live is set.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live bootstrap orchestration to refresh canonicals when missing/out-of-date.",
    )
    parser.add_argument(
        "--ingest-mode",
        choices=["rest", "flat_file", "auto"],
        default="rest",
        help="Ingestion mode forwarded to the live bootstrap utility.",
    )
    parser.add_argument(
        "--canonical-domain",
        default="equity_ohlcv",
        help="Canonical domain identifier used for bootstrap manifest checks.",
    )
    parser.add_argument(
        "--force-ingest",
        action="store_true",
        help="Force running live ingestion before the training run.",
    )
    parser.add_argument(
        "--force-canonical-build",
        action="store_true",
        help="Force rebuilding canonicals before training.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_training(args)
    print(json.dumps(payload, sort_keys=True))
    return 0


def run_training(args: argparse.Namespace) -> Dict[str, Any]:
    symbol = args.symbol.upper().strip()
    if not symbol:
        raise SystemExit("symbol must be provided")
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    if end < start:
        raise SystemExit("end-date cannot be before start-date")
    interval = str(args.interval).strip().lower()
    if interval != "daily":
        raise SystemExit("Only the daily interval is supported for PPO training in v1.")
    if args.fast_window <= 0 or args.slow_window <= 0:
        raise SystemExit("SMA windows must be positive.")
    if args.fast_window >= args.slow_window:
        raise SystemExit("fast-window must be strictly less than slow-window.")
    data_root = resolve_data_root(args.data_root)
    os.environ["QUANTO_DATA_ROOT"] = str(data_root)

    if args.live:
        bootstrap = maybe_run_live_bootstrap(
            symbols=[symbol],
            start=start,
            end=end,
            data_root=data_root,
            domain=args.canonical_domain,
            vendor=args.vendor,
            ingest_mode=args.ingest_mode,
            force_ingest=args.force_ingest,
            force_canonical=args.force_canonical_build,
            run_id_seed=args.run_id,
        )
    else:
        bootstrap = BootstrapMetadata(mode="none")

    canonical_manifest = bootstrap.canonical_manifest or _locate_canonical_manifest(
        data_root, args.canonical_domain, start, end
    )
    if not canonical_manifest or not _canonical_files_exist(data_root, args.canonical_domain, [symbol], start, end):
        raise SystemExit(
            "No canonical data available for the requested window. "
            "Provide --live to bootstrap or refresh canonicals."
        )
    bootstrap.canonical_manifest = canonical_manifest

    slices, canonical_hashes = load_canonical_equity(
        [symbol],
        start,
        end,
        data_root=data_root,
        interval=interval,
    )
    slice_data = slices.get(symbol)
    if not slice_data or not slice_data.rows:
        raise SystemExit(f"No canonical data found for symbol {symbol}")

    sma_config = SMAStrategyConfig(fast_window=args.fast_window, slow_window=args.slow_window)
    strategy = run_sma_crossover(symbol, slice_data.timestamps, slice_data.closes, sma_config)
    strategy_frame = strategy_to_feature_frame(strategy)
    if len(strategy_frame) < 2:
        raise SystemExit("Not enough rows after SMA alignment to train PPO.")
    feature_result = build_features(
        args.feature_set,
        strategy_frame,
        underlying_symbol=symbol,
        start_date=start,
        end_date=end,
        data_root=data_root,
    )
    rows = feature_result.frame.to_dict("records")
    if len(rows) < 2:
        raise SystemExit("Not enough feature rows to train PPO.")

    env_config = SignalWeightEnvConfig(transaction_cost_bp=args.transaction_cost_bp)
    train_env = GymWeightTradingEnv(rows, config=env_config, observation_columns=feature_result.observation_columns)
    eval_env = GymWeightTradingEnv(rows, config=env_config, observation_columns=feature_result.observation_columns)

    try:
        train_start = time.perf_counter()
        model = train_ppo(
            train_env,
            total_timesteps=args.timesteps,
            seed=args.seed,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            policy=args.policy,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    duration_seconds = time.perf_counter() - train_start

    eval_result = evaluate(model, eval_env)
    combined_hashes = dict(canonical_hashes)
    combined_hashes.update(feature_result.inputs_used)
    run_id = args.run_id or derive_run_id(
        symbol,
        start,
        end,
        interval,
        args.timesteps,
        args.seed,
        combined_hashes,
        args.fast_window,
        args.slow_window,
        args.transaction_cost_bp,
        feature_result,
    )

    reports_dir = data_root / "monitoring" / "reports"
    plots_dir = data_root / "monitoring" / "plots"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    train_report_path = reports_dir / f"ppo_train_{run_id}.json"
    eval_report_path = reports_dir / f"ppo_eval_{run_id}.json"
    plot_path = plots_dir / f"ppo_eval_{run_id}.png"

    _render_account_weight_plot(plot_path, eval_result.account_values, eval_result.weights)
    plot_hash = compute_file_hash(plot_path)

    model_path = save_model(model, data_root / "models" / f"ppo_{run_id}.zip")

    train_payload = build_train_report(
        run_id=run_id,
        symbol=symbol,
        start=start,
        end=end,
        interval=interval,
        args=args,
        env_config=env_config,
        sma_config=sma_config,
        canonical_hashes=combined_hashes,
        bootstrap=bootstrap,
        data_root=data_root,
        report_path=train_report_path,
        model_path=model_path,
        duration_seconds=duration_seconds,
        feature_result=feature_result,
    )
    eval_payload = build_eval_report(
        run_id=run_id,
        symbol=symbol,
        start=start,
        end=end,
        interval=interval,
        env_config=env_config,
        sma_config=sma_config,
        eval_result=eval_result,
        canonical_hashes=combined_hashes,
        bootstrap=bootstrap,
        data_root=data_root,
        report_path=eval_report_path,
        plot_path=plot_path,
        plot_hash=plot_hash,
        feature_result=feature_result,
    )

    _write_report(train_report_path, train_payload)
    _write_report(eval_report_path, eval_payload)

    return {
        "run_id": run_id,
        "interval": interval,
        "artifacts": {
            "train_report": _rel_path(train_report_path, data_root),
            "eval_report": _rel_path(eval_report_path, data_root),
            "eval_plot": _rel_path(plot_path, data_root),
            "model": _rel_path(model_path, data_root) if model_path else None,
        },
    }


def save_model(model, path: Path) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save"):
        try:
            model.save(str(path))
            return path
        except Exception:  # pragma: no cover - saving is best-effort
            return None
    return None


def build_train_report(
    *,
    run_id: str,
    symbol: str,
    start: date,
    end: date,
    interval: str,
    args: argparse.Namespace,
    env_config: SignalWeightEnvConfig,
    sma_config: SMAStrategyConfig,
    canonical_hashes: Dict[str, str],
    bootstrap: BootstrapMetadata,
    data_root: Path,
    report_path: Path,
    model_path: Path | None,
    duration_seconds: float,
    feature_result: FeatureSetResult,
) -> Dict[str, Any]:
    hashes = {
        "canonical_files": dict(sorted(canonical_hashes.items())),
        "report_json": "",
    }
    training_config = {
        "total_timesteps": args.timesteps,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "policy": args.policy,
        "duration_seconds": duration_seconds,
    }
    artifacts = {"report": _rel_path(report_path, data_root)}
    if model_path:
        artifacts["model"] = _rel_path(model_path, data_root)
        hashes["model_zip"] = compute_file_hash(model_path)
    return {
        "run_id": run_id,
        "symbol": symbol,
        "interval": interval,
        "date_range": {"start": start.isoformat(), "end": end.isoformat()},
        "training": training_config,
        "env_config": {
            "initial_cash": env_config.initial_cash,
            "transaction_cost_bp": env_config.transaction_cost_bp,
        },
        "features": {
            "fast_window": sma_config.fast_window,
            "slow_window": sma_config.slow_window,
            "feature_set": feature_result.feature_set,
            "observation_columns": list(feature_result.observation_columns),
        },
        "bootstrap_mode": bootstrap.mode,
        "bootstrap": bootstrap.as_payload(data_root),
        "hashes": hashes,
        "inputs_used": hashes["canonical_files"],
        "artifacts": artifacts,
    }


def build_eval_report(
    *,
    run_id: str,
    symbol: str,
    start: date,
    end: date,
    interval: str,
    env_config: SignalWeightEnvConfig,
    sma_config: SMAStrategyConfig,
    eval_result: EvaluationResult,
    canonical_hashes: Dict[str, str],
    bootstrap: BootstrapMetadata,
    data_root: Path,
    report_path: Path,
    plot_path: Path,
    plot_hash: str,
    feature_result: FeatureSetResult,
) -> Dict[str, Any]:
    hashes = {
        "canonical_files": dict(sorted(canonical_hashes.items())),
        "plot_png": plot_hash,
        "report_json": "",
    }
    series = {
        "timestamps": eval_result.timestamps,
        "account_value": eval_result.account_values,
        "weights": eval_result.weights,
        "log_returns": eval_result.log_returns,
    }
    artifacts = {
        "report": _rel_path(report_path, data_root),
        "plot": _rel_path(plot_path, data_root),
    }
    return {
        "run_id": run_id,
        "symbol": symbol,
        "interval": interval,
        "date_range": {"start": start.isoformat(), "end": end.isoformat()},
        "metrics": eval_result.metrics,
        "series": series,
        "steps": eval_result.steps,
        "env_config": {
            "initial_cash": env_config.initial_cash,
            "transaction_cost_bp": env_config.transaction_cost_bp,
        },
        "features": {
            "fast_window": sma_config.fast_window,
            "slow_window": sma_config.slow_window,
            "feature_set": feature_result.feature_set,
            "observation_columns": list(feature_result.observation_columns),
        },
        "bootstrap_mode": bootstrap.mode,
        "bootstrap": bootstrap.as_payload(data_root),
        "hashes": hashes,
        "inputs_used": hashes["canonical_files"],
        "artifacts": artifacts,
    }


def derive_run_id(
    symbol: str,
    start: date,
    end: date,
    interval: str,
    timesteps: int,
    seed: int,
    canonical_hashes: Dict[str, str],
    fast_window: int,
    slow_window: int,
    transaction_cost_bp: float,
    feature_result: FeatureSetResult,
) -> str:
    canonical = {
        "symbol": symbol,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "interval": interval,
        "timesteps": timesteps,
        "seed": seed,
        "fast_window": fast_window,
        "slow_window": slow_window,
        "transaction_cost_bp": transaction_cost_bp,
        "canonical_hashes": {key: canonical_hashes[key] for key in sorted(canonical_hashes)},
        "feature_set": feature_result.feature_set,
        "observation_columns": list(feature_result.observation_columns),
    }
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return f"ppo_weight_{digest[:12]}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
