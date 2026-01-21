#!/usr/bin/env python3
"""Train a PPO weight agent on canonical equity data with FinRL-style artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence
from types import SimpleNamespace

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(PROJECT_ROOT))

from infra.normalization.lineage import compute_file_hash
from research.datasets.canonical_equity_loader import build_union_calendar, load_canonical_equity
from research.envs.gym_weight_env import GymWeightTradingEnv
from research.envs.signal_weight_env import SignalWeightEnvConfig
from research.features.feature_eng import build_universe_feature_results
from research.features.feature_registry import (
    FeatureSetResult,
    build_universe_feature_panel,
    default_regime_for_feature_set,
    build_features,
    is_universe_feature_set,
    normalize_feature_set_name,
    strategy_to_feature_frame,
)
from research.strategies.sma_crossover import SMAStrategyConfig, run_sma_crossover
from research.training.ppo_trainer import DEFAULT_REWARD_VERSION, EvaluationResult, evaluate, train_ppo
from research.risk import RiskConfig
from research.training.reward_registry import get_reward_spec, reward_versions
from scripts.run_sma_finrl_rollout import (  # type: ignore
    BootstrapMetadata,
    _canonical_files_exist as _rollout_canonical_files_exist,
    _locate_canonical_manifest,
    _rel_path,
    _render_account_weight_plot,
    _write_report,
    ensure_yearly_daily_coverage,
    maybe_run_live_bootstrap,
    resolve_data_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO agent on canonical SMA-derived features.")
    parser.add_argument("--symbol", default="AAPL", help="Equity symbol to train on.")
    parser.add_argument(
        "--symbols",
        action="append",
        help="Universe mode symbols (comma-separated list when repeated).",
    )
    parser.add_argument("--start-date", required=True, help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", required=True, help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument("--train-start-date", help="Optional in-sample start date (YYYY-MM-DD).")
    parser.add_argument("--train-end-date", help="Optional in-sample end date (YYYY-MM-DD).")
    parser.add_argument("--test-start-date", help="Optional OOS start date (YYYY-MM-DD).")
    parser.add_argument("--test-end-date", help="Optional OOS end date (YYYY-MM-DD).")
    parser.add_argument("--train-ratio", type=float, help="Optional train ratio metadata override.")
    parser.add_argument("--test-ratio", type=float, help="Optional test ratio metadata override.")
    parser.add_argument("--test-window-months", type=int, help="Optional test window metadata override.")
    parser.add_argument("--interval", default="daily", help="Bar interval to use (daily only in v1).")
    parser.add_argument("--fast-window", type=int, default=20, help="Fast SMA window for features.")
    parser.add_argument("--slow-window", type=int, default=50, help="Slow SMA window for features.")
    parser.add_argument(
        "--feature-set",
        choices=[
            "sma_v1",
            "sma_universe_v1",
            "options_v1",
            "sma_plus_options_v1",
            "equity_xsec_v1",
            "sma_plus_xsec_v1",
            "core_v1_regime",
        ],
        default="sma_v1",
        help="Feature set used for environment observations.",
    )
    parser.add_argument("--transaction-cost-bp", type=float, default=1.0, help="Transaction cost in basis points.")
    parser.add_argument("--max-weight", type=float, default=1.0, help="Per-asset weight cap enforced by projection.")
    parser.add_argument("--exposure-cap", type=float, default=1.0, help="Total exposure cap enforced by projection.")
    parser.add_argument("--min-cash", type=float, default=0.0, help="Minimum cash allocation reserved each step.")
    parser.add_argument("--max-turnover-1d", type=float, help="Optional turnover cap per rebalance (L1 distance).")
    parser.add_argument("--allow-short", action="store_true", help="Disable long-only constraint in the projection.")
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
    parser.add_argument(
        "--reward-version",
        default=DEFAULT_REWARD_VERSION,
        help=f"Reward shaping version to apply (default: {DEFAULT_REWARD_VERSION}).",
    )
    return parser.parse_args()


def _canonical_files_exist(*args, **kwargs):
    return _rollout_canonical_files_exist(*args, **kwargs)


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


def _ensure_canonical_slice(slice_data: Any, symbol: str):
    """Normalize canonical slice payloads for compatibility with older tests."""

    has_frame = hasattr(slice_data, "frame")
    has_timestamps = hasattr(slice_data, "timestamps")
    has_closes = hasattr(slice_data, "closes")
    if has_frame and has_timestamps and has_closes:
        # Ensure downstream code can rely on .symbol even for legacy slices.
        if getattr(slice_data, "symbol", None):
            return slice_data
        return SimpleNamespace(
            frame=slice_data.frame,
            timestamps=slice_data.timestamps,
            closes=slice_data.closes,
            symbol=getattr(slice_data, "symbol", symbol),
            rows=getattr(slice_data, "rows", None),
        )

    rows = getattr(slice_data, "rows", None)
    if rows is None:
        raise SystemExit("Canonical slice is missing 'frame' and 'rows' payloads.")
    frame = getattr(slice_data, "frame", None)
    if frame is None:
        frame = pd.DataFrame(rows)
    timestamps = getattr(slice_data, "timestamps", None)
    if timestamps is None:
        timestamps = [row.get("timestamp") for row in rows if "timestamp" in row]
    closes = getattr(slice_data, "closes", None)
    if closes is None:
        closes = [row.get("close") for row in rows if "close" in row]
    if not timestamps or not closes:
        raise SystemExit("Canonical slice must include timestamps and close prices.")
    return SimpleNamespace(
        frame=frame,
        timestamps=timestamps,
        closes=closes,
        symbol=getattr(slice_data, "symbol", symbol),
        rows=rows,
    )


def _build_single_symbol_feature_result(
    slice_data: Any,
    *,
    fast_window: int,
    slow_window: int,
    feature_set: str,
    start_date: date,
    end_date: date,
    data_root: Path,
) -> FeatureSetResult:
    """Local SMA feature builder to keep trainer tests lightweight."""

    if fast_window <= 0 or slow_window <= 0:
        raise SystemExit("SMA windows must be positive.")
    if fast_window >= slow_window:
        raise SystemExit("fast-window must be strictly less than slow-window.")

    frame = getattr(slice_data, "frame", None)
    if frame is None or frame.empty:
        raise SystemExit("Canonical slice must contain at least two rows.")
    timestamps = getattr(slice_data, "timestamps", None)
    closes = getattr(slice_data, "closes", None)
    if not timestamps or not closes or len(timestamps) != len(closes):
        raise SystemExit("Canonical slice must include aligned timestamps and closes.")
    if len(timestamps) < 2:
        raise SystemExit("At least two bars are required to build SMA features.")

    normalized_feature_set = normalize_feature_set_name(feature_set)
    if normalized_feature_set == "core_v1":
        equity_df = frame.reset_index(drop=False)
        if "index" in equity_df.columns:
            equity_df.rename(columns={"index": "timestamp"}, inplace=True)
        return build_features(
            "core_v1",
            equity_df,
            underlying_symbol=getattr(slice_data, "symbol", "UNKNOWN"),
            start_date=start_date,
            end_date=end_date,
            data_root=data_root,
        )

    sma_config = SMAStrategyConfig(fast_window=fast_window, slow_window=slow_window)
    strategy = run_sma_crossover(getattr(slice_data, "symbol", "UNKNOWN"), timestamps, closes, sma_config)
    strategy_frame = strategy_to_feature_frame(strategy)
    if len(strategy_frame) < 2:
        raise SystemExit("Not enough SMA-aligned rows to build features.")

    return build_features(
        feature_set,
        strategy_frame,
        underlying_symbol=getattr(slice_data, "symbol", "UNKNOWN"),
        start_date=start_date,
        end_date=end_date,
        data_root=data_root,
    )


def _build_risk_config(args: argparse.Namespace) -> RiskConfig:
    return RiskConfig(
        long_only=not bool(getattr(args, "allow_short", False)),
        max_weight=None if args.max_weight is None else float(args.max_weight),
        exposure_cap=None if args.exposure_cap is None else float(args.exposure_cap),
        min_cash=None if args.min_cash is None else float(args.min_cash),
        max_turnover_1d=None if args.max_turnover_1d is None else float(args.max_turnover_1d),
    )


def _normalize_reward_version(value: str | None) -> str:
    token = (value or DEFAULT_REWARD_VERSION).strip()
    return token or DEFAULT_REWARD_VERSION


def _validate_reward_version(value: str) -> str:
    normalized = _normalize_reward_version(value)
    try:
        get_reward_spec(normalized)
    except KeyError as exc:  # pragma: no cover - defensive, exercised via CLI
        available = ", ".join(sorted(reward_versions()))
        raise SystemExit(
            f"Unknown reward_version '{value}'. Available reward versions: {available}"
        ) from exc
    return normalized


def main() -> int:
    args = parse_args()
    payload = run_training(args)
    print(json.dumps(payload, sort_keys=True))
    return 0


def run_training(args: argparse.Namespace) -> Dict[str, Any]:
    symbols = _resolve_symbol_list(args.symbol, getattr(args, "symbols", None))
    universe_mode = bool(getattr(args, "symbols", None))
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
    if universe_mode:
        ensure_yearly_daily_coverage(
            symbols=symbols,
            start=start,
            end=end,
            data_root=data_root,
            auto_build=True,
            run_id_seed=args.run_id,
        )
    reward_version = _validate_reward_version(getattr(args, "reward_version", DEFAULT_REWARD_VERSION))
    args.reward_version = reward_version

    if args.live:
        bootstrap = maybe_run_live_bootstrap(
            symbols=symbols,
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
    ensure_yearly_daily_coverage(
        symbols=symbols,
        start=start,
        end=end,
        data_root=data_root,
        auto_build=False,
        run_id_seed=args.run_id,
    )

    split_requested = any(
        [
            getattr(args, "train_start_date", None),
            getattr(args, "train_end_date", None),
            getattr(args, "test_start_date", None),
            getattr(args, "test_end_date", None),
        ]
    )
    if split_requested:
        train_start = _parse_optional_date(getattr(args, "train_start_date", None))
        train_end = _parse_optional_date(getattr(args, "train_end_date", None))
        test_start = _parse_optional_date(getattr(args, "test_start_date", None))
        test_end = _parse_optional_date(getattr(args, "test_end_date", None))
        if (train_start is None) ^ (train_end is None):
            raise SystemExit("train-start-date and train-end-date must be provided together.")
        if (test_start is None) ^ (test_end is None):
            raise SystemExit("test-start-date and test-end-date must be provided together.")
        if not train_start or not train_end or not test_start or not test_end:
            raise SystemExit("Both train and test windows must be provided when using split dates.")
        if train_end < train_start:
            raise SystemExit("train-end-date cannot be before train-start-date")
        if test_end < test_start:
            raise SystemExit("test-end-date cannot be before test-start-date")
        if train_start < start or train_end > end:
            raise SystemExit("Train window must fall within the requested start/end range.")
        if test_start < start or test_end > end:
            raise SystemExit("Test window must fall within the requested start/end range.")
        if train_end >= test_start:
            raise SystemExit("Train window must end before test window starts.")
    else:
        train_start, train_end = start, end
        test_start, test_end = start, end

    canonical_manifest = bootstrap.canonical_manifest or _locate_canonical_manifest(
        data_root, args.canonical_domain, start, end
    )
    if not canonical_manifest:
        raise SystemExit(
            "No canonical data available for the requested window. "
            "Provide --live to bootstrap or refresh canonicals."
        )
    bootstrap.canonical_manifest = canonical_manifest

    slices, canonical_hashes = load_canonical_equity(symbols, start, end, data_root=data_root, interval=interval)
    for symbol in symbols:
        slice_data = slices.get(symbol)
        if not slice_data:
            raise SystemExit(f"No canonical data found for symbol {symbol}")
        rows = getattr(slice_data, "rows", None)
        frame = getattr(slice_data, "frame", None)
        if not rows and (frame is None or frame.empty):
            raise SystemExit(f"No canonical data found for symbol {symbol}")

    sma_config = SMAStrategyConfig(fast_window=args.fast_window, slow_window=args.slow_window)
    per_symbol_features: Dict[str, FeatureSetResult] = {}
    feature_hashes: Dict[str, str] = {}
    feature_set_name: str | None = None
    multi_symbol = len(symbols) > 1
    normalized_feature_set = normalize_feature_set_name(args.feature_set)
    panel_regime_feature = default_regime_for_feature_set(normalized_feature_set)
    if not multi_symbol and is_universe_feature_set(normalized_feature_set):
        raise SystemExit(f"Feature set '{args.feature_set}' requires at least two symbols (--symbols).")
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
            slice_data = _ensure_canonical_slice(slices[symbol], symbol)
            feature_result = _build_single_symbol_feature_result(
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
        raise SystemExit("Failed to construct feature frames for PPO training.")
    feature_set_name = next(iter(per_symbol_features.values())).feature_set
    for symbol, feature_result in per_symbol_features.items():
        if feature_result.feature_set != feature_set_name:
            raise SystemExit("Feature set mismatch across symbols; ensure a consistent registry entry.")
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
        base_observation_columns = panel.observation_columns
    else:
        primary = symbols[0]
        rows = per_symbol_features[primary].frame.to_dict("records")
        base_observation_columns = per_symbol_features[primary].observation_columns

    if len(rows) < 2:
        raise SystemExit("Not enough feature rows to train PPO.")

    risk_config = _build_risk_config(args)
    env_config = SignalWeightEnvConfig(transaction_cost_bp=args.transaction_cost_bp, risk_config=risk_config)
    train_rows = _slice_rows_by_date(rows, train_start, train_end)
    test_rows = _slice_rows_by_date(rows, test_start, test_end)
    if len(train_rows) < 2:
        raise SystemExit("Not enough feature rows in the training window.")
    if len(test_rows) < 2:
        raise SystemExit("Not enough feature rows in the test window.")
    train_env = GymWeightTradingEnv(train_rows, config=env_config, observation_columns=base_observation_columns)
    eval_env = GymWeightTradingEnv(test_rows, config=env_config, observation_columns=base_observation_columns)
    observation_headers = train_env.inner_env.observation_columns

    try:
        train_start_time = time.perf_counter()
        model = train_ppo(
            train_env,
            total_timesteps=args.timesteps,
            seed=args.seed,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            policy=args.policy,
            reward_version=args.reward_version,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    duration_seconds = time.perf_counter() - train_start_time

    eval_result = evaluate(model, eval_env)
    combined_hashes = dict(canonical_hashes)
    combined_hashes.update(feature_hashes)
    run_id = args.run_id or derive_run_id(
        symbols,
        start,
        end,
        interval,
        args.timesteps,
        args.seed,
        combined_hashes,
        args.fast_window,
        args.slow_window,
        args.transaction_cost_bp,
        risk_config,
        feature_set_name or args.feature_set,
        base_observation_columns,
    )

    reports_dir = data_root / "monitoring" / "reports"
    plots_dir = data_root / "monitoring" / "plots"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    train_report_path = reports_dir / f"ppo_train_{run_id}.json"
    eval_report_path = reports_dir / f"ppo_eval_{run_id}.json"
    plot_path = plots_dir / f"ppo_eval_{run_id}.png"

    plot_weights = _weights_for_plot(eval_result.weights, eval_result.symbols)
    _render_account_weight_plot(plot_path, eval_result.account_values, plot_weights)
    plot_hash = compute_file_hash(plot_path)

    model_path = save_model(model, data_root / "models" / f"ppo_{run_id}.zip")

    data_split = _build_data_split_payload(args, train_start, train_end, test_start, test_end)
    train_payload = build_train_report(
        run_id=run_id,
        symbols=symbols,
        start=train_start,
        end=train_end,
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
        feature_set=feature_set_name or args.feature_set,
        base_observation_columns=base_observation_columns,
        observation_headers=observation_headers,
        data_split=data_split,
    )
    eval_payload = build_eval_report(
        run_id=run_id,
        symbols=symbols,
        start=test_start,
        end=test_end,
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
        feature_set=feature_set_name or args.feature_set,
        base_observation_columns=base_observation_columns,
        observation_headers=observation_headers,
        data_split=data_split,
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


def _weights_series_for_report(
    weight_entries: Sequence[Dict[str, float]],
    symbol_order: Sequence[str],
) -> object:
    if not weight_entries:
        return [] if len(symbol_order) <= 1 else {symbol: [] for symbol in symbol_order}
    if len(symbol_order) <= 1:
        symbol = symbol_order[0] if symbol_order else "asset"
        return [entry.get(symbol, 0.0) for entry in weight_entries]
    return {
        symbol: [entry.get(symbol, 0.0) for entry in weight_entries] for symbol in symbol_order
    }


def _weights_for_plot(weight_entries: Sequence[Dict[str, float]], symbols: Sequence[str]) -> List[float]:
    if not weight_entries:
        return []
    if not symbols:
        return [sum(entry.values()) for entry in weight_entries]
    totals: List[float] = []
    for entry in weight_entries:
        totals.append(sum(entry.get(symbol, 0.0) for symbol in symbols))
    return totals


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
    symbols: Sequence[str],
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
    feature_set: str,
    base_observation_columns: Sequence[str],
    observation_headers: Sequence[str],
    data_split: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    symbol_list = list(symbols)
    primary_symbol = symbol_list[0] if symbol_list else ""
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
        "reward_version": getattr(args, "reward_version", DEFAULT_REWARD_VERSION),
        "duration_seconds": duration_seconds,
    }
    artifacts = {"report": _rel_path(report_path, data_root)}
    if model_path:
        artifacts["model"] = _rel_path(model_path, data_root)
        hashes["model_zip"] = compute_file_hash(model_path)
    payload = {
        "run_id": run_id,
        "symbol": primary_symbol,
        "symbols": symbol_list,
        "interval": interval,
        "date_range": {"start": start.isoformat(), "end": end.isoformat()},
        "training": training_config,
        "env_config": {
            "initial_cash": env_config.initial_cash,
            "transaction_cost_bp": env_config.transaction_cost_bp,
            "risk_config": env_config.risk_config.to_dict(),
        },
        "features": {
            "fast_window": sma_config.fast_window,
            "slow_window": sma_config.slow_window,
            "feature_set": feature_set,
            "observation_columns": list(base_observation_columns),
            "panel_observation_columns": list(observation_headers),
        },
        "bootstrap_mode": bootstrap.mode,
        "bootstrap": bootstrap.as_payload(data_root),
        "hashes": hashes,
        "inputs_used": hashes["canonical_files"],
        "artifacts": artifacts,
    }
    if data_split is not None:
        payload["data_split"] = data_split
    return payload


def build_eval_report(
    *,
    run_id: str,
    symbols: Sequence[str],
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
    feature_set: str,
    base_observation_columns: Sequence[str],
    observation_headers: Sequence[str],
    data_split: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    symbol_list = list(symbols)
    primary_symbol = symbol_list[0] if symbol_list else ""
    hashes = {
        "canonical_files": dict(sorted(canonical_hashes.items())),
        "plot_png": plot_hash,
        "report_json": "",
    }
    weights_series = _weights_series_for_report(eval_result.weights, eval_result.symbols)
    series = {
        "timestamps": eval_result.timestamps,
        "account_value": eval_result.account_values,
        "weights": weights_series,
        "log_returns": eval_result.log_returns,
    }
    if eval_result.regime_features and eval_result.regime_feature_names:
        series["regime"] = {
            "feature_names": list(eval_result.regime_feature_names),
            "values": [list(snapshot) for snapshot in eval_result.regime_features],
        }
    artifacts = {
        "report": _rel_path(report_path, data_root),
        "plot": _rel_path(plot_path, data_root),
    }
    payload = {
        "run_id": run_id,
        "symbol": primary_symbol,
        "symbols": symbol_list,
        "interval": interval,
        "date_range": {"start": start.isoformat(), "end": end.isoformat()},
        "metrics": eval_result.metrics,
        "series": series,
        "steps": eval_result.steps,
        "env_config": {
            "initial_cash": env_config.initial_cash,
            "transaction_cost_bp": env_config.transaction_cost_bp,
            "risk_config": env_config.risk_config.to_dict(),
        },
        "features": {
            "fast_window": sma_config.fast_window,
            "slow_window": sma_config.slow_window,
            "feature_set": feature_set,
            "observation_columns": list(base_observation_columns),
            "panel_observation_columns": list(observation_headers),
        },
        "bootstrap_mode": bootstrap.mode,
        "bootstrap": bootstrap.as_payload(data_root),
        "hashes": hashes,
        "inputs_used": hashes["canonical_files"],
        "artifacts": artifacts,
    }
    if data_split is not None:
        payload["data_split"] = data_split
    return payload


def derive_run_id(
    symbols: Sequence[str],
    start: date,
    end: date,
    interval: str,
    timesteps: int,
    seed: int,
    canonical_hashes: Dict[str, str],
    fast_window: int,
    slow_window: int,
    transaction_cost_bp: float,
    risk_config: RiskConfig,
    feature_set: str,
    observation_columns: Sequence[str],
) -> str:
    ordered_symbols = list(symbols)
    canonical: Dict[str, Any]
    if len(ordered_symbols) == 1:
        canonical = {
            "symbol": ordered_symbols[0],
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "interval": interval,
            "timesteps": timesteps,
            "seed": seed,
            "fast_window": fast_window,
            "slow_window": slow_window,
            "transaction_cost_bp": transaction_cost_bp,
            "risk_config": risk_config.to_dict(),
            "canonical_hashes": {key: canonical_hashes[key] for key in sorted(canonical_hashes)},
            "feature_set": feature_set,
            "observation_columns": list(observation_columns),
        }
    else:
        canonical = {
            "symbols": ordered_symbols,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "interval": interval,
            "timesteps": timesteps,
            "seed": seed,
            "fast_window": fast_window,
            "slow_window": slow_window,
            "transaction_cost_bp": transaction_cost_bp,
            "risk_config": risk_config.to_dict(),
            "canonical_hashes": {key: canonical_hashes[key] for key in sorted(canonical_hashes)},
            "feature_set": feature_set,
            "observation_columns": list(observation_columns),
        }
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return f"ppo_weight_{digest[:12]}"


def _parse_optional_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(str(value))


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
    raise SystemExit("Row timestamp is missing or invalid.")


def _build_data_split_payload(
    args: argparse.Namespace,
    train_start: date,
    train_end: date,
    test_start: date,
    test_end: date,
) -> Dict[str, Any] | None:
    if not any(
        [
            getattr(args, "train_start_date", None),
            getattr(args, "train_end_date", None),
            getattr(args, "test_start_date", None),
            getattr(args, "test_end_date", None),
        ]
    ):
        return None
    return {
        "train_start": train_start.isoformat(),
        "train_end": train_end.isoformat(),
        "test_start": test_start.isoformat(),
        "test_end": test_end.isoformat(),
        "train_ratio": float(args.train_ratio) if args.train_ratio is not None else None,
        "test_ratio": float(args.test_ratio) if args.test_ratio is not None else None,
        "test_window_months": int(args.test_window_months) if args.test_window_months is not None else None,
    }


__all__ = ["parse_args", "run_training", "main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
