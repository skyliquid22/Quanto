#!/usr/bin/env python3
"""SMA crossover backtest CLI on canonical equity data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import date
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import guard
    sys.path.insert(0, str(PROJECT_ROOT))

from infra.paths import get_data_root as default_data_root
from infra.normalization.lineage import compute_file_hash
from plots.equity_curves import render_equity_curves
from research.backtests.sma_backtest import (
    AggregatedBacktest,
    BacktestResult,
    aggregate_results,
    run_backtest,
    serialize_metrics,
)
from research.datasets.canonical_equity_loader import load_canonical_equity
from research.strategies.sma_crossover import SMAStrategyConfig, SMAStrategyResult, run_sma_crossover


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a deterministic SMA crossover backtest.")
    parser.add_argument("--symbols", required=True, help="Comma separated list of symbols.")
    parser.add_argument("--start-date", required=True, help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", required=True, help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument("--fast", type=int, default=20, help="Fast SMA window.")
    parser.add_argument("--slow", type=int, default=50, help="Slow SMA window.")
    parser.add_argument("--run-id", help="Optional run identifier; derived deterministically if omitted.")
    parser.add_argument("--data-root", help="Override data root directory.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="If set, run the v1 slice offline workflow to synthesize canonical data first.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    symbols = _parse_symbols(args.symbols)
    if not symbols:
        raise SystemExit("No symbols were provided.")
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    if end < start:
        raise SystemExit("end-date cannot be before start-date")
    if args.fast >= args.slow:
        raise SystemExit("fast window must be smaller than slow window")

    data_root = resolve_data_root(args.data_root)
    os.environ["QUANTO_DATA_ROOT"] = str(data_root)
    if args.offline:
        _run_offline_slice(symbols, start, end, data_root)

    slices, canonical_hashes = load_canonical_equity(symbols, start, end, data_root=data_root)
    if any(not slice.rows for slice in slices.values()):
        missing = [symbol for symbol, slice in slices.items() if not slice.rows]
        raise SystemExit(f"No canonical data found for: {', '.join(missing)}")
    config = SMAStrategyConfig(fast_window=args.fast, slow_window=args.slow)
    strategy_results = {
        symbol: run_sma_crossover(symbol, slice.timestamps, slice.closes, config) for symbol, slice in slices.items()
    }
    backtest_results: Dict[str, BacktestResult] = {
        symbol: run_backtest(strategy_results[symbol]) for symbol in strategy_results
    }
    aggregate = aggregate_results(list(backtest_results.values()))

    run_id = args.run_id or derive_run_id(symbols, start, end, config, canonical_hashes)
    report_path = data_root / "monitoring" / "reports" / f"sma_crossover_{run_id}.json"
    plot_path = data_root / "monitoring" / "plots" / f"sma_crossover_{run_id}.png"
    plot_symbol = sorted(strategy_results)[0]
    _render_plot(plot_path, backtest_results[plot_symbol])

    hashes = {
        "canonical_files": canonical_hashes,
        "plot_png": compute_file_hash(plot_path),
        "report_json": "",
    }
    artifacts = {"report": _rel_path(report_path, data_root), "plot": _rel_path(plot_path, data_root)}
    payload = build_report_payload(
        run_id,
        data_root,
        start,
        end,
        config,
        strategy_results,
        backtest_results,
        aggregate,
        hashes,
        artifacts,
    )
    write_report(payload, report_path)
    print(json.dumps({"report": str(report_path), "plot": str(plot_path)}, separators=(",", ":"), sort_keys=True))
    return 0


def resolve_data_root(arg_root: str | None) -> Path:
    if arg_root:
        return Path(arg_root).expanduser()
    env_override = os.environ.get("QUANTO_DATA_ROOT")
    if env_override:
        return Path(env_override).expanduser()
    return default_data_root()


def _parse_symbols(raw: str) -> List[str]:
    seen = set()
    parsed: List[str] = []
    for part in raw.split(","):
        clean = part.strip().upper()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        parsed.append(clean)
    return parsed


def _run_offline_slice(symbols: List[str], start: date, end: date, data_root: Path) -> None:
    config = {
        "offline_ingestion": {
            "vendor": "polygon",
            "symbols": symbols,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        },
        "canonical": {
            "config_path": "configs/data_sources.json",
            "domains": ["equity_ohlcv"],
        },
        "reporting": {"plot_symbol": symbols[0]},
    }
    text = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        handle.write(text)
        config_path = Path(handle.name)
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(PROJECT_ROOT))
    env["QUANTO_DATA_ROOT"] = str(data_root)
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_v1_slice",
        "--config",
        str(config_path),
        "--data-root",
        str(data_root),
        "--offline",
    ]
    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            sys.stderr.write(result.stdout)
            sys.stderr.write(result.stderr)
            raise RuntimeError("offline slice failed")
    finally:
        try:
            config_path.unlink()
        except FileNotFoundError:
            pass


def derive_run_id(
    symbols: List[str],
    start: date,
    end: date,
    config: SMAStrategyConfig,
    canonical_hashes: Mapping[str, str],
) -> str:
    canonical = {
        "symbols": symbols,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "fast": config.fast_window,
        "slow": config.slow_window,
        "canonical_hashes": {path: canonical_hashes[path] for path in sorted(canonical_hashes)},
    }
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return f"sma_{digest[:12]}"


def _render_plot(path: Path, result: BacktestResult) -> Path:
    curves = {
        "strategy": result.equity_curve,
        "buy_and_hold": result.buy_and_hold_curve,
    }
    return render_equity_curves(path, curves)


def build_report_payload(
    run_id: str,
    data_root: Path,
    start: date,
    end: date,
    config: SMAStrategyConfig,
    strategies: Mapping[str, SMAStrategyResult],
    backtests: Mapping[str, BacktestResult],
    aggregate: AggregatedBacktest,
    hashes: Mapping[str, Any],
    artifacts: Mapping[str, str],
) -> Dict[str, Any]:
    ordered_symbols = sorted(strategies)
    per_symbol = {}
    for symbol in ordered_symbols:
        strategy = strategies[symbol]
        backtest = backtests[symbol]
        per_symbol[symbol] = {
            "timestamps": [ts.isoformat() for ts in strategy.timestamps],
            "close": strategy.closes,
            "fast_sma": strategy.fast_sma,
            "slow_sma": strategy.slow_sma,
            "signal": strategy.signal,
            "positions": strategy.positions,
            "equity_curve": backtest.equity_curve,
            "buy_and_hold_curve": backtest.buy_and_hold_curve,
            "metrics": serialize_metrics(backtest.metrics),
        }
    aggregate_payload = {
        "symbols": ordered_symbols,
        "metrics": serialize_metrics(aggregate.metrics),
        "equity_curve": aggregate.equity_curve,
    }
    return {
        "run_id": run_id,
        "data_root": str(data_root),
        "parameters": {
            "symbols": ordered_symbols,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "fast_window": config.fast_window,
            "slow_window": config.slow_window,
        },
        "artifacts": artifacts,
        "hashes": dict(hashes),
        "symbols": per_symbol,
        "aggregate": aggregate_payload,
    }


def write_report(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    copy_payload = json.loads(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    if "hashes" in copy_payload:
        copy_payload["hashes"]["report_json"] = ""
    canonical_bytes = json.dumps(copy_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    payload["hashes"]["report_json"] = f"sha256:{hashlib.sha256(canonical_bytes).hexdigest()}"
    final_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    path.write_bytes(final_bytes + b"\n")


def _rel_path(path: Path, data_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(data_root.resolve()))
    except ValueError:
        return str(path)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
