#!/usr/bin/env python3
"""SMA crossover backtest CLI on canonical equity data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import date
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import guard
    sys.path.insert(0, str(PROJECT_ROOT))

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
from scripts.run_sma_finrl_rollout import BootstrapMetadata, ensure_yearly_daily_coverage, maybe_run_live_bootstrap, resolve_data_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a deterministic SMA crossover backtest.")
    parser.add_argument("--symbols", required=True, help="Comma separated list of symbols.")
    parser.add_argument("--start-date", required=True, help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", required=True, help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument("--fast", type=int, default=20, help="Fast SMA window.")
    parser.add_argument("--slow", type=int, default=50, help="Slow SMA window.")
    parser.add_argument("--run-id", help="Optional run identifier; derived deterministically if omitted.")
    parser.add_argument("--data-root", help="Override data root directory.")
    parser.add_argument("--vendor", default="polygon", help="Vendor used for live ingestion bootstrap.")
    parser.add_argument(
        "--ingest-mode",
        choices=["rest", "flat_file", "auto"],
        default="rest",
        help="Force a deterministic ingestion mode during live bootstraps.",
    )
    parser.add_argument(
        "--canonical-domain",
        default="equity_ohlcv",
        help="Canonical domain identifier used for freshness checks and builds.",
    )
    parser.add_argument(
        "--force-ingest",
        action="store_true",
        help="Always run live ingestion before the backtest (implies canonical rebuild).",
    )
    parser.add_argument(
        "--force-canonical-build",
        action="store_true",
        help="Always rebuild canonicals before the backtest.",
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
    ensure_yearly_daily_coverage(
        symbols=symbols,
        start=start,
        end=end,
        data_root=data_root,
        auto_build=True,
        run_id_seed=args.run_id,
    )
    os.environ["QUANTO_DATA_ROOT"] = str(data_root)
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
    ensure_yearly_daily_coverage(
        symbols=symbols,
        start=start,
        end=end,
        data_root=data_root,
        auto_build=False,
        run_id_seed=args.run_id,
    )

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
        bootstrap,
    )
    write_report(payload, report_path)
    print(json.dumps({"report": str(report_path), "plot": str(plot_path)}, separators=(",", ":"), sort_keys=True))
    return 0


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
    bootstrap: BootstrapMetadata | None = None,
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
    payload = {
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
    if bootstrap:
        payload["bootstrap"] = bootstrap.as_payload(data_root)
    return payload


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
