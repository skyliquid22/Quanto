#!/usr/bin/env python3
"""SMA-driven FinRL rollout producing deterministic monitoring artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import date
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(PROJECT_ROOT))

from infra.normalization.lineage import compute_file_hash
from infra.paths import get_data_root as default_data_root
from research.datasets.canonical_equity_loader import load_canonical_equity
from research.envs.signal_weight_env import SignalWeightEnvConfig, SignalWeightTradingEnv
from research.policies.sma_weight_policy import SMAWeightPolicy, SMAWeightPolicyConfig
from research.runners.rollout import RolloutResult, run_rollout
from research.strategies.sma_crossover import SMAStrategyConfig, SMAStrategyResult, run_sma_crossover


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic SMA rollout in a FinRL-style environment.")
    parser.add_argument("--symbol", default="AAPL", help="Single equity symbol to backtest.")
    parser.add_argument("--start-date", required=True, help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", required=True, help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument("--fast", type=int, default=20, help="Fast SMA window.")
    parser.add_argument("--slow", type=int, default=50, help="Slow SMA window.")
    parser.add_argument("--transaction-cost-bp", type=float, default=1.0, help="Round-trip transaction cost in basis points.")
    parser.add_argument("--policy-mode", choices=["hard", "sigmoid"], default="hard", help="Mapping used by the SMA policy.")
    parser.add_argument("--sigmoid-scale", type=float, default=5.0, help="Scale factor for sigmoid mode.")
    parser.add_argument("--run-id", help="Optional run identifier; derived deterministically if omitted.")
    parser.add_argument("--data-root", help="Override data root directory.")
    parser.add_argument("--offline", action="store_true", help="If set, run the offline slice to materialize canonicals.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    symbol = args.symbol.upper().strip()
    if not symbol:
        raise SystemExit("symbol must be provided")
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    if end < start:
        raise SystemExit("end-date cannot be before start-date")
    if args.fast >= args.slow:
        raise SystemExit("fast window must be smaller than slow window")
    if args.policy_mode == "sigmoid" and args.sigmoid_scale <= 0:
        raise SystemExit("sigmoid-scale must be positive when using sigmoid mode")

    data_root = resolve_data_root(args.data_root)
    os.environ["QUANTO_DATA_ROOT"] = str(data_root)
    if args.offline:
        _run_offline_slice(symbol, start, end, data_root)

    slices, canonical_hashes = load_canonical_equity([symbol], start, end, data_root=data_root)
    if not slices.get(symbol) or not slices[symbol].rows:
        raise SystemExit(f"No canonical data found for symbol {symbol}")

    sma_config = SMAStrategyConfig(fast_window=args.fast, slow_window=args.slow)
    strategy = run_sma_crossover(symbol, slices[symbol].timestamps, slices[symbol].closes, sma_config)
    rows = _build_feature_rows(strategy)
    if len(rows) < 2:
        raise SystemExit("Not enough SMA-aligned rows to run the rollout")

    env_config = SignalWeightEnvConfig(transaction_cost_bp=args.transaction_cost_bp)
    env = SignalWeightTradingEnv(rows, env_config)
    policy = SMAWeightPolicy(SMAWeightPolicyConfig(mode=args.policy_mode, sigmoid_scale=args.sigmoid_scale))
    result = run_rollout(env, policy, inputs_used=canonical_hashes)

    run_id = args.run_id or derive_run_id(
        symbol,
        start,
        end,
        sma_config,
        env_config,
        policy,
        result.inputs_used,
    )

    report_path = data_root / "monitoring" / "reports" / f"sma_finrl_rollout_{run_id}.json"
    plot_path = data_root / "monitoring" / "plots" / f"sma_finrl_rollout_{run_id}.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _render_account_weight_plot(plot_path, result.account_values, result.weights)

    hashes = {
        "canonical_files": dict(sorted(result.inputs_used.items())),
        "plot_png": compute_file_hash(plot_path),
        "report_json": "",
    }
    payload = build_report_payload(
        run_id=run_id,
        symbol=symbol,
        start=start,
        end=end,
        sma_config=sma_config,
        env_config=env_config,
        policy=policy,
        result=result,
        hashes=hashes,
        data_root=data_root,
        report_path=report_path,
        plot_path=plot_path,
    )
    _write_report(report_path, payload)
    print(
        json.dumps(
            {"report": str(report_path), "plot": str(plot_path), "run_id": run_id},
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


def resolve_data_root(arg_root: str | None) -> Path:
    if arg_root:
        return Path(arg_root).expanduser()
    override = os.environ.get("QUANTO_DATA_ROOT")
    if override:
        return Path(override).expanduser()
    return default_data_root()


def _run_offline_slice(symbol: str, start: date, end: date, data_root: Path) -> None:
    config = {
        "offline_ingestion": {
            "vendor": "polygon",
            "symbols": [symbol],
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        },
        "canonical": {
            "config_path": "configs/data_sources.json",
            "domains": ["equity_ohlcv"],
        },
        "reporting": {"plot_symbol": symbol},
    }
    config_path = data_root / "tmp_sma_rollout_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
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
        completed = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
        if completed.returncode != 0:
            sys.stderr.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            raise RuntimeError("offline slice failed")
    finally:
        try:
            config_path.unlink()
        except FileNotFoundError:
            pass


def _build_feature_rows(strategy: SMAStrategyResult) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, timestamp in enumerate(strategy.timestamps):
        fast = strategy.fast_sma[idx]
        slow = strategy.slow_sma[idx]
        if fast is None or slow is None:
            continue
        close = strategy.closes[idx]
        rows.append(
            {
                "timestamp": timestamp,
                "close": float(close),
                "sma_fast": float(fast),
                "sma_slow": float(slow),
                "sma_diff": float(fast - slow),
                "sma_signal": float(strategy.signal[idx]),
            }
        )
    return rows


def derive_run_id(
    symbol: str,
    start: date,
    end: date,
    sma_config: SMAStrategyConfig,
    env_config: SignalWeightEnvConfig,
    policy: SMAWeightPolicy,
    canonical_hashes: Dict[str, str],
) -> str:
    canonical = {
        "symbol": symbol,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "fast_window": sma_config.fast_window,
        "slow_window": sma_config.slow_window,
        "transaction_cost_bp": env_config.transaction_cost_bp,
        "policy_mode": policy.mode,
        "sigmoid_scale": policy.config.sigmoid_scale,
        "canonical_hashes": {key: canonical_hashes[key] for key in sorted(canonical_hashes)},
    }
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return f"sma_finrl_{digest[:12]}"


def build_report_payload(
    *,
    run_id: str,
    symbol: str,
    start: date,
    end: date,
    sma_config: SMAStrategyConfig,
    env_config: SignalWeightEnvConfig,
    policy: SMAWeightPolicy,
    result: RolloutResult,
    hashes: Dict[str, Any],
    data_root: Path,
    report_path: Path,
    plot_path: Path,
) -> Dict[str, Any]:
    parameters = {
        "symbol": symbol,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "fast_window": sma_config.fast_window,
        "slow_window": sma_config.slow_window,
        "transaction_cost_bp": env_config.transaction_cost_bp,
        "initial_cash": env_config.initial_cash,
        "policy_mode": policy.mode,
        "sigmoid_scale": policy.config.sigmoid_scale,
    }
    series = {
        "timestamps": result.timestamps,
        "account_value": result.account_values,
        "weights": result.weights,
        "log_returns": result.log_returns,
    }
    artifacts = {
        "report": _rel_path(report_path, data_root),
        "plot": _rel_path(plot_path, data_root),
    }
    return {
        "run_id": run_id,
        "symbol": symbol,
        "date_range": {"start": start.isoformat(), "end": end.isoformat()},
        "parameters": parameters,
        "metrics": result.metrics,
        "series": series,
        "steps": result.steps,
        "inputs_used": result.inputs_used,
        "artifacts": artifacts,
        "hashes": hashes,
    }


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    copy_payload = json.loads(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
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


def _render_account_weight_plot(path: Path, account_values: List[float], weights: List[float]) -> None:
    width, height = 760, 480
    pixels = bytearray([255] * width * height * 3)
    margin_left, margin_right = 60, 20
    top_height = height // 2 - 20
    bottom_height = height - top_height - 40

    def draw_axes(y_offset: int, panel_height: int) -> None:
        axis_color = (0, 0, 0)
        x_axis_y = y_offset + panel_height - 20
        for x in range(margin_left, width - margin_right):
            _set_pixel(pixels, width, height, x, x_axis_y, axis_color)
        for y in range(y_offset + 10, x_axis_y + 1):
            _set_pixel(pixels, width, height, margin_left, y, axis_color)

    def draw_series(series: List[float], color, y_offset: int, panel_height: int, value_range: tuple[float, float]) -> None:
        if not series:
            return
        min_val, max_val = value_range
        if min_val == max_val:
            min_val -= 0.5
            max_val += 0.5
        usable_height = panel_height - 40
        total_points = max(len(series) - 1, 1)
        x_span = width - margin_left - margin_right
        for idx in range(len(series)):
            ratio = (series[idx] - min_val) / (max_val - min_val)
            ratio = max(0.0, min(1.0, ratio))
            x = margin_left + int(round(x_span * (idx / total_points)))
            y = y_offset + 20 + int(round((1 - ratio) * usable_height))
            if idx > 0:
                prev_ratio = (series[idx - 1] - min_val) / (max_val - min_val)
                prev_ratio = max(0.0, min(1.0, prev_ratio))
                prev_x = margin_left + int(round(x_span * ((idx - 1) / total_points)))
                prev_y = y_offset + 20 + int(round((1 - prev_ratio) * usable_height))
                _draw_line(pixels, width, height, prev_x, prev_y, x, y, color)
            _set_pixel(pixels, width, height, x, y, color)

    draw_axes(0, top_height)
    draw_axes(height - bottom_height, bottom_height)
    if account_values:
        draw_series(account_values, (31, 119, 180), 0, top_height, (min(account_values), max(account_values)))
    if weights:
        draw_series(weights, (214, 39, 40), height - bottom_height, bottom_height, (0.0, 1.0))
    _write_png(path, pixels, width, height)


def _set_pixel(pixels: bytearray, width: int, height: int, x: int, y: int, color) -> None:
    if not (0 <= x < width and 0 <= y < height):
        return
    index = (y * width + x) * 3
    pixels[index : index + 3] = bytes(color)


def _draw_line(pixels: bytearray, width: int, height: int, x0: int, y0: int, x1: int, y1: int, color) -> None:
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    steps = max(dx, dy, 1)
    for step in range(steps + 1):
        t = step / steps
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        _set_pixel(pixels, width, height, x, y, color)


def _write_png(path: Path, pixels: bytearray, width: int, height: int) -> None:
    import struct
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    rows = bytearray()
    row_bytes = width * 3
    for y in range(height):
        start = y * row_bytes
        rows.append(0)
        rows.extend(pixels[start : start + row_bytes])

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(rows), level=9)
    with path.open("wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        handle.write(chunk(b"IHDR", ihdr))
        handle.write(chunk(b"IDAT", idat))
        handle.write(chunk(b"IEND", b""))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
