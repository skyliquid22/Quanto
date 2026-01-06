#!/usr/bin/env python3
"""SMA-driven FinRL rollout producing deterministic monitoring artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(PROJECT_ROOT))

from infra.normalization.lineage import compute_file_hash
from infra.paths import get_data_root as default_data_root
from research.datasets.canonical_equity_loader import load_canonical_equity
from research.envs.signal_weight_env import SignalWeightEnvConfig, SignalWeightTradingEnv
from research.features.feature_registry import FeatureSetResult, build_features, strategy_to_feature_frame
from research.policies.sma_weight_policy import SMAWeightPolicy, SMAWeightPolicyConfig
from research.runners.rollout import RolloutResult, run_rollout
from research.strategies.sma_crossover import SMAStrategyConfig, run_sma_crossover


@dataclass(frozen=True)
class ManifestMetadata:
    path: Path
    run_id: str
    start_date: date
    end_date: date
    created_at: Optional[datetime]
    domain: str
    vendor: Optional[str] = None

    def covers(self, start: date, end: date) -> bool:
        return self.start_date <= start and self.end_date >= end

    def as_report(self, data_root: Path) -> Dict[str, str]:
        return {
            "run_id": self.run_id,
            "path": _rel_path(self.path, data_root),
            "hash": compute_file_hash(self.path),
        }


@dataclass
class BootstrapMetadata:
    mode: str
    raw_manifests: List[ManifestMetadata] = field(default_factory=list)
    canonical_manifest: ManifestMetadata | None = None
    refreshed: bool = False

    def as_payload(self, data_root: Path) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "refreshed": self.refreshed,
            "raw_manifests": [manifest.as_report(data_root) for manifest in self.raw_manifests],
            "canonical_manifest": self.canonical_manifest.as_report(data_root) if self.canonical_manifest else None,
        }


@dataclass(frozen=True)
class BootstrapDecision:
    should_ingest: bool
    should_build: bool
    raw_manifest: ManifestMetadata | None
    canonical_manifest: ManifestMetadata | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic SMA rollout in a FinRL-style environment.")
    parser.add_argument("--symbol", default="AAPL", help="Single equity symbol to backtest.")
    parser.add_argument("--start-date", required=True, help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", required=True, help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument("--interval", default="daily", help="Bar interval to use (daily only in v1).")
    parser.add_argument("--fast", type=int, default=20, help="Fast SMA window.")
    parser.add_argument("--slow", type=int, default=50, help="Slow SMA window.")
    parser.add_argument("--transaction-cost-bp", type=float, default=1.0, help="Round-trip transaction cost in basis points.")
    parser.add_argument("--policy-mode", choices=["hard", "sigmoid"], default="hard", help="Mapping used by the SMA policy.")
    parser.add_argument("--sigmoid-scale", type=float, default=5.0, help="Scale factor for sigmoid mode.")
    parser.add_argument(
        "--feature-set",
        choices=["sma_v1", "options_v1", "sma_plus_options_v1"],
        default="sma_v1",
        help="Feature set used for environment observations.",
    )
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
        help="Always run live ingestion before the rollout (implies canonical rebuild).",
    )
    parser.add_argument(
        "--force-canonical-build",
        action="store_true",
        help="Always rebuild canonicals before the rollout.",
    )
    return parser.parse_args()


def needs_canonical_refresh(
    *,
    data_root: Path,
    domain: str,
    symbols: Sequence[str],
    start: date,
    end: date,
    vendor: str,
    force_ingest: bool = False,
    force_canonical: bool = False,
) -> BootstrapDecision:
    canonical_manifest = _locate_canonical_manifest(data_root, domain, start, end)
    outputs_present = _canonical_files_exist(data_root, domain, symbols, start, end)
    raw_manifest = _locate_raw_manifest(data_root, domain, vendor, start, end)

    coverage_missing = canonical_manifest is None
    files_missing = not outputs_present
    canonical_older_than_raw = False
    if canonical_manifest and raw_manifest and raw_manifest.created_at:
        if canonical_manifest.created_at:
            canonical_older_than_raw = canonical_manifest.created_at < raw_manifest.created_at
        else:
            canonical_older_than_raw = True

    should_build = force_canonical or coverage_missing or files_missing or canonical_older_than_raw
    should_ingest = force_ingest or raw_manifest is None
    if should_ingest:
        should_build = True
    return BootstrapDecision(
        should_ingest=should_ingest,
        should_build=should_build,
        raw_manifest=raw_manifest,
        canonical_manifest=canonical_manifest,
    )


def _locate_canonical_manifest(data_root: Path, domain: str, start: date, end: date) -> ManifestMetadata | None:
    manifest_dir = data_root / "canonical" / "manifests" / domain
    return _find_covering_manifest(manifest_dir, domain=domain, start=start, end=end)


def _locate_raw_manifest(data_root: Path, domain: str, vendor: str, start: date, end: date) -> ManifestMetadata | None:
    manifest_dir = data_root / "raw" / vendor / domain / "manifests"
    return _find_covering_manifest(manifest_dir, domain=domain, start=start, end=end, vendor=vendor)


def _find_covering_manifest(
    manifest_dir: Path,
    *,
    domain: str,
    start: date,
    end: date,
    vendor: str | None = None,
) -> ManifestMetadata | None:
    if not manifest_dir.exists():
        return None
    candidates: List[ManifestMetadata] = []
    for path in sorted(manifest_dir.glob("*.json")):
        try:
            metadata = _load_manifest_metadata(path, domain=domain, vendor=vendor)
        except Exception:
            continue
        if metadata.covers(start, end):
            candidates.append(metadata)
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item.created_at or datetime.min, item.run_id), reverse=True)
    return candidates[0]


def _canonical_files_exist(data_root: Path, domain: str, symbols: Sequence[str], start: date, end: date) -> bool:
    if not symbols:
        return False
    if domain != "equity_ohlcv":
        base = data_root / "canonical" / domain
        # Unknown layout; best-effort existence check.
        return base.exists()
    for symbol in symbols:
        base = data_root / "canonical" / domain / symbol / "daily"
        has_data = False
        for day in _iter_days(start, end):
            path = base / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.day:02d}.parquet"
            if path.exists():
                has_data = True
                break
        if not has_data:
            return False
    return True


def _load_manifest_metadata(path: Path, *, domain: str, vendor: str | None = None) -> ManifestMetadata:
    payload = json.loads(path.read_text())
    run_id = str(payload.get("run_id") or path.stem)
    start = date.fromisoformat(str(payload["start_date"]))
    end = date.fromisoformat(str(payload["end_date"]))
    created_text = payload.get("created_at") or payload.get("creation_timestamp")
    created_at = _parse_iso_datetime(created_text) if created_text else None
    return ManifestMetadata(
        path=path,
        run_id=run_id,
        start_date=start,
        end_date=end,
        created_at=created_at,
        domain=domain,
        vendor=vendor,
    )


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def maybe_run_live_bootstrap(
    *,
    symbols: Sequence[str],
    start: date,
    end: date,
    data_root: Path,
    domain: str,
    vendor: str,
    ingest_mode: str,
    force_ingest: bool,
    force_canonical: bool,
    run_id_seed: str | None = None,
) -> BootstrapMetadata:
    if domain != "equity_ohlcv":
        raise ValueError("Live bootstrap currently supports only the equity_ohlcv domain")
    seen = set()
    ordered_symbols: List[str] = []
    for symbol in symbols:
        clean = str(symbol).strip().upper()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        ordered_symbols.append(clean)
    if not ordered_symbols:
        raise ValueError("At least one symbol is required for live bootstrap")
    decision = needs_canonical_refresh(
        data_root=data_root,
        domain=domain,
        symbols=ordered_symbols,
        start=start,
        end=end,
        vendor=vendor,
        force_ingest=force_ingest,
        force_canonical=force_canonical,
    )
    requires_refresh = decision.should_ingest or decision.should_build
    bootstrap = BootstrapMetadata(
        mode="live" if requires_refresh else "none",
        canonical_manifest=decision.canonical_manifest,
    )
    refreshed = False
    raw_manifests: List[ManifestMetadata] = []
    latest_raw_manifest = decision.raw_manifest

    if decision.should_ingest:
        _ensure_polygon_credentials_available(ingest_mode)
        ingest_run_id = _derive_bootstrap_run_id("live_ingest", ordered_symbols, start, end, vendor, seed=run_id_seed)
        ingest_manifest_path = _run_live_ingestion(
            symbols=ordered_symbols,
            start=start,
            end=end,
            data_root=data_root,
            vendor=vendor,
            run_id=ingest_run_id,
            ingest_mode=ingest_mode,
        )
        latest_raw_manifest = _load_manifest_metadata(ingest_manifest_path, domain=domain, vendor=vendor)
        raw_manifests = [latest_raw_manifest]
        refreshed = True
    elif latest_raw_manifest:
        raw_manifests = [latest_raw_manifest]

    canonical_manifest = decision.canonical_manifest
    if decision.should_build:
        canonical_run_id = _derive_bootstrap_run_id("live_canonical", ordered_symbols, start, end, vendor, seed=run_id_seed)
        canonical_manifest_path = _run_canonical_build(
            data_root=data_root,
            domain=domain,
            start=start,
            end=end,
            run_id=canonical_run_id,
        )
        canonical_manifest = _load_manifest_metadata(canonical_manifest_path, domain=domain)
        refreshed = True

    bootstrap.raw_manifests = raw_manifests
    bootstrap.canonical_manifest = canonical_manifest
    bootstrap.refreshed = refreshed
    return bootstrap


def _derive_bootstrap_run_id(
    prefix: str,
    symbols: Sequence[str],
    start: date,
    end: date,
    vendor: str,
    *,
    seed: str | None = None,
) -> str:
    symbol_key = ",".join(sorted(symbols)) or "*"
    base = seed or f"{symbol_key}_{start.isoformat()}_{end.isoformat()}_{vendor}"
    digest = hashlib.sha256(f"{prefix}:{base}".encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:10]}"


def _ensure_polygon_credentials_available(ingest_mode: str) -> None:
    if ingest_mode == "flat_file":
        return
    api_key = os.environ.get("POLYGON_API_KEY")
    if api_key:
        return
    raise SystemExit(
        "Live bootstrap requires the POLYGON_API_KEY environment variable when canonical data must be ingested. "
        "Set the key or rerun with --ingest-mode flat_file if you have prepared local fixtures."
    )


def _run_live_ingestion(
    *,
    symbols: Sequence[str],
    start: date,
    end: date,
    data_root: Path,
    vendor: str,
    run_id: str,
    ingest_mode: str,
) -> Path:
    config_payload: Dict[str, Any] = {
        "symbols": list(symbols),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "vendor": vendor,
        "raw_base_path": str(data_root / "raw"),
        "manifest_dir": str(data_root / "raw" / vendor / "equity_ohlcv" / "manifests"),
    }
    tmp_config = _write_temporary_config(config_payload)
    cmd = [
        sys.executable,
        "-m",
        "scripts.ingest_equity_ohlcv",
        "--config",
        str(tmp_config),
        "--run-id",
        run_id,
        "--mode",
        ingest_mode,
        "--data-root",
        str(data_root),
    ]
    env = _subprocess_env(data_root)
    try:
        subprocess.check_call(cmd, cwd=PROJECT_ROOT, env=env)
    finally:
        try:
            tmp_config.unlink()
        except FileNotFoundError:
            pass

    manifest_path = data_root / "raw" / vendor / "equity_ohlcv" / "manifests" / f"{run_id}.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Ingestion manifest missing at {manifest_path}")
    return manifest_path


def _run_canonical_build(
    *,
    data_root: Path,
    domain: str,
    start: date,
    end: date,
    run_id: str,
) -> Path:
    cmd = [
        sys.executable,
        "-m",
        "scripts.build_canonical_datasets",
        "--domains",
        domain,
        "--start-date",
        start.isoformat(),
        "--end-date",
        end.isoformat(),
        "--run-id",
        run_id,
        "--raw-root",
        str(data_root / "raw"),
        "--canonical-root",
        str(data_root / "canonical"),
        "--manifest-root",
        str(data_root / "canonical" / "manifests"),
        "--metrics-root",
        str(data_root / "canonical" / "metrics"),
    ]
    env = _subprocess_env(data_root)
    subprocess.check_call(cmd, cwd=PROJECT_ROOT, env=env)
    manifest_path = data_root / "canonical" / "manifests" / domain / f"{run_id}.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Canonical manifest missing at {manifest_path}")
    return manifest_path


def _subprocess_env(data_root: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(PROJECT_ROOT))
    env["QUANTO_DATA_ROOT"] = str(data_root)
    return env


def _write_temporary_config(payload: Mapping[str, Any]) -> Path:
    handle = tempfile.NamedTemporaryFile("w", delete=False, prefix="quanto_live_bootstrap_", suffix=".json")
    with handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return Path(handle.name)


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

    interval = str(args.interval).strip().lower()
    if interval != "daily":
        raise SystemExit("Only the daily interval is supported for the SMA FinRL rollout in v1.")
    data_root = resolve_data_root(args.data_root)
    os.environ["QUANTO_DATA_ROOT"] = str(data_root)
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

    canonical_manifest = bootstrap.canonical_manifest or _locate_canonical_manifest(
        data_root, args.canonical_domain, start, end
    )
    if not canonical_manifest or not _canonical_files_exist(data_root, args.canonical_domain, [symbol], start, end):
        raise SystemExit(
            f"No canonical data found for {symbol}. Live bootstrap failed or data is missing; inspect vendor credentials "
            "and canonical build logs."
        )
    bootstrap.canonical_manifest = canonical_manifest

    slices, canonical_hashes = load_canonical_equity([symbol], start, end, data_root=data_root, interval=interval)
    if not slices.get(symbol) or not slices[symbol].rows:
        raise SystemExit(f"No canonical data found for symbol {symbol}")

    sma_config = SMAStrategyConfig(fast_window=args.fast, slow_window=args.slow)
    strategy = run_sma_crossover(symbol, slices[symbol].timestamps, slices[symbol].closes, sma_config)
    strategy_frame = strategy_to_feature_frame(strategy)
    if len(strategy_frame) < 2:
        raise SystemExit("Not enough SMA-aligned rows to build features")
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
        raise SystemExit("Not enough feature rows to run the rollout")

    env_config = SignalWeightEnvConfig(transaction_cost_bp=args.transaction_cost_bp)
    env = SignalWeightTradingEnv(rows, env_config, observation_columns=feature_result.observation_columns)
    policy = SMAWeightPolicy(SMAWeightPolicyConfig(mode=args.policy_mode, sigmoid_scale=args.sigmoid_scale))
    combined_hashes = dict(canonical_hashes)
    combined_hashes.update(feature_result.inputs_used)
    result = run_rollout(env, policy, inputs_used=combined_hashes)

    run_id = args.run_id or derive_run_id(
        symbol,
        start,
        end,
        sma_config,
        env_config,
        policy,
        result.inputs_used,
        feature_result,
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
        bootstrap=bootstrap,
        interval=interval,
        feature_result=feature_result,
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


def derive_run_id(
    symbol: str,
    start: date,
    end: date,
    sma_config: SMAStrategyConfig,
    env_config: SignalWeightEnvConfig,
    policy: SMAWeightPolicy,
    canonical_hashes: Dict[str, str],
    feature_result: FeatureSetResult,
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
        "feature_set": feature_result.feature_set,
        "observation_columns": list(feature_result.observation_columns),
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
    bootstrap: BootstrapMetadata,
    interval: str,
    feature_result: FeatureSetResult,
) -> Dict[str, Any]:
    parameters = {
        "symbol": symbol,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "interval": interval,
        "fast_window": sma_config.fast_window,
        "slow_window": sma_config.slow_window,
        "transaction_cost_bp": env_config.transaction_cost_bp,
        "initial_cash": env_config.initial_cash,
        "policy_mode": policy.mode,
        "sigmoid_scale": policy.config.sigmoid_scale,
        "feature_set": feature_result.feature_set,
        "observation_columns": list(feature_result.observation_columns),
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
        "interval": interval,
        "symbol": symbol,
        "date_range": {"start": start.isoformat(), "end": end.isoformat()},
        "parameters": parameters,
        "metrics": result.metrics,
        "series": series,
        "steps": result.steps,
        "inputs_used": result.inputs_used,
        "artifacts": artifacts,
        "hashes": hashes,
        "bootstrap": bootstrap.as_payload(data_root),
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


def _iter_days(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
