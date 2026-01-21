#!/usr/bin/env python3
"""SMA-driven FinRL rollout producing deterministic monitoring artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import os
from datetime import date, datetime
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(PROJECT_ROOT))
CANONICAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "data_sources.yml"
_UNIVERSE_BOOTSTRAP_LOG_PREFIX = "[universe-bootstrap]"

from infra.normalization.lineage import compute_file_hash
from infra.paths import get_data_root as default_data_root
from research.datasets.canonical_equity_loader import build_union_calendar, load_canonical_equity, verify_coverage
from research.envs.signal_weight_env import SignalWeightEnvConfig, SignalWeightTradingEnv
from research.features.feature_eng import build_universe_feature_results
from research.features.feature_registry import (
    FeatureSetResult,
    build_features,
    build_universe_feature_panel,
    default_regime_for_feature_set,
    is_universe_feature_set,
    normalize_feature_set_name,
    strategy_to_feature_frame,
)
from research.policies.sma_weight_policy import SMAWeightPolicy, SMAWeightPolicyConfig
from research.runners.rollout import RolloutResult, run_rollout
from research.risk import RiskConfig
from research.strategies.sma_crossover import SMAStrategyConfig, run_sma_crossover
from research.validation.data_health import run_data_health_preflight
from scripts.build_canonical_datasets import build_missing_equity_ohlcv_canonical


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


def ensure_yearly_daily_coverage(
    *,
    symbols: Sequence[str],
    start: date,
    end: date,
    data_root: Path,
    auto_build: bool,
    run_id_seed: str | None = None,
) -> Dict[str, Any]:
    """Ensure YEARLY-daily canonical coverage for the requested symbols."""

    coverage = verify_coverage(list(symbols), start.isoformat(), end.isoformat(), data_root=data_root)
    missing_pairs: List[Tuple[str, int]] = list(coverage["missing_pairs"])
    if not missing_pairs:
        return coverage
    formatted = _format_missing_shards(missing_pairs)
    print(f"{_UNIVERSE_BOOTSTRAP_LOG_PREFIX} Missing canonical shards: {', '.join(formatted)}")
    if auto_build:
        years = sorted({year for _, year in missing_pairs})
        missing_symbols = sorted({symbol for symbol, _ in missing_pairs})
        run_id = _derive_universe_canonical_run_id(missing_symbols, years, seed=run_id_seed)
        print(f"{_UNIVERSE_BOOTSTRAP_LOG_PREFIX} Building canonical shards: {', '.join(formatted)}")
        build_missing_equity_ohlcv_canonical(
            symbols=missing_symbols,
            years=years,
            config_path=str(CANONICAL_CONFIG_PATH),
            raw_root=data_root,
            data_root=data_root,
            run_id=run_id,
        )
        coverage = verify_coverage(list(symbols), start.isoformat(), end.isoformat(), data_root=data_root)
        missing_pairs = list(coverage["missing_pairs"])
    if missing_pairs:
        formatted = _format_missing_shards(missing_pairs)
        raise SystemExit(
            "Missing canonical yearly shards: "
            + ", ".join(formatted)
            + ". Run live bootstrap or inspect canonical builder logs."
        )
    return coverage


def _format_missing_shards(pairs: Sequence[Tuple[str, int]]) -> List[str]:
    return [f"{symbol}:{year}" for symbol, year in pairs]


def _derive_universe_canonical_run_id(symbols: Sequence[str], years: Sequence[int], *, seed: str | None = None) -> str:
    payload = {
        "symbols": sorted(symbols),
        "years": sorted(years),
        "seed": seed or "",
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"universe_canonical_{digest[:10]}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic SMA rollout in a FinRL-style environment.")
    parser.add_argument("--symbol", default="AAPL", help="Single equity symbol to backtest.")
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
    parser.add_argument("--fast", type=int, default=20, help="Fast SMA window.")
    parser.add_argument("--slow", type=int, default=50, help="Slow SMA window.")
    parser.add_argument("--transaction-cost-bp", type=float, default=1.0, help="Round-trip transaction cost in basis points.")
    parser.add_argument("--max-weight", type=float, default=1.0, help="Per-asset weight cap enforced during projection.")
    parser.add_argument("--exposure-cap", type=float, default=1.0, help="Total exposure cap enforced during projection.")
    parser.add_argument("--min-cash", type=float, default=0.0, help="Minimum cash allocation reserved each step.")
    parser.add_argument("--max-turnover-1d", type=float, help="Optional turnover cap (L1 distance) per rebalance.")
    parser.add_argument("--allow-short", action="store_true", help="Disable long-only constraint in projection.")
    parser.add_argument("--policy-mode", choices=["hard", "sigmoid"], default="hard", help="Mapping used by the SMA policy.")
    parser.add_argument("--sigmoid-scale", type=float, default=5.0, help="Scale factor for sigmoid mode.")
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
            raise SystemExit("At least one valid symbol must be provided for --symbols")
        return ordered
    clean = str(symbol).upper().strip()
    if not clean:
        raise SystemExit("symbol must be provided")
    return [clean]


def _build_risk_config(args: argparse.Namespace) -> RiskConfig:
    return RiskConfig(
        long_only=not bool(getattr(args, "allow_short", False)),
        max_weight=None if args.max_weight is None else float(args.max_weight),
        exposure_cap=None if args.exposure_cap is None else float(args.exposure_cap),
        min_cash=None if args.min_cash is None else float(args.min_cash),
        max_turnover_1d=None if args.max_turnover_1d is None else float(args.max_turnover_1d),
    )


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
        missing_years = [
            year for year in range(start.year, end.year + 1) if not (base / f"{year}.parquet").exists()
        ]
        if missing_years:
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
    symbols = _resolve_symbol_list(args.symbol, args.symbols)
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
    universe_mode = bool(args.symbols)
    if universe_mode:
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

    try:
        run_data_health_preflight(
            symbols=symbols,
            start_date=start,
            end_date=end,
            feature_set=args.feature_set,
            data_root=data_root,
            interval=interval,
            calendar_mode=os.environ.get("QUANTO_DATA_HEALTH_CALENDAR_MODE", "union"),
            parquet_engine=os.environ.get("QUANTO_DATA_HEALTH_PARQUET_ENGINE", "fastparquet"),
            max_missing_ratio=_env_float("QUANTO_DATA_HEALTH_MAX_MISSING_RATIO", 0.01),
            max_nan_ratio=_env_float("QUANTO_DATA_HEALTH_MAX_NAN_RATIO", 0.05),
            strict=_env_flag("QUANTO_DATA_HEALTH_STRICT"),
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

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
            "No canonical manifest found for the requested symbols. Live bootstrap failed or data is missing; inspect "
            "vendor credentials and canonical build logs."
        )
    bootstrap.canonical_manifest = canonical_manifest

    slices, canonical_hashes = load_canonical_equity(symbols, start, end, data_root=data_root, interval=interval)
    for symbol in symbols:
        if not slices.get(symbol) or not slices[symbol].rows:
            raise SystemExit(f"No canonical data found for symbol {symbol}")

    sma_config = SMAStrategyConfig(fast_window=args.fast, slow_window=args.slow)
    per_symbol_features: Dict[str, FeatureSetResult] = {}
    feature_hashes: Dict[str, str] = {}
    feature_set_name: str | None = None
    multi_symbol = len(symbols) > 1
    normalized_feature_set = normalize_feature_set_name(args.feature_set)
    panel_regime_feature = default_regime_for_feature_set(normalized_feature_set)
    if not multi_symbol and is_universe_feature_set(normalized_feature_set):
        raise SystemExit(f"Feature set '{args.feature_set}' requires at least two symbols (--symbols)")
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
            strategy = run_sma_crossover(symbol, slices[symbol].timestamps, slices[symbol].closes, sma_config)
            strategy_frame = strategy_to_feature_frame(strategy)
            if len(strategy_frame) < 2:
                raise SystemExit(f"Not enough SMA-aligned rows to build features for {symbol}")
            feature_result = build_features(
                args.feature_set,
                strategy_frame,
                underlying_symbol=symbol,
                start_date=start,
                end_date=end,
                data_root=data_root,
            )
            per_symbol_features[symbol] = feature_result
    if not per_symbol_features:
        raise SystemExit("Failed to build observation features")
    feature_set_name = next(iter(per_symbol_features.values())).feature_set
    for symbol, feature_result in per_symbol_features.items():
        if feature_result.feature_set != feature_set_name:
            raise SystemExit("Feature set mismatch across symbols; ensure a consistent registry entry")
        if multi_symbol:
            for key, value in feature_result.inputs_used.items():
                feature_hashes[f"{symbol}:{key}"] = value
        else:
            feature_hashes.update(feature_result.inputs_used)

    if len(symbols) == 1:
        primary = symbols[0]
        rows = per_symbol_features[primary].frame.to_dict("records")
        base_observation_columns = per_symbol_features[primary].observation_columns
    else:
        calendar = build_union_calendar(slices, start_date=start, end_date=end)
        panel = build_universe_feature_panel(
            per_symbol_features,
            symbol_order=symbols,
            calendar=calendar,
            forward_fill_limit=3,
            regime_feature_set=panel_regime_feature,
            data_root=data_root,
        )
        rows = panel.rows
        base_observation_columns = panel.observation_columns

    rows = _slice_rows_by_date(rows, test_start, test_end)
    if len(rows) < 2:
        raise SystemExit("Not enough aligned feature rows to run the rollout")

    risk_config = _build_risk_config(args)
    env_config = SignalWeightEnvConfig(transaction_cost_bp=args.transaction_cost_bp, risk_config=risk_config)
    env = SignalWeightTradingEnv(rows, env_config, observation_columns=base_observation_columns)
    observation_headers = env.observation_columns
    policy = SMAWeightPolicy(SMAWeightPolicyConfig(mode=args.policy_mode, sigmoid_scale=args.sigmoid_scale))
    combined_hashes = dict(canonical_hashes)
    combined_hashes.update(feature_hashes)
    rollout_metadata = {
        "feature_set": feature_set_name or args.feature_set,
        "interval": interval,
        "start_date": test_start.isoformat(),
        "end_date": test_end.isoformat(),
        "transaction_cost_bp": env_config.transaction_cost_bp,
        "risk_config": env_config.risk_config.to_dict(),
        "policy": {
            "type": "sma_weight",
            "mode": policy.mode,
            "sigmoid_scale": policy.config.sigmoid_scale,
        },
        "symbols": list(symbols),
    }
    result = run_rollout(env, policy, inputs_used=combined_hashes, metadata=rollout_metadata)

    run_id = args.run_id or derive_run_id(
        symbols,
        test_start,
        test_end,
        sma_config,
        env_config,
        policy,
        result.inputs_used,
        feature_set_name or args.feature_set,
        base_observation_columns,
    )

    report_path = data_root / "monitoring" / "reports" / f"sma_finrl_rollout_{run_id}.json"
    plot_path = data_root / "monitoring" / "plots" / f"sma_finrl_rollout_{run_id}.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    weights_for_plot = _weights_for_plot(result.weights, result.symbols)
    _render_account_weight_plot(plot_path, result.account_values, weights_for_plot)

    hashes = {
        "canonical_files": dict(sorted(result.inputs_used.items())),
        "plot_png": compute_file_hash(plot_path),
        "report_json": "",
    }
    payload = build_report_payload(
        run_id=run_id,
        symbols=result.symbols,
        start=test_start,
        end=test_end,
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
        feature_set=feature_set_name or args.feature_set,
        base_observation_columns=base_observation_columns,
        observation_headers=observation_headers,
        data_split=_build_data_split_payload(args, train_start, train_end, test_start, test_end),
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
    symbols: Sequence[str],
    start: date,
    end: date,
    sma_config: SMAStrategyConfig,
    env_config: SignalWeightEnvConfig,
    policy: SMAWeightPolicy,
    canonical_hashes: Dict[str, str],
    feature_set: str,
    observation_columns: Sequence[str],
) -> str:
    ordered_symbols = list(symbols)
    if len(ordered_symbols) == 1:
        canonical = {
            "symbol": ordered_symbols[0],
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "fast_window": sma_config.fast_window,
            "slow_window": sma_config.slow_window,
            "transaction_cost_bp": env_config.transaction_cost_bp,
            "policy_mode": policy.mode,
            "sigmoid_scale": policy.config.sigmoid_scale,
            "canonical_hashes": {key: canonical_hashes[key] for key in sorted(canonical_hashes)},
            "feature_set": feature_set,
            "observation_columns": list(observation_columns),
            "risk_config": env_config.risk_config.to_dict(),
        }
    else:
        canonical = {
            "symbols": ordered_symbols,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "fast_window": sma_config.fast_window,
            "slow_window": sma_config.slow_window,
            "transaction_cost_bp": env_config.transaction_cost_bp,
            "policy_mode": policy.mode,
            "sigmoid_scale": policy.config.sigmoid_scale,
            "canonical_hashes": {key: canonical_hashes[key] for key in sorted(canonical_hashes)},
            "feature_set": feature_set,
            "observation_columns": list(observation_columns),
            "risk_config": env_config.risk_config.to_dict(),
        }
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return f"sma_finrl_{digest[:12]}"


def build_report_payload(
    *,
    run_id: str,
    symbols: Sequence[str],
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
    feature_set: str,
    base_observation_columns: Sequence[str],
    observation_headers: Sequence[str],
    data_split: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    symbol_list = list(symbols)
    primary_symbol = symbol_list[0] if symbol_list else ""
    parameters = {
        "symbol": primary_symbol,
        "symbols": symbol_list,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "interval": interval,
        "fast_window": sma_config.fast_window,
        "slow_window": sma_config.slow_window,
        "transaction_cost_bp": env_config.transaction_cost_bp,
        "initial_cash": env_config.initial_cash,
        "policy_mode": policy.mode,
        "sigmoid_scale": policy.config.sigmoid_scale,
        "feature_set": feature_set,
        "observation_columns": list(base_observation_columns),
        "panel_observation_columns": list(observation_headers),
        "risk_config": env_config.risk_config.to_dict(),
    }
    weights_series = _weights_series_for_report(result)
    series = {
        "timestamps": result.timestamps,
        "account_value": result.account_values,
        "weights": weights_series,
        "log_returns": result.log_returns,
        "transaction_costs": result.transaction_costs,
    }
    artifacts = {
        "report": _rel_path(report_path, data_root),
        "plot": _rel_path(plot_path, data_root),
    }
    payload = {
        "run_id": run_id,
        "interval": interval,
        "symbol": primary_symbol,
        "date_range": {"start": start.isoformat(), "end": end.isoformat()},
        "parameters": parameters,
        "metrics": result.metrics,
        "series": series,
        "steps": result.steps,
        "inputs_used": result.inputs_used,
        "artifacts": artifacts,
        "hashes": hashes,
        "bootstrap": bootstrap.as_payload(data_root),
        "rollout_metadata": result.metadata,
    }
    if data_split is not None:
        payload["data_split"] = data_split
    return payload


def _weights_series_for_report(result: RolloutResult) -> object:
    if not result.weights:
        return [] if len(result.symbols) <= 1 else {symbol: [] for symbol in result.symbols}
    if len(result.symbols) <= 1:
        symbol = result.symbols[0] if result.symbols else "asset"
        return [entry.get(symbol, 0.0) for entry in result.weights]
    return {
        symbol: [entry.get(symbol, 0.0) for entry in result.weights] for symbol in result.symbols
    }


def _weights_for_plot(weight_entries: Sequence[Mapping[str, float]], symbols: Sequence[str]) -> List[float]:
    if not weight_entries:
        return []
    if not symbols:
        return [sum(entry.values()) for entry in weight_entries]
    totals: List[float] = []
    for entry in weight_entries:
        totals.append(sum(entry.get(symbol, 0.0) for symbol in symbols))
    return totals


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


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float | None) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


__all__ = ["parse_args", "main", "ensure_yearly_daily_coverage", "resolve_data_root"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
