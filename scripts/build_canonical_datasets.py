#!/usr/bin/env python3
"""CLI entrypoint orchestrating canonical reconciliation runs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from infra.normalization import ReconciliationBuilder
from research.datasets.canonical_equity_loader import verify_coverage

UTC = timezone.utc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical datasets across vendors deterministically.")
    parser.add_argument(
        "--config",
        default="configs/data_sources.yml",
        help="Path to the data sources configuration file (YAML or JSON).",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        help="Optional list of domains to reconcile (default: all configured domains).",
    )
    parser.add_argument("--start-date", required=True, help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", required=True, help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument(
        "--run-id",
        help="Deterministic run identifier; defaults to canonical_{timestamp} if omitted.",
    )
    parser.add_argument("--raw-root", help="Override raw data root directory.")
    parser.add_argument("--canonical-root", help="Override canonical output root directory.")
    parser.add_argument("--manifest-root", help="Override validation manifest directory.")
    parser.add_argument("--metrics-root", help="Override reconciliation metrics output directory.")
    return parser.parse_args()


def _resolve_layer_root(path: Path, layer: str) -> Path:
    expanded = path.expanduser()
    if expanded.name == layer:
        return expanded
    return expanded / layer


def build_builder(args: argparse.Namespace) -> ReconciliationBuilder:
    kwargs: Dict[str, Any] = {}
    raw_root_resolved: Path | None = None
    canonical_root_resolved: Path | None = None
    if args.raw_root:
        raw_root_resolved = _resolve_layer_root(Path(args.raw_root), "raw")
        kwargs["raw_data_root"] = raw_root_resolved
    if args.canonical_root:
        canonical_root_resolved = _resolve_layer_root(Path(args.canonical_root), "canonical")
        kwargs["canonical_root"] = canonical_root_resolved
    if args.manifest_root:
        kwargs["validation_manifest_root_path"] = Path(args.manifest_root).expanduser()
    if args.metrics_root:
        kwargs["metrics_root"] = Path(args.metrics_root).expanduser()
    builder = ReconciliationBuilder.from_config_file(args.config, **kwargs)
    builder._raw_root_hint = str(raw_root_resolved or builder.raw_data_root)  # type: ignore[attr-defined]
    return builder


def build_missing_equity_ohlcv_canonical(
    *,
    symbols: Sequence[str],
    years: Sequence[int],
    config_path: str,
    raw_root: Path,
    data_root: Path,
    run_id: str,
) -> None:
    """Programmatic helper to build only the missing equity OHLCV yearly shards."""

    ordered_symbols = _ordered_symbols(symbols)
    ordered_years = sorted({int(year) for year in years if int(year) > 0})
    if not ordered_symbols or not ordered_years:
        return

    data_root_resolved = Path(data_root).expanduser()
    raw_base_resolved = Path(raw_root).expanduser()
    canonical_root = _resolve_layer_root(data_root_resolved, "canonical")
    raw_data_root = _resolve_layer_root(raw_base_resolved, "raw")
    config_resolved = str(Path(config_path).expanduser())

    start_year = ordered_years[0]
    end_year = ordered_years[-1]
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    coverage = verify_coverage(
        ordered_symbols,
        start.isoformat(),
        end.isoformat(),
        data_root=data_root_resolved,
    )
    requested_years = set(ordered_years)
    missing_pairs: List[Tuple[str, int]] = []
    for symbol in ordered_symbols:
        for year in coverage["missing_by_symbol"].get(symbol, []):
            if year in requested_years:
                missing_pairs.append((symbol, year))
    if not missing_pairs:
        return

    formatted = ", ".join(f"{symbol}:{year}" for symbol, year in missing_pairs)
    print(f"[canonical-build] Building equity_ohlcv shards: {formatted}")
    builder = ReconciliationBuilder.from_config_file(
        config_resolved,
        raw_data_root=raw_data_root,
        canonical_root=canonical_root,
    )
    builder._raw_root_hint = str(raw_data_root)  # type: ignore[attr-defined]
    cfg = builder.domain_configs.get("equity_ohlcv")
    if not cfg:
        raise ValueError("equity_ohlcv domain is not configured")
    vendor_priority = builder._resolve_vendor_priority(cfg)
    vendor_data = builder._load_equity_snapshots(vendor_priority, start, end)
    target_pairs = {(symbol, year) for symbol, year in missing_pairs}
    filtered_vendor_data: Dict[str, Any] = {}
    for vendor, snapshot in vendor_data.items():
        filtered_records = {}
        for key, record in snapshot.records_by_key.items():
            symbol, ts = key
            try:
                year = int(str(ts)[:4])
            except Exception:
                continue
            if (symbol, year) not in target_pairs:
                continue
            filtered_records[key] = record
        if not filtered_records:
            continue
        snapshot.records_by_key = filtered_records
        filtered_vendor_data[vendor] = snapshot
    if not filtered_vendor_data:
        print("[canonical-build] No raw inputs found for requested shards; skipping build.")
        return
    builder._materialize_equity_canonical(
        domain="equity_ohlcv",
        vendor_priority=vendor_priority,
        vendor_data=filtered_vendor_data,
        cfg=cfg,
        start=start,
        end=end,
        run_id=run_id,
    )
    print(f"[canonical-build] Completed run {run_id} ({len(missing_pairs)} shard(s) refreshed).")


def _ordered_symbols(symbols: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for symbol in symbols:
        clean = str(symbol).strip().upper()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        ordered.append(clean)
    return ordered


def main() -> int:
    args = parse_args()
    builder = build_builder(args)
    run_id = args.run_id or datetime.now(tz=UTC).strftime("canonical_%Y%m%dT%H%M%S")
    manifests = builder.run(
        domains=args.domains,
        start_date=args.start_date,
        end_date=args.end_date,
        run_id=run_id,
    )
    print(json.dumps(manifests, indent=2, sort_keys=True))
    raw_hint = getattr(builder, "_raw_root_hint", builder.raw_data_root)  # type: ignore[attr-defined]
    for domain, payload in manifests.items():
        if payload.get("inputs") in (None, [], {}):
            print(
                f"Warning: no raw inputs discovered for domain '{domain}'. "
                f"Checked raw root hint '{raw_hint}'.",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
