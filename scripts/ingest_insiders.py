#!/usr/bin/env python3
"""Ingest insider trades into the raw layer and emit a manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from infra.ingestion.adapters import FinancialDatasetsAdapter, FinancialDatasetsRESTClient
from infra.ingestion.insiders_pipeline import InsiderTradesIngestionPipeline
from infra.ingestion.request import IngestionRequest
from infra.paths import get_data_root

try:  # pragma: no cover - PyYAML optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Financial Datasets insider trades.")
    parser.add_argument("--config", required=True, help="Path to JSON or YAML config describing the ingest run.")
    parser.add_argument("--run-id", help="Optional deterministic run identifier.")
    parser.add_argument("--data-root", help="Optional override for runtime data root.")
    parser.add_argument("--mode", choices=("rest",), default="rest", help="Ingestion mode (rest only).")
    parser.add_argument("--force", action="store_true", help="Overwrite existing manifest for the run-id.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve config and print summary only.")
    return parser.parse_args(argv)


def load_config(path: Path) -> Mapping[str, Any]:
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        if not yaml:
            raise RuntimeError("PyYAML must be installed to load YAML ingestion configs")
        return yaml.safe_load(text)
    return json.loads(text)


def _resolve_layer_root(path: Path, layer: str) -> Path:
    expanded = path.expanduser()
    if expanded.name == layer:
        return expanded
    return expanded / layer


def _resolve_run_id(run_id: str | None) -> str:
    if run_id:
        return run_id
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"insider_trades-{stamp}"


def _manifest_path(raw_root: Path, vendor: str, run_id: str) -> Path:
    return raw_root / vendor / "insider_trades" / "manifests" / f"{run_id}.json"


def _instantiate_adapter(config: Mapping[str, Any], vendor: str) -> FinancialDatasetsAdapter:
    rest_cfg = dict(config.get("financialdatasets", {}).get("rest", {}))
    api_key = rest_cfg.pop("api_key", None) or os.environ.get("FINANCIALDATASETS_API_KEY")
    if not api_key:
        raise RuntimeError("Financial Datasets REST ingestion requested but no FINANCIALDATASETS_API_KEY provided")
    rest_client = FinancialDatasetsRESTClient(api_key, timeout=float(rest_cfg.get("timeout", 30.0)))
    return FinancialDatasetsAdapter(rest_client=rest_client, rest_config=rest_cfg or None, vendor=vendor)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config)
    config = load_config(config_path)
    request = IngestionRequest.from_mapping("insider_trades", config)
    vendor = request.vendor
    if vendor != "financialdatasets":
        raise SystemExit("insider trades ingestion currently supports vendor=financialdatasets only")

    data_root = Path(args.data_root).expanduser() if args.data_root else get_data_root()
    raw_root = _resolve_layer_root(data_root, "raw")
    run_id = _resolve_run_id(args.run_id)
    manifest_path = _manifest_path(raw_root, vendor, run_id)

    if manifest_path.exists() and not args.force:
        summary = {
            "adapter": None,
            "config_path": str(config_path),
            "domain": "insider_trades",
            "error": f"Manifest {manifest_path} already exists. Re-run with --force to overwrite.",
            "mode": None,
            "run_id": run_id,
            "status": "failed",
            "vendor": vendor,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1

    if args.dry_run:
        summary = {
            "adapter": "FinancialDatasetsAdapter",
            "config_path": str(config_path),
            "domain": "insider_trades",
            "mode": args.mode,
            "run_id": run_id,
            "status": "dry_run",
            "vendor": vendor,
            "symbols": list(request.symbols),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    try:
        adapter = _instantiate_adapter(config, vendor)
        pipeline = InsiderTradesIngestionPipeline(
            adapter=adapter,
            raw_writer=None,
            manifest_base_dir=raw_root,
        )
        manifest = pipeline.run(request, run_id=run_id, forced_mode=args.mode)
        summary = {
            "adapter": "FinancialDatasetsAdapter",
            "config_path": str(config_path),
            "domain": "insider_trades",
            "files_written": manifest.get("files_written", []),
            "manifest_path": manifest.get("manifest_path"),
            "mode": args.mode,
            "run_id": run_id,
            "status": manifest.get("status"),
            "vendor": vendor,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        summary = {
            "adapter": None,
            "config_path": str(config_path),
            "domain": "insider_trades",
            "error": str(exc),
            "mode": None,
            "run_id": run_id,
            "status": "failed",
            "vendor": vendor,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    sys.exit(main())
