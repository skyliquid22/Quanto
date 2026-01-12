#!/usr/bin/env python3
"""CLI entrypoint for fundamentals ingestion runs."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import date
from pathlib import Path
import sys
from typing import Any, Mapping

from infra.ingestion.adapters import (
    FundamentalsIngestionRequest,
    PolygonFundamentalsAdapter,
    PolygonFundamentalsRESTClient,
)
from infra.ingestion.fundamentals_pipeline import FundamentalsIngestionPipeline
from infra.ingestion.router import IngestionRouter
from infra.paths import get_data_root
from infra.storage.raw_writer import RawFundamentalsWriter
from infra.validation import ValidationError

try:  # pragma: no cover - optional dependency
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Polygon fundamentals into raw storage.")
    parser.add_argument("--config", required=True, help="Path to JSON or YAML config file.")
    parser.add_argument("--run-id", required=True, help="Deterministic run identifier.")
    parser.add_argument(
        "--mode",
        choices=("auto", "rest", "flat_file"),
        default="auto",
        help="Override router decision.",
    )
    parser.add_argument(
        "--data-root",
        help="Optional override for runtime data root.",
    )
    return parser.parse_args()


def load_config(path: Path) -> Mapping[str, Any]:
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        if not yaml:
            raise RuntimeError("PyYAML must be installed to parse YAML configs")
        return yaml.safe_load(text)
    return json.loads(text)


def _warn_unified_cli() -> None:
    print(_DEPRECATION_NOTICE, file=sys.stderr)


def build_request(config: Mapping[str, Any]) -> FundamentalsIngestionRequest:
    statement_types = tuple(config.get("statement_types", ("quarterly", "annual")))
    return FundamentalsIngestionRequest(
        symbols=tuple(config["symbols"]),
        start_date=date.fromisoformat(config["start_date"]),
        end_date=date.fromisoformat(config["end_date"]),
        statement_types=statement_types,
        flat_file_uris=tuple(config.get("flat_file_uris", [])),
        vendor=config.get("vendor", "polygon"),
    )


def main() -> int:
    _warn_unified_cli()
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    request = build_request(config)
    data_root = Path(args.data_root).expanduser() if args.data_root else get_data_root()

    router = IngestionRouter(config.get("router"))
    raw_base_cfg = config.get("raw_base_path")
    raw_base = Path(raw_base_cfg).expanduser() if raw_base_cfg else data_root / "raw"
    manifest_base_cfg = config.get("manifest_base_dir")
    manifest_base = Path(manifest_base_cfg).expanduser() if manifest_base_cfg else raw_base
    checkpoint_cfg = config.get("checkpoint_dir")
    checkpoint_dir = Path(checkpoint_cfg).expanduser() if checkpoint_cfg else None
    raw_writer = RawFundamentalsWriter(base_path=raw_base)

    forced_mode = None if args.mode == "auto" else args.mode
    rest_cfg = dict(config.get("rest", {}))
    flat_cfg = dict(config.get("flat_file", {}))
    api_key = rest_cfg.pop("api_key", None) or os.environ.get("POLYGON_API_KEY")

    predicted_mode = forced_mode or router.route_fundamentals(request)
    rest_client = None
    if predicted_mode == "rest":
        if not api_key:
            print("REST mode selected but no Polygon API key provided.", file=sys.stderr)
            return 1
        rest_client = PolygonFundamentalsRESTClient(api_key)

    adapter = PolygonFundamentalsAdapter(
        rest_client=rest_client,
        rest_config=rest_cfg or None,
        flat_file_config=flat_cfg or None,
        vendor=request.vendor,
    )
    pipeline = FundamentalsIngestionPipeline(
        adapter=adapter,
        router=router,
        raw_writer=raw_writer,
        manifest_base_dir=manifest_base,
        checkpoint_dir=checkpoint_dir,
    )

    try:
        manifest = pipeline.run(request, run_id=args.run_id, forced_mode=forced_mode)
    except ValidationError as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - safety net for unexpected issues
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if rest_client:
            asyncio.run(rest_client.aclose())

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())

_DEPRECATION_NOTICE = (
    "Legacy ingest_fundamentals entrypoint detected. Prefer `python -m scripts.ingest "
    "--domain fundamentals ...` for multi-vendor ingestion."
)
