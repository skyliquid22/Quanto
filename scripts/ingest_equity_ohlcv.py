#!/usr/bin/env python3
"""CLI entrypoint for deterministic equity OHLCV ingestion runs."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import date
from pathlib import Path
import sys
from typing import Any, Mapping

from infra.ingestion.adapters import EquityIngestionRequest, PolygonEquityAdapter, PolygonRESTClient
from infra.ingestion.data_pipeline import EquityIngestionPipeline
from infra.ingestion.router import IngestionRouter
from infra.paths import get_data_root
from infra.storage.raw_writer import RawEquityOHLCVWriter
from infra.validation import ValidationError

try:  # pragma: no cover - YAML support is optional
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Polygon equity OHLCV bars into raw storage.")
    parser.add_argument("--config", required=True, help="Path to JSON or YAML config describing the run.")
    parser.add_argument("--run-id", required=True, help="Deterministic run identifier.")
    parser.add_argument(
        "--mode",
        choices=("auto", "rest", "flat_file"),
        default="auto",
        help="Override routing decision.",
    )
    parser.add_argument(
        "--data-root",
        help="Optional override for runtime data root; defaults to QUANTO_DATA_ROOT or .quanto_data",
    )
    return parser.parse_args()


def load_config(path: Path) -> Mapping[str, Any]:
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        if not yaml:
            raise RuntimeError("PyYAML must be installed to consume YAML configs")
        return yaml.safe_load(text)
    return json.loads(text)


def build_request(config: Mapping[str, Any]) -> EquityIngestionRequest:
    return EquityIngestionRequest(
        symbols=config["symbols"],
        start_date=date.fromisoformat(config["start_date"]),
        end_date=date.fromisoformat(config["end_date"]),
        flat_file_uris=tuple(config.get("flat_file_uris", [])),
        vendor=config.get("vendor", "polygon"),
    )


async def _amain(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    config = load_config(config_path)
    request = build_request(config)
    data_root = Path(args.data_root).expanduser() if args.data_root else get_data_root()
    router = IngestionRouter(config.get("router"))

    raw_base_cfg = config.get("raw_base_path")
    raw_base = Path(raw_base_cfg).expanduser() if raw_base_cfg else data_root / "raw"
    manifest_dir_cfg = config.get("manifest_dir")
    default_manifest_dir = raw_base / request.vendor / "equity_ohlcv" / "manifests"
    manifest_dir = Path(manifest_dir_cfg).expanduser() if manifest_dir_cfg else default_manifest_dir
    raw_writer = RawEquityOHLCVWriter(base_path=raw_base)

    forced_mode = None if args.mode == "auto" else args.mode
    rest_cfg = dict(config.get("rest", {}))
    flat_cfg = dict(config.get("flat_file", {}))
    api_key = rest_cfg.pop("api_key", None) or os.environ.get("POLYGON_API_KEY")
    predicted_mode = forced_mode or router.route_equity_ohlcv(request)
    rest_client = None
    if predicted_mode == "rest":
        if not api_key:
            print("REST mode selected but no Polygon API key provided.", file=sys.stderr)
            return 1
        rest_client = PolygonRESTClient(api_key)

    adapter = PolygonEquityAdapter(
        rest_client=rest_client,
        rest_config=rest_cfg or None,
        flat_file_config=flat_cfg or None,
        vendor=request.vendor,
    )
    pipeline = EquityIngestionPipeline(adapter=adapter, router=router, raw_writer=raw_writer, manifest_dir=manifest_dir)

    loop = asyncio.get_running_loop()
    manifest: Mapping[str, Any] | None = None
    exit_code: int
    try:
        manifest = await loop.run_in_executor(None, lambda: pipeline.run(request, run_id=args.run_id, forced_mode=forced_mode))
    except ValidationError as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        exit_code = 1
    except Exception as exc:  # pragma: no cover - surface unexpected errors
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        exit_code = 1
    else:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        exit_code = 0 if manifest["status"] == "succeeded" else 1
    finally:
        if rest_client:
            try:
                await rest_client.aclose()
            except RuntimeError as exc:
                if "Event loop is closed" not in str(exc):
                    raise

    return exit_code


def main() -> int:
    args = parse_args()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
