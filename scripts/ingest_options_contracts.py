#!/usr/bin/env python3
"""CLI entrypoint orchestrating Polygon option contract ingestion."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

from infra.ingestion.adapters import PolygonOptionsAdapter, PolygonOptionsRESTClient
from infra.ingestion.options_pipeline import OptionsIngestionPipeline, OptionsIngestionPlan
from infra.ingestion.router import IngestionRouter
from infra.paths import get_data_root
from infra.storage.raw_writer import RawOptionsWriter
from infra.validation import ValidationError

try:  # pragma: no cover - YAML support optional
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Polygon option contract datasets into raw storage.")
    parser.add_argument("--config", required=True, help="Path to JSON or YAML config describing the run plan.")
    parser.add_argument("--run-id", required=True, help="Deterministic run identifier shared across domains.")
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


def plan_requires_rest(plan: OptionsIngestionPlan) -> bool:
    for partition in (*plan.reference, *plan.ohlcv, *plan.open_interest):
        if partition.mode == "rest":
            return True
        if partition.mode is None and not partition.request.flat_file_uris:
            return True
    return False


def build_pipeline(
    config: Mapping[str, Any],
    plan: OptionsIngestionPlan,
    *,
    data_root: Path,
    rest_client: PolygonOptionsRESTClient | None,
    rest_config: Mapping[str, Any] | None,
    flat_file_config: Mapping[str, Any] | None,
) -> OptionsIngestionPipeline:
    router = IngestionRouter(config.get("router"))
    raw_base_cfg = config.get("raw_base_path")
    raw_base = Path(raw_base_cfg).expanduser() if raw_base_cfg else data_root / "raw"
    manifest_base_cfg = config.get("manifest_base_dir")
    manifest_base = Path(manifest_base_cfg).expanduser() if manifest_base_cfg else raw_base
    checkpoint_cfg = config.get("checkpoint_dir")
    checkpoint_dir = Path(checkpoint_cfg).expanduser() if checkpoint_cfg else None
    raw_writer = RawOptionsWriter(base_path=raw_base)
    adapter = PolygonOptionsAdapter(
        rest_client=rest_client,
        rest_config=rest_config,
        flat_file_config=flat_file_config,
        vendor=plan.vendor,
    )
    return OptionsIngestionPipeline(
        adapter=adapter,
        router=router,
        raw_writer=raw_writer,
        manifest_base_dir=manifest_base,
        checkpoint_dir=checkpoint_dir,
    )


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    plan = OptionsIngestionPlan.from_mapping(config)
    data_root = Path(args.data_root).expanduser() if args.data_root else get_data_root()

    rest_cfg = dict(config.get("rest", {}))
    flat_cfg = dict(config.get("flat_file", {}))
    api_key = rest_cfg.pop("api_key", None) or os.environ.get("POLYGON_API_KEY")
    rest_client: PolygonOptionsRESTClient | None = None
    needs_rest = plan_requires_rest(plan)
    if needs_rest:
        if not api_key:
            print("REST ingestion required but no Polygon API key provided.", file=sys.stderr)
            return 1
        rest_client = PolygonOptionsRESTClient(api_key, timeout=float(rest_cfg.get("timeout", 30.0)))
    pipeline = build_pipeline(
        config,
        plan,
        data_root=data_root,
        rest_client=rest_client,
        rest_config=rest_cfg or None,
        flat_file_config=flat_cfg or None,
    )

    try:
        manifest = pipeline.run(plan, run_id=args.run_id)
    except ValidationError as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - unexpected failure surface
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if rest_client:
            asyncio.run(rest_client.aclose())

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest.get("status") == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
