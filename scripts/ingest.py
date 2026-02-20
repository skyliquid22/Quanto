#!/usr/bin/env python3
"""Unified ingestion CLI that routes across vendors and domains."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from infra.ingestion.adapters import (
    EquityIngestionRequest,
    FinancialDatasetsAdapter,
    FinancialDatasetsRESTClient,
    FundamentalsIngestionRequest,
    IvolatilityOptionsSurfaceAdapter,
    OptionReferenceIngestionRequest,
    OptionTimeseriesIngestionRequest,
    OptionsSurfaceIngestionRequest,
    OptionsSurfaceStorage,
    PolygonEquityAdapter,
    PolygonFundamentalsAdapter,
    PolygonFundamentalsRESTClient,
    PolygonOptionsAdapter,
    PolygonOptionsRESTClient,
    PolygonRESTClient,
)
from infra.ingestion.data_pipeline import EquityIngestionPipeline
from infra.ingestion.fundamentals_pipeline import FinancialDatasetsRawPipeline, FundamentalsIngestionPipeline
from infra.ingestion.ivolatility_client import IvolatilityClient, IvolatilityClientError
from infra.ingestion.options_pipeline import OptionPartition, OptionsIngestionPipeline, OptionsIngestionPlan
from infra.ingestion.request import IngestionRequest as NormalizedIngestionRequest
from infra.ingestion.router import AdapterRoute, IngestionRouter, Mode
from infra.storage.raw_writer import (
    RawEquityOHLCVWriter,
    RawFinancialDatasetsWriter,
    RawFundamentalsWriter,
    RawOptionsWriter,
)
from infra.paths import get_data_root
from infra.validation import ValidationError

try:  # pragma: no cover - PyYAML optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

SUPPORTED_DOMAINS = {
    "equity_ohlcv",
    "fundamentals",
    "financial_statements",
    "company_facts",
    "financial_metrics",
    "financial_metrics_snapshot",
    "institutional_ownership",
    "news",
    "option_contract_reference",
    "option_contract_ohlcv",
    "option_open_interest",
    "options_surface_v1",
}

FINANCIALDATASETS_RAW_DOMAINS = {
    "financial_statements",
    "company_facts",
    "financial_metrics",
    "financial_metrics_snapshot",
    "institutional_ownership",
    "news",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified ingestion CLI supporting multiple vendors.")
    parser.add_argument("--config", required=True, help="Path to JSON or YAML config describing the ingestion run.")
    parser.add_argument("--domain", required=True, choices=sorted(SUPPORTED_DOMAINS), help="Canonical domain to ingest.")
    parser.add_argument("--run-id", help="Optional deterministic run identifier. Auto-generated when omitted.")
    parser.add_argument("--data-root", help="Optional override for runtime data root.")
    parser.add_argument(
        "--mode",
        choices=("auto", "rest", "flat_file"),
        default="auto",
        help="Force a specific ingestion mode (defaults to auto).",
    )
    parser.add_argument("--force", action="store_true", help="Allow overwriting existing manifests for the run-id.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve routing and print summary without writing data.")
    return parser.parse_args(argv)


def load_config(path: Path) -> Mapping[str, Any]:
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        if not yaml:
            raise RuntimeError("PyYAML must be installed to load YAML ingestion configs")
        return yaml.safe_load(text)
    return json.loads(text)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config)
    config = load_config(config_path)
    normalized = NormalizedIngestionRequest.from_mapping(args.domain, config)
    router = IngestionRouter(config.get("router"))
    run_id = _resolve_run_id(args.run_id, normalized.domain)

    try:
        summary = _dispatch_domain(args, config, normalized, router, run_id, config_path)
    except ValidationError as exc:
        summary = _failure_summary(
            normalized,
            run_id,
            mode=None,
            adapter_name=None,
            config_path=config_path,
            error=str(exc),
            extra={"validation_manifest": exc.manifest.get("manifest_path")},
        )
    except IvolatilityClientError as exc:
        summary = _failure_summary(
            normalized,
            run_id,
            mode=None,
            adapter_name=None,
            config_path=config_path,
            error=str(exc),
        )
    except Exception as exc:  # pragma: no cover - defensive catch-all
        summary = _failure_summary(
            normalized,
            run_id,
            mode=None,
            adapter_name=None,
            config_path=config_path,
            error=str(exc),
        )
    return _emit_summary(summary)


def _dispatch_domain(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    normalized: NormalizedIngestionRequest,
    router: IngestionRouter,
    run_id: str,
    config_path: Path,
) -> Mapping[str, Any]:
    domain = normalized.domain
    if domain == "equity_ohlcv":
        return _handle_equity(args, config, normalized, router, run_id, config_path)
    if domain == "fundamentals":
        return _handle_fundamentals(args, config, normalized, router, run_id, config_path)
    if domain in FINANCIALDATASETS_RAW_DOMAINS:
        return _handle_financialdatasets_raw(args, config, normalized, router, run_id, config_path)
    if domain in {"option_contract_reference", "option_contract_ohlcv", "option_open_interest"}:
        return _handle_options(args, config, normalized, router, run_id, config_path)
    if domain == "options_surface_v1":
        return _handle_options_surface(args, config, normalized, router, run_id, config_path)
    raise ValueError(f"Unsupported ingestion domain '{domain}'")


def _handle_equity(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    normalized: NormalizedIngestionRequest,
    router: IngestionRouter,
    run_id: str,
    config_path: Path,
) -> Mapping[str, Any]:
    request = _build_equity_request(normalized)
    forced_mode = _forced_mode(args.mode) or _forced_mode(normalized.mode)
    mode = forced_mode or router.route_equity_ohlcv(request)
    try:
        route = router.resolve_vendor_adapter("equity_ohlcv", normalized.vendor, mode)
    except ValueError as exc:
        return _failure_summary(normalized, run_id, mode, None, config_path, str(exc))

    data_root = _resolve_data_root(args.data_root)
    raw_base = _resolve_raw_base(config, data_root)
    manifest_dir = _resolve_manifest_dir(
        config.get("manifest_dir"),
        raw_base / normalized.vendor / "equity_ohlcv" / "manifests",
    )
    manifest_path = manifest_dir / f"{run_id}.json"

    if args.dry_run:
        return _dry_run_summary(normalized, run_id, mode, route.adapter_name, config_path, manifest_path, args.force)

    if not args.force and manifest_path.exists():
        return _failure_summary(
            normalized,
            run_id,
            mode,
            route.adapter_name,
            config_path,
            f"Manifest {manifest_path} already exists. Re-run with --force to overwrite.",
        )

    rest_cfg, flat_cfg = _resolve_ingestion_configs(config, normalized.vendor, normalized.vendor_params)
    raw_writer = RawEquityOHLCVWriter(base_path=raw_base, shard_yearly_daily=config.get("raw_shard_yearly_daily"))
    adapter, cleanup_callbacks = _instantiate_equity_adapter(
        route,
        normalized.vendor,
        rest_cfg,
        flat_cfg,
        mode,
        normalized.vendor_params,
    )
    pipeline = EquityIngestionPipeline(adapter=adapter, router=router, raw_writer=raw_writer, manifest_dir=manifest_dir)

    try:
        manifest = pipeline.run(request, run_id=run_id, forced_mode=forced_mode)
    finally:
        for callback in cleanup_callbacks:
            callback()

    manifest["manifest_path"] = str(manifest_path)
    return _success_summary(normalized, run_id, mode, route.adapter_name, config_path, manifest)


def _handle_fundamentals(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    normalized: NormalizedIngestionRequest,
    router: IngestionRouter,
    run_id: str,
    config_path: Path,
) -> Mapping[str, Any]:
    request = _build_fundamentals_request(normalized, config)
    forced_mode = _forced_mode(args.mode) or _forced_mode(normalized.mode)
    mode = forced_mode or router.route_fundamentals(request)
    try:
        route = router.resolve_vendor_adapter("fundamentals", normalized.vendor, mode)
    except ValueError as exc:
        return _failure_summary(normalized, run_id, mode, None, config_path, str(exc))

    data_root = _resolve_data_root(args.data_root)
    raw_base = _resolve_raw_base(config, data_root)
    manifest_base_cfg = config.get("manifest_base_dir")
    manifest_base = Path(manifest_base_cfg).expanduser() if manifest_base_cfg else raw_base
    checkpoint_cfg = config.get("checkpoint_dir")
    checkpoint_dir = Path(checkpoint_cfg).expanduser() if checkpoint_cfg else None
    manifest_path = manifest_base / normalized.vendor / "fundamentals" / "manifests" / f"{run_id}.json"

    if args.dry_run:
        return _dry_run_summary(normalized, run_id, mode, route.adapter_name, config_path, manifest_path, args.force)

    if not args.force and manifest_path.exists():
        return _failure_summary(
            normalized,
            run_id,
            mode,
            route.adapter_name,
            config_path,
            f"Manifest {manifest_path} already exists. Re-run with --force to overwrite.",
        )

    rest_cfg, flat_cfg = _resolve_ingestion_configs(config, normalized.vendor, normalized.vendor_params)
    raw_writer = RawFundamentalsWriter(base_path=raw_base)
    adapter, cleanup_callbacks = _instantiate_fundamentals_adapter(
        route,
        normalized.vendor,
        rest_cfg,
        flat_cfg,
    )
    pipeline = FundamentalsIngestionPipeline(
        adapter=adapter,
        router=router,
        raw_writer=raw_writer,
        manifest_base_dir=manifest_base,
        checkpoint_dir=checkpoint_dir,
    )
    try:
        manifest = pipeline.run(request, run_id=run_id, forced_mode=forced_mode)
    finally:
        for callback in cleanup_callbacks:
            callback()
    return _success_summary(normalized, run_id, mode, route.adapter_name, config_path, manifest)


def _handle_financialdatasets_raw(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    normalized: NormalizedIngestionRequest,
    router: IngestionRouter,
    run_id: str,
    config_path: Path,
) -> Mapping[str, Any]:
    domain = normalized.domain
    forced_mode = _forced_mode(args.mode) or _forced_mode(normalized.mode)
    mode = forced_mode or router.route_financialdatasets_raw()
    try:
        route = router.resolve_vendor_adapter(domain, normalized.vendor, mode)
    except ValueError as exc:
        return _failure_summary(normalized, run_id, mode, None, config_path, str(exc))

    data_root = _resolve_data_root(args.data_root)
    raw_base = _resolve_raw_base(config, data_root)
    manifest_base_cfg = config.get("manifest_base_dir")
    manifest_base = Path(manifest_base_cfg).expanduser() if manifest_base_cfg else raw_base
    manifest_path = manifest_base / normalized.vendor / domain / "manifests" / f"{run_id}.json"

    if args.dry_run:
        return _dry_run_summary(normalized, run_id, mode, route.adapter_name, config_path, manifest_path, args.force)

    if not args.force and manifest_path.exists():
        return _failure_summary(
            normalized,
            run_id,
            mode,
            route.adapter_name,
            config_path,
            f"Manifest {manifest_path} already exists. Re-run with --force to overwrite.",
        )

    rest_cfg, _ = _resolve_ingestion_configs(config, normalized.vendor, normalized.vendor_params)
    raw_writer = RawFinancialDatasetsWriter(base_path=raw_base)
    adapter, cleanup_callbacks = _instantiate_financialdatasets_adapter(normalized.vendor, rest_cfg)
    pipeline = FinancialDatasetsRawPipeline(
        adapter=adapter,
        raw_writer=raw_writer,
        manifest_base_dir=manifest_base,
        router=router,
    )

    try:
        manifest = pipeline.run(normalized, run_id=run_id, forced_mode=forced_mode)
    finally:
        for callback in cleanup_callbacks:
            callback()

    return _success_summary(normalized, run_id, mode, route.adapter_name, config_path, manifest)


def _handle_options(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    normalized: NormalizedIngestionRequest,
    router: IngestionRouter,
    run_id: str,
    config_path: Path,
) -> Mapping[str, Any]:
    domain = normalized.domain
    request = _build_option_request(normalized, domain)
    forced_mode = _forced_mode(args.mode) or _forced_mode(normalized.mode)
    if domain == "option_contract_reference":
        mode = forced_mode or router.route_option_contract_reference(request)
    elif domain == "option_contract_ohlcv":
        mode = forced_mode or router.route_option_contract_ohlcv(request)
    else:
        mode = forced_mode or router.route_option_open_interest(request)

    try:
        route = router.resolve_vendor_adapter(domain, normalized.vendor, mode)
    except ValueError as exc:
        return _failure_summary(normalized, run_id, mode, None, config_path, str(exc))

    data_root = _resolve_data_root(args.data_root)
    raw_base = _resolve_raw_base(config, data_root)
    manifest_base_cfg = config.get("manifest_base_dir")
    manifest_base = Path(manifest_base_cfg).expanduser() if manifest_base_cfg else raw_base
    checkpoint_cfg = config.get("checkpoint_dir")
    checkpoint_dir = Path(checkpoint_cfg).expanduser() if checkpoint_cfg else None
    manifest_path = manifest_base / normalized.vendor / domain / "manifests" / f"{run_id}.json"

    if args.dry_run:
        return _dry_run_summary(normalized, run_id, mode, route.adapter_name, config_path, manifest_path, args.force)

    if not args.force and manifest_path.exists():
        return _failure_summary(
            normalized,
            run_id,
            mode,
            route.adapter_name,
            config_path,
            f"Manifest {manifest_path} already exists. Re-run with --force to overwrite.",
        )

    rest_cfg, flat_cfg = _resolve_ingestion_configs(config, normalized.vendor, normalized.vendor_params)
    raw_writer = RawOptionsWriter(base_path=raw_base)
    adapter, cleanup_callbacks = _instantiate_options_adapter(route, normalized.vendor, rest_cfg, flat_cfg, mode, normalized.vendor_params)
    pipeline = OptionsIngestionPipeline(
        adapter=adapter,
        router=router,
        raw_writer=raw_writer,
        manifest_base_dir=manifest_base,
        checkpoint_dir=checkpoint_dir,
    )
    partition_id = str(config.get("partition_id") or f"{domain}-{run_id}")
    partition = OptionPartition(partition_id=partition_id, request=request, mode=forced_mode)
    if domain == "option_contract_reference":
        plan = OptionsIngestionPlan(vendor=normalized.vendor, reference=(partition,), ohlcv=tuple(), open_interest=tuple())
    elif domain == "option_contract_ohlcv":
        plan = OptionsIngestionPlan(vendor=normalized.vendor, reference=tuple(), ohlcv=(partition,), open_interest=tuple())
    else:
        plan = OptionsIngestionPlan(vendor=normalized.vendor, reference=tuple(), ohlcv=tuple(), open_interest=(partition,))

    try:
        manifest = pipeline.run(plan, run_id=run_id)
    finally:
        for callback in cleanup_callbacks:
            callback()

    domain_manifest = manifest["domains"][domain]
    return _success_summary(normalized, run_id, mode, route.adapter_name, config_path, domain_manifest)


def _handle_options_surface(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    normalized: NormalizedIngestionRequest,
    router: IngestionRouter,
    run_id: str,
    config_path: Path,
) -> Mapping[str, Any]:
    request = _build_options_surface_request(normalized)
    forced_mode = _forced_mode(args.mode) or _forced_mode(normalized.mode)
    if forced_mode and forced_mode != "rest":
        raise ValueError("options_surface_v1 only supports REST ingestion mode")
    mode: Mode = "rest"
    try:
        route = router.resolve_vendor_adapter("options_surface_v1", normalized.vendor, mode)
    except ValueError as exc:
        return _failure_summary(normalized, run_id, mode, None, config_path, str(exc))

    data_root = _resolve_data_root(args.data_root)
    derived_base_cfg = config.get("derived_base_path")
    derived_base = Path(derived_base_cfg).expanduser() if derived_base_cfg else data_root / "derived" / "options_surface_v1"
    manifest_dir = _resolve_manifest_dir(config.get("manifest_dir"), derived_base / "manifests")
    storage = OptionsSurfaceStorage(base_dir=derived_base, manifest_dir=manifest_dir)
    manifest_path = storage.manifest_path(run_id)

    if args.dry_run:
        return _dry_run_summary(normalized, run_id, mode, route.adapter_name, config_path, manifest_path, args.force)

    if not args.force and manifest_path.exists():
        return _failure_summary(
            normalized,
            run_id,
            mode,
            route.adapter_name,
            config_path,
            f"Manifest {manifest_path} already exists. Re-run with --force to overwrite.",
        )

    ivol_client = _build_ivol_client(normalized.vendor_params)
    adapter: IvolatilityOptionsSurfaceAdapter = route.adapter(  # type: ignore[assignment]
        client=ivol_client,
        vendor=normalized.vendor,
        config=normalized.vendor_params or None,
    )
    rows, metadata = adapter.fetch_surface(request)
    creation_ts = _deterministic_creation_timestamp(run_id)
    manifest = storage.persist(
        rows,
        request=request,
        run_id=run_id,
        endpoint=adapter.endpoint,
        params=metadata or {},
        created_at=creation_ts,
    )
    return _success_summary(normalized, run_id, mode, route.adapter_name, config_path, manifest)


def _resolve_run_id(run_id: str | None, domain: str) -> str:
    if run_id:
        return run_id
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{domain}-{timestamp}"


def _deterministic_creation_timestamp(seed: str) -> str:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    seconds = int(digest[:8], 16) % (365 * 24 * 60 * 60)
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    return (base + timedelta(seconds=seconds)).isoformat()


def _forced_mode(value: str | None) -> Mode | None:
    if not value:
        return None
    normalized = value.strip().lower()
    if normalized in {"", "auto"}:
        return None
    if normalized not in {"rest", "flat_file"}:
        raise ValueError(f"Unsupported mode '{value}'. Expected 'rest', 'flat_file', or 'auto'.")
    return normalized  # type: ignore[return-value]


def _build_equity_request(normalized: NormalizedIngestionRequest) -> EquityIngestionRequest:
    if not normalized.symbols or not normalized.start_date or not normalized.end_date:
        raise ValueError("equity_ohlcv config must specify symbols, start_date, and end_date")
    return EquityIngestionRequest(
        symbols=normalized.symbols,
        start_date=normalized.start_date,
        end_date=normalized.end_date,
        frequency=normalized.interval or "daily",
        flat_file_uris=normalized.flat_file_uris,
        vendor=normalized.vendor,
        options=normalized.options,
    )


def _build_fundamentals_request(
    normalized: NormalizedIngestionRequest,
    config: Mapping[str, Any],
) -> FundamentalsIngestionRequest:
    if not normalized.symbols or not normalized.start_date or not normalized.end_date:
        raise ValueError("fundamentals config must include symbols, start_date, and end_date")
    statement_types = tuple(config.get("statement_types", ("quarterly", "annual")))
    return FundamentalsIngestionRequest(
        symbols=normalized.symbols,
        start_date=normalized.start_date,
        end_date=normalized.end_date,
        statement_types=statement_types,
        flat_file_uris=normalized.flat_file_uris,
        vendor=normalized.vendor,
        options=normalized.options,
    )


def _build_option_request(
    normalized: NormalizedIngestionRequest,
    domain: str,
) -> OptionReferenceIngestionRequest | OptionTimeseriesIngestionRequest:
    if domain == "option_contract_reference":
        if not normalized.as_of_date:
            raise ValueError("option_contract_reference configs must include as_of_date")
        if not normalized.symbols:
            raise ValueError("option_contract_reference configs must include underlying symbols")
        return OptionReferenceIngestionRequest(
            underlying_symbols=normalized.symbols,
            as_of_date=normalized.as_of_date,
            flat_file_uris=normalized.flat_file_uris,
            vendor=normalized.vendor,
            options=normalized.options,
        )

    if not normalized.start_date or not normalized.end_date:
        raise ValueError(f"{domain} configs must include start_date and end_date")
    if not normalized.symbols:
        raise ValueError(f"{domain} configs must include option_symbols")
    return OptionTimeseriesIngestionRequest(
        domain=domain,
        option_symbols=normalized.symbols,
        start_date=normalized.start_date,
        end_date=normalized.end_date,
        flat_file_uris=normalized.flat_file_uris,
        vendor=normalized.vendor,
        options=normalized.options,
    )


def _build_options_surface_request(normalized: NormalizedIngestionRequest) -> OptionsSurfaceIngestionRequest:
    if not normalized.start_date or not normalized.end_date:
        raise ValueError("options_surface_v1 configs must include start_date and end_date")
    if not normalized.symbols:
        raise ValueError("options_surface_v1 configs must include symbols")
    return OptionsSurfaceIngestionRequest(
        symbols=normalized.symbols,
        start_date=normalized.start_date,
        end_date=normalized.end_date,
        vendor=normalized.vendor,
        options=normalized.options,
        vendor_params=normalized.vendor_params,
    )


def _resolve_data_root(arg_value: str | None) -> Path:
    if arg_value:
        return Path(arg_value).expanduser()
    return get_data_root()


def _resolve_raw_base(config: Mapping[str, Any], data_root: Path) -> Path:
    raw_base_cfg = config.get("raw_base_path")
    return Path(raw_base_cfg).expanduser() if raw_base_cfg else data_root / "raw"


def _resolve_manifest_dir(config_value: Any, default: Path) -> Path:
    if config_value:
        return Path(config_value).expanduser()
    return default


def _resolve_ingestion_configs(
    config: Mapping[str, Any],
    vendor: str,
    vendor_block: Mapping[str, Any],
) -> tuple[MutableMapping[str, Any], MutableMapping[str, Any]]:
    rest_cfg: MutableMapping[str, Any] = dict(config.get("rest") or {})
    flat_cfg: MutableMapping[str, Any] = dict(config.get("flat_file") or {})
    if vendor == "polygon":
        rest_override = vendor_block.get("rest")
        if isinstance(rest_override, Mapping):
            rest_cfg.update(rest_override)
        else:
            for key, value in vendor_block.items():
                if key not in {"flat_file"}:
                    rest_cfg[key] = value
        flat_override = vendor_block.get("flat_file")
        if isinstance(flat_override, Mapping):
            flat_cfg.update(flat_override)
    return rest_cfg, flat_cfg


def _instantiate_equity_adapter(
    route: AdapterRoute,
    vendor: str,
    rest_cfg: Mapping[str, Any],
    flat_cfg: Mapping[str, Any],
    mode: Mode,
    vendor_params: Mapping[str, Any],
) -> tuple[PolygonEquityAdapter | Any, list[Callable[[], None]]]:
    cleanup_callbacks: list[Callable[[], None]] = []
    if vendor == "polygon":
        rest_client = None
        rest_cfg_local = dict(rest_cfg)
        api_key = rest_cfg_local.pop("api_key", None) or os.environ.get("POLYGON_API_KEY")
        if mode == "rest":
            if not api_key:
                raise RuntimeError("Polygon REST ingestion requested but no POLYGON_API_KEY provided")
            rest_client = PolygonRESTClient(api_key, timeout=float(rest_cfg_local.get("timeout", 30.0)))
        adapter = PolygonEquityAdapter(
            rest_client=rest_client,
            rest_config=rest_cfg_local or None,
            flat_file_config=dict(flat_cfg) or None,
            vendor=vendor,
        )
        return adapter, cleanup_callbacks

    ivol_client = _build_ivol_client(vendor_params)
    adapter = route.adapter(client=ivol_client, vendor=vendor, config=vendor_params or None)
    return adapter, cleanup_callbacks


def _instantiate_fundamentals_adapter(
    route: AdapterRoute,
    vendor: str,
    rest_cfg: Mapping[str, Any],
    flat_cfg: Mapping[str, Any],
) -> tuple[PolygonFundamentalsAdapter, list[Callable[[], None]]]:
    cleanup_callbacks: list[Callable[[], None]] = []
    rest_cfg_local = dict(rest_cfg)
    rest_client = None
    if route.mode == "rest":
        api_key = rest_cfg_local.pop("api_key", None) or os.environ.get("POLYGON_API_KEY")
        if not api_key:
            raise RuntimeError("Polygon REST ingestion requested but no POLYGON_API_KEY provided")
        rest_client = PolygonFundamentalsRESTClient(api_key, timeout=float(rest_cfg_local.get("timeout", 30.0)))
        cleanup_callbacks.append(lambda rc=rest_client: asyncio.run(rc.aclose()))
    adapter = PolygonFundamentalsAdapter(
        rest_client=rest_client,
        rest_config=rest_cfg_local or None,
        flat_file_config=dict(flat_cfg) or None,
        vendor=vendor,
    )
    return adapter, cleanup_callbacks


def _instantiate_financialdatasets_adapter(
    vendor: str,
    rest_cfg: Mapping[str, Any],
) -> tuple[FinancialDatasetsAdapter, list[Callable[[], None]]]:
    cleanup_callbacks: list[Callable[[], None]] = []
    rest_cfg_local = dict(rest_cfg)
    api_key = rest_cfg_local.pop("api_key", None) or os.environ.get("FINANCIALDATASETS_API_KEY")
    if not api_key:
        raise RuntimeError("Financial Datasets REST ingestion requested but no FINANCIALDATASETS_API_KEY provided")
    rest_client = FinancialDatasetsRESTClient(api_key, timeout=float(rest_cfg_local.get("timeout", 30.0)))
    adapter = FinancialDatasetsAdapter(rest_client=rest_client, rest_config=rest_cfg_local or None, vendor=vendor)
    return adapter, cleanup_callbacks


def _instantiate_options_adapter(
    route: AdapterRoute,
    vendor: str,
    rest_cfg: Mapping[str, Any],
    flat_cfg: Mapping[str, Any],
    mode: Mode,
    vendor_params: Mapping[str, Any],
) -> tuple[PolygonOptionsAdapter | Any, list[Callable[[], None]]]:
    cleanup_callbacks: list[Callable[[], None]] = []
    if vendor == "polygon":
        rest_client = None
        rest_cfg_local = dict(rest_cfg)
        api_key = rest_cfg_local.pop("api_key", None) or os.environ.get("POLYGON_API_KEY")
        if mode == "rest":
            if not api_key:
                raise RuntimeError("Polygon REST ingestion requested but no POLYGON_API_KEY provided")
            rest_client = PolygonOptionsRESTClient(api_key, timeout=float(rest_cfg_local.get("timeout", 30.0)))
            cleanup_callbacks.append(lambda rc=rest_client: asyncio.run(rc.aclose()))
        adapter = PolygonOptionsAdapter(
            rest_client=rest_client,
            rest_config=rest_cfg_local or None,
            flat_file_config=dict(flat_cfg) or None,
            vendor=vendor,
        )
        return adapter, cleanup_callbacks

    ivol_client = _build_ivol_client(vendor_params)
    adapter = route.adapter(client=ivol_client, vendor=vendor, config=vendor_params or None)
    return adapter, cleanup_callbacks


def _build_ivol_client(vendor_params: Mapping[str, Any]) -> IvolatilityClient:
    params: dict[str, Any] = {}
    for key in (
        "api_key",
        "api_secret",
        "base_url",
        "timeout",
        "max_retries",
        "backoff_factor",
        "max_backoff",
        "cache_dir",
        "transport",
    ):
        if key in vendor_params:
            params[key] = vendor_params[key]
    return IvolatilityClient(**params)


def _dry_run_summary(
    normalized: NormalizedIngestionRequest,
    run_id: str,
    mode: Mode,
    adapter_name: str | None,
    config_path: Path,
    manifest_path: Path,
    force: bool,
) -> Mapping[str, Any]:
    return {
        "status": "dry_run",
        "domain": normalized.domain,
        "vendor": normalized.vendor,
        "run_id": run_id,
        "mode": mode,
        "adapter": adapter_name,
        "manifest_path": str(manifest_path),
        "symbols": list(normalized.symbols),
        "start_date": normalized.start_date.isoformat() if normalized.start_date else None,
        "end_date": normalized.end_date.isoformat() if normalized.end_date else None,
        "as_of_date": normalized.as_of_date.isoformat() if normalized.as_of_date else None,
        "config_path": str(config_path),
        "force": force,
        "dry_run": True,
    }


def _success_summary(
    normalized: NormalizedIngestionRequest,
    run_id: str,
    mode: Mode,
    adapter_name: str | None,
    config_path: Path,
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    files = _sorted_files(manifest.get("files_written") or [])
    validation_entries = list(manifest.get("validation_manifests", []))
    single_manifest = manifest.get("validation_manifest")
    if single_manifest:
        validation_entries.append(single_manifest)
    return {
        "status": manifest.get("status", "succeeded"),
        "domain": normalized.domain,
        "vendor": normalized.vendor,
        "run_id": run_id,
        "mode": mode,
        "adapter": adapter_name,
        "manifest_path": manifest.get("manifest_path"),
        "files_written": files,
        "config_path": str(config_path),
        "validation_manifests": sorted(validation_entries),
    }


def _failure_summary(
    normalized: NormalizedIngestionRequest,
    run_id: str,
    mode: Mode | None,
    adapter_name: str | None,
    config_path: Path,
    error: str,
    extra: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    payload = {
        "status": "failed",
        "domain": normalized.domain,
        "vendor": normalized.vendor,
        "run_id": run_id,
        "mode": mode,
        "adapter": adapter_name,
        "config_path": str(config_path),
        "error": error,
    }
    if extra:
        payload.update(extra)
    return payload


def _sorted_files(items: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    serializable = [dict(item) for item in items]
    return sorted(serializable, key=lambda entry: entry.get("path", ""))


def _emit_summary(summary: Mapping[str, Any]) -> int:
    print(json.dumps(summary, indent=2, sort_keys=True))
    status = summary.get("status")
    if status in {"succeeded", "dry_run"}:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
