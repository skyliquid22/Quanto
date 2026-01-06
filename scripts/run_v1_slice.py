#!/usr/bin/env python3
"""Offline vertical slice runner wiring ingestion → canonicalization → reporting."""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
from typing import Any, Dict, List, Mapping, Sequence
import zlib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import guard
    sys.path.insert(0, str(PROJECT_ROOT))

from infra.ingestion.adapters import EquityIngestionRequest, PolygonEquityAdapter
from infra.ingestion.data_pipeline import EquityIngestionPipeline
from infra.ingestion.router import IngestionRouter
from infra.normalization import ReconciliationBuilder
from infra.normalization.lineage import compute_file_hash
from infra.paths import get_data_root as default_data_root
from infra.storage.raw_writer import RawEquityOHLCVWriter

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - simplifies local environments without PyYAML
    yaml = None

try:  # pragma: no cover - optional dependency
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover - fallback when pyarrow is unavailable
    pq = None

UTC = timezone.utc
EQUITY_DOMAIN = "equity_ohlcv"
REPORT_NAME = "v1_slice_report.json"
PLOT_NAME = "v1_slice_equity.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the offline v1 vertical slice deterministically.")
    parser.add_argument(
        "--config",
        default="tests/fixtures/configs/v1_slice.json",
        help="Path to the slice configuration file (YAML or JSON).",
    )
    parser.add_argument("--run-id", help="Deterministic run identifier. Defaults to a hash of the config.")
    parser.add_argument(
        "--data-root",
        help="Optional data root override. Defaults to QUANTO_DATA_ROOT or infra.paths.get_data_root().",
    )
    parser.add_argument(
        "--offline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force offline synthetic ingestion. Online mode is not implemented for the slice.",
    )
    return parser.parse_args()


def _disable_parquet_dependencies() -> None:
    """Force JSON fallback for offline workflows to avoid binary parquet IO."""

    global pq
    pq = None
    try:
        import infra.storage.parquet as storage_parquet

        storage_parquet._PARQUET_AVAILABLE = False  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        import infra.normalization.reconciliation_builder as recon_builder

        recon_builder.pq = None  # type: ignore[attr-defined]
    except Exception:
        pass


def load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text()
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        if not yaml:
            raise RuntimeError("PyYAML is required to parse YAML configs; provide JSON instead or install PyYAML")
        return yaml.safe_load(text)
    return json.loads(text)


def derive_run_id(config: Mapping[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"v1slice_{digest[:12]}"


def _deterministic_timestamp(seed: str) -> datetime:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    seconds = int(digest[:8], 16) % (365 * 24 * 60 * 60)
    base = datetime(2020, 1, 1, tzinfo=UTC)
    return base + timedelta(seconds=seconds)


def resolve_data_root(arg_root: str | None) -> Path:
    if arg_root:
        return Path(arg_root).expanduser()
    env_override = os.environ.get("QUANTO_DATA_ROOT")
    if env_override:
        return Path(env_override).expanduser()
    return default_data_root()


def synthesize_equity_records(cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    symbols = [str(symbol) for symbol in cfg["symbols"]]
    start = date.fromisoformat(cfg["start_date"])
    end = date.fromisoformat(cfg["end_date"])
    base_price = float(cfg.get("base_price", 100.0))
    price_increment = float(cfg.get("price_increment", 1.25))
    volume_base = int(cfg.get("base_volume", 1_000))
    volume_increment = int(cfg.get("volume_increment", 25))
    hours = int(cfg.get("close_hour", 16))
    minutes = int(cfg.get("close_minute", 0))

    current = start
    index = 0
    payload: List[Dict[str, Any]] = []
    while current <= end:
        for symbol in symbols:
            offset = _symbol_offset(symbol)
            opening = base_price + offset + index * price_increment
            closing = opening + 0.75
            high = closing + 0.25
            low = opening - 0.5
            vol = volume_base + offset * 10 + index * volume_increment
            timestamp = datetime.combine(current, datetime.min.time(), tzinfo=UTC).replace(hour=hours, minute=minutes)
            payload.append(
                {
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "open": round(opening, 6),
                    "high": round(high, 6),
                    "low": round(low, 6),
                    "close": round(closing, 6),
                    "volume": float(vol),
                }
            )
        current += timedelta(days=1)
        index += 1
    payload.sort(key=lambda item: (item["symbol"], item["timestamp"]))
    return payload


def _symbol_offset(symbol: str) -> float:
    digest = hashlib.sha256(symbol.encode("utf-8")).hexdigest()
    return (int(digest[:6], 16) % 25) * 0.5


def materialize_flat_file(path: Path, records: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["symbol", "timestamp", "open", "high", "low", "close", "volume"])
        for record in records:
            writer.writerow(
                [
                    record["symbol"],
                    record["timestamp"].astimezone(UTC).isoformat(),
                    record["open"],
                    record["high"],
                    record["low"],
                    record["close"],
                    record["volume"],
                ]
            )
    return path


def run_offline_ingestion(cfg: Mapping[str, Any], data_root: Path, run_id: str) -> Dict[str, Any]:
    records = synthesize_equity_records(cfg)
    snapshot = [dict(record) for record in records]
    fixture_path = data_root / "offline" / "equity" / f"{run_id}_equity.csv"
    materialize_flat_file(fixture_path, records)

    vendor = str(cfg.get("vendor", "polygon"))
    request = EquityIngestionRequest(
        symbols=tuple(cfg["symbols"]),
        start_date=date.fromisoformat(cfg["start_date"]),
        end_date=date.fromisoformat(cfg["end_date"]),
        flat_file_uris=(str(fixture_path),),
        vendor=vendor,
    )
    router = IngestionRouter({"force_mode": "flat_file"})
    raw_writer = RawEquityOHLCVWriter(base_path=data_root / "raw")
    manifest_dir = data_root / "raw" / vendor / EQUITY_DOMAIN / "manifests"
    pipeline = EquityIngestionPipeline(
        adapter=PolygonEquityAdapter(),
        router=router,
        raw_writer=raw_writer,
        manifest_dir=manifest_dir,
    )
    manifest = pipeline.run(request, run_id=run_id, forced_mode="flat_file")
    validation_root = manifest_dir / "validation"
    return {"manifest": manifest, "validation_root": validation_root, "records": snapshot}


def run_canonical_build(
    canonical_cfg: Mapping[str, Any],
    slice_cfg: Mapping[str, Any],
    data_root: Path,
    validation_root: Path,
    run_id: str,
) -> Dict[str, Any]:
    config_path = canonical_cfg.get("config_path", "configs/data_sources.json")
    domains = list(canonical_cfg.get("domains", [EQUITY_DOMAIN]))
    builder = ReconciliationBuilder.from_config_file(
        config_path,
        raw_data_root=data_root / "raw",
        canonical_root=data_root / "canonical",
        validation_manifest_root_path=validation_root,
        metrics_root=data_root / "monitoring" / "metrics" / "reconciliation",
        now=_deterministic_timestamp(run_id),
    )
    manifests = builder.run(
        domains=domains,
        start_date=slice_cfg["start_date"],
        end_date=slice_cfg["end_date"],
        run_id=run_id,
    )
    return {"builder": builder, "manifests": manifests}


def collect_equity_records(
    base: Path,
    *,
    fallback_records: Sequence[Mapping[str, Any]] | None = None,
    fallback_vendor: str | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    records_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    if pq is not None and base.exists():
        files = sorted(base.rglob("*.parquet"))
        for file_path in files:
            try:
                table = pq.read_table(file_path)
            except Exception:
                records_by_symbol.clear()
                break
            parts = file_path.parts
            try:
                symbol_index = parts.index("equity_ohlcv") + 1
                symbol = parts[symbol_index]
            except (ValueError, IndexError):
                continue
            payload = table.to_pylist()
            bucket = records_by_symbol.setdefault(symbol, [])
            bucket.extend(payload)
        if records_by_symbol:
            for symbol in list(records_by_symbol):
                records_by_symbol[symbol].sort(key=lambda entry: (entry.get("timestamp"), entry.get("symbol")))
            return records_by_symbol

    if not fallback_records or not fallback_vendor:
        return {}

    for raw in fallback_records:
        symbol = str(raw.get("symbol"))
        if not symbol:
            continue
        enriched = _attach_fallback_metadata(raw, fallback_vendor)
        records_by_symbol.setdefault(symbol, []).append(enriched)
    for symbol in list(records_by_symbol):
        records_by_symbol[symbol].sort(key=lambda entry: (entry.get("timestamp"), entry.get("symbol")))
    return records_by_symbol


def _attach_fallback_metadata(record: Mapping[str, Any], vendor: str) -> Dict[str, Any]:
    normalized = dict(record)
    normalized["primary_source_vendor"] = vendor
    normalized["selected_source_vendor"] = vendor
    normalized["fallback_source_vendor"] = None
    normalized["reconcile_method"] = "primary"
    return normalized


def collect_raw_counts(raw_root: Path, vendor: str) -> Dict[str, Dict[str, int]]:
    base = raw_root / vendor / EQUITY_DOMAIN
    manifest_dir = base / "manifests"
    return summarize_counts(base, manifest_dir=manifest_dir)


def collect_canonical_counts(canonical_root: Path) -> Dict[str, Dict[str, int]]:
    base = canonical_root / EQUITY_DOMAIN
    manifest_dir = canonical_root / "manifests" / EQUITY_DOMAIN
    return summarize_counts(base, manifest_dir=manifest_dir)


def summarize_counts(base: Path, *, manifest_dir: Path | None = None) -> Dict[str, Dict[str, int]]:
    by_symbol: Dict[str, Dict[str, int]] = {}
    if manifest_dir and manifest_dir.exists():
        _accumulate_from_manifests(by_symbol, manifest_dir)
    elif base.exists():
        _accumulate_from_parquet(by_symbol, base)
    ordered: Dict[str, Dict[str, int]] = {}
    for symbol in sorted(by_symbol):
        per_day = by_symbol[symbol]
        ordered[symbol] = {day: per_day[day] for day in sorted(per_day)}
    return ordered


def _accumulate_from_manifests(counter: Dict[str, Dict[str, int]], manifest_dir: Path) -> None:
    manifest_files = sorted(manifest_dir.glob("*.json"))
    for manifest_path in manifest_files:
        try:
            payload = json.loads(manifest_path.read_text())
        except Exception:
            continue
        entries: Sequence[Mapping[str, Any]] = ()
        if "files_written" in payload:
            entries = payload.get("files_written") or ()
        elif "outputs" in payload:
            entries = payload.get("outputs") or ()
        for entry in entries:
            path_str = entry.get("path")
            if not path_str:
                continue
            key = _symbol_day_from_path(Path(path_str))
            if not key:
                continue
            symbol, iso_day = key
            count = int(entry.get("records") or entry.get("record_count") or 0)
            per_symbol = counter.setdefault(symbol, {})
            per_symbol[iso_day] = per_symbol.get(iso_day, 0) + count


def _accumulate_from_parquet(counter: Dict[str, Dict[str, int]], base: Path) -> None:
    for file_path in sorted(base.rglob("*.parquet")):
        key = _symbol_day_from_path(file_path)
        if not key:
            continue
        symbol, iso_day = key
        count = _row_count(file_path)
        per_symbol = counter.setdefault(symbol, {})
        per_symbol[iso_day] = per_symbol.get(iso_day, 0) + count


def _symbol_day_from_path(path: Path) -> tuple[str, str] | None:
    parts = path.parts
    try:
        domain_index = parts.index(EQUITY_DOMAIN)
    except ValueError:
        return None
    try:
        symbol = parts[domain_index + 1]
    except IndexError:
        return None
    cursor = domain_index + 2
    if cursor >= len(parts):
        return None
    if parts[cursor] == "daily":
        cursor += 1
    if cursor + 2 >= len(parts):
        return None
    year, month, day_segment = parts[cursor], parts[cursor + 1], parts[cursor + 2]
    day = day_segment.split(".")[0]
    return symbol, f"{year}-{month}-{day}"


def _row_count(path: Path) -> int:
    if pq is not None:
        try:
            parquet_file = pq.ParquetFile(path)
            metadata = parquet_file.metadata
            if metadata is not None:
                return metadata.num_rows
        except Exception:
            return 0
    return 0


def _record_day(record: Mapping[str, Any]) -> str:
    timestamp = record.get("timestamp")
    to_pydatetime = getattr(timestamp, "to_pydatetime", None)
    if callable(to_pydatetime):
        timestamp = to_pydatetime()
    if isinstance(timestamp, datetime):
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        return timestamp.astimezone(UTC).date().isoformat()
    if isinstance(timestamp, str):
        clean = timestamp.strip()
        if len(clean) >= 10:
            return clean[:10]
    if record.get("snapshot_date"):
        return str(record["snapshot_date"])[:10]
    raise ValueError("Record missing timestamp information")


def compute_percent_missing(symbols: Sequence[str], start: str, end: str, canonical_counts: Mapping[str, Mapping[str, int]]) -> Dict[str, float]:
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    total_days = (end_date - start_date).days + 1
    percents: Dict[str, float] = {}
    for symbol in sorted(symbols):
        observed = len(canonical_counts.get(symbol, {}))
        missing = max(total_days - observed, 0)
        percents[symbol] = round((missing / total_days) * 100.0, 6) if total_days else 0.0
    return percents


def build_vendor_decisions(manifest: Mapping[str, Any], canonical_records: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    summary: List[Dict[str, Any]] = []
    for symbol in sorted(canonical_records):
        for record in canonical_records[symbol]:
            summary.append(
                {
                    "symbol": symbol,
                    "date": _record_day(record),
                    "method": record.get("reconcile_method"),
                    "selected_vendor": record.get("selected_source_vendor"),
                    "fallback_vendor": record.get("fallback_source_vendor"),
                    "primary_vendor": record.get("primary_source_vendor"),
                }
            )
    summary.sort(key=lambda item: (item["symbol"], item["date"]))
    fallback_records = sum(1 for entry in summary if entry.get("method") == "fallback")
    return {
        "domain": manifest.get("domain"),
        "vendor_priority": manifest.get("vendor_priority", []),
        "vendor_usage": manifest.get("vendor_usage", {}),
        "records": summary,
        "fallback_records": fallback_records,
    }


def render_equity_plot(symbol: str, records: Sequence[Mapping[str, Any]], path: Path) -> Path:
    if not records:
        raise RuntimeError(f"No canonical data available to plot for symbol {symbol}")
    width, height = 720, 360
    margin_left, margin_right, margin_top, margin_bottom = 60, 20, 20, 40
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    pixels = bytearray([255] * width * height * 3)

    def set_pixel(x: int, y: int, color: Sequence[int]) -> None:
        if not (0 <= x < width and 0 <= y < height):
            return
        index = (y * width + x) * 3
        pixels[index : index + 3] = bytes(color)

    # Draw axes
    axis_color = (0, 0, 0)
    x_axis_y = height - margin_bottom
    for x in range(margin_left, width - margin_right):
        set_pixel(x, x_axis_y, axis_color)
    for y in range(margin_top, height - margin_bottom + 1):
        set_pixel(margin_left, y, axis_color)

    closes = [float(record.get("close", 0.0)) for record in records]
    min_close = min(closes)
    max_close = max(closes)
    span = max(max_close - min_close, 1e-6)
    total_points = len(records)
    x_span = max(total_points - 1, 1)

    def to_coords(idx: int, close_value: float) -> tuple[int, int]:
        x = margin_left + int(round((idx / x_span) * plot_width))
        relative = (close_value - min_close) / span
        y = height - margin_bottom - int(round(relative * plot_height))
        return x, y

    line_color = (0, 90, 181)
    points = [to_coords(idx, closes[idx]) for idx in range(total_points)]
    for start, end in zip(points, points[1:]):
        _draw_line(pixels, width, height, start, end, line_color)

    write_png(path, width, height, pixels)
    return path


def _draw_line(
    pixels: bytearray,
    width: int,
    height: int,
    start: tuple[int, int],
    end: tuple[int, int],
    color: Sequence[int],
) -> None:
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if 0 <= x0 < width and 0 <= y0 < height:
            index = (y0 * width + x0) * 3
            pixels[index : index + 3] = bytes(color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def write_png(path: Path, width: int, height: int, pixels: bytearray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_rows = bytearray()
    stride = width * 3
    for y in range(height):
        row_start = y * stride
        raw_rows.append(0)
        raw_rows.extend(pixels[row_start : row_start + stride])

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(raw_rows), level=9)
    with path.open("wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        handle.write(chunk(b"IHDR", ihdr))
        handle.write(chunk(b"IDAT", idat))
        handle.write(chunk(b"IEND", b""))


def build_hashes(
    data_root: Path,
    manifest: Mapping[str, Any],
    plot_path: Path,
    report_path: Path,
) -> Dict[str, Any]:
    canonical_hashes: Dict[str, str] = {}
    for entry in manifest.get("outputs", []):
        path = Path(entry["path"]).resolve()
        canonical_hashes[_rel_path(path, data_root)] = entry.get("file_hash") or compute_file_hash(path)

    manifest_path = data_root / "canonical" / "manifests" / manifest.get("domain", "") / f"{manifest['run_id']}.json"
    manifest_hashes = {_rel_path(manifest_path, data_root): compute_file_hash(manifest_path)}
    return {
        "canonical_files": canonical_hashes,
        "canonical_manifests": manifest_hashes,
        "plot_png": compute_file_hash(plot_path),
        "report_json": "",  # populated just before writing to avoid self-referential hashing
    }


def _rel_path(path: Path, data_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(data_root.resolve()))
    except ValueError:
        return str(path)


def write_report(report_payload: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    hashes = report_payload.setdefault("hashes", {})
    # Hash the payload with the report hash field cleared to avoid recursive dependencies.
    hash_ready = copy.deepcopy(report_payload)
    if "hashes" in hash_ready:
        hash_ready["hashes"]["report_json"] = ""
    canonical_bytes = json.dumps(hash_ready, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    hashes["report_json"] = f"sha256:{hashlib.sha256(canonical_bytes).hexdigest()}"
    final_bytes = json.dumps(report_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    path.write_bytes(final_bytes + b"\n")
    return path


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    run_id = args.run_id or derive_run_id(config)
    data_root = resolve_data_root(args.data_root)
    os.environ["QUANTO_DATA_ROOT"] = str(data_root)

    offline_cfg = dict(config.get("offline_ingestion", {}))
    canonical_cfg = dict(config.get("canonical", {}))
    reporting_cfg = dict(config.get("reporting", {}))
    offline_cfg.setdefault("vendor", "polygon")

    if not args.offline:
        raise RuntimeError("Online mode is not implemented for the v1 slice")

    _disable_parquet_dependencies()

    ingestion = run_offline_ingestion(offline_cfg, data_root, run_id)
    canonical = run_canonical_build(canonical_cfg, offline_cfg, data_root, ingestion["validation_root"], run_id)

    raw_counts = collect_raw_counts(data_root / "raw", offline_cfg["vendor"])
    canonical_counts = collect_canonical_counts(data_root / "canonical")
    canonical_records = collect_equity_records(
        data_root / "canonical" / EQUITY_DOMAIN,
        fallback_records=ingestion.get("records"),
        fallback_vendor=offline_cfg["vendor"],
    )
    percent_missing = compute_percent_missing(offline_cfg["symbols"], offline_cfg["start_date"], offline_cfg["end_date"], canonical_counts)
    equity_manifest = canonical["manifests"].get(EQUITY_DOMAIN)
    if not equity_manifest:
        raise RuntimeError("Canonical builder did not return an equity manifest")
    vendor_decisions = build_vendor_decisions(equity_manifest, canonical_records)

    plot_symbol = reporting_cfg.get("plot_symbol") or (sorted(canonical_records) or [offline_cfg["symbols"][0]])[0]
    plot_path = data_root / "monitoring" / "plots" / PLOT_NAME
    render_equity_plot(plot_symbol, canonical_records.get(plot_symbol, []), plot_path)

    report_path = data_root / "monitoring" / "reports" / REPORT_NAME
    hashes = build_hashes(data_root, equity_manifest, plot_path, report_path)
    report_payload = {
        "run_id": run_id,
        "config_path": str(config_path),
        "data_root": str(data_root),
        "domains": canonical_cfg.get("domains", [EQUITY_DOMAIN]),
        "parameters": {
            "symbols": offline_cfg["symbols"],
            "start_date": offline_cfg["start_date"],
            "end_date": offline_cfg["end_date"],
            "vendor": offline_cfg["vendor"],
            "offline": args.offline,
        },
        "raw_counts": raw_counts,
        "canonical_counts": canonical_counts,
        "percent_missing_days": percent_missing,
        "vendor_decisions": vendor_decisions,
        "hashes": hashes,
        "artifacts": {
            "report": _rel_path(report_path, data_root),
            "plot": _rel_path(plot_path, data_root),
            "canonical_manifest": _rel_path(
                data_root / "canonical" / "manifests" / EQUITY_DOMAIN / f"{run_id}.json", data_root
            ),
        },
    }
    write_report(report_payload, report_path)
    print(json.dumps({"report": str(report_path), "plot": str(plot_path)}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
