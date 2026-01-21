"""Canonical reconciliation builder consolidating multi-vendor datasets."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import math
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence
import sys

from infra.paths import get_data_root, get_repo_root, raw_root, validation_manifest_root
from infra.storage.parquet import write_parquet_atomic
from infra.storage.parquet_writer import merge_write_parquet, write_normalized_parquet

try:  # pragma: no cover - optional dependency chain
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover - keep tests lightweight
    pq = None

from .lineage import LineageInput, LineageOutput, build_lineage_payload, compute_file_hash

UTC = timezone.utc
MAX_OHLC_RATIO = 50.0
MAX_DAILY_CLOSE_JUMP = 20.0
_OPTION_DOMAIN_DIRS = {
    "option_contract_reference": "option_contract_reference",
    "option_contract_ohlcv": "option_contract_ohlcv",
    "option_open_interest": "option_open_interest",
}

_LEGACY_OPTIONS_DOMAINS = {
    "options_reference": "option_contract_reference",
    "options_ohlcv": "option_contract_ohlcv",
    "options_oi": "option_open_interest",
}


@dataclass
class ManifestInfo:
    """Materialized validation manifest used as reconciliation input."""

    vendor: str
    path: Path
    payload: Dict[str, Any]
    creation_timestamp: str


@dataclass
class VendorSnapshot:
    """Domain-specific raw snapshot for a given vendor."""

    manifest: ManifestInfo
    records_by_key: Dict[Any, Any] = field(default_factory=dict)
    records_by_underlying: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    files: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DomainMetrics:
    """Aggregate reconciliation metrics persisted for monitoring."""

    domain: str
    run_id: str
    close_diff_bp: float | None = None
    close_diff_samples: int = 0
    volume_diff_pct: float | None = None
    volume_diff_samples: int = 0
    percent_missing_primary: float = 0.0
    fallback_usage_rate: float = 0.0
    total_records: int = 0
    fallback_count: int = 0

    def as_dict(self) -> Dict[str, Any]:
        payload = {
            "domain": self.domain,
            "run_id": self.run_id,
            "percent_missing_primary": self.percent_missing_primary,
            "fallback_usage_rate": self.fallback_usage_rate,
            "total_records": self.total_records,
            "fallback_count": self.fallback_count,
        }
        if self.close_diff_bp is not None:
            payload["close_diff_bp"] = self.close_diff_bp
            payload["close_diff_samples"] = self.close_diff_samples
        if self.volume_diff_pct is not None:
            payload["volume_diff_pct"] = self.volume_diff_pct
            payload["volume_diff_samples"] = self.volume_diff_samples
        return payload


class ReconciliationError(RuntimeError):
    """Raised when deterministic reconciliation cannot continue."""


class ReconciliationBuilder:
    """Builder orchestrating domain-specific canonical reconciliation rules."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        raw_data_root: Path | str | None = None,
        canonical_root: Path | str | None = None,
        validation_manifest_root_path: Path | str | None = None,
        metrics_root: Path | str | None = None,
        now: datetime | None = None,
    ) -> None:
        reconciliation_cfg = dict(config.get("reconciliation", config))
        price_sanity_cfg = dict(reconciliation_cfg.get("price_sanity", {}))
        domain_cfg = reconciliation_cfg.get("domains")
        if domain_cfg is None:
            # Backwards compatible with configs that keep domains at top level.
            domain_cfg = {
                key: value for key, value in reconciliation_cfg.items() if isinstance(value, Mapping)
            }
        if not domain_cfg:
            raise ValueError("No reconciliation domains configured")
        self.domain_configs: Dict[str, Dict[str, Any]] = {str(k): dict(v) for k, v in domain_cfg.items()}
        legacy_domains = sorted(name for name in self.domain_configs if name in _LEGACY_OPTIONS_DOMAINS)
        if legacy_domains:
            canonical = ", ".join(sorted(_OPTION_DOMAIN_DIRS))
            raise ValueError(
                "Legacy options domains detected in reconciliation config: "
                f"{legacy_domains}. Replace them with canonical names: {canonical}."
            )
        self.default_allowed_statuses: Sequence[str] = tuple(
            reconciliation_cfg.get("allowed_validation_statuses", ["passed"])
        )
        self.price_sanity = {
            "enabled": bool(price_sanity_cfg.get("enabled", True)),
            "strict": bool(price_sanity_cfg.get("strict", False)),
            "max_return_sigma": float(price_sanity_cfg.get("max_return_sigma", 24.0)),
            "min_abs_return": float(price_sanity_cfg.get("min_abs_return", 0.45)),
            "median_jump": float(price_sanity_cfg.get("median_jump", 3.6)),
            "detect_splits": bool(price_sanity_cfg.get("detect_splits", True)),
            "action": str(price_sanity_cfg.get("action", "report")).strip().lower(),
        }
        if self.price_sanity["action"] not in {"report", "fail", "drop", "clip"}:
            raise ValueError("price_sanity.action must be one of: report, fail, drop, clip")
        self.raw_data_root = _resolve_runtime_root(Path(raw_data_root)) if raw_data_root else raw_root()
        self.canonical_root = _resolve_runtime_root(Path(canonical_root), layer="canonical") if canonical_root else get_data_root() / "canonical"
        self.validation_manifest_root = (
            Path(validation_manifest_root_path)
            if validation_manifest_root_path
            else validation_manifest_root()
        )
        metrics_base = metrics_root if metrics_root else get_repo_root() / "monitoring" / "metrics" / "reconciliation"
        self.metrics_root = Path(metrics_base)
        self.run_timestamp = (now or datetime.now(tz=UTC)).astimezone(UTC).isoformat()

    @classmethod
    def from_config_file(
        cls,
        config_path: Path | str,
        **kwargs: Any,
    ) -> "ReconciliationBuilder":
        path = Path(config_path)
        text = path.read_text()
        data: Mapping[str, Any]
        suffix = path.suffix.lower()
        if suffix == ".json":
            data = json.loads(text)
        elif suffix in {".yml", ".yaml"}:
            try:
                import yaml  # type: ignore
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "PyYAML must be installed to parse YAML configs; provide JSON instead or install PyYAML"
                ) from exc
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls(data, **kwargs)

    def run(
        self,
        *,
        domains: Sequence[str] | None,
        start_date: date | str | datetime,
        end_date: date | str | datetime,
        run_id: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Execute reconciliation for the requested domains."""

        requested = list(domains) if domains else sorted(self.domain_configs)
        start = self._coerce_date(start_date, "start_date")
        end = self._coerce_date(end_date, "end_date")
        if end < start:
            raise ValueError("end_date must be greater than or equal to start_date")

        manifests: Dict[str, Dict[str, Any]] = {}
        for domain in requested:
            cfg = self.domain_configs.get(domain)
            if not cfg:
                raise ValueError(f"Domain '{domain}' is not configured")
            handler = getattr(self, f"_run_{domain}", None)
            if handler is None:
                handler = self._run_generic_domain
            manifest = handler(domain=domain, cfg=cfg, start=start, end=end, run_id=run_id)
            manifests[domain] = manifest
        return manifests

    # ------------------------------------------------------------------
    # Domain handlers
    # ------------------------------------------------------------------

    def _run_equity_ohlcv(self, *, domain: str, cfg: Mapping[str, Any], start: date, end: date, run_id: str) -> Dict[str, Any]:
        vendor_priority = self._resolve_vendor_priority(cfg)
        vendor_data = self._load_equity_snapshots(vendor_priority, start, end)
        return self._materialize_equity_canonical(
            domain=domain,
            vendor_priority=vendor_priority,
            vendor_data=vendor_data,
            cfg=cfg,
            start=start,
            end=end,
            run_id=run_id,
        )

    def _run_option_contract_reference(self, **kwargs: Any) -> Dict[str, Any]:
        return self._run_options_domain(**kwargs)

    def _run_option_contract_ohlcv(self, **kwargs: Any) -> Dict[str, Any]:
        return self._run_options_domain(**kwargs)

    def _run_option_open_interest(self, **kwargs: Any) -> Dict[str, Any]:
        return self._run_options_domain(**kwargs)

    def _run_options_reference(self, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover - shim
        raise ReconciliationError(
            "options_reference domain alias is no longer supported. Use option_contract_reference instead."
        )

    def _run_options_ohlcv(self, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover - shim
        raise ReconciliationError(
            "options_ohlcv domain alias is no longer supported. Use option_contract_ohlcv instead."
        )

    def _run_options_oi(self, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover - shim
        raise ReconciliationError(
            "options_oi domain alias is no longer supported. Use option_open_interest instead."
        )

    def _run_options_domain(
        self,
        *,
        domain: str,
        cfg: Mapping[str, Any],
        start: date,
        end: date,
        run_id: str,
    ) -> Dict[str, Any]:
        vendor_priority = self._resolve_vendor_priority(cfg)
        vendor_data = self._load_options_snapshots(domain, vendor_priority, start, end)
        return self._materialize_options_canonical(
            domain=domain,
            cfg=cfg,
            vendor_priority=vendor_priority,
            vendor_data=vendor_data,
            start=start,
            end=end,
            run_id=run_id,
        )

    def _run_fundamentals(self, *, domain: str, cfg: Mapping[str, Any], start: date, end: date, run_id: str) -> Dict[str, Any]:
        vendor_priority = self._resolve_vendor_priority(cfg)
        vendor_data = self._load_fundamentals_snapshots(vendor_priority, start, end)
        return self._materialize_fundamentals_canonical(
            domain=domain,
            cfg=cfg,
            vendor_priority=vendor_priority,
            vendor_data=vendor_data,
            start=start,
            end=end,
            run_id=run_id,
        )

    def _run_generic_domain(
        self,
        *,
        domain: str,
        cfg: Mapping[str, Any],
        start: date,
        end: date,
        run_id: str,
    ) -> Dict[str, Any]:
        raise ReconciliationError(f"Domain '{domain}' does not have a reconciliation strategy")

    # ------------------------------------------------------------------
    # Equity reconciliation
    # ------------------------------------------------------------------

    def _load_equity_snapshots(
        self,
        vendor_priority: Sequence[str],
        start: date,
        end: date,
    ) -> Dict[str, VendorSnapshot]:
        snapshots: Dict[str, VendorSnapshot] = {}
        manifests = self._load_manifests("equity_ohlcv")
        for vendor in vendor_priority:
            manifest = manifests.get(vendor)
            manifest_missing = manifest is None
            vendor_root = self.raw_data_root / vendor / "equity_ohlcv"
            if not vendor_root.exists():
                continue
            if manifest_missing:
                manifest = ManifestInfo(
                    vendor=vendor,
                    path=vendor_root / "manifests" / "missing_manifest.json",
                    payload={"source_vendor": vendor, "domain": "equity_ohlcv", "run_id": "unknown"},
                    creation_timestamp="",
                )
            records: Dict[tuple[str, str], Dict[str, Any]] = {}
            files: List[Dict[str, Any]] = []
            years = range(start.year, end.year + 1)
            # Prefer yearly shards when present (daily interval).
            for symbol_dir in vendor_root.iterdir():
                if not symbol_dir.is_dir():
                    continue
                daily_dir = symbol_dir / "daily"
                if not daily_dir.exists():
                    continue
                for year in years:
                    yearly_path = daily_dir / f"{year}.parquet"
                    if not yearly_path.exists():
                        continue
                    raw_records = self._read_records(yearly_path)
                    file_hash = compute_file_hash(yearly_path)
                    files.append({"path": str(yearly_path), "file_hash": file_hash, "records": len(raw_records)})
                    for row in raw_records:
                        normalized = dict(row)
                        normalized_symbol = str(normalized.get("symbol") or symbol_dir.name)
                        normalized_ts = self._resolve_timestamp(normalized)
                        partition_day = normalized_ts[:10]
                        partition_date = date.fromisoformat(partition_day)
                        if not (start <= partition_date <= end):
                            continue
                        normalized["symbol"] = normalized_symbol
                        normalized["timestamp"] = normalized_ts
                        normalized["source_vendor"] = vendor
                        normalized["__partition_day"] = partition_day
                        normalized["__input_file_hash"] = file_hash
                        key = (normalized_symbol, normalized_ts)
                        if key not in records:
                            records[key] = normalized
            if records:
                snapshots[vendor] = VendorSnapshot(manifest=manifest, records_by_key=records, files=files)
                if manifest_missing:
                    print(
                        f"Warning: no validation manifest found for vendor '{vendor}'. "
                        "Using raw files without manifest metadata.",
                        file=sys.stderr,
                    )
                continue
            for file_path in vendor_root.glob("*/daily/*/*/*.parquet"):
                partition_day = self._parse_daily_partition(file_path)
                if not partition_day or not (start <= partition_day <= end):
                    continue
                symbol = file_path.parents[4].name
                raw_records = self._read_records(file_path)
                file_hash = compute_file_hash(file_path)
                files.append({"path": str(file_path), "file_hash": file_hash, "records": len(raw_records)})
                for row in raw_records:
                    normalized = dict(row)
                    normalized["symbol"] = str(normalized.get("symbol") or symbol)
                    normalized["timestamp"] = self._resolve_timestamp(normalized)
                    normalized["source_vendor"] = vendor
                    normalized["__partition_day"] = partition_day.isoformat()
                    normalized["__input_file_hash"] = file_hash
                    key = (normalized["symbol"], normalized["timestamp"])
                    if key not in records:
                        records[key] = normalized
            if records:
                snapshots[vendor] = VendorSnapshot(manifest=manifest, records_by_key=records, files=files)
        return snapshots

    def _materialize_equity_canonical(
        self,
        *,
        domain: str,
        vendor_priority: Sequence[str],
        vendor_data: Mapping[str, VendorSnapshot],
        cfg: Mapping[str, Any],
        start: date,
        end: date,
        run_id: str,
    ) -> Dict[str, Any]:
        if not vendor_data:
            return self._persist_manifest(
                domain=domain,
                run_id=run_id,
                start=start,
                end=end,
                records_written=0,
                vendor_priority=vendor_priority,
                vendor_usage={},
                inputs=[],
                outputs=[],
                metrics=self._write_metrics(DomainMetrics(domain=domain, run_id=run_id)),
                lineage_metadata={},
            )

        all_keys = set()
        for snapshot in vendor_data.values():
            all_keys.update(snapshot.records_by_key.keys())
        sorted_keys = sorted(all_keys, key=lambda item: (item[0], item[1]))
        selected_records: List[Dict[str, Any]] = []
        vendor_usage: MutableMapping[str, int] = defaultdict(int)
        missing_primary = 0
        fallback_count = 0
        close_diffs: List[float] = []
        volume_diffs: List[float] = []
        for symbol, ts in sorted_keys:
            candidates: List[tuple[str, Dict[str, Any]]] = []
            for vendor in vendor_priority:
                snapshot = vendor_data.get(vendor)
                if not snapshot:
                    continue
                record = snapshot.records_by_key.get((symbol, ts))
                if record:
                    candidates.append((vendor, record))
            if not candidates:
                continue
            if vendor_priority and candidates[0][0] != vendor_priority[0]:
                missing_primary += 1
            chosen_vendor, record = candidates[0]
            method = "primary" if chosen_vendor == vendor_priority[0] else "fallback"
            metadata_manifest = vendor_data[chosen_vendor].manifest
            enriched = self._attach_metadata(
                record,
                manifest=metadata_manifest,
                method=method,
                primary_vendor=vendor_priority[0] if vendor_priority else chosen_vendor,
                selected_vendor=chosen_vendor,
            )
            vendor_usage[chosen_vendor] += 1
            if method == "fallback":
                fallback_count += 1
            selected_records.append(enriched)
            if len(candidates) > 1:
                top = candidates[0][1]
                second = candidates[1][1]
                if "close" in top and "close" in second:
                    top_close = float(top["close"])
                    sec_close = float(second["close"])
                    if top_close and sec_close:
                        diff_bp = abs(top_close - sec_close) / top_close * 10000.0
                        close_diffs.append(diff_bp)
                if "volume" in top and "volume" in second:
                    top_vol = float(top["volume"]) or 0.0
                    sec_vol = float(second["volume"]) or 0.0
                    if top_vol and sec_vol:
                        volume_diffs.append(abs(top_vol - sec_vol) / top_vol * 100.0)

        selected_records, price_sanity_report = self._enforce_equity_price_sanity(selected_records)

        metrics = DomainMetrics(domain=domain, run_id=run_id)
        total = len(selected_records)
        metrics.total_records = total
        metrics.fallback_count = fallback_count
        if total:
            metrics.percent_missing_primary = missing_primary / total
            metrics.fallback_usage_rate = fallback_count / total
        if close_diffs:
            metrics.close_diff_bp = sum(close_diffs) / len(close_diffs)
            metrics.close_diff_samples = len(close_diffs)
        if volume_diffs:
            metrics.volume_diff_pct = sum(volume_diffs) / len(volume_diffs)
            metrics.volume_diff_samples = len(volume_diffs)

        outputs = self._write_equity_outputs(selected_records)
        inputs = self._build_lineage_inputs(domain="equity_ohlcv", vendor_data=vendor_data)
        metrics_path = self._write_metrics(metrics)
        lineage_metadata = {
            "missing_primary_records": missing_primary,
            "fallback_records": fallback_count,
            "manifest_paths": self._manifest_path_map(vendor_data),
        }
        if price_sanity_report:
            lineage_metadata["price_sanity"] = price_sanity_report
        return self._persist_manifest(
            domain=domain,
            run_id=run_id,
            start=start,
            end=end,
            records_written=total,
            vendor_priority=vendor_priority,
            vendor_usage=dict(vendor_usage),
            inputs=inputs,
            outputs=outputs,
            metrics=metrics_path,
            lineage_metadata=lineage_metadata,
            price_sanity=price_sanity_report,
        )

    def _write_equity_outputs(self, records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        return self._write_equity_outputs_yearly(records)

    def _enforce_equity_price_sanity(
        self,
        records: Sequence[Mapping[str, Any]],
    ) -> tuple[List[Mapping[str, Any]], Dict[str, Any] | None]:
        if not records:
            return list(records), None
        cfg = dict(self.price_sanity)
        if not cfg.get("enabled", True):
            return list(records), None
        violations: Dict[str, List[str]] = defaultdict(list)
        violation_types: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        split_like: Dict[str, List[str]] = defaultdict(list)
        per_symbol: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for record in records:
            symbol = str(record.get("symbol") or "").upper()
            if symbol:
                per_symbol[symbol].append(record)

        for record in records:
            symbol = str(record.get("symbol") or "").upper()
            timestamp = str(record.get("timestamp") or "")
            open_value = _coerce_float(record.get("open"))
            high_value = _coerce_float(record.get("high"))
            low_value = _coerce_float(record.get("low"))
            close_value = _coerce_float(record.get("close"))
            if any(value is None or value <= 0 for value in (open_value, high_value, low_value, close_value)):
                violations[symbol].append(timestamp)
                violation_types["ohlc_invalid"][symbol].append(timestamp)
                continue
            if high_value < max(open_value, close_value, low_value) or low_value > min(open_value, close_value):
                violations[symbol].append(timestamp)
                violation_types["ohlc_invalid"][symbol].append(timestamp)
                continue
            if low_value > 0 and high_value / low_value > MAX_OHLC_RATIO:
                violations[symbol].append(timestamp)
                violation_types["ohlc_invalid"][symbol].append(timestamp)
                continue

        for symbol, rows in per_symbol.items():
            ordered = sorted(
                rows,
                key=lambda item: str(item.get("timestamp") or ""),
            )
            prev_close: float | None = None
            returns_window: List[float] = []
            close_window: List[float] = []
            for record in ordered:
                timestamp = str(record.get("timestamp") or "")
                close_value = _coerce_float(record.get("close"))
                if close_value is None or close_value <= 0:
                    prev_close = close_value
                    continue
                if prev_close is not None and prev_close > 0:
                    ratio = close_value / prev_close
                    if cfg.get("detect_splits", True) and _is_split_like(ratio):
                        split_like[symbol].append(timestamp)
                    else:
                        log_return = math.log(ratio)
                        returns_window.append(log_return)
                        if len(returns_window) > 60:
                            returns_window.pop(0)
                        rolling_std = _rolling_std(returns_window)
                        threshold = max(cfg["min_abs_return"], cfg["max_return_sigma"] * rolling_std)
                        if abs(log_return) > threshold:
                            violations[symbol].append(timestamp)
                            violation_types["return_outlier"][symbol].append(timestamp)
                close_window.append(float(close_value))
                if len(close_window) > 20:
                    close_window.pop(0)
                prev_close = close_value
                if close_window:
                    mean_close = sum(close_window) / len(close_window)
                    if mean_close > 0:
                        if abs(close_value / mean_close - 1.0) > cfg["median_jump"]:
                            violations[symbol].append(timestamp)
                            violation_types["median_jump"][symbol].append(timestamp)

        if violations:
            report = _build_price_sanity_report(
                violations=violations,
                violation_types=violation_types,
                split_like=split_like,
                cfg=cfg,
            )
            action = cfg["action"]
            if cfg.get("strict") or action == "fail":
                raise ValueError(
                    "Equity price sanity gate failed; "
                    f"{report.get('total_anomalies', 0)} record(s) flagged: {report.get('by_symbol', {})}"
                )
            if action == "drop":
                filtered = [record for record in records if not _is_record_flagged(record, violations)]
                return filtered, report
            if action == "clip":
                _clip_price_anomalies(records, violations)
            return list(records), report
        if split_like:
            report = _build_price_sanity_report(
                violations={},
                violation_types={},
                split_like=split_like,
                cfg=cfg,
            )
            return list(records), report
        return list(records), None

    def _write_equity_outputs_yearly(self, records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        grouped: MutableMapping[tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
        for record in records:
            symbol = str(record["symbol"])
            timestamp = str(record["timestamp"])
            year = timestamp[:4]
            grouped[(symbol, year)].append(record)
        outputs: List[Dict[str, Any]] = []
        for (symbol, year), entries in grouped.items():
            base_dir = self.canonical_root / "equity_ohlcv" / symbol / "daily"
            path = base_dir / f"{year}.parquet"
            if path.parent != base_dir:
                raise ValueError(f"Unexpected canonical path layout: {path}")
            dedup_cols = ["symbol", "timestamp"]
            if entries and "source_vendor" in entries[0]:
                dedup_cols.append("source_vendor")
            if path.exists():
                path.unlink()
            result = merge_write_parquet(path, entries, dedup_cols=dedup_cols, sort_key="timestamp")
            outputs.append(
                {
                    "path": str(path),
                    "file_hash": result["file_hash"],
                    "content_hash": result["content_hash"],
                    "records": result.get("records", len(entries)),
                }
            )
        outputs.sort(key=lambda item: item["path"])
        return outputs

    # ------------------------------------------------------------------
    # Options reconciliation
    # ------------------------------------------------------------------

    def _load_options_snapshots(
        self,
        domain: str,
        vendor_priority: Sequence[str],
        start: date,
        end: date,
    ) -> Dict[str, VendorSnapshot]:
        snapshots: Dict[str, VendorSnapshot] = {}
        manifests = self._load_manifests(domain)
        domain_dir = _OPTION_DOMAIN_DIRS.get(domain)
        if not domain_dir:
            legacy_target = _LEGACY_OPTIONS_DOMAINS.get(domain)
            if legacy_target:
                raise ReconciliationError(
                    f"Unsupported legacy options domain '{domain}'. Use '{legacy_target}' instead."
                )
            raise ReconciliationError(f"Unsupported options domain '{domain}'")
        for vendor in vendor_priority:
            manifest = manifests.get(vendor)
            if not manifest:
                continue
            vendor_root = self.raw_data_root / vendor / domain_dir
            if not vendor_root.exists():
                continue
            files: List[Dict[str, Any]] = []
            records_by_underlying: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            records_by_key: Dict[str, Dict[str, Any]] = {}
            is_timeseries = domain != "option_contract_reference"
            glob_pattern = "*/daily/*/*/*.parquet" if is_timeseries else "*/*/*/*.parquet"
            for file_path in vendor_root.glob(glob_pattern):
                partition_day = (
                    self._parse_daily_partition(file_path) if is_timeseries else self._parse_simple_partition(file_path)
                )
                if not partition_day or not (start <= partition_day <= end):
                    continue
                underlying_from_path = file_path.parents[3].name if not is_timeseries else ""
                raw_records = self._read_records(file_path)
                file_hash = compute_file_hash(file_path)
                files.append({"path": str(file_path), "file_hash": file_hash, "records": len(raw_records)})
                for row in raw_records:
                    normalized = dict(row)
                    normalized["source_vendor"] = vendor
                    if is_timeseries:
                        normalized["timestamp"] = self._normalize_timestamp(normalized.get("timestamp"))
                        day_field = normalized["timestamp"][:10]
                        underlying_symbol = str(normalized.get("underlying_symbol") or "").strip()
                        if not underlying_symbol:
                            raise ReconciliationError(
                                f"underlying_symbol field missing in {domain} record for vendor {vendor}"
                            )
                        normalized["underlying_symbol"] = underlying_symbol
                        key = f"{normalized.get('option_symbol')}|{normalized['timestamp']}"
                    else:
                        day_field = str(normalized.get("snapshot_date") or partition_day.isoformat())
                        normalized["snapshot_date"] = day_field
                        underlying_symbol = str(normalized.get("underlying_symbol") or underlying_from_path)
                        normalized["underlying_symbol"] = underlying_symbol
                        key = str(normalized.get("option_symbol"))
                    normalized["__partition_day"] = day_field
                    normalized["__input_file_hash"] = file_hash
                    records_by_underlying[normalized["underlying_symbol"]].append(normalized)
                    records_by_key[key] = normalized
            if records_by_underlying:
                snapshots[vendor] = VendorSnapshot(
                    manifest=manifest,
                    records_by_key=records_by_key,
                    records_by_underlying=records_by_underlying,
                    files=files,
                )
        return snapshots

    def _materialize_options_canonical(
        self,
        *,
        domain: str,
        cfg: Mapping[str, Any],
        vendor_priority: Sequence[str],
        vendor_data: Mapping[str, VendorSnapshot],
        start: date,
        end: date,
        run_id: str,
    ) -> Dict[str, Any]:
        if not vendor_data:
            metrics = DomainMetrics(domain=domain, run_id=run_id)
            metrics_path = self._write_metrics(metrics)
            return self._persist_manifest(
                domain=domain,
                run_id=run_id,
                start=start,
                end=end,
                records_written=0,
                vendor_priority=vendor_priority,
                vendor_usage={},
                inputs=[],
                outputs=[],
                metrics=metrics_path,
                lineage_metadata={},
            )

        underlying_union: set[str] = set()
        for snapshot in vendor_data.values():
            underlying_union.update(snapshot.records_by_underlying.keys())
        ordered_underlyings = sorted(underlying_union)
        selected_records: List[Dict[str, Any]] = []
        vendor_usage: MutableMapping[str, int] = defaultdict(int)
        missing_primary = 0
        fallback_count = 0
        for underlying in ordered_underlyings:
            available: List[tuple[str, List[Dict[str, Any]]]] = []
            for vendor in vendor_priority:
                snapshot = vendor_data.get(vendor)
                if not snapshot:
                    continue
                rows = snapshot.records_by_underlying.get(underlying)
                if rows:
                    available.append((vendor, rows))
            if not available:
                continue
            if vendor_priority and available[0][0] != vendor_priority[0]:
                missing_primary += 1
            chosen_vendor, rows = available[0]
            method = "primary" if chosen_vendor == vendor_priority[0] else "fallback"
            manifest = vendor_data[chosen_vendor].manifest
            for row in rows:
                enriched = self._attach_metadata(
                    row,
                    manifest=manifest,
                    method=method,
                    primary_vendor=vendor_priority[0] if vendor_priority else chosen_vendor,
                    selected_vendor=chosen_vendor,
                )
                selected_records.append(enriched)
                vendor_usage[chosen_vendor] += 1
                if method == "fallback":
                    fallback_count += 1
        metrics = DomainMetrics(domain=domain, run_id=run_id)
        total = len(selected_records)
        metrics.total_records = total
        metrics.fallback_count = fallback_count
        if total:
            metrics.percent_missing_primary = missing_primary / max(len(ordered_underlyings), 1)
            metrics.fallback_usage_rate = fallback_count / total
        outputs = self._write_options_outputs(domain, selected_records)
        inputs = self._build_lineage_inputs(domain=domain, vendor_data=vendor_data)
        metrics_path = self._write_metrics(metrics)
        lineage_metadata = {
            "missing_primary_underlyings": missing_primary,
            "fallback_records": fallback_count,
            "manifest_paths": self._manifest_path_map(vendor_data),
        }
        return self._persist_manifest(
            domain=domain,
            run_id=run_id,
            start=start,
            end=end,
            records_written=total,
            vendor_priority=vendor_priority,
            vendor_usage=dict(vendor_usage),
            inputs=inputs,
            outputs=outputs,
            metrics=metrics_path,
            lineage_metadata=lineage_metadata,
        )

    def _write_options_outputs(self, domain: str, records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        grouped: MutableMapping[tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
        if domain == "option_contract_reference":
            for record in records:
                underlying = str(record["underlying_symbol"])
                iso_day = str(record["snapshot_date"])[:10]
                grouped[(underlying, iso_day)].append(record)
        else:
            for record in records:
                symbol = str(record.get("option_symbol"))
                iso_day = str(record["timestamp"])[:10]
                grouped[(symbol, iso_day)].append(record)
        outputs: List[Dict[str, Any]] = []
        subdir = {
            "option_contract_reference": "option_contract_reference",
            "option_contract_ohlcv": "option_contract_ohlcv",
            "option_open_interest": "option_open_interest",
        }[domain]
        for (identifier, iso_day), entries in grouped.items():
            year, month, day = iso_day.split("-")
            if domain == "option_contract_reference":
                path = self.canonical_root / subdir / identifier / year / month / f"{day}.parquet"
            else:
                path = (
                    self.canonical_root
                    / subdir
                    / identifier
                    / "daily"
                    / year
                    / month
                    / f"{day}.parquet"
                )
            if domain == "option_contract_reference":
                ordered = sorted(entries, key=lambda item: (item.get("option_symbol"), item.get("strike_price")))
            else:
                ordered = sorted(entries, key=lambda item: (item.get("option_symbol"), item["timestamp"]))
            result = write_parquet_atomic(ordered, path)
            outputs.append({"path": str(path), "file_hash": result["file_hash"], "records": len(ordered)})
        outputs.sort(key=lambda item: item["path"])
        return outputs

    # ------------------------------------------------------------------
    # Fundamentals reconciliation
    # ------------------------------------------------------------------

    def _load_fundamentals_snapshots(
        self,
        vendor_priority: Sequence[str],
        start: date,
        end: date,
    ) -> Dict[str, VendorSnapshot]:
        snapshots: Dict[str, VendorSnapshot] = {}
        manifests = self._load_manifests("fundamentals")
        for vendor in vendor_priority:
            manifest = manifests.get(vendor)
            if not manifest:
                continue
            vendor_root = self.raw_data_root / vendor / "fundamentals"
            if not vendor_root.exists():
                continue
            files: List[Dict[str, Any]] = []
            records: Dict[tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
            for file_path in vendor_root.glob("*/*/*/*.parquet"):
                partition_day = self._parse_simple_partition(file_path)
                if not partition_day or not (start <= partition_day <= end):
                    continue
                symbol = file_path.parents[3].name
                raw_records = self._read_records(file_path)
                file_hash = compute_file_hash(file_path)
                files.append({"path": str(file_path), "file_hash": file_hash, "records": len(raw_records)})
                for row in raw_records:
                    normalized = dict(row)
                    normalized["symbol"] = str(normalized.get("symbol") or symbol)
                    normalized["report_date"] = self._normalize_date(normalized.get("report_date"))
                    normalized["filing_date"] = self._normalize_date(normalized.get("filing_date"))
                    normalized["source_vendor"] = vendor
                    normalized["__input_file_hash"] = file_hash
                    statement = str(normalized.get("statement_type") or "")
                    key = (normalized["symbol"], normalized["report_date"], statement)
                    records[key].append(normalized)
            if records:
                snapshots[vendor] = VendorSnapshot(manifest=manifest, records_by_key=records, files=files)
        return snapshots

    def _materialize_fundamentals_canonical(
        self,
        *,
        domain: str,
        cfg: Mapping[str, Any],
        vendor_priority: Sequence[str],
        vendor_data: Mapping[str, VendorSnapshot],
        start: date,
        end: date,
        run_id: str,
    ) -> Dict[str, Any]:
        if not vendor_data:
            metrics = DomainMetrics(domain=domain, run_id=run_id)
            metrics_path = self._write_metrics(metrics)
            return self._persist_manifest(
                domain=domain,
                run_id=run_id,
                start=start,
                end=end,
                records_written=0,
                vendor_priority=vendor_priority,
                vendor_usage={},
                inputs=[],
                outputs=[],
                metrics=metrics_path,
                lineage_metadata={},
            )

        record_vendor = str(cfg.get("fundamentals_of_record") or vendor_priority[0])
        all_keys = set()
        for snapshot in vendor_data.values():
            all_keys.update(snapshot.records_by_key.keys())
        ordered_keys = sorted(all_keys)
        selected_records: List[Dict[str, Any]] = []
        vendor_usage: MutableMapping[str, int] = defaultdict(int)
        fallback_count = 0
        superseded_log: Dict[str, List[str]] = {}
        for key in ordered_keys:
            candidates: List[tuple[str, List[Dict[str, Any]]]] = []
            for vendor in vendor_priority:
                snapshot = vendor_data.get(vendor)
                if not snapshot:
                    continue
                rows = snapshot.records_by_key.get(key)
                if rows:
                    candidates.append((vendor, rows))
            if not candidates:
                continue
            chosen_vendor, rows = candidates[0]
            selected, superseded = self._select_latest_filing(rows)
            method = "record_vendor" if chosen_vendor == record_vendor else "fallback"
            if superseded:
                method = "restatement"
                superseded_log["|".join(key)] = superseded
            if method == "fallback":
                fallback_count += 1
            manifest = vendor_data[chosen_vendor].manifest
            enriched = self._attach_metadata(
                selected,
                manifest=manifest,
                method=method,
                primary_vendor=record_vendor,
                selected_vendor=chosen_vendor,
            )
            selected_records.append(enriched)
            vendor_usage[chosen_vendor] += 1

        metrics = DomainMetrics(domain=domain, run_id=run_id)
        metrics.total_records = len(selected_records)
        metrics.fallback_count = fallback_count
        if metrics.total_records:
            metrics.fallback_usage_rate = fallback_count / metrics.total_records
        outputs = self._write_fundamentals_outputs(selected_records)
        inputs = self._build_lineage_inputs(domain=domain, vendor_data=vendor_data)
        metrics_path = self._write_metrics(metrics)
        lineage_metadata = {
            "superseded_filings": superseded_log,
            "manifest_paths": self._manifest_path_map(vendor_data),
        }
        return self._persist_manifest(
            domain=domain,
            run_id=run_id,
            start=start,
            end=end,
            records_written=len(selected_records),
            vendor_priority=vendor_priority,
            vendor_usage=dict(vendor_usage),
            inputs=inputs,
            outputs=outputs,
            metrics=metrics_path,
            lineage_metadata=lineage_metadata,
        )

    def _select_latest_filing(self, rows: Sequence[Mapping[str, Any]]) -> tuple[Dict[str, Any], List[str]]:
        ordered = []
        for row in rows:
            filing_date = self._normalize_date(row.get("filing_date"))
            filing_id = str(row.get("filing_id") or "")
            ordered.append((filing_date, filing_id, row))
        ordered.sort(key=lambda item: (item[0], item[1]))
        selected = dict(ordered[-1][2])
        superseded = [item[1] for item in ordered[:-1] if item[1]]
        return selected, superseded

    def _write_fundamentals_outputs(self, records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        grouped: MutableMapping[tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
        for record in records:
            symbol = str(record["symbol"])
            iso_day = str(record["report_date"])[:10]
            grouped[(symbol, iso_day)].append(record)
        outputs: List[Dict[str, Any]] = []
        for (symbol, iso_day), entries in grouped.items():
            year, month, day = iso_day.split("-")
            path = self.canonical_root / "fundamentals" / symbol / year / month / f"{day}.parquet"
            ordered = sorted(entries, key=lambda item: (item.get("statement_type"), item.get("filing_id")))
            result = write_parquet_atomic(ordered, path)
            outputs.append({"path": str(path), "file_hash": result["file_hash"], "records": len(ordered)})
        outputs.sort(key=lambda item: item["path"])
        return outputs

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _resolve_vendor_priority(self, cfg: Mapping[str, Any]) -> Sequence[str]:
        vendors = cfg.get("vendor_priority") or cfg.get("vendors")
        if not vendors:
            raise ValueError("vendor_priority must be configured for each domain")
        return [str(v) for v in vendors]

    def _load_manifests(self, domain: str) -> Dict[str, ManifestInfo]:
        manifests: Dict[str, ManifestInfo] = {}
        seen_paths: set[Path] = set()
        candidate_dirs = [self.validation_manifest_root / domain, self.validation_manifest_root]
        if self.raw_data_root.exists():
            for vendor_dir in self.raw_data_root.iterdir():
                if not vendor_dir.is_dir():
                    continue
                manifest_dir = vendor_dir / domain / "manifests"
                candidate_dirs.append(manifest_dir)
                validation_dir = manifest_dir / "validation"
                candidate_dirs.append(validation_dir)
                candidate_dirs.append(validation_dir / domain)
        for directory in candidate_dirs:
            if not directory.exists():
                continue
            for manifest_path in sorted(directory.glob("*.json")):
                if manifest_path in seen_paths:
                    continue
                seen_paths.add(manifest_path)
                payload = json.loads(manifest_path.read_text())
                status = payload.get("validation_status")
                if status not in self.default_allowed_statuses:
                    continue
                vendor = str(payload.get("source_vendor") or manifest_path.parents[2].name)
                if not vendor:
                    continue
                timestamp = str(payload.get("creation_timestamp") or "")
                existing = manifests.get(vendor)
                if existing is None or timestamp >= existing.creation_timestamp:
                    manifests[vendor] = ManifestInfo(
                        vendor=vendor,
                        path=manifest_path,
                        payload=payload,
                        creation_timestamp=timestamp,
                    )
        return manifests

    def _read_records(self, path: Path) -> List[Dict[str, Any]]:
        if pq is not None:
            try:
                table = pq.read_table(path)
                return table.to_pylist()
            except Exception:
                # Fall back to JSON payloads produced by the lightweight writer.
                pass
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return []
        return json.loads(text)

    def _parse_daily_partition(self, path: Path) -> date | None:
        try:
            day = int(path.stem.split(".")[0])
            month = int(path.parent.name)
            year = int(path.parent.parent.name)
            return date(year, month, day)
        except Exception:
            return None

    def _parse_simple_partition(self, path: Path) -> date | None:
        try:
            day = int(path.stem.split(".")[0])
            month = int(path.parent.name)
            year = int(path.parent.parent.name)
            return date(year, month, day)
        except Exception:
            return None

    def _resolve_timestamp(self, record: Mapping[str, Any]) -> str:
        value = record.get("timestamp")
        if value is None:
            raw = record.get("t")
            if raw is None:
                raise ValueError("timestamp field is required for OHLCV records")
            if isinstance(raw, (int, float)):
                seconds = float(raw)
                if seconds > 1e12:
                    seconds /= 1000.0
                return datetime.fromtimestamp(seconds, tz=UTC).isoformat()
            if isinstance(raw, str) and raw.strip().isdigit():
                digits = float(raw.strip())
                if digits > 1e12:
                    digits /= 1000.0
                return datetime.fromtimestamp(digits, tz=UTC).isoformat()
            value = raw
        return self._normalize_timestamp(value)

    def _resolve_timestamp(self, record: Mapping[str, Any]) -> str:
        value = record.get("timestamp")
        if value is None:
            raw_value = record.get("t")
            if raw_value is None:
                raise ValueError("timestamp field is required for OHLCV records")
            if isinstance(raw_value, (int, float)):
                seconds = float(raw_value)
                if seconds > 1e12:
                    seconds /= 1000.0
                return datetime.fromtimestamp(seconds, tz=UTC).isoformat()
            if isinstance(raw_value, str):
                trimmed = raw_value.strip()
                if trimmed.isdigit():
                    seconds = float(trimmed)
                    if seconds > 1e12:
                        seconds /= 1000.0
                    return datetime.fromtimestamp(seconds, tz=UTC).isoformat()
                value = trimmed
            else:
                value = raw_value
        return self._normalize_timestamp(value)

    def _normalize_timestamp(self, value: Any) -> str:
        to_pydatetime = getattr(value, "to_pydatetime", None)
        if callable(to_pydatetime):
            candidate = to_pydatetime()
            if isinstance(candidate, datetime):
                if candidate.tzinfo is None:
                    candidate = candidate.replace(tzinfo=UTC)
                return candidate.astimezone(UTC).isoformat()
        if isinstance(value, datetime):
            return value.astimezone(UTC).isoformat()
        if isinstance(value, str):
            text = value.strip()
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                return datetime.fromisoformat(text).astimezone(UTC).isoformat()
            except ValueError:
                return text
        raise ValueError("timestamp field is required for OHLCV records")

    def _normalize_date(self, value: Any) -> str:
        if isinstance(value, datetime):
            return value.date().isoformat()
        if isinstance(value, date):
            return value.isoformat()
        if isinstance(value, str):
            return date.fromisoformat(value).isoformat()
        raise ValueError("Date fields must be ISO formatted strings or date objects")

    def _attach_metadata(
        self,
        record: Mapping[str, Any],
        *,
        manifest: ManifestInfo,
        method: str,
        primary_vendor: str,
        selected_vendor: str,
    ) -> Dict[str, Any]:
        normalized = dict(record)
        partition_marker = normalized.pop("__partition_day", None)
        input_hash = normalized.pop("__input_file_hash", None)
        normalized["primary_source_vendor"] = selected_vendor if method == "primary" else primary_vendor
        normalized["fallback_source_vendor"] = selected_vendor if method == "fallback" else None
        normalized["reconcile_method"] = method
        manifest_hashes = list(manifest.payload.get("input_file_hashes", []))
        if input_hash and input_hash not in manifest_hashes:
            manifest_hashes.append(input_hash)
        normalized["input_file_hashes"] = sorted(set(manifest_hashes))
        normalized["validation_status"] = manifest.payload.get("validation_status")
        normalized["creation_timestamp"] = self.run_timestamp
        normalized["selected_source_vendor"] = selected_vendor
        if partition_marker:
            normalized.setdefault("partition_day", partition_marker)
        return normalized

    def _build_lineage_inputs(self, *, domain: str, vendor_data: Mapping[str, VendorSnapshot]) -> List[LineageInput]:
        inputs: List[LineageInput] = []
        for snapshot in vendor_data.values():
            for file_info in snapshot.files:
                inputs.append(
                    LineageInput(
                        path=file_info["path"],
                        vendor=snapshot.manifest.vendor,
                        domain=domain,
                        file_hash=file_info["file_hash"],
                        record_count=file_info.get("records"),
                    )
                )
        return inputs

    def _write_metrics(self, metrics: DomainMetrics) -> str:
        self.metrics_root.mkdir(parents=True, exist_ok=True)
        metrics_path = self.metrics_root / f"{metrics.domain}_{metrics.run_id}.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics.as_dict(), handle, indent=2, sort_keys=True)
        return str(metrics_path)

    def _manifest_path_map(self, vendor_data: Mapping[str, VendorSnapshot]) -> Dict[str, str]:
        return {snapshot.manifest.vendor: str(snapshot.manifest.path) for snapshot in vendor_data.values()}

    def _persist_manifest(
        self,
        *,
        domain: str,
        run_id: str,
        start: date,
        end: date,
        records_written: int,
        vendor_priority: Sequence[str],
        vendor_usage: Mapping[str, int],
        inputs: Sequence[LineageInput],
        outputs: Sequence[Dict[str, Any]],
        metrics: str,
        lineage_metadata: Mapping[str, Any],
        price_sanity: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        manifest_dir = self.canonical_root / "manifests" / domain
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"{run_id}.json"
        output_objects = [
            LineageOutput(path=entry["path"], file_hash=entry.get("file_hash", ""), record_count=entry.get("records"))
            for entry in outputs
        ]
        lineage_payload = build_lineage_payload(
            domain=domain,
            run_id=run_id,
            inputs=inputs,
            outputs=output_objects,
            metadata=lineage_metadata,
        )
        manifest_payload = {
            "domain": domain,
            "run_id": run_id,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "creation_timestamp": self.run_timestamp,
            "records_written": records_written,
            "vendor_priority": list(vendor_priority),
            "vendor_usage": dict(vendor_usage),
            "inputs": lineage_payload["inputs"],
            "outputs": outputs,
            "metrics_path": metrics,
            "lineage": lineage_payload,
        }
        if price_sanity:
            manifest_payload["price_sanity"] = dict(price_sanity)
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest_payload, handle, indent=2, sort_keys=True)
        return manifest_payload

    def _coerce_date(self, value: date | str | datetime, field: str) -> date:
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, str):
            try:
                return date.fromisoformat(value)
            except ValueError as exc:  # pragma: no cover - defensive
                raise ValueError(f"{field} must be YYYY-MM-DD text (got {value!r})") from exc
        raise ValueError(f"{field} must be a date compatible value (got {value!r})")


def _rolling_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _is_split_like(ratio: float) -> bool:
    for multiple in (2, 3, 4, 5):
        if abs(ratio - multiple) <= 0.05 * multiple:
            return True
        inverse = 1.0 / multiple
        if abs(ratio - inverse) <= 0.05 * inverse:
            return True
    return False


def _build_price_sanity_report(
    *,
    violations: Mapping[str, Sequence[str]],
    violation_types: Mapping[str, Mapping[str, Sequence[str]]],
    split_like: Mapping[str, Sequence[str]],
    cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    by_symbol = {
        symbol: {
            "count": len(timestamps),
            "min_ts": min(timestamps),
            "max_ts": max(timestamps),
        }
        for symbol, timestamps in violations.items()
    }
    by_type: Dict[str, Any] = {}
    for anomaly_type, symbols in violation_types.items():
        by_type[anomaly_type] = {
            symbol: {
                "count": len(timestamps),
                "min_ts": min(timestamps),
                "max_ts": max(timestamps),
            }
            for symbol, timestamps in symbols.items()
        }
    split_summary = {
        symbol: {
            "count": len(timestamps),
            "min_ts": min(timestamps),
            "max_ts": max(timestamps),
        }
        for symbol, timestamps in split_like.items()
    }
    return {
        "action": cfg.get("action"),
        "strict": cfg.get("strict"),
        "total_anomalies": sum(len(ts) for ts in violations.values()),
        "by_symbol": by_symbol,
        "by_type": by_type,
        "split_like": split_summary,
    }


def _is_record_flagged(record: Mapping[str, Any], violations: Mapping[str, Sequence[str]]) -> bool:
    symbol = str(record.get("symbol") or "").upper()
    timestamp = str(record.get("timestamp") or "")
    return timestamp in set(violations.get(symbol, ()))


def _clip_price_anomalies(
    records: Sequence[Mapping[str, Any]],
    violations: Mapping[str, Sequence[str]],
) -> None:
    if not violations:
        return
    per_symbol: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        symbol = str(record.get("symbol") or "").upper()
        if symbol:
            per_symbol[symbol].append(record)
    for symbol, rows in per_symbol.items():
        ordered = sorted(rows, key=lambda item: str(item.get("timestamp") or ""))
        close_window: List[float] = []
        for record in ordered:
            timestamp = str(record.get("timestamp") or "")
            close_value = _coerce_float(record.get("close"))
            if close_value is not None and close_value > 0:
                close_window.append(float(close_value))
                if len(close_window) > 20:
                    close_window.pop(0)
            if timestamp not in violations.get(symbol, ()):
                continue
            if close_window:
                mean_close = sum(close_window) / len(close_window)
                record["open"] = mean_close
                record["high"] = mean_close
                record["low"] = mean_close
                record["close"] = mean_close

def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


__all__ = ["ReconciliationBuilder", "ReconciliationError"]


def _canonical_legacy_daily_enabled() -> bool:
    value = os.environ.get("QUANTO_CANONICAL_LEGACY_DAILY_SHARDS")
    if value is None:
        return False
    normalized = value.strip().lower()
    return normalized not in {"0", "false", "no"}


def _resolve_runtime_root(path: Path, *, layer: str = "raw") -> Path:
    expanded = path.expanduser()
    if expanded.name == layer:
        return expanded
    candidate = expanded / layer
    if candidate.exists():
        return candidate
    return expanded
