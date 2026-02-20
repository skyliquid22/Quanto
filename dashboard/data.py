"""Data loading and caching for the Quanto dashboard."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None

try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    st = None  # type: ignore

from dashboard.config import CACHE_TTL_SECONDS, resolve_data_root


@dataclass(frozen=True)
class ExperimentSummary:
    experiment_id: str
    experiment_name: str | None
    feature_set: str | None
    regime_feature_set: str | None
    policy: str | None
    reward_version: str | None
    start_date: str | None
    end_date: str | None
    sharpe: float | None
    total_return: float | None
    max_drawdown: float | None
    turnover_1d_mean: float | None
    avg_exposure: float | None
    tx_cost_bps: float | None
    recorded_at: str | None
    qualification_passed: bool | None


def _cache_data(func):
    if st is None:
        return func
    return st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)(func)


@_cache_data
def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - log only
        print(f"[dashboard] Failed to read {path}: {exc}")
        return None


def _get_nested(payload: Mapping[str, Any] | None, path: str) -> Any:
    if not payload:
        return None
    node: Any = payload
    for part in path.split("."):
        if not isinstance(node, Mapping):
            return None
        node = node.get(part)
    return node


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


def _parse_recorded_at(payload: Mapping[str, Any] | None) -> str | None:
    if not payload:
        return None
    recorded = payload.get("recorded_at")
    if not recorded:
        return None
    try:
        return datetime.fromisoformat(str(recorded)).isoformat()
    except ValueError:
        return None


def _extract_reward_version(spec_payload: Mapping[str, Any] | None) -> str | None:
    if not spec_payload:
        return None
    policy_params = spec_payload.get("policy_params")
    if isinstance(policy_params, Mapping):
        value = policy_params.get("reward_version")
        return str(value) if value else None
    policy = spec_payload.get("policy")
    if str(policy).strip().lower() == "ppo":
        return "reward_v1"
    return None


def _qualification_passed(exp_dir: Path) -> bool | None:
    payload = read_json(exp_dir / "promotion" / "qualification_report.json")
    if not payload:
        return None
    if "passed" in payload:
        return bool(payload.get("passed"))
    return None


@_cache_data
def load_experiment_summaries(data_root: Path | None = None) -> list[ExperimentSummary]:
    root = data_root or resolve_data_root()
    registry_root = root / "experiments"
    summaries: list[ExperimentSummary] = []
    if not registry_root.exists():
        return summaries
    for exp_dir in sorted(registry_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        metrics_path = exp_dir / "evaluation" / "metrics.json"
        if not metrics_path.exists():
            continue
        spec_payload = read_json(exp_dir / "spec" / "experiment_spec.json") or {}
        metrics_payload = read_json(metrics_path) or {}
        run_summary = read_json(exp_dir / "logs" / "run_summary.json")
        summaries.append(
            ExperimentSummary(
                experiment_id=exp_dir.name,
                experiment_name=spec_payload.get("experiment_name"),
                feature_set=spec_payload.get("feature_set"),
                regime_feature_set=spec_payload.get("regime_feature_set") or None,
                policy=spec_payload.get("policy"),
                reward_version=_extract_reward_version(spec_payload),
                start_date=spec_payload.get("start_date"),
                end_date=spec_payload.get("end_date"),
                sharpe=_coerce_float(_get_nested(metrics_payload, "performance.sharpe")),
                total_return=_coerce_float(_get_nested(metrics_payload, "performance.total_return")),
                max_drawdown=_coerce_float(_get_nested(metrics_payload, "performance.max_drawdown")),
                turnover_1d_mean=_coerce_float(_get_nested(metrics_payload, "trading.turnover_1d_mean")),
                avg_exposure=_coerce_float(_get_nested(metrics_payload, "trading.avg_exposure")),
                tx_cost_bps=_coerce_float(_get_nested(metrics_payload, "trading.tx_cost_bps")),
                recorded_at=_parse_recorded_at(run_summary),
                qualification_passed=_qualification_passed(exp_dir),
            )
        )
    summaries.sort(key=lambda entry: (entry.recorded_at or "", entry.experiment_id))
    return summaries


def load_experiment_payload(exp_id: str, data_root: Path | None = None) -> dict[str, Any]:
    root = data_root or resolve_data_root()
    exp_dir = root / "experiments" / exp_id
    payload = {
        "spec": read_json(exp_dir / "spec" / "experiment_spec.json") or {},
        "metrics": read_json(exp_dir / "evaluation" / "metrics.json") or {},
        "regime_slices": read_json(exp_dir / "evaluation" / "regime_slices.json"),
        "qualification": read_json(exp_dir / "promotion" / "qualification_report.json"),
    }
    shadow_root = root / "shadow" / exp_id
    payload["shadow"] = _load_latest_shadow(shadow_root)
    return payload


def _load_latest_shadow(shadow_root: Path) -> dict[str, Any] | None:
    if not shadow_root.exists():
        return None
    run_dirs = [entry for entry in shadow_root.iterdir() if entry.is_dir()]
    run_dirs.sort(key=lambda entry: entry.stat().st_mtime, reverse=True)
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics_sim.json"
        if metrics_path.exists():
            return read_json(metrics_path)
    return None


def load_recent_promotions(data_root: Path | None = None) -> Sequence[Mapping[str, Any]]:
    root = data_root or resolve_data_root()
    promotions_root = root / "promotions"
    if not promotions_root.exists():
        return []
    items: list[Mapping[str, Any]] = []
    for path in promotions_root.rglob("*.json"):
        payload = read_json(path)
        if payload:
            entry = dict(payload)
            entry["__mtime"] = path.stat().st_mtime
            items.append(entry)
    items.sort(key=lambda item: item.get("__mtime", 0), reverse=True)
    return items


def load_latest_data_health(data_root: Path | None = None) -> dict[str, Any] | None:
    root = data_root or resolve_data_root()
    health_root = root / "monitoring" / "data_health"
    if not health_root.exists():
        return None
    run_dirs = [entry for entry in health_root.iterdir() if entry.is_dir()]
    if not run_dirs:
        return None
    run_dirs.sort(key=lambda entry: entry.stat().st_mtime, reverse=True)
    for run_dir in run_dirs:
        canonical_path = run_dir / "canonical_summary.json"
        if canonical_path.exists():
            payload = read_json(canonical_path)
            if payload is not None:
                payload["__mtime"] = run_dir.stat().st_mtime
                return payload
    return None


def _scan_partitioned_dates(domain_root: Path) -> dict[str, set[date]]:
    symbol_dates: dict[str, set[date]] = {}
    if not domain_root.exists():
        return symbol_dates
    for path in domain_root.glob("*/*/*/*.parquet"):
        parts = path.parts
        try:
            symbol = parts[-4]
            year = int(parts[-3])
            month = int(parts[-2])
            day = int(path.stem)
            symbol_dates.setdefault(symbol, set()).add(date(year, month, day))
        except Exception:
            continue
    return symbol_dates


def _scan_options_surface_dates(domain_root: Path) -> dict[str, set[date]]:
    symbol_dates: dict[str, set[date]] = {}
    if not domain_root.exists():
        return symbol_dates
    if pd is None:
        return symbol_dates
    for path in domain_root.glob("*/daily/*.parquet"):
        symbol = path.parents[1].name
        try:
            frame = pd.read_parquet(path, columns=["date"])
        except Exception:
            continue
        if frame is None or frame.empty or "date" not in frame.columns:
            continue
        dates = pd.to_datetime(frame["date"], errors="coerce").dt.date.dropna()
        if dates.empty:
            continue
        symbol_dates.setdefault(symbol, set()).update(dates.tolist())
    return symbol_dates


def _coverage_from_symbol_dates(symbol_dates: Mapping[str, set[date]]) -> dict[str, Any]:
    if not symbol_dates:
        return {"symbols": 0, "min_date": None, "max_date": None, "coverage": None}
    union_dates: set[date] = set()
    for dates in symbol_dates.values():
        union_dates.update(dates)
    if not union_dates:
        return {"symbols": len(symbol_dates), "min_date": None, "max_date": None, "coverage": None}
    min_date = min(union_dates)
    max_date = max(union_dates)
    union_count = len(union_dates)
    avg_ratio = 0.0
    for dates in symbol_dates.values():
        avg_ratio += len(dates) / union_count if union_count else 0.0
    coverage = (avg_ratio / len(symbol_dates)) * 100.0 if symbol_dates else None
    return {
        "symbols": len(symbol_dates),
        "min_date": min_date.isoformat(),
        "max_date": max_date.isoformat(),
        "coverage": coverage,
    }


def load_domain_coverage(data_root: Path | None = None) -> dict[str, dict[str, Any]]:
    root = data_root or resolve_data_root()
    coverage: dict[str, dict[str, Any]] = {}

    health = load_latest_data_health(root)
    if health:
        summary = (health.get("canonical_summary") or {}).get("summary_by_symbol")
        if isinstance(summary, Mapping) and summary:
            symbols = list(summary.keys())
            min_date = min((item.get("first_observed") for item in summary.values() if item.get("first_observed")), default=None)
            max_date = max((item.get("last_observed") for item in summary.values() if item.get("last_observed")), default=None)
            missing_ratio = (health.get("canonical_summary") or {}).get("overall", {}).get("missing_ratio")
            coverage_pct = None
            if missing_ratio is not None:
                coverage_pct = (1.0 - float(missing_ratio)) * 100.0
            coverage["equity_ohlcv"] = {
                "symbols": len(symbols),
                "min_date": min_date,
                "max_date": max_date,
                "coverage": coverage_pct,
            }

    fundamentals_root = root / "canonical" / "fundamentals"
    coverage["fundamentals"] = _coverage_from_symbol_dates(_scan_partitioned_dates(fundamentals_root))

    insiders_root = root / "canonical" / "insiders"
    coverage["insiders"] = _coverage_from_symbol_dates(_scan_partitioned_dates(insiders_root))

    options_root = root / "derived" / "options_surface_v1"
    coverage["options_surface"] = _coverage_from_symbol_dates(_scan_options_surface_dates(options_root))

    return coverage


__all__ = [
    "ExperimentSummary",
    "load_experiment_summaries",
    "load_experiment_payload",
    "load_recent_promotions",
    "load_latest_data_health",
    "load_domain_coverage",
]
