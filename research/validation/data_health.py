"""Deterministic data health reporting for canonical and feature layers."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from infra.normalization.lineage import compute_file_hash
from infra.paths import get_data_root
from research.datasets.canonical_equity_loader import CanonicalEquitySlice
from research.features.feature_registry import (
    build_features,
    is_universe_feature_set,
)

try:  # pragma: no cover - optional import in minimal environments
    import pandas as pd  # type: ignore
except Exception as exc:  # pragma: no cover
    pd = None
    _PANDAS_ERROR: Exception | None = exc
else:  # pragma: no cover
    _PANDAS_ERROR = None


CalendarMode = str


@dataclass(frozen=True)
class CanonicalLoadResult:
    slices: Mapping[str, CanonicalEquitySlice]
    file_hashes: Mapping[str, str]


@dataclass(frozen=True)
class FundamentalsLoadResult:
    frames: Mapping[str, "pd.DataFrame"]
    file_hashes: Mapping[str, str]


def load_canonical_equity_pandas(
    symbols: Sequence[str],
    start_date: date,
    end_date: date,
    *,
    data_root: Path | None = None,
    interval: str = "daily",
    parquet_engine: str = "fastparquet",
) -> CanonicalLoadResult:
    _ensure_pandas_available()
    resolved_root = Path(data_root) if data_root else get_data_root()
    normalized_interval = str(interval).strip().lower() or "daily"
    if normalized_interval != "daily":
        raise ValueError(f"Unsupported interval '{interval}'; only daily data is available in v1.")
    ordered = _ordered_symbols(symbols)
    base = resolved_root / "canonical" / "equity_ohlcv"
    slices: Dict[str, CanonicalEquitySlice] = {}
    hashes: Dict[str, str] = {}
    for symbol in ordered:
        frames: List["pd.DataFrame"] = []
        files: List[Path] = []
        for year in range(start_date.year, end_date.year + 1):
            path = base / symbol / normalized_interval / f"{year}.parquet"
            if not path.exists():
                raise FileNotFoundError(
                    f"Canonical yearly shard missing for {symbol} {year}. "
                    f"Expected file: {path}."
                )
            files.append(path)
            hashes[_relative_to(path, resolved_root)] = compute_file_hash(path)
            try:
                frame = pd.read_parquet(path, engine=parquet_engine)
            except Exception as exc:
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    raise exc
                if isinstance(raw, dict) and "data" in raw:
                    raw = raw["data"]
                if not isinstance(raw, list):
                    raise exc
                frame = pd.DataFrame(raw)
            if "timestamp" not in frame.columns:
                if frame.index.name == "timestamp":
                    frame = frame.reset_index()
                else:
                    raise ValueError(f"Canonical shard missing timestamp column: {path}")
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            frame = frame.dropna(subset=["timestamp"])
            frame = frame[(frame["timestamp"].dt.date >= start_date) & (frame["timestamp"].dt.date <= end_date)]
            frames.append(frame)
        merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not merged.empty:
            merged.sort_values("timestamp", inplace=True, kind="mergesort")
            merged.reset_index(drop=True, inplace=True)
            merged.set_index("timestamp", inplace=True)
            merged.index = pd.DatetimeIndex(merged.index, tz="UTC", name="timestamp")
        slices[symbol] = CanonicalEquitySlice(symbol=symbol, frame=merged, file_paths=files)
    return CanonicalLoadResult(slices=slices, file_hashes=hashes)


def load_canonical_fundamentals_pandas(
    symbols: Sequence[str],
    start_date: date,
    end_date: date,
    *,
    data_root: Path | None = None,
    parquet_engine: str = "fastparquet",
    lookback_days: int = 400,
) -> FundamentalsLoadResult:
    _ensure_pandas_available()
    resolved_root = Path(data_root) if data_root else get_data_root()
    ordered = _ordered_symbols(symbols)
    base = resolved_root / "canonical" / "fundamentals"
    frames: Dict[str, "pd.DataFrame"] = {}
    hashes: Dict[str, str] = {}
    lookback_start = start_date - timedelta(days=max(0, int(lookback_days)))
    for symbol in ordered:
        symbol_dir = base / symbol
        symbol_frames: List["pd.DataFrame"] = []
        if symbol_dir.exists():
            for year in range(lookback_start.year, end_date.year + 1):
                year_dir = symbol_dir / f"{year}"
                if not year_dir.exists():
                    continue
                for path in year_dir.rglob("*.parquet"):
                    partition_day = _parse_fundamentals_partition(path)
                    if not partition_day:
                        continue
                    if partition_day < lookback_start or partition_day > end_date:
                        continue
                    hashes[_relative_to(path, resolved_root)] = compute_file_hash(path)
                    try:
                        frame = pd.read_parquet(path, engine=parquet_engine)
                    except Exception as exc:
                        raise exc
                    if frame is None or frame.empty:
                        continue
                    symbol_frames.append(frame)
        merged = pd.concat(symbol_frames, ignore_index=True) if symbol_frames else pd.DataFrame()
        frames[symbol] = merged
    return FundamentalsLoadResult(frames=frames, file_hashes=hashes)


def compute_canonical_health(
    slices: Mapping[str, CanonicalEquitySlice],
    *,
    start_date: date,
    end_date: date,
    calendar_mode: CalendarMode = "union",
) -> Dict[str, Any]:
    _ensure_pandas_available()
    symbol_dates = _extract_symbol_dates(slices, start_date, end_date)
    union_dates = _sorted_dates(_union_dates(symbol_dates))
    intersection_dates = _sorted_dates(_intersection_dates(symbol_dates))
    summary_by_symbol: Dict[str, Any] = {}
    total_expected = 0
    total_missing = 0
    for symbol in sorted(symbol_dates):
        observed = symbol_dates[symbol]
        expected = _expected_dates(
            calendar_mode,
            union_dates=union_dates,
            intersection_dates=intersection_dates,
            observed_dates=observed,
        )
        missing_ranges, missing_count = _missing_ranges(expected, observed)
        expected_count = len(expected)
        observed_count = len(observed)
        total_expected += expected_count
        total_missing += missing_count
        summary_by_symbol[symbol] = {
            "expected_count": expected_count,
            "observed_count": observed_count,
            "missing_count": missing_count,
            "missing_ratio": _safe_ratio(missing_count, expected_count),
            "first_observed": _format_date(min(observed) if observed else None),
            "last_observed": _format_date(max(observed) if observed else None),
            "missing_ranges": missing_ranges,
        }
    return {
        "calendar_mode": calendar_mode,
        "calendar_size": len(union_dates) if calendar_mode in ("union", "symbol") else len(intersection_dates),
        "union_calendar_size": len(union_dates),
        "intersection_calendar_size": len(intersection_dates),
        "summary_by_symbol": summary_by_symbol,
        "overall": {
            "missing_ratio": _safe_ratio(total_missing, total_expected),
            "missing_count": total_missing,
            "expected_count": total_expected,
        },
    }


def compute_feature_health(
    frames_by_symbol: Mapping[str, "pd.DataFrame"],
    observation_columns: Sequence[str],
) -> Dict[str, Any]:
    _ensure_pandas_available()
    columns = list(observation_columns)
    summary_by_symbol: Dict[str, Any] = {}
    column_totals: Dict[str, Dict[str, int]] = {column: {"rows": 0, "nan_count": 0} for column in columns}
    total_rows = 0
    total_nans = 0
    for symbol in sorted(frames_by_symbol):
        frame = frames_by_symbol[symbol]
        data = frame[columns] if columns else pd.DataFrame(index=frame.index)
        rows = int(len(data))
        if rows > 0 and columns:
            nan_counts = data.isna().sum()
            row_valid_ratio = float(data.notna().all(axis=1).mean())
        else:
            nan_counts = pd.Series({column: 0 for column in columns})
            row_valid_ratio = None
        nan_summary = {}
        for column in columns:
            nan_count = int(nan_counts.get(column, 0))
            column_totals[column]["rows"] += rows
            column_totals[column]["nan_count"] += nan_count
            total_nans += nan_count
            nan_summary[column] = {
                "nan_count": nan_count,
                "nan_ratio": _safe_ratio(nan_count, rows),
            }
        total_rows += rows
        summary_by_symbol[symbol] = {
            "rows": rows,
            "row_valid_ratio": row_valid_ratio,
            "columns": nan_summary,
        }
    summary_by_column: Dict[str, Any] = {}
    for column in columns:
        totals = column_totals[column]
        summary_by_column[column] = {
            "rows": totals["rows"],
            "nan_count": totals["nan_count"],
            "nan_ratio": _safe_ratio(totals["nan_count"], totals["rows"]),
        }
    total_cells = total_rows * len(columns) if columns else 0
    overall_nan_ratio = _safe_ratio(total_nans, total_cells)
    return {
        "columns": columns,
        "summary_by_symbol": summary_by_symbol,
        "summary_by_column": summary_by_column,
        "overall": {
            "rows": total_rows,
            "nan_count": total_nans,
            "nan_ratio": overall_nan_ratio,
        },
    }


def compute_fundamentals_health(
    frames_by_symbol: Mapping[str, "pd.DataFrame"],
    *,
    start_date: date,
    end_date: date,
    stale_days: int = 180,
) -> Dict[str, Any]:
    _ensure_pandas_available()
    summary_by_symbol: Dict[str, Any] = {}
    stale_count = 0
    missing_count = 0
    for symbol in sorted(frames_by_symbol):
        frame = frames_by_symbol[symbol]
        if frame.empty:
            summary_by_symbol[symbol] = {
                "rows": 0,
                "rows_in_window": 0,
                "first_report_date": None,
                "last_report_date": None,
                "last_as_of_date": None,
                "stale_days_end": None,
                "stale_at_end": None,
            }
            missing_count += 1
            continue
        prepared = _prepare_fundamentals_for_health(frame)
        if prepared.empty:
            summary_by_symbol[symbol] = {
                "rows": 0,
                "rows_in_window": 0,
                "first_report_date": None,
                "last_report_date": None,
                "last_as_of_date": None,
                "stale_days_end": None,
                "stale_at_end": None,
            }
            missing_count += 1
            continue
        report_dates = prepared["report_date"]
        rows_in_window = prepared[
            (report_dates.dt.date >= start_date) & (report_dates.dt.date <= end_date)
        ]
        last_report = report_dates.max()
        last_as_of = prepared["as_of_date"].max()
        stale_days_end = None
        stale_at_end = None
        if pd.notna(last_as_of):
            stale_days_end = (end_date - last_as_of.date()).days
            stale_at_end = stale_days_end > stale_days
            if stale_at_end:
                stale_count += 1
        summary_by_symbol[symbol] = {
            "rows": int(len(prepared)),
            "rows_in_window": int(len(rows_in_window)),
            "first_report_date": _format_date(report_dates.min().date() if pd.notna(report_dates.min()) else None),
            "last_report_date": _format_date(last_report.date() if pd.notna(last_report) else None),
            "last_as_of_date": _format_date(last_as_of.date() if pd.notna(last_as_of) else None),
            "stale_days_end": stale_days_end,
            "stale_at_end": stale_at_end,
        }
    total_symbols = len(frames_by_symbol)
    return {
        "stale_days_threshold": stale_days,
        "summary_by_symbol": summary_by_symbol,
        "overall": {
            "symbols": total_symbols,
            "missing_symbols": missing_count,
            "stale_symbols": stale_count,
            "stale_ratio": _safe_ratio(stale_count, total_symbols),
        },
    }


def build_feature_frames(
    *,
    feature_set: str,
    slices: Mapping[str, CanonicalEquitySlice],
    start_date: date,
    end_date: date,
    data_root: Path | None = None,
) -> Tuple[Mapping[str, "pd.DataFrame"], Tuple[str, ...]]:
    if is_universe_feature_set(feature_set):
        raise ValueError(f"Feature set '{feature_set}' requires universe context and is not supported here.")
    frames: Dict[str, "pd.DataFrame"] = {}
    observation_columns: Tuple[str, ...] | None = None
    for symbol in sorted(slices):
        slice_data = slices[symbol]
        equity_df = slice_data.frame.reset_index()
        result = build_features(
            feature_set,
            equity_df,
            underlying_symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            data_root=data_root,
        )
        if observation_columns is None:
            observation_columns = result.observation_columns
        elif observation_columns != result.observation_columns:
            raise ValueError("Feature set observation columns are inconsistent across symbols")
        frames[symbol] = result.frame
    return frames, observation_columns or tuple()


def evaluate_thresholds(
    *,
    canonical_report: Mapping[str, Any],
    feature_report: Mapping[str, Any] | None,
    max_missing_ratio: float | None,
    max_nan_ratio: float | None,
) -> List[str]:
    failures: List[str] = []
    missing_ratio = canonical_report.get("overall", {}).get("missing_ratio")
    if max_missing_ratio is not None:
        if missing_ratio is None or missing_ratio > max_missing_ratio:
            failures.append("missing_ratio_exceeded")
    if feature_report is not None and max_nan_ratio is not None:
        nan_ratio = _max_column_nan_ratio(feature_report)
        if nan_ratio is None:
            nan_ratio = feature_report.get("overall", {}).get("nan_ratio")
        if nan_ratio is None or nan_ratio > max_nan_ratio:
            failures.append("nan_ratio_exceeded")
    return failures


def run_data_health_preflight(
    *,
    symbols: Sequence[str],
    start_date: date,
    end_date: date,
    feature_set: str | None,
    data_root: Path | None = None,
    interval: str = "daily",
    calendar_mode: CalendarMode = "union",
    parquet_engine: str = "fastparquet",
    max_missing_ratio: float | None = 0.01,
    max_nan_ratio: float | None = 0.05,
    strict: bool = False,
) -> Dict[str, Any]:
    """Run a lightweight data health check before training/evaluation."""

    try:
        load_result = load_canonical_equity_pandas(
            symbols,
            start_date,
            end_date,
            data_root=data_root,
            interval=interval,
            parquet_engine=parquet_engine,
        )
    except OSError as exc:
        if str(parquet_engine).strip().lower() != "auto":
            try:
                load_result = load_canonical_equity_pandas(
                    symbols,
                    start_date,
                    end_date,
                    data_root=data_root,
                    interval=interval,
                    parquet_engine="auto",
                )
            except FileNotFoundError as missing_exc:
                exc = missing_exc
            except OSError:
                raise
            else:
                exc = None
        if exc is None:
            pass
        else:
            if isinstance(exc, FileNotFoundError):
                payload = {
                    "canonical": {"error": str(exc)},
                    "features": None,
                    "failures": ["missing_data"],
                    "settings": {
                        "calendar_mode": calendar_mode,
                        "max_missing_ratio": max_missing_ratio,
                        "max_nan_ratio": max_nan_ratio,
                        "strict": strict,
                    },
                }
                message = f"Data health preflight skipped: {exc}"
                if strict:
                    raise RuntimeError(message) from exc
                print(message, file=sys.stderr)  # noqa: T201 - CLI warning
                return payload
            raise
    except FileNotFoundError as exc:
        payload = {
            "canonical": {"error": str(exc)},
            "features": None,
            "failures": ["missing_data"],
            "settings": {
                "calendar_mode": calendar_mode,
                "max_missing_ratio": max_missing_ratio,
                "max_nan_ratio": max_nan_ratio,
                "strict": strict,
            },
        }
        message = f"Data health preflight skipped: {exc}"
        if strict:
            raise RuntimeError(message) from exc
        print(message, file=sys.stderr)  # noqa: T201 - CLI warning
        return payload

    canonical_report = compute_canonical_health(
        load_result.slices,
        start_date=start_date,
        end_date=end_date,
        calendar_mode=calendar_mode,
    )

    feature_report = None
    fundamentals_report = None
    feature_set_name = str(feature_set).strip() if feature_set else ""
    if feature_set_name:
        if is_universe_feature_set(feature_set_name):
            feature_report = None
        else:
            try:
                frames, observation_columns = build_feature_frames(
                    feature_set=feature_set_name,
                    slices=load_result.slices,
                    start_date=start_date,
                    end_date=end_date,
                    data_root=data_root,
                )
                feature_report = compute_feature_health(frames, observation_columns)
            except Exception as exc:
                message = f"Data health preflight skipped feature checks: {exc}"
                if strict:
                    raise RuntimeError(message) from exc
                print(message, file=sys.stderr)  # noqa: T201 - CLI warning
                feature_report = None
        if _feature_set_uses_fundamentals(feature_set_name):
            try:
                fundamentals_load = load_canonical_fundamentals_pandas(
                    symbols,
                    start_date,
                    end_date,
                    data_root=data_root,
                    parquet_engine=parquet_engine,
                )
                fundamentals_report = compute_fundamentals_health(
                    fundamentals_load.frames,
                    start_date=start_date,
                    end_date=end_date,
                )
            except Exception as exc:
                message = f"Data health preflight skipped fundamentals checks: {exc}"
                if strict:
                    raise RuntimeError(message) from exc
                print(message, file=sys.stderr)  # noqa: T201 - CLI warning
                fundamentals_report = {"error": str(exc)}

    failures = evaluate_thresholds(
        canonical_report=canonical_report,
        feature_report=feature_report,
        max_missing_ratio=max_missing_ratio,
        max_nan_ratio=max_nan_ratio,
    )
    payload = {
        "canonical": canonical_report,
        "features": feature_report,
        "fundamentals": fundamentals_report,
        "failures": failures,
        "settings": {
            "calendar_mode": calendar_mode,
            "max_missing_ratio": max_missing_ratio,
            "max_nan_ratio": max_nan_ratio,
            "strict": strict,
        },
    }
    if failures:
        message = f"Data health preflight failed: {', '.join(failures)}"
        if strict:
            raise RuntimeError(message)
        print(message, file=sys.stderr)  # noqa: T201 - CLI warning
    return payload


def resolve_run_id(payload: Mapping[str, Any], *, prefix: str = "data_health") -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str)
    digest = sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _extract_symbol_dates(
    slices: Mapping[str, CanonicalEquitySlice],
    start: date,
    end: date,
) -> Dict[str, List[date]]:
    symbol_dates: Dict[str, List[date]] = {}
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    for symbol, slice_data in slices.items():
        frame = slice_data.frame
        if frame.empty:
            symbol_dates[symbol] = []
            continue
        index = frame.index
        index = index[(index >= start_ts) & (index <= end_ts)]
        dates = sorted({ts.date() for ts in index})
        symbol_dates[symbol] = dates
    return symbol_dates


def _union_dates(symbol_dates: Mapping[str, Sequence[date]]) -> Iterable[date]:
    dates: set[date] = set()
    for values in symbol_dates.values():
        dates.update(values)
    return dates


def _intersection_dates(symbol_dates: Mapping[str, Sequence[date]]) -> Iterable[date]:
    iterator = iter(symbol_dates.values())
    try:
        first = set(next(iterator))
    except StopIteration:
        return []
    intersection = first
    for values in iterator:
        intersection = intersection.intersection(values)
    return intersection


def _expected_dates(
    calendar_mode: CalendarMode,
    *,
    union_dates: Sequence[date],
    intersection_dates: Sequence[date],
    observed_dates: Sequence[date],
) -> List[date]:
    if calendar_mode == "intersection":
        return list(intersection_dates)
    if calendar_mode == "symbol":
        if not observed_dates:
            return []
        start = min(observed_dates)
        end = max(observed_dates)
        return [value for value in union_dates if start <= value <= end]
    return list(union_dates)


def _missing_ranges(expected: Sequence[date], observed: Sequence[date]) -> Tuple[List[Dict[str, Any]], int]:
    observed_set = set(observed)
    ranges: List[Dict[str, Any]] = []
    current_start: date | None = None
    current_end: date | None = None
    current_count = 0
    total_missing = 0
    for day in expected:
        missing = day not in observed_set
        if missing:
            total_missing += 1
            if current_start is None:
                current_start = day
                current_end = day
                current_count = 1
            else:
                current_end = day
                current_count += 1
        elif current_start is not None:
            ranges.append(
                {
                    "start": current_start.isoformat(),
                    "end": current_end.isoformat() if current_end else current_start.isoformat(),
                    "count": current_count,
                }
            )
            current_start = None
            current_end = None
            current_count = 0
    if current_start is not None:
        ranges.append(
            {
                "start": current_start.isoformat(),
                "end": current_end.isoformat() if current_end else current_start.isoformat(),
                "count": current_count,
            }
        )
    return ranges, total_missing


def _max_column_nan_ratio(feature_report: Mapping[str, Any]) -> float | None:
    summary = feature_report.get("summary_by_column")
    if not isinstance(summary, Mapping):
        return None
    ratios: List[float] = []
    for value in summary.values():
        if isinstance(value, Mapping):
            ratio = value.get("nan_ratio")
            if isinstance(ratio, (int, float)):
                ratios.append(float(ratio))
    if not ratios:
        return None
    return max(ratios)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _format_date(value: date | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _feature_set_uses_fundamentals(feature_set: str) -> bool:
    name = str(feature_set or "").strip().lower()
    return "fundamentals" in name


def _parse_fundamentals_partition(path: Path) -> date | None:
    try:
        day = int(path.stem)
        month = int(path.parent.name)
        year = int(path.parent.parent.name)
        return date(year, month, day)
    except Exception:
        return None


def _prepare_fundamentals_for_health(frame: "pd.DataFrame") -> "pd.DataFrame":
    report = frame.get("report_date")
    if report is None:
        report = frame.get("report_period")
    report = pd.to_datetime(report, utc=True, errors="coerce")
    report = report.dt.normalize()
    filing = pd.to_datetime(frame.get("filing_date"), utc=True, errors="coerce")
    filing = filing.dt.normalize()
    data = frame.copy()
    data["report_date"] = report
    data["filing_date"] = filing
    data = data.dropna(subset=["report_date"])
    period = data.get("period")
    fiscal = data.get("fiscal_period")
    if period is None:
        period = [""] * len(data)
    if fiscal is None:
        fiscal = [""] * len(data)
    normalized = [_normalize_period(p, f) for p, f in zip(period, fiscal)]
    data["period"] = normalized
    lag_days = pd.Series(normalized).map({"annual": 90, "quarterly": 45}).fillna(45)
    fallback = data["report_date"] + pd.to_timedelta(lag_days, unit="D")
    data["as_of_date"] = data["filing_date"].fillna(fallback)
    return data


def _normalize_period(period: object, fiscal_period: object) -> str:
    text = str(period or "").strip().lower()
    if text in {"quarter", "quarterly", "q", "qtr"}:
        return "quarterly"
    if text in {"annual", "fy", "year", "yearly"}:
        return "annual"
    fiscal = str(fiscal_period or "").strip().upper()
    if fiscal.startswith("Q"):
        return "quarterly"
    if fiscal.startswith("FY") or fiscal.startswith("A"):
        return "annual"
    return text


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


def _relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _sorted_dates(values: Iterable[date]) -> List[date]:
    return sorted(values)


def _ensure_pandas_available() -> None:
    if pd is None:
        raise RuntimeError("pandas is required for data health reporting") from _PANDAS_ERROR


__all__ = [
    "CanonicalLoadResult",
    "FundamentalsLoadResult",
    "build_feature_frames",
    "compute_canonical_health",
    "compute_fundamentals_health",
    "compute_feature_health",
    "evaluate_thresholds",
    "load_canonical_equity_pandas",
    "load_canonical_fundamentals_pandas",
    "run_data_health_preflight",
    "resolve_run_id",
]
