"""User-facing experiment monitoring helpers."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import pandas as pd

from infra.paths import get_data_root
from research.validation.data_health import run_data_health_preflight


@dataclass(frozen=True)
class ExperimentArtifacts:
    experiment_id: str
    experiment_dir: Path
    metrics_path: Path
    rollout_path: Optional[Path]
    qualification_path: Optional[Path]
    metrics: Mapping[str, Any]
    rollout: Mapping[str, Any] | None
    qualification: Mapping[str, Any] | None


@dataclass(frozen=True)
class BaselineArtifacts:
    experiment_id: str
    metrics_path: Path
    rollout_path: Optional[Path]
    metrics: Mapping[str, Any]
    rollout: Mapping[str, Any] | None


def resolve_experiment_dir(experiment_id: str, *, data_root: Optional[Path] = None) -> Path:
    base = Path(data_root) if data_root else get_data_root()
    return base / "experiments" / experiment_id


def load_json(path: Path, *, strict: bool) -> Mapping[str, Any] | None:
    if not path.exists():
        if strict:
            raise FileNotFoundError(str(path))
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_experiment_artifacts(
    experiment_id: str,
    *,
    data_root: Optional[Path] = None,
    strict: bool = False,
) -> ExperimentArtifacts:
    experiment_dir = resolve_experiment_dir(experiment_id, data_root=data_root)
    metrics_path = experiment_dir / "evaluation" / "metrics.json"
    metrics = load_json(metrics_path, strict=True)
    rollout_path = experiment_dir / "runs" / "rollout.json"
    rollout = load_json(rollout_path, strict=False)
    qualification_path = experiment_dir / "promotion" / "qualification_report.json"
    qualification = load_json(qualification_path, strict=False)
    if metrics is None:
        raise FileNotFoundError(str(metrics_path))
    return ExperimentArtifacts(
        experiment_id=experiment_id,
        experiment_dir=experiment_dir,
        metrics_path=metrics_path,
        rollout_path=rollout_path if rollout_path.exists() else None,
        qualification_path=qualification_path if qualification_path.exists() else None,
        metrics=metrics,
        rollout=rollout,
        qualification=qualification,
    )


def load_baseline_artifacts(
    baseline_id: str,
    *,
    data_root: Optional[Path] = None,
    strict: bool = False,
) -> BaselineArtifacts | None:
    experiment_dir = resolve_experiment_dir(baseline_id, data_root=data_root)
    metrics_path = experiment_dir / "evaluation" / "metrics.json"
    metrics = load_json(metrics_path, strict=strict)
    if metrics is None:
        return None
    rollout_path = experiment_dir / "runs" / "rollout.json"
    rollout = load_json(rollout_path, strict=False)
    return BaselineArtifacts(
        experiment_id=baseline_id,
        metrics_path=metrics_path,
        rollout_path=rollout_path if rollout_path.exists() else None,
        metrics=metrics,
        rollout=rollout,
    )


def _is_sma_metadata(metadata: Mapping[str, Any]) -> bool:
    feature_set = str(metadata.get("feature_set") or "").strip().lower()
    if feature_set != "sma_v1":
        return False
    policy_id = metadata.get("policy_id")
    if isinstance(policy_id, str) and "sma" in policy_id.lower():
        return True
    policy = metadata.get("policy")
    if isinstance(policy, Mapping) and policy.get("type") == "sma_weight":
        return True
    if isinstance(policy, str) and "sma" in policy.lower():
        return True
    return False


def resolve_default_baseline(
    *,
    data_root: Optional[Path] = None,
    exclude_id: str | None = None,
) -> str | None:
    base = Path(data_root) if data_root else get_data_root()
    experiments_dir = base / "experiments"
    if not experiments_dir.exists():
        return None
    candidates = sorted(
        [path for path in experiments_dir.iterdir() if path.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for entry in candidates:
        if exclude_id and entry.name == exclude_id:
            continue
        metrics_path = entry / "evaluation" / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        metadata = payload.get("metadata")
        if isinstance(metadata, Mapping) and _is_sma_metadata(metadata):
            return entry.name
    return None


def extract_metrics(payload: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = payload.get("performance")
    if isinstance(metrics, Mapping):
        return dict(metrics)
    metrics = payload.get("metrics")
    if isinstance(metrics, Mapping):
        return dict(metrics)
    return {}


def extract_metadata(payload: Mapping[str, Any]) -> Dict[str, Any]:
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        return dict(metadata)
    return {}


def extract_series(payload: Mapping[str, Any], fallback: Mapping[str, Any] | None) -> Dict[str, Any]:
    series = payload.get("series")
    if isinstance(series, Mapping):
        return dict(series)
    if fallback:
        series = fallback.get("series")
        if isinstance(series, Mapping):
            return dict(series)
    return {}


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.6f}"
    return str(value)


def build_metrics_table(metrics: Mapping[str, Any]) -> pd.DataFrame:
    rows = [{"metric": key, "value": value} for key, value in sorted(metrics.items())]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["value"] = df["value"].map(_format_value)
    return df


def build_comparison_table(
    candidate_metrics: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
) -> pd.DataFrame:
    keys = sorted(set(candidate_metrics) & set(baseline_metrics))
    rows = []
    for key in keys:
        cand = candidate_metrics.get(key)
        base = baseline_metrics.get(key)
        delta = None
        delta_pct = None
        if isinstance(cand, (int, float)) and isinstance(base, (int, float)):
            delta = cand - base
            delta_pct = (delta / base) * 100 if base else None
        rows.append(
            {
                "metric": key,
                "candidate": _format_value(cand),
                "baseline": _format_value(base),
                "delta": delta,
                "delta_pct": delta_pct,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ("delta", "delta_pct"):
        df[col] = df[col].map(_format_value)
    return df


def build_winner_table(
    candidate_metrics: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
) -> pd.DataFrame:
    keys = sorted(set(candidate_metrics) & set(baseline_metrics))
    rows = []
    for key in keys:
        cand = candidate_metrics.get(key)
        base = baseline_metrics.get(key)
        cand_win = ""
        base_win = ""
        if isinstance(cand, (int, float)) and isinstance(base, (int, float)) and cand != base:
            lower_better = _metric_prefers_lower(key)
            base_den = abs(base)
            if base_den > 0:
                if lower_better:
                    cand_improve = (base - cand) / base_den
                    base_improve = (cand - base) / base_den
                else:
                    cand_improve = (cand - base) / base_den
                    base_improve = (base - cand) / base_den
                cand_win = _tick_marks(cand_improve)
                base_win = _tick_marks(base_improve)
        rows.append(
            {
                "metric": key,
                "candidate": cand_win,
                "baseline": base_win,
            }
        )
    return pd.DataFrame(rows)


def _metric_prefers_lower(name: str) -> bool:
    lowered = str(name or "").lower()
    if lowered in {
        "max_drawdown",
        "volatility_ann",
        "turnover_1d_mean",
        "turnover",
        "tx_cost_bps",
        "tx_cost_total",
    }:
        return True
    for token in ("drawdown", "volatility", "turnover", "cost", "fee", "slippage"):
        if token in lowered:
            return True
    return False


def _tick_marks(improvement: float | None) -> str:
    if improvement is None or improvement <= 0:
        return ""
    thresholds = (0.2, 0.4, 0.6, 0.8, 1.0)
    ticks = sum(1 for threshold in thresholds if improvement >= threshold)
    return "✓" * max(1, ticks)


def resolve_total_return(metrics: Mapping[str, Any]) -> float | None:
    value = metrics.get("total_return")
    return float(value) if isinstance(value, (int, float)) else None


def derive_turnover_series(weights: Any) -> Sequence[float] | None:
    if not weights:
        return None
    if isinstance(weights, Mapping):
        series = [list(series) for series in weights.values()]
        if not series:
            return None
        length = min(len(values) for values in series)
        if length < 2:
            return None
        turnover = []
        for idx in range(1, length):
            delta = 0.0
            for values in series:
                delta += abs(values[idx] - values[idx - 1])
            turnover.append(0.5 * delta)
        return turnover
    if isinstance(weights, Sequence):
        if len(weights) < 2:
            return None
        return [abs(weights[idx] - weights[idx - 1]) for idx in range(1, len(weights))]
    return None


def resolve_timestamps(series: Mapping[str, Any]) -> Sequence[Any]:
    timestamps = series.get("timestamps")
    if isinstance(timestamps, Sequence) and not isinstance(timestamps, (str, bytes)):
        return timestamps
    account_values = series.get("account_value")
    if isinstance(account_values, Sequence) and not isinstance(account_values, (str, bytes)):
        return list(range(len(account_values)))
    return []


def _resolve_datetime_index(timestamps: Sequence[Any]) -> Sequence[Any]:
    if timestamps is None or len(timestamps) == 0:
        return []
    parsed = pd.to_datetime(list(timestamps), errors="coerce", utc=True)
    if parsed.notna().any():
        return parsed
    return list(range(len(timestamps)))


def _plot_series_with_stats(
    ax,
    x,
    y,
    *,
    label: str,
    color: str | None = None,
    rolling_window: int = 20,
    label_stats: bool = True,
    include_stats: bool = True,
) -> None:
    if x is None or y is None:
        return
    if len(x) == 0 or len(y) == 0:
        return
    values = [float(entry) for entry in y]
    ax.plot(x, values, label=label, linewidth=2.0, color=color)
    if not values:
        return
    max_value = max(values)
    min_value = min(values)
    if include_stats:
        rolling = pd.Series(values).rolling(window=rolling_window, min_periods=1).mean().tolist()
        mean_label = f"{rolling_window}d mean" if label_stats else "_nolegend_"
        max_label = "max" if label_stats else "_nolegend_"
        min_label = "min" if label_stats else "_nolegend_"
        ax.plot(
            x,
            rolling,
            label=mean_label,
            linestyle="--",
            linewidth=1.4,
            alpha=0.75,
            color="#6e6e6e",
        )
        ax.axhline(max_value, linestyle=":", linewidth=1.0, alpha=0.5, color="#9a9a9a", label=max_label)
        ax.axhline(min_value, linestyle=":", linewidth=1.0, alpha=0.5, color="#9a9a9a", label=min_label)
        max_idx = values.index(max_value)
        min_idx = values.index(min_value)
        ax.scatter([x[max_idx]], [max_value], s=26, color="#6e6e6e", zorder=3)
        ax.scatter([x[min_idx]], [min_value], s=26, color="#6e6e6e", zorder=3)
        ax.text(x[max_idx], max_value, f" max {max_value:.4f}", fontsize=8, ha="left", va="bottom")
        ax.text(x[min_idx], min_value, f" min {min_value:.4f}", fontsize=8, ha="left", va="top")


def is_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore
    except Exception:
        return False
    shell = get_ipython()
    if shell is None:
        return False
    return "IPKernelApp" in sys.modules


def summarize_metadata(metadata: Mapping[str, Any]) -> Dict[str, str]:
    summary = {
        "feature_set": str(metadata.get("feature_set") or "n/a"),
        "policy": str(metadata.get("policy_id") or metadata.get("policy") or "n/a"),
        "start_date": str(metadata.get("start_date") or "n/a"),
        "end_date": str(metadata.get("end_date") or "n/a"),
    }
    data_split = metadata.get("data_split") or {}
    if isinstance(data_split, Mapping):
        summary["test_window"] = f"{data_split.get('test_start','?')} → {data_split.get('test_end','?')}"
    return summary


def _ascii_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "No metrics available."
    rendered_rows = []
    for row in rows:
        rendered_rows.append({col: _format_value(row.get(col)) for col in columns})
    widths = {col: len(col) for col in columns}
    for row in rendered_rows:
        for col in columns:
            widths[col] = max(widths[col], len(str(row.get(col, ""))))
    sep = "+-" + "-+-".join("-" * widths[col] for col in columns) + "-+"
    header = "| " + " | ".join(col.ljust(widths[col]) for col in columns) + " |"
    lines = [sep, header, sep]
    for row in rendered_rows:
        line = "| " + " | ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns) + " |"
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)


def format_table(table: pd.DataFrame | Sequence[Mapping[str, Any]] | None) -> str:
    if table is None:
        return "No metrics available."
    if isinstance(table, pd.DataFrame):
        if table.empty:
            return "No metrics available."
        rows = table.to_dict("records")
        columns = list(table.columns)
        return _ascii_table(rows, columns)
    if isinstance(table, Sequence) and table:
        first = table[0]
        if isinstance(first, Mapping):
            columns = list(first.keys())
            return _ascii_table(table, columns)
    return "No metrics available."


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def render_account_value_plot(
    series: Mapping[str, Any],
    *,
    label: str,
    ax,
    color: str | None = None,
    label_stats: bool = True,
    include_stats: bool = True,
) -> None:
    timestamps = _resolve_datetime_index(resolve_timestamps(series))
    values = series.get("account_value")
    if not isinstance(values, Sequence):
        return
    if timestamps is None or len(timestamps) == 0:
        return
    y_values = list(values)
    if len(y_values) == 0:
        return
    window = max(5, min(20, len(y_values) // 4 or 5))
    _plot_series_with_stats(
        ax,
        list(timestamps),
        y_values,
        label=label,
        color=color,
        rolling_window=window,
        label_stats=label_stats,
        include_stats=include_stats,
    )
    ax.set_title("Account Value")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.2)


def render_turnover_plot(
    series: Mapping[str, Any],
    *,
    label: str,
    ax,
    color: str | None = None,
    label_stats: bool = True,
    include_stats: bool = True,
) -> None:
    turnover = series.get("turnover_1d")
    if not turnover:
        turnover = derive_turnover_series(series.get("weights"))
    if turnover is None:
        return
    y_values = list(turnover)
    if len(y_values) == 0:
        return
    timestamps = _resolve_datetime_index(resolve_timestamps(series))
    if timestamps is not None and len(timestamps) > 0:
        x = list(timestamps)[1 : len(y_values) + 1]
    else:
        x = list(range(len(y_values)))
    window = max(5, min(20, len(y_values) // 4 or 5))
    _plot_series_with_stats(
        ax,
        x,
        y_values,
        label=label,
        color=color,
        rolling_window=window,
        label_stats=label_stats,
        include_stats=include_stats,
    )
    ax.set_title("Turnover (1d)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Turnover")
    ax.grid(True, alpha=0.2)


def render_total_return_bar(
    candidate: float | None,
    baseline: float | None,
    ax,
    *,
    candidate_label: str,
    baseline_label: str,
    candidate_color: str,
    baseline_color: str,
) -> None:
    labels = [candidate_label, baseline_label]
    values = [candidate, baseline]
    ax.bar(labels, [value or 0.0 for value in values], color=[candidate_color, baseline_color])
    ax.set_title("Total Return")
    ax.set_ylabel("Total Return")
    ax.grid(True, axis="y", alpha=0.2)
    for idx, value in enumerate(values):
        ax.text(idx, (value or 0.0), _format_value(value), ha="center", va="bottom", fontsize=9)


def generate_experiment_report(
    experiment_id: str,
    *,
    data_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    strict: bool = False,
    inline: Optional[bool] = None,
) -> Dict[str, Any]:
    artifacts = load_experiment_artifacts(experiment_id, data_root=data_root, strict=strict)
    metadata = extract_metadata(artifacts.metrics)
    metrics = extract_metrics(artifacts.metrics)
    series = extract_series(artifacts.metrics, artifacts.rollout)

    baseline_artifacts = None
    comparison_table = pd.DataFrame()
    winner_table = pd.DataFrame()
    baseline_metrics = {}
    baseline_series = {}
    candidate_label = f"candidate ({experiment_id[:6]})"
    baseline_label = "baseline"
    baseline_id = None
    if artifacts.qualification and artifacts.qualification.get("baseline_experiment_id"):
        baseline_id = str(artifacts.qualification.get("baseline_experiment_id"))
    if not baseline_id:
        baseline_id = resolve_default_baseline(data_root=data_root, exclude_id=experiment_id)
    if baseline_id:
        baseline_artifacts = load_baseline_artifacts(baseline_id, data_root=data_root, strict=strict)
        if baseline_artifacts:
            baseline_label = f"baseline ({baseline_artifacts.experiment_id[:6]})"
            baseline_metrics = extract_metrics(baseline_artifacts.metrics)
            baseline_series = extract_series(baseline_artifacts.metrics, baseline_artifacts.rollout)
            comparison_table = build_comparison_table(metrics, baseline_metrics)
            winner_table = build_winner_table(metrics, baseline_metrics)

    inline_mode = inline if inline is not None else is_notebook()
    output_dir = Path(output_dir) if output_dir else get_data_root() / "monitoring" / "plots" / experiment_id
    output_dir = ensure_output_dir(output_dir)

    metrics_table = build_metrics_table(metrics)

    figures: Dict[str, Path] = {}
    try:
        import matplotlib.pyplot as plt  # type: ignore
        try:
            import seaborn as sns  # type: ignore

            sns.set_theme(style="whitegrid", palette="Set2")
        except Exception:
            try:
                plt.style.use("seaborn-v0_8")
            except Exception:
                pass

        if metrics_table.empty:
            pass
        fig, ax = plt.subplots(figsize=(10, 4))
        render_account_value_plot(
            series,
            label=candidate_label,
            ax=ax,
            color="#1b9e77",
            label_stats=True,
            include_stats=True,
        )
        if baseline_series:
            render_account_value_plot(
                baseline_series,
                label=baseline_label,
                ax=ax,
                color="#d95f02",
                label_stats=False,
                include_stats=False,
            )
        ax.legend(loc="best")
        try:
            import matplotlib.dates as mdates  # type: ignore

            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        except Exception:
            pass
        if inline_mode:
            plt.show()
        else:
            path = output_dir / "account_value.png"
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            figures["account_value"] = path
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 3.5))
        render_turnover_plot(
            series,
            label=candidate_label,
            ax=ax,
            color="#1b9e77",
            label_stats=True,
            include_stats=True,
        )
        if baseline_series:
            render_turnover_plot(
                baseline_series,
                label=baseline_label,
                ax=ax,
                color="#d95f02",
                label_stats=False,
                include_stats=False,
            )
        ax.legend(loc="best")
        try:
            import matplotlib.dates as mdates  # type: ignore

            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        except Exception:
            pass
        if inline_mode:
            plt.show()
        else:
            path = output_dir / "turnover.png"
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            figures["turnover"] = path
        plt.close(fig)

        if baseline_artifacts:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            render_total_return_bar(
                resolve_total_return(metrics),
                resolve_total_return(baseline_metrics),
                ax,
                candidate_label=candidate_label,
                baseline_label=baseline_label,
                candidate_color="#1b9e77",
                baseline_color="#d95f02",
            )
            if inline_mode:
                plt.show()
            else:
                path = output_dir / "total_return_comparison.png"
                fig.tight_layout()
                fig.savefig(path, dpi=150)
                figures["total_return"] = path
            plt.close(fig)
    except Exception as exc:  # pragma: no cover - plotting is best effort
        if strict:
            raise
        print(f"Plotting skipped: {exc}")

    data_health_payload = _data_health_payload(metadata, data_root=data_root)
    data_health_table = _data_health_table(data_health_payload)

    return {
        "metadata": summarize_metadata(metadata),
        "metrics_table": metrics_table,
        "comparison_table": comparison_table,
        "winner_table": winner_table,
        "data_health": data_health_payload,
        "data_health_table": data_health_table,
        "figures": figures,
        "baseline_experiment_id": baseline_artifacts.experiment_id if baseline_artifacts else None,
        "output_dir": output_dir,
    }


def _data_health_payload(metadata: Mapping[str, Any], *, data_root: Optional[Path]) -> Dict[str, Any] | None:
    symbols = metadata.get("symbols")
    if not isinstance(symbols, list):
        return None
    start = _parse_iso_date(metadata.get("start_date"))
    end = _parse_iso_date(metadata.get("end_date"))
    if not start or not end:
        return None
    feature_set = metadata.get("feature_set")
    try:
        return run_data_health_preflight(
            symbols=symbols,
            start_date=start,
            end_date=end,
            feature_set=str(feature_set) if feature_set else None,
            data_root=data_root,
            max_missing_ratio=None,
            max_nan_ratio=None,
            strict=False,
        )
    except Exception as exc:
        return {"error": str(exc)}


def _data_health_table(payload: Mapping[str, Any] | None) -> Sequence[Mapping[str, Any]] | None:
    if not payload:
        return None
    if "error" in payload:
        return [{"check": "data_health_error", "value": payload.get("error")}]
    rows: List[Dict[str, Any]] = []
    canonical = payload.get("canonical") or {}
    missing_ratio = canonical.get("overall", {}).get("missing_ratio")
    rows.append({"check": "equity_missing_ratio", "value": _format_value(missing_ratio)})

    features = payload.get("features") or {}
    if isinstance(features, Mapping) and features:
        nan_ratio = features.get("overall", {}).get("nan_ratio")
        rows.append({"check": "feature_nan_ratio", "value": _format_value(nan_ratio)})
        max_column = _max_column_nan_ratio(features)
        rows.append({"check": "feature_max_column_nan", "value": _format_value(max_column)})

    fundamentals = payload.get("fundamentals") or {}
    if isinstance(fundamentals, Mapping) and fundamentals:
        overall = fundamentals.get("overall", {})
        rows.append({"check": "fund_stale_ratio", "value": _format_value(overall.get("stale_ratio"))})
        rows.append({"check": "fund_missing_symbols", "value": overall.get("missing_symbols")})
    return rows


def _parse_iso_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _max_column_nan_ratio(feature_report: Mapping[str, Any]) -> float | None:
    summary = feature_report.get("summary_by_column")
    if not isinstance(summary, Mapping):
        return None
    ratios = []
    for entry in summary.values():
        if isinstance(entry, Mapping):
            ratio = entry.get("nan_ratio")
            if isinstance(ratio, (int, float)):
                ratios.append(float(ratio))
    if not ratios:
        return None
    return max(ratios)
