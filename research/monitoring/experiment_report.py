"""User-facing experiment monitoring helpers."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import pandas as pd

from infra.paths import get_data_root


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
                "candidate": cand,
                "baseline": base,
                "delta": delta,
                "delta_pct": delta_pct,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ("candidate", "baseline", "delta", "delta_pct"):
        df[col] = df[col].map(_format_value)
    return df


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


def format_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No metrics available."
    with pd.option_context("display.max_rows", 200, "display.max_colwidth", 80):
        return df.to_string(index=False)


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def render_account_value_plot(
    series: Mapping[str, Any],
    *,
    label: str,
    ax,
) -> None:
    timestamps = resolve_timestamps(series)
    values = series.get("account_value")
    if not isinstance(values, Sequence) or not timestamps:
        return
    ax.plot(timestamps, values, label=label, linewidth=2.0)
    ax.set_title("Account Value")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.2)


def render_turnover_plot(
    series: Mapping[str, Any],
    *,
    label: str,
    ax,
) -> None:
    turnover = series.get("turnover_1d")
    if not turnover:
        turnover = derive_turnover_series(series.get("weights"))
    if not turnover:
        return
    x = list(range(len(turnover)))
    ax.plot(x, turnover, label=label, linewidth=1.5)
    ax.set_title("Turnover (1d)")
    ax.set_ylabel("Turnover")
    ax.grid(True, alpha=0.2)


def render_total_return_bar(candidate: float | None, baseline: float | None, ax) -> None:
    labels = ["candidate", "baseline"]
    values = [candidate, baseline]
    ax.bar(labels, [value or 0.0 for value in values], color=["#1f77b4", "#ff7f0e"])
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
    baseline_metrics = {}
    baseline_series = {}
    if artifacts.qualification and artifacts.qualification.get("baseline_experiment_id"):
        baseline_id = str(artifacts.qualification.get("baseline_experiment_id"))
        baseline_artifacts = load_baseline_artifacts(baseline_id, data_root=data_root, strict=strict)
        if baseline_artifacts:
            baseline_metrics = extract_metrics(baseline_artifacts.metrics)
            baseline_series = extract_series(baseline_artifacts.metrics, baseline_artifacts.rollout)
            comparison_table = build_comparison_table(metrics, baseline_metrics)

    inline_mode = inline if inline is not None else is_notebook()
    output_dir = Path(output_dir) if output_dir else get_data_root() / "monitoring" / "plots" / experiment_id
    output_dir = ensure_output_dir(output_dir)

    metrics_table = build_metrics_table(metrics)

    figures: Dict[str, Path] = {}
    try:
        import matplotlib.pyplot as plt  # type: ignore
        try:
            plt.style.use("seaborn-v0_8")
        except Exception:
            pass

        if metrics_table.empty:
            pass
        fig, ax = plt.subplots(figsize=(10, 4))
        render_account_value_plot(series, label="candidate", ax=ax)
        if baseline_series:
            render_account_value_plot(baseline_series, label="baseline", ax=ax)
        ax.legend(loc="best")
        if inline_mode:
            plt.show()
        else:
            path = output_dir / "account_value.png"
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            figures["account_value"] = path
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 3.5))
        render_turnover_plot(series, label="candidate", ax=ax)
        if baseline_series:
            render_turnover_plot(baseline_series, label="baseline", ax=ax)
        ax.legend(loc="best")
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

    return {
        "metadata": summarize_metadata(metadata),
        "metrics_table": metrics_table,
        "comparison_table": comparison_table,
        "figures": figures,
        "baseline_experiment_id": baseline_artifacts.experiment_id if baseline_artifacts else None,
        "output_dir": output_dir,
    }
