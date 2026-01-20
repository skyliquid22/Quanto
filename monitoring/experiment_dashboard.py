"""Streamlit dashboard for browsing experiment artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

try:
    import streamlit as st
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit("Streamlit is required. Install with: pip install streamlit") from exc

from infra.paths import get_data_root


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
    recorded_at: str | None
    sharpe: float | None
    total_return: float | None
    max_drawdown: float | None
    turnover_1d_mean: float | None
    tx_cost_bps: float | None
    qualification_passed: bool | None

    @property
    def as_row(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "feature_set": self.feature_set,
            "regime_feature_set": self.regime_feature_set,
            "policy": self.policy,
            "reward_version": self.reward_version,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "recorded_at": self.recorded_at,
            "sharpe": self.sharpe,
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "turnover_1d_mean": self.turnover_1d_mean,
            "tx_cost_bps": self.tx_cost_bps,
            "qualification_passed": self.qualification_passed,
        }


@st.cache_data(show_spinner=False)
def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
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
    return None


def _qualification_passed(exp_dir: Path) -> bool | None:
    payload = read_json(exp_dir / "promotion" / "qualification_report.json")
    if not payload:
        return None
    return bool(payload.get("passed")) if "passed" in payload else None


def _extract_summary(exp_dir: Path) -> ExperimentSummary | None:
    spec_path = exp_dir / "spec" / "experiment_spec.json"
    metrics_path = exp_dir / "evaluation" / "metrics.json"
    if not spec_path.exists() or not metrics_path.exists():
        return None
    spec_payload = read_json(spec_path) or {}
    metrics_payload = read_json(metrics_path) or {}
    run_summary = read_json(exp_dir / "logs" / "run_summary.json")
    return ExperimentSummary(
        experiment_id=exp_dir.name,
        experiment_name=spec_payload.get("experiment_name"),
        feature_set=spec_payload.get("feature_set"),
        regime_feature_set=spec_payload.get("regime_feature_set") or None,
        policy=spec_payload.get("policy"),
        reward_version=_extract_reward_version(spec_payload),
        start_date=spec_payload.get("start_date"),
        end_date=spec_payload.get("end_date"),
        recorded_at=_parse_recorded_at(run_summary),
        sharpe=_coerce_float(_get_nested(metrics_payload, "performance.sharpe")),
        total_return=_coerce_float(_get_nested(metrics_payload, "performance.total_return")),
        max_drawdown=_coerce_float(_get_nested(metrics_payload, "performance.max_drawdown")),
        turnover_1d_mean=_coerce_float(_get_nested(metrics_payload, "trading.turnover_1d_mean")),
        tx_cost_bps=_coerce_float(_get_nested(metrics_payload, "trading.tx_cost_bps")),
        qualification_passed=_qualification_passed(exp_dir),
    )


@st.cache_data(show_spinner=False)
def load_experiment_summaries(registry_root: Path) -> list[ExperimentSummary]:
    summaries: list[ExperimentSummary] = []
    if not registry_root.exists():
        return summaries
    for exp_dir in sorted(registry_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        summary = _extract_summary(exp_dir)
        if summary is not None:
            summaries.append(summary)
    summaries.sort(key=lambda entry: (entry.recorded_at or "", entry.experiment_id))
    return summaries


def _latest_shadow_metrics(data_root: Path, experiment_id: str) -> tuple[Path | None, dict[str, Any] | None]:
    shadow_root = data_root / "shadow" / experiment_id
    if not shadow_root.exists():
        return None, None
    run_dirs = [entry for entry in shadow_root.iterdir() if entry.is_dir()]
    run_dirs.sort(key=lambda entry: entry.stat().st_mtime, reverse=True)
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics_sim.json"
        if metrics_path.exists():
            return metrics_path, read_json(metrics_path)
    return None, None


def _resolve_regime_payload(metrics_payload: Mapping[str, Any] | None, slices_payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    for payload in (slices_payload, metrics_payload):
        if not isinstance(payload, Mapping):
            continue
        section = payload.get("performance_by_regime")
        if isinstance(section, Mapping) and section:
            return dict(section)
    return None


def _extract_regime_table(regime_payload: Mapping[str, Any] | None) -> pd.DataFrame | None:
    if not regime_payload:
        return None
    rows: list[dict[str, Any]] = []
    for bucket, metrics in regime_payload.items():
        if not isinstance(metrics, Mapping):
            continue
        row = {"regime": bucket}
        for key, value in metrics.items():
            row[key] = _coerce_float(value)
        rows.append(row)
    if not rows:
        return None
    frame = pd.DataFrame(rows).set_index("regime")
    return frame.sort_index()


def _compare_metrics(
    candidate_payload: Mapping[str, Any] | None,
    baseline_payload: Mapping[str, Any] | None,
    metric_paths: Iterable[tuple[str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, path in metric_paths:
        candidate_value = _coerce_float(_get_nested(candidate_payload, path))
        baseline_value = _coerce_float(_get_nested(baseline_payload, path))
        delta = None
        delta_pct = None
        if candidate_value is not None and baseline_value is not None:
            delta = candidate_value - baseline_value
            if baseline_value:
                delta_pct = (delta / baseline_value) * 100.0
        rows.append(
            {
                "metric": label,
                "candidate": candidate_value,
                "baseline": baseline_value,
                "delta": delta,
                "delta_pct": delta_pct,
            }
        )
    return pd.DataFrame(rows)


def _format_symbols(spec_payload: Mapping[str, Any] | None) -> str:
    if not spec_payload:
        return ""
    symbols = spec_payload.get("symbols")
    if not isinstance(symbols, list):
        return ""
    return ", ".join(str(symbol) for symbol in symbols[:10]) + (" ..." if len(symbols) > 10 else "")


def main() -> None:
    st.set_page_config(page_title="Quanto Experiment Dashboard", layout="wide")
    st.title("Quanto Experiment Dashboard")

    default_root = get_data_root()
    data_root_input = st.sidebar.text_input("Data root", str(default_root))
    data_root = Path(data_root_input).expanduser()
    registry_root = Path(st.sidebar.text_input("Experiments root", str(data_root / "experiments"))).expanduser()

    if st.sidebar.button("Refresh data"):
        st.cache_data.clear()

    summaries = load_experiment_summaries(registry_root)
    if not summaries:
        st.warning(f"No experiments found under {registry_root}.")
        return

    df = pd.DataFrame([entry.as_row for entry in summaries])

    name_options = sorted({value for value in df["experiment_name"].dropna().unique()})
    feature_options = sorted({value for value in df["feature_set"].dropna().unique()})
    policy_options = sorted({value for value in df["policy"].dropna().unique()})
    reward_options = sorted({value for value in df["reward_version"].dropna().unique()})

    selected_names = st.sidebar.multiselect("Experiment name", name_options)
    selected_features = st.sidebar.multiselect("Feature set", feature_options)
    selected_policies = st.sidebar.multiselect("Policy", policy_options)
    selected_rewards = st.sidebar.multiselect("Reward version", reward_options)
    only_qualified = st.sidebar.checkbox("Only qualified (passed)", value=False)

    filtered = df.copy()
    if selected_names:
        filtered = filtered[filtered["experiment_name"].isin(selected_names)]
    if selected_features:
        filtered = filtered[filtered["feature_set"].isin(selected_features)]
    if selected_policies:
        filtered = filtered[filtered["policy"].isin(selected_policies)]
    if selected_rewards:
        filtered = filtered[filtered["reward_version"].isin(selected_rewards)]
    if only_qualified:
        filtered = filtered[filtered["qualification_passed"] == True]  # noqa: E712

    st.subheader("Experiments")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    experiment_ids = filtered["experiment_id"].tolist()
    selected_id = st.selectbox("Select experiment", experiment_ids, index=0)
    exp_dir = registry_root / selected_id
    spec_payload = read_json(exp_dir / "spec" / "experiment_spec.json") or {}
    metrics_payload = read_json(exp_dir / "evaluation" / "metrics.json") or {}
    regime_slices_payload = read_json(exp_dir / "evaluation" / "regime_slices.json")
    qualification_payload = read_json(exp_dir / "promotion" / "qualification_report.json")
    shadow_path, shadow_payload = _latest_shadow_metrics(data_root, selected_id)

    st.subheader("Selected Experiment")
    summary_cols = st.columns(3)
    summary_cols[0].metric("Experiment ID", selected_id)
    summary_cols[1].metric("Feature set", spec_payload.get("feature_set") or "unknown")
    summary_cols[2].metric("Policy", spec_payload.get("policy") or "unknown")

    detail_cols = st.columns(3)
    detail_cols[0].metric("Sharpe", _coerce_float(_get_nested(metrics_payload, "performance.sharpe")))
    detail_cols[1].metric("Total Return", _coerce_float(_get_nested(metrics_payload, "performance.total_return")))
    detail_cols[2].metric("Max Drawdown", _coerce_float(_get_nested(metrics_payload, "performance.max_drawdown")))

    detail_cols = st.columns(3)
    detail_cols[0].metric("Turnover 1D Mean", _coerce_float(_get_nested(metrics_payload, "trading.turnover_1d_mean")))
    detail_cols[1].metric("Avg Exposure", _coerce_float(_get_nested(metrics_payload, "trading.avg_exposure")))
    detail_cols[2].metric("Tx Cost (bps)", _coerce_float(_get_nested(metrics_payload, "trading.tx_cost_bps")))

    with st.expander("Spec and Metadata", expanded=False):
        st.write(f"Symbols: {_format_symbols(spec_payload)}")
        st.json(spec_payload)

    tabs = st.tabs(["Regime Slices", "Qualification", "Shadow Metrics", "Compare"])

    with tabs[0]:
        regime_payload = _resolve_regime_payload(metrics_payload, regime_slices_payload)
        regime_table = _extract_regime_table(regime_payload)
        if regime_table is None:
            st.info("No regime slices found for this experiment.")
        else:
            st.dataframe(regime_table, use_container_width=True)
            plot_metrics = [col for col in ("sharpe", "total_return", "max_drawdown") if col in regime_table.columns]
            if plot_metrics:
                st.bar_chart(regime_table[plot_metrics])

    with tabs[1]:
        if not qualification_payload:
            st.info("No qualification report found.")
        else:
            st.metric("Qualified", bool(qualification_payload.get("passed")))
            st.write("Failed hard gates:", qualification_payload.get("failed_hard", []))
            st.write("Failed soft gates:", qualification_payload.get("failed_soft", []))
            st.json(qualification_payload.get("gate_summary", {}))

    with tabs[2]:
        if not shadow_payload:
            st.info("No shadow metrics found.")
        else:
            st.write(f"Latest shadow metrics: {shadow_path}")
            shadow_cols = st.columns(3)
            shadow_cols[0].metric("Sharpe", _coerce_float(_get_nested(shadow_payload, "performance.sharpe")))
            shadow_cols[1].metric("Total Return", _coerce_float(_get_nested(shadow_payload, "performance.total_return")))
            shadow_cols[2].metric("Max Drawdown", _coerce_float(_get_nested(shadow_payload, "performance.max_drawdown")))
            st.json(shadow_payload.get("performance", {}))

    with tabs[3]:
        baseline_id = st.selectbox(
            "Baseline experiment",
            [""] + [exp_id for exp_id in df["experiment_id"].tolist() if exp_id != selected_id],
            index=0,
        )
        if baseline_id:
            baseline_dir = registry_root / baseline_id
            baseline_metrics = read_json(baseline_dir / "evaluation" / "metrics.json") or {}
            metric_paths = [
                ("Sharpe", "performance.sharpe"),
                ("Total Return", "performance.total_return"),
                ("Max Drawdown", "performance.max_drawdown"),
                ("Turnover 1D Mean", "trading.turnover_1d_mean"),
                ("Avg Exposure", "trading.avg_exposure"),
                ("Tx Cost (bps)", "trading.tx_cost_bps"),
            ]
            compare_table = _compare_metrics(metrics_payload, baseline_metrics, metric_paths)
            st.dataframe(compare_table, use_container_width=True, hide_index=True)
        else:
            st.info("Select a baseline experiment to compare.")


if __name__ == "__main__":
    main()
