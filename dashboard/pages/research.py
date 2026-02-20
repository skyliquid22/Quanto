"""Research page with experiment browser and detail view."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import streamlit as st

from dashboard.components.formatted_table import render_table
from dashboard.components.metric_cards import render_kpi_strip
from dashboard.components.regime_heatmap import render_regime_heatmap
from dashboard.components.status_badge import render_badge
from dashboard.data import load_experiment_payload, load_experiment_summaries
from dashboard.styles import (
    column_label,
    delta_color,
    format_bps,
    format_delta,
    format_delta_pct,
    format_exposure,
    format_pct,
    format_sharpe,
    format_turnover,
)


def _short_id(exp_id: str) -> str:
    return exp_id[:8]


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    name_filter = st.sidebar.text_input("Experiment name")
    feature_filter = st.sidebar.multiselect("Feature set", sorted(df["feature_set"].dropna().unique()))
    policy_filter = st.sidebar.multiselect("Policy", sorted(df["policy"].dropna().unique()))
    reward_filter = st.sidebar.multiselect("Reward version", sorted(df["reward_version"].dropna().unique()))
    only_qualified = st.sidebar.checkbox("Only qualified", value=False)

    filtered = df.copy()
    if name_filter:
        filtered = filtered[filtered["experiment_name"].str.contains(name_filter, case=False, na=False)]
    if feature_filter:
        filtered = filtered[filtered["feature_set"].isin(feature_filter)]
    if policy_filter:
        filtered = filtered[filtered["policy"].isin(policy_filter)]
    if reward_filter:
        filtered = filtered[filtered["reward_version"].isin(reward_filter)]
    if only_qualified:
        filtered = filtered[filtered["qualification_passed"] == True]  # noqa: E712
    return filtered


def _format_list_table(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    display["id"] = display["experiment_id"].apply(_short_id)
    display["max_drawdown"] = pd.to_numeric(display["max_drawdown"], errors="coerce") * 100.0
    display["total_return"] = pd.to_numeric(display["total_return"], errors="coerce") * 100.0
    display = display.rename(
        columns={
            "experiment_name": "Name",
            "id": "ID",
            "feature_set": "Feature Set",
            "policy": "Policy",
            "sharpe": "Sharpe",
            "max_drawdown": "Max Drawdown",
            "total_return": "Total Return",
        }
    )
    display = display[["Name", "ID", "Feature Set", "Policy", "Sharpe", "Max Drawdown", "Total Return"]]
    return display


def _render_experiment_table(df: pd.DataFrame) -> None:
    table = _format_list_table(df)
    column_config = {
        "ID": st.column_config.Column(label="ID", width="small"),
        "Sharpe": st.column_config.NumberColumn(label="Sharpe", format="%.2f"),
        "Max Drawdown": st.column_config.NumberColumn(label="Max Drawdown", format="%.2f%%"),
        "Total Return": st.column_config.NumberColumn(label="Total Return", format="%.2f%%"),
    }
    selection = st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
        on_select="rerun",
        selection_mode="single-row",
        key="experiment_table",
    )
    selected_rows = None
    if hasattr(selection, "selection"):
        selection_state = getattr(selection, "selection", None)
        if isinstance(selection_state, dict):
            selected_rows = selection_state.get("rows")
    elif isinstance(selection, dict):
        selected_rows = selection.get("rows")
    if selected_rows:
        idx = selected_rows[0]
        exp_id = df.iloc[idx]["experiment_id"]
        st.session_state["selected_experiment_id"] = exp_id


def _render_detail_header(spec: Mapping[str, Any], exp_id: str) -> None:
    name = spec.get("experiment_name") or "Experiment"
    st.header(name)
    metadata = f"ID: {exp_id} | Feature Set: {spec.get('feature_set', 'n/a')} | Policy: {spec.get('policy', 'n/a')} | Window: {spec.get('start_date', 'n/a')} → {spec.get('end_date', 'n/a')}"
    st.caption(metadata)


def _metric_payload(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sharpe": metrics.get("performance", {}).get("sharpe"),
        "total_return": metrics.get("performance", {}).get("total_return"),
        "max_drawdown": metrics.get("performance", {}).get("max_drawdown"),
        "turnover_1d_mean": metrics.get("trading", {}).get("turnover_1d_mean"),
        "avg_exposure": metrics.get("trading", {}).get("avg_exposure"),
        "tx_cost_bps": metrics.get("trading", {}).get("tx_cost_bps"),
    }


def _render_regime_tab(regime_payload: Mapping[str, Any] | None) -> None:
    if not regime_payload:
        st.info("Regime slices not available for this experiment.")
        return
    regimes = regime_payload.get("performance_by_regime")
    if not isinstance(regimes, Mapping):
        st.info("Regime slices not available for this experiment.")
        return
    rows = []
    for regime, metrics in regimes.items():
        if not isinstance(metrics, Mapping):
            continue
        rows.append(
            {
                "regime": regime,
                "avg_exposure": format_exposure(metrics.get("avg_exposure")),
                "avg_turnover": format_turnover(metrics.get("avg_turnover")),
                "max_drawdown": format_pct(metrics.get("max_drawdown")),
                "sharpe": format_sharpe(metrics.get("sharpe")),
                "total_return": format_pct(metrics.get("total_return")),
                "volatility_ann": format_pct(metrics.get("volatility_ann")),
            }
        )
    render_table(
        columns=[
            {"key": "regime", "label": "Regime"},
            {"key": "avg_exposure", "label": "Avg Exposure"},
            {"key": "avg_turnover", "label": "Avg Turnover"},
            {"key": "max_drawdown", "label": "Max Drawdown"},
            {"key": "sharpe", "label": "Sharpe"},
            {"key": "total_return", "label": "Total Return"},
            {"key": "volatility_ann", "label": "Ann. Volatility"},
        ],
        rows=rows,
        right_align=[
            "avg_exposure",
            "avg_turnover",
            "max_drawdown",
            "sharpe",
            "total_return",
            "volatility_ann",
        ],
    )

    frame = pd.DataFrame(regimes).T
    if not frame.empty:
        desired_cols = [
            "sharpe",
            "max_drawdown",
            "total_return",
            "volatility_ann",
            "avg_turnover",
            "avg_exposure",
        ]
        for col in desired_cols:
            if col not in frame.columns:
                frame[col] = None
        render_regime_heatmap(
            frame,
            desired_cols,
        )


def _extract_gate_rows(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    gate_summary = payload.get("gate_summary") or {}
    evaluations = gate_summary.get("evaluations") if isinstance(gate_summary, Mapping) else None
    if not isinstance(evaluations, list):
        return []
    rows = []
    for entry in evaluations:
        if not isinstance(entry, Mapping):
            continue
        thresholds = entry.get("thresholds") or {}
        observed = entry.get("observed") or {}
        threshold = None
        actual = None
        if thresholds.get("max_degradation_pct") is not None:
            threshold = thresholds.get("max_degradation_pct")
            actual = observed.get("degradation_pct") or observed.get("delta_pct")
        elif thresholds.get("max_value") is not None:
            threshold = thresholds.get("max_value")
            actual = observed.get("candidate")
        rows.append(
            {
                "gate": entry.get("gate_id") or entry.get("metric_id") or "gate",
                "type": entry.get("gate_type", "n/a"),
                "threshold": str(threshold) if threshold is not None else "n/a",
                "actual": str(actual) if actual is not None else "n/a",
                "status": render_badge(entry.get("status", "n/a").upper(), entry.get("status", "warn")),
            }
        )
    return rows


def _render_qualification_tab(payload: Mapping[str, Any] | None) -> None:
    if not payload:
        st.info("Qualification report not available.")
        return
    passed = payload.get("passed")
    status_label = "QUALIFIED" if passed else "NOT QUALIFIED"
    status_kind = "pass" if passed else "fail"
    st.markdown(render_badge(status_label, status_kind), unsafe_allow_html=True)

    rows = _extract_gate_rows(payload)
    if not rows:
        st.info("No gate details available.")
        return
    render_table(
        columns=[
            {"key": "gate", "label": "Gate"},
            {"key": "type", "label": "Type"},
            {"key": "threshold", "label": "Threshold"},
            {"key": "actual", "label": "Actual"},
            {"key": "status", "label": "Status"},
        ],
        rows=rows,
        right_align=["threshold", "actual"],
    )


def _render_shadow_tab(payload: Mapping[str, Any] | None) -> None:
    if not payload:
        st.info("Shadow replay data not available.")
        return
    rows = [
        {
            "metric": "Sharpe",
            "value": format_sharpe(payload.get("performance", {}).get("sharpe")),
        },
        {
            "metric": "Total Return",
            "value": format_pct(payload.get("performance", {}).get("total_return")),
        },
        {
            "metric": "Max Drawdown",
            "value": format_pct(payload.get("performance", {}).get("max_drawdown")),
        },
    ]
    render_table(
        columns=[
            {"key": "metric", "label": "Metric"},
            {"key": "value", "label": "Value"},
        ],
        rows=rows,
        right_align=["value"],
    )


def _render_compare_tab(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
) -> None:
    if not baseline:
        st.info("Select a baseline experiment to compare.")
        return
    metrics = [
        ("sharpe", "Sharpe"),
        ("total_return", "Total Return"),
        ("max_drawdown", "Max Drawdown"),
        ("turnover_1d_mean", "Turnover 1D Mean"),
        ("avg_exposure", "Avg Exposure"),
        ("tx_cost_bps", "Tx Cost (bps)"),
    ]
    rows = []
    for key, label in metrics:
        cand_val = candidate.get(key)
        base_val = baseline.get(key)
        delta = None
        delta_pct = None
        if cand_val is not None and base_val not in (None, 0):
            try:
                delta = cand_val - base_val
                delta_pct = delta / base_val
            except Exception:
                delta = None
        color = delta_color(key, delta)
        delta_text = format_delta(key, delta)
        delta_pct_text = format_delta_pct(delta_pct) if delta_pct is not None else "n/a"
        if color:
            delta_text = f"<span style=\"color:{color};\">{delta_text}</span>"
            delta_pct_text = f"<span style=\"color:{color};\">{delta_pct_text}</span>"
        rows.append(
            {
                "metric": label,
                "candidate": _format_value(key, cand_val),
                "baseline": _format_value(key, base_val),
                "delta": delta_text,
                "delta_pct": delta_pct_text,
            }
        )
    render_table(
        columns=[
            {"key": "metric", "label": "Metric"},
            {"key": "candidate", "label": "Candidate"},
            {"key": "baseline", "label": "Baseline"},
            {"key": "delta", "label": "Δ"},
            {"key": "delta_pct", "label": "Δ%"},
        ],
        rows=rows,
        right_align=["candidate", "baseline", "delta", "delta_pct"],
    )


def _format_value(key: str, value: Any) -> str:
    if key == "sharpe":
        return format_sharpe(value)
    if key in {"total_return", "max_drawdown"}:
        return format_pct(value)
    if key == "turnover_1d_mean":
        return format_turnover(value)
    if key == "avg_exposure":
        return format_exposure(value)
    if key == "tx_cost_bps":
        return format_bps(value)
    return "n/a"


def render(data_root: Path) -> None:
    st.header("Research")
    summaries = load_experiment_summaries(data_root)
    if not summaries:
        st.info("No experiments found.")
        return
    df = pd.DataFrame([entry.__dict__ for entry in summaries])
    filtered = _apply_filters(df)
    if filtered.empty:
        st.info("No experiments match the current filters.")
        return

    selected_id = st.session_state.get("selected_experiment_id")
    if not selected_id:
        st.subheader("Experiment Browser")
        _render_experiment_table(filtered)
        def _format_option(exp_id: str) -> str:
            name = filtered.loc[filtered["experiment_id"] == exp_id, "experiment_name"].values
            label = name[0] if len(name) else "unknown"
            return f"{label} [{_short_id(exp_id)}]"

        st.selectbox(
            "Open experiment",
            filtered["experiment_id"].tolist(),
            format_func=_format_option,
            key="experiment_selector",
            on_change=lambda: st.session_state.update(
                {"selected_experiment_id": st.session_state["experiment_selector"]}
            ),
        )
        return

    payload = load_experiment_payload(selected_id, data_root)
    spec = payload["spec"]
    metrics = payload["metrics"]
    st.button("← Back to experiments", on_click=lambda: st.session_state.pop("selected_experiment_id", None))
    _render_detail_header(spec, selected_id)

    kpi_metrics = [
        {"key": "sharpe", "label": "Sharpe", "value": metrics.get("performance", {}).get("sharpe")},
        {"key": "total_return", "label": "Total Return", "value": metrics.get("performance", {}).get("total_return")},
        {"key": "max_drawdown", "label": "Max Drawdown", "value": metrics.get("performance", {}).get("max_drawdown")},
        {"key": "turnover_1d_mean", "label": "Avg Turnover", "value": metrics.get("trading", {}).get("turnover_1d_mean")},
        {"key": "avg_exposure", "label": "Avg Exposure", "value": metrics.get("trading", {}).get("avg_exposure")},
        {"key": "tx_cost_bps", "label": "Tx Cost", "value": metrics.get("trading", {}).get("tx_cost_bps")},
    ]
    render_kpi_strip(kpi_metrics)

    tabs = st.tabs(["Regime Slices", "Qualification", "Shadow Metrics", "Compare"])
    with tabs[0]:
        _render_regime_tab(payload.get("regime_slices"))
    with tabs[1]:
        _render_qualification_tab(payload.get("qualification"))
    with tabs[2]:
        _render_shadow_tab(payload.get("shadow"))
    with tabs[3]:
        baseline_id = st.selectbox(
            "Baseline experiment",
            [None] + filtered["experiment_id"].tolist(),
            format_func=lambda exp_id: "Select" if exp_id is None else f"{filtered.loc[filtered['experiment_id']==exp_id, 'experiment_name'].values[0]} [{_short_id(exp_id)}]",
            key="baseline_selector",
        )
        baseline_payload = None
        if baseline_id:
            baseline_data = load_experiment_payload(baseline_id, data_root)
            baseline_payload = _metric_payload(baseline_data["metrics"])
        _render_compare_tab(_metric_payload(metrics), baseline_payload)


__all__ = ["render"]
