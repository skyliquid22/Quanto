"""Evaluation harness producing deterministic metrics JSON."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from research.eval.metrics import MetricConfig, MetricResult, compute_metric_bundle


@dataclass(frozen=True)
class EvalSeries:
    timestamps: Sequence[str]
    account_values: Sequence[float]
    weights: Sequence[Mapping[str, float]]
    transaction_costs: Sequence[float]
    symbols: Sequence[str]
    rollout_metadata: Mapping[str, object] | None = None
    regime_features: Sequence[Sequence[float]] | None = None
    regime_feature_names: Sequence[str] | None = None
    modes: Sequence[str] | None = None


@dataclass(frozen=True)
class EvaluationMetadata:
    symbols: Tuple[str, ...]
    start_date: str
    end_date: str
    interval: str
    feature_set: str | None
    policy_id: str
    run_id: str
    policy_details: Mapping[str, Any] | None = None
    data_split: Mapping[str, Any] | None = None


def evaluation_payload(
    series: EvalSeries,
    metadata: EvaluationMetadata,
    *,
    inputs_used: Mapping[str, str] | None = None,
    config: MetricConfig | None = None,
) -> Dict[str, Any]:
    """Compute evaluation metrics and assemble a deterministic payload."""

    cfg = config or MetricConfig()
    metrics = compute_metric_bundle(
        series.account_values,
        series.weights,
        transaction_costs=_align_costs(series.account_values, series.transaction_costs),
        symbols=series.symbols,
        config=cfg,
        regime_feature_series=series.regime_features,
        regime_feature_names=series.regime_feature_names,
        mode_series=series.modes,
        timestamps=series.timestamps,
    )
    metadata_section = _metadata_section(metadata, series.rollout_metadata)
    payload = {
        "metadata": metadata_section,
        "performance": metrics.performance,
        "trading": metrics.trading,
        "safety": metrics.safety,
        "series": _series_section(series, metrics),
        "inputs_used": dict(sorted((inputs_used or {}).items())),
        "config": {
            "evaluation": {
                "annualization_days": cfg.annualization_days,
                "min_cagr_periods": cfg.min_cagr_periods,
                "risk_free_rate": cfg.risk_free_rate,
                "float_precision": cfg.float_precision,
            }
        },
    }
    if cfg.risk_config is not None:
        payload["config"]["risk"] = cfg.risk_config.to_dict()
    if metrics.regime_slicing is not None:
        payload["regime_slicing"] = metrics.regime_slicing
    if metrics.performance_by_regime is not None:
        payload["performance_by_regime"] = metrics.performance_by_regime
    return payload


def from_rollout(
    *,
    timestamps: Sequence[str],
    account_values: Sequence[float],
    weights: Sequence[Mapping[str, float]],
    transaction_costs: Sequence[float] | None,
    symbols: Sequence[str],
    rollout_metadata: Mapping[str, object] | None = None,
    regime_features: Sequence[Sequence[float]] | None = None,
    regime_feature_names: Sequence[str] | None = None,
    modes: Sequence[str] | None = None,
) -> EvalSeries:
    costs = _align_costs(account_values, transaction_costs)
    return EvalSeries(
        timestamps=list(timestamps),
        account_values=list(account_values),
        weights=list(weights),
        transaction_costs=costs,
        symbols=tuple(symbols),
        rollout_metadata=dict(rollout_metadata or {}),
        regime_features=list(regime_features) if regime_features else None,
        regime_feature_names=tuple(regime_feature_names) if regime_feature_names else None,
        modes=list(modes) if modes else None,
    )


def _metadata_section(metadata: EvaluationMetadata, rollout_metadata: Mapping[str, object] | None) -> Dict[str, Any]:
    payload = {
        "symbols": list(metadata.symbols),
        "start_date": metadata.start_date,
        "end_date": metadata.end_date,
        "interval": metadata.interval,
        "feature_set": metadata.feature_set,
        "policy_id": metadata.policy_id,
        "run_id": metadata.run_id,
    }
    if metadata.policy_details:
        payload["policy"] = dict(metadata.policy_details)
    if metadata.data_split:
        payload["data_split"] = dict(metadata.data_split)
    if rollout_metadata:
        payload["rollout"] = dict(rollout_metadata)
    return payload


def _series_section(series: EvalSeries, metrics: MetricResult) -> Dict[str, Any]:
    weights_series = _weights_series(series.weights, series.symbols)
    payload = {
        "timestamps": list(series.timestamps),
        "account_value": list(series.account_values),
        "returns": metrics.returns,
        "transaction_costs": list(series.transaction_costs),
        "weights": weights_series,
    }
    if series.modes:
        payload["modes"] = list(series.modes)
    if series.regime_features and series.regime_feature_names:
        regime_values = [
            [float(value) for value in snapshot]
            for snapshot in series.regime_features
        ]
        if len(regime_values) != len(payload["returns"]):
            raise ValueError(
                "regime series length does not match returns length "
                f"({len(regime_values)} != {len(payload['returns'])})"
            )
        payload["regime"] = {
            "feature_names": list(series.regime_feature_names),
            "values": regime_values,
        }
    return payload


def _weights_series(weights: Sequence[Mapping[str, float]], symbols: Sequence[str]) -> Any:  # type: ignore[override]
    if not weights:
        return [] if len(symbols) <= 1 else {symbol: [] for symbol in symbols}
    if len(symbols) <= 1:
        symbol = symbols[0] if symbols else "asset"
        return [float(entry.get(symbol, 0.0)) for entry in weights]
    panel: Dict[str, list[float]] = {symbol: [] for symbol in symbols}
    for entry in weights:
        for symbol in symbols:
            panel[symbol].append(float(entry.get(symbol, 0.0)))
    return panel


def _align_costs(account_values: Sequence[float], transaction_costs: Sequence[float] | None) -> List[float]:
    expected = max(len(account_values) - 1, 0)
    costs = list(transaction_costs or [])
    if len(costs) < expected:
        costs.extend([0.0] * (expected - len(costs)))
    elif len(costs) > expected:
        costs = costs[:expected]
    return [float(value) for value in costs]


__all__ = ["MetricConfig", "EvalSeries", "EvaluationMetadata", "evaluation_payload", "from_rollout"]
