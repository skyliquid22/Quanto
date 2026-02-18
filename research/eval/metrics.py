"""Deterministic evaluation metrics for rollout results."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime
import math
from statistics import median
from typing import Dict, List, Mapping, Sequence, Tuple, Literal

from research.eval.regime_slicing import compute_regime_slices
from research.risk import PROJECTION_TOLERANCE, RiskConfig
from research.hierarchy.modes import normalize_mode

_DEFAULT_PRECISION = 10


@dataclass(frozen=True)
class MetricConfig:
    """Configuration controlling deterministic metric computation."""

    annualization_days: int = 252
    min_cagr_periods: int = 252
    risk_free_rate: float = 0.0
    float_precision: int = _DEFAULT_PRECISION
    turnover_tolerance: float = 1e-12
    risk_config: RiskConfig | None = None
    regime_labeling_version: str = "v1"
    regime_thresholds_path: str | None = None


@dataclass(frozen=True)
class MetricResult:
    performance: Dict[str, float | None]
    trading: Dict[str, object]
    safety: Dict[str, float]
    returns: List[float]
    regime_slicing: Dict[str, object] | None = None
    performance_by_regime: Dict[str, Dict[str, float | None]] | None = None
    stability: Dict[str, float] | None = None


@dataclass(frozen=True)
class MetricSchemaEntry:
    """Metadata describing how a scalar metric should be interpreted."""

    key: str
    category: str
    direction: Literal["higher_is_better", "lower_is_better", "neutral"]
    unit: str
    description: str

    @property
    def metric_id(self) -> str:
        return f"{self.category}.{self.key}"


METRIC_SCHEMA: Tuple[MetricSchemaEntry, ...] = (
    MetricSchemaEntry("total_return", "performance", "higher_is_better", "ratio", "Portfolio total return."),
    MetricSchemaEntry("cagr", "performance", "higher_is_better", "ratio", "Compound annual growth rate."),
    MetricSchemaEntry("volatility_ann", "performance", "neutral", "ratio", "Annualized volatility."),
    MetricSchemaEntry("sharpe", "performance", "higher_is_better", "ratio", "Annualized Sharpe ratio."),
    MetricSchemaEntry("max_drawdown", "performance", "lower_is_better", "ratio", "Maximum peak-to-trough drawdown."),
    MetricSchemaEntry("calmar", "performance", "higher_is_better", "ratio", "Calmar ratio."),
    MetricSchemaEntry("turnover_1d_mean", "trading", "lower_is_better", "ratio", "Mean 1-day turnover."),
    MetricSchemaEntry("turnover_1d_median", "trading", "lower_is_better", "ratio", "Median 1-day turnover."),
    MetricSchemaEntry("turnover_1d_std", "trading", "lower_is_better", "ratio", "Std dev of 1-day turnover."),
    MetricSchemaEntry("turnover_1d_p95", "trading", "lower_is_better", "ratio", "95th percentile 1-day turnover."),
    MetricSchemaEntry("avg_exposure", "trading", "neutral", "ratio", "Mean total exposure."),
    MetricSchemaEntry("max_concentration", "trading", "lower_is_better", "ratio", "Average max symbol weight."),
    MetricSchemaEntry("hhi_mean", "trading", "lower_is_better", "ratio", "Mean Herfindahl-Hirschman index."),
    MetricSchemaEntry("tx_cost_total", "trading", "lower_is_better", "currency", "Total transaction costs."),
    MetricSchemaEntry("tx_cost_bps", "trading", "lower_is_better", "bps", "Transaction costs in basis points."),
    MetricSchemaEntry("avg_cash", "trading", "neutral", "ratio", "Average idle cash allocation."),
    MetricSchemaEntry("nan_inf_violations", "safety", "lower_is_better", "count", "Number of NaN/Inf values detected."),
    MetricSchemaEntry("action_bounds_violations", "safety", "lower_is_better", "count", "Invalid action magnitudes."),
    MetricSchemaEntry("constraint_violations_count", "safety", "lower_is_better", "count", "Total constraint violations detected post-projection."),
    MetricSchemaEntry("max_weight_violation_count", "safety", "lower_is_better", "count", "Per-asset cap violations."),
    MetricSchemaEntry("exposure_violation_count", "safety", "lower_is_better", "count", "Exposure cap or min cash violations."),
    MetricSchemaEntry("turnover_violation_count", "safety", "lower_is_better", "count", "Turnover cap violations."),
    MetricSchemaEntry("mode_churn_rate", "stability", "lower_is_better", "ratio", "Normalized hierarchy mode churn."),
    MetricSchemaEntry("mode_set_size", "stability", "neutral", "count", "Number of active hierarchy modes."),
    MetricSchemaEntry("cost_curve_span", "stability", "lower_is_better", "ratio", "Net return span across cost multipliers."),
    MetricSchemaEntry("summary.fill_rate", "execution", "higher_is_better", "ratio", "Execution fill rate."),
    MetricSchemaEntry("summary.reject_rate", "execution", "lower_is_better", "ratio", "Execution reject rate."),
    MetricSchemaEntry("summary.avg_slippage_bps", "execution", "lower_is_better", "bps", "Average slippage for fills."),
    MetricSchemaEntry("summary.p95_slippage_bps", "execution", "lower_is_better", "bps", "95th percentile slippage."),
    MetricSchemaEntry("summary.total_fees", "execution", "lower_is_better", "currency", "Total execution fees."),
    MetricSchemaEntry("summary.turnover_realized", "execution", "lower_is_better", "ratio", "Realized turnover attributed to execution."),
    MetricSchemaEntry("summary.execution_halts", "execution", "lower_is_better", "count", "Execution halt events."),
    MetricSchemaEntry("summary.partial_fill_rate", "execution", "lower_is_better", "ratio", "Fraction of partially filled orders."),
)


def metric_schema_entries() -> Tuple[MetricSchemaEntry, ...]:
    """Return the ordered tuple containing the metric metadata."""

    return METRIC_SCHEMA


def compute_metric_bundle(
    account_values: Sequence[float],
    weights: Sequence[Mapping[str, float]],
    *,
    transaction_costs: Sequence[float] | None = None,
    symbols: Sequence[str] | None = None,
    config: MetricConfig | None = None,
    regime_feature_series: Sequence[Sequence[float]] | None = None,
    regime_feature_names: Sequence[str] | None = None,
    mode_series: Sequence[str] | None = None,
    timestamps: Sequence[object] | None = None,
    window: Tuple[date, date] | None = None,
) -> MetricResult:
    """Compute portfolio, trading, and safety metrics."""

    cfg = config or MetricConfig()
    if window is not None and timestamps is not None:
        account_values, weights, transaction_costs, regime_feature_series, mode_series = _slice_metric_inputs(
            account_values,
            weights,
            transaction_costs,
            regime_feature_series,
            mode_series,
            timestamps,
            window,
        )
    returns = _simple_returns(account_values)
    perf = _performance_metrics(account_values, returns, cfg)
    trading, constraint_diag, exposures, turnover_by_step = _trading_metrics(
        account_values,
        weights,
        transaction_costs,
        symbols or (),
        cfg,
        regime_feature_series,
        regime_feature_names,
    )
    mode_diag = _mode_diagnostics(
        mode_series,
        exposures,
        turnover_by_step,
        returns,
        cfg.float_precision,
    )
    trading.update(
        {
            "mode_counts": mode_diag["counts"],
            "mode_transitions": mode_diag["transitions"],
            "avg_exposure_by_mode": mode_diag["exposure"],
            "avg_turnover_by_mode": mode_diag["turnover"],
        }
    )
    if mode_diag["performance"]:
        perf["performance_by_mode"] = mode_diag["performance"]
    safety = _safety_checks(account_values, weights, returns, transaction_costs)
    for key, value in constraint_diag.items():
        safety[key] = float(value)
    rounded_returns = [_round(value, cfg.float_precision) for value in returns]
    cost_curve, cost_span = _cost_sensitivity_curve(account_values, transaction_costs, returns, cfg.float_precision)
    if cost_curve:
        trading["cost_sensitivity_curve"] = cost_curve
    stability = _stability_metrics(trading, mode_diag, cfg.float_precision, cost_span)
    regime_result = compute_regime_slices(
        regime_feature_series,
        regime_feature_names,
        returns=returns,
        exposures=exposures,
        turnover_by_step=turnover_by_step,
        annualization_days=cfg.annualization_days,
        float_precision=cfg.float_precision,
        labeling_version=cfg.regime_labeling_version,
        thresholds_path=cfg.regime_thresholds_path,
    )
    return MetricResult(
        performance=perf,
        trading=trading,
        safety=safety,
        returns=rounded_returns,
        regime_slicing=regime_result.metadata if regime_result else None,
        performance_by_regime=regime_result.performance_by_regime if regime_result else None,
        stability=stability,
    )


def _slice_metric_inputs(
    account_values: Sequence[float],
    weights: Sequence[Mapping[str, float]],
    transaction_costs: Sequence[float] | None,
    regime_feature_series: Sequence[Sequence[float]] | None,
    mode_series: Sequence[str] | None,
    timestamps: Sequence[object],
    window: Tuple[date, date],
) -> Tuple[
    Sequence[float],
    Sequence[Mapping[str, float]],
    Sequence[float] | None,
    Sequence[Sequence[float]] | None,
    Sequence[str] | None,
]:
    if not account_values or not timestamps:
        return account_values, weights, transaction_costs, regime_feature_series, mode_series
    if len(timestamps) != len(account_values):
        raise ValueError("timestamps length must match account_values length for window slicing")
    start_date, end_date = window
    if end_date < start_date:
        raise ValueError("window end_date cannot be earlier than start_date")
    indices = [
        idx
        for idx, value in enumerate(timestamps)
        if start_date <= _coerce_date(value) <= end_date
    ]
    if not indices:
        raise ValueError("window does not overlap provided timestamps")
    start_idx = indices[0]
    end_idx = indices[-1]
    sliced_account_values = list(account_values[start_idx : end_idx + 1])
    sliced_weights = list(weights[start_idx : end_idx + 1])
    sliced_costs = None
    if transaction_costs is not None:
        sliced_costs = list(transaction_costs[start_idx:end_idx])
    sliced_regime = None
    if regime_feature_series is not None:
        sliced_regime = list(regime_feature_series[start_idx:end_idx])
    sliced_modes = None
    if mode_series is not None:
        sliced_modes = list(mode_series[start_idx:end_idx])
    return sliced_account_values, sliced_weights, sliced_costs, sliced_regime, sliced_modes


def _coerce_date(value: object) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime().date()
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    raise ValueError("Unable to parse timestamp for window slicing.")


def _performance_metrics(
    account_values: Sequence[float],
    returns: Sequence[float],
    cfg: MetricConfig,
) -> Dict[str, float | None]:
    if not account_values:
        return {
            "total_return": 0.0,
            "cagr": None,
            "volatility_ann": 0.0,
            "sharpe": None,
            "max_drawdown": 0.0,
            "calmar": None,
        }
    start_value = account_values[0]
    end_value = account_values[-1]
    total_return = (end_value / start_value - 1.0) if start_value else 0.0
    volatility = _ann_volatility(returns, cfg.annualization_days)
    sharpe = None
    mean_daily = _mean(returns)
    if volatility and cfg.annualization_days > 0:
        excess = mean_daily * cfg.annualization_days - cfg.risk_free_rate
        sharpe = excess / volatility
    num_periods = max(len(account_values) - 1, 0)
    cagr = None
    if num_periods >= cfg.min_cagr_periods and start_value > 0 and end_value > 0:
        years = num_periods / cfg.annualization_days if cfg.annualization_days > 0 else 0.0
        if years > 0:
            cagr = (end_value / start_value) ** (1.0 / years) - 1.0
    max_drawdown = _max_drawdown(account_values)
    calmar = None
    if cagr is not None and max_drawdown > 0:
        calmar = cagr / max_drawdown
    precision = cfg.float_precision
    return {
        "total_return": _round(total_return, precision),
        "cagr": None if cagr is None else _round(cagr, precision),
        "volatility_ann": _round(volatility, precision),
        "sharpe": None if sharpe is None else _round(sharpe, precision),
        "max_drawdown": _round(max_drawdown, precision),
        "calmar": None if calmar is None else _round(calmar, precision),
    }


def _trading_metrics(
    account_values: Sequence[float],
    weights: Sequence[Mapping[str, float]],
    transaction_costs: Sequence[float] | None,
    symbols: Sequence[str],
    cfg: MetricConfig,
    regime_feature_series: Sequence[Sequence[float]] | None,
    regime_feature_names: Sequence[str] | None,
) -> Tuple[Dict[str, float], Dict[str, float], List[float], List[float]]:
    symbol_order = tuple(dict.fromkeys(symbols))
    turnover_steps: List[float] = []
    exposures: List[float] = []
    concentrations: List[float] = []
    hhi_values: List[float] = []
    turnover_by_step: List[float] = [0.0 for _ in weights]
    tolerance = cfg.turnover_tolerance
    resolved_weights = [_fill_weights(entry, symbol_order) for entry in weights]
    if not resolved_weights:
        resolved_weights = [{symbol: 0.0 for symbol in symbol_order}] if symbol_order else []
    for idx, entry in enumerate(resolved_weights):
        exposures.append(sum(entry.values()))
        concentrations.append(max(entry.values()) if entry else 0.0)
        hhi_values.append(sum(value * value for value in entry.values()))
        if idx == 0:
            continue
        prev = resolved_weights[idx - 1]
        total = 0.0
        keys = symbol_order or tuple(sorted(set(prev) | set(entry)))
        for symbol in keys:
            total += abs(entry.get(symbol, 0.0) - prev.get(symbol, 0.0))
        turnover_value = total if total > tolerance else 0.0
        turnover_steps.append(turnover_value)
        if idx < len(turnover_by_step):
            turnover_by_step[idx] = turnover_value
    tx_costs = list(transaction_costs or [])
    tx_total = sum(float(value) for value in tx_costs)
    avg_account = _mean(account_values)
    tx_bps = (tx_total / avg_account * 1e4) if avg_account else 0.0
    constraint_diag = _constraint_diagnostics(resolved_weights, cfg.risk_config, cfg.turnover_tolerance)
    avg_cash = constraint_diag.pop("_avg_cash", 0.0)
    precision = cfg.float_precision
    trading_metrics = {
        "turnover_1d_mean": _round(_mean(turnover_steps), precision),
        "turnover_1d_median": _round(median(turnover_steps) if turnover_steps else 0.0, precision),
        "turnover_1d_std": _round(_std(turnover_steps), precision),
        "turnover_1d_p95": _round(_percentile(turnover_steps, 0.95), precision),
        "avg_exposure": _round(_mean(exposures), precision),
        "max_concentration": _round(_mean(concentrations), precision),
        "hhi_mean": _round(_mean(hhi_values), precision),
        "tx_cost_total": _round(tx_total, precision),
        "tx_cost_bps": _round(tx_bps, precision),
        "avg_cash": _round(avg_cash, precision),
    }
    regime_diag = _regime_diagnostics(
        regime_feature_series,
        regime_feature_names,
        exposures,
        turnover_by_step,
        precision,
    )
    trading_metrics.update(regime_diag)
    return trading_metrics, constraint_diag, exposures, turnover_by_step


def _safety_checks(
    account_values: Sequence[float],
    weights: Sequence[Mapping[str, float]],
    returns: Sequence[float],
    transaction_costs: Sequence[float] | None,
) -> Dict[str, float]:
    nan_inf = 0
    for value in account_values:
        if not math.isfinite(float(value)):
            nan_inf += 1
    for value in returns:
        if not math.isfinite(float(value)):
            nan_inf += 1
    for value in transaction_costs or ():
        if not math.isfinite(float(value)):
            nan_inf += 1
    action_violations = 0
    for entry in weights:
        for value in entry.values():
            if not math.isfinite(float(value)):
                nan_inf += 1
                continue
            if value < -1e-9 or value > 1.0 + 1e-9:
                action_violations += 1
    return {
        "nan_inf_violations": float(nan_inf),
        "action_bounds_violations": float(action_violations),
    }


def _constraint_diagnostics(
    weights: Sequence[Mapping[str, float]],
    risk_config: RiskConfig | None,
    turnover_tolerance: float,
) -> Dict[str, float]:
    cfg = risk_config or RiskConfig()
    long_only_count = 0
    max_weight_count = 0
    exposure_count = 0
    turnover_count = 0
    cash_levels: List[float] = []
    exposure_cap = None
    if cfg.exposure_cap is not None or cfg.min_cash is not None:
        caps: List[float] = []
        if cfg.exposure_cap is not None:
            caps.append(max(cfg.exposure_cap, 0.0))
        if cfg.min_cash is not None:
            caps.append(max(0.0, 1.0 - cfg.min_cash))
        if caps:
            exposure_cap = min(caps)
    prev_entry: Mapping[str, float] | None = None
    for entry in weights:
        total = sum(entry.values())
        cash_levels.append(max(0.0, 1.0 - total))
        if cfg.long_only:
            for value in entry.values():
                if value < -PROJECTION_TOLERANCE:
                    long_only_count += 1
        if cfg.max_weight is not None:
            limit = cfg.max_weight + PROJECTION_TOLERANCE
            for value in entry.values():
                if value > limit:
                    max_weight_count += 1
        if exposure_cap is not None and total > exposure_cap + PROJECTION_TOLERANCE:
            exposure_count += 1
        if prev_entry is not None and cfg.max_turnover_1d is not None:
            turnover = 0.0
            keys = set(prev_entry) | set(entry)
            for symbol in keys:
                turnover += abs(entry.get(symbol, 0.0) - prev_entry.get(symbol, 0.0))
            if turnover > cfg.max_turnover_1d + max(turnover_tolerance, PROJECTION_TOLERANCE):
                turnover_count += 1
        prev_entry = entry
    constraint_total = long_only_count + max_weight_count + exposure_count
    if cfg.max_turnover_1d is not None:
        constraint_total += turnover_count
    diag = {
        "constraint_violations_count": float(constraint_total),
        "max_weight_violation_count": float(max_weight_count),
        "exposure_violation_count": float(exposure_count),
        "turnover_violation_count": float(turnover_count),
        "_avg_cash": _mean(cash_levels),
    }
    return diag


def _regime_diagnostics(
    regime_series: Sequence[Sequence[float]] | None,
    feature_names: Sequence[str] | None,
    exposures: Sequence[float],
    turnover_by_step: Sequence[float],
    precision: int,
) -> Dict[str, object]:
    diagnostics: Dict[str, object] = {
        "avg_exposure_by_regime": {},
        "avg_turnover_by_regime": {},
        "regime_feature_summary": {},
    }
    if not regime_series or not feature_names:
        return diagnostics
    names = tuple(feature_names)
    available = min(len(regime_series), len(exposures), len(turnover_by_step))
    if available == 0:
        return diagnostics
    trimmed_regime = [tuple(float(value) for value in regime_series[idx]) for idx in range(available)]
    trimmed_exposure = [float(exposures[idx]) for idx in range(available)]
    trimmed_turnover = [float(turnover_by_step[idx]) for idx in range(available)]
    exposure_diag: Dict[str, Dict[str, float]] = {}
    turnover_diag: Dict[str, Dict[str, float]] = {}
    summary_diag: Dict[str, Dict[str, float]] = {}

    for feature_idx, name in enumerate(names):
        column = [entry[feature_idx] if feature_idx < len(entry) else 0.0 for entry in trimmed_regime]
        if not column:
            exposure_diag[name] = {"low": 0.0, "high": 0.0}
            turnover_diag[name] = {"low": 0.0, "high": 0.0}
            summary_diag[name] = {"mean": 0.0, "stdev": 0.0}
            continue
        pivot = median(column)
        low_indices = [idx for idx, value in enumerate(column) if value <= pivot]
        high_indices = [idx for idx, value in enumerate(column) if value > pivot]
        low_exposure = _mean([trimmed_exposure[idx] for idx in low_indices]) if low_indices else 0.0
        high_exposure = _mean([trimmed_exposure[idx] for idx in high_indices]) if high_indices else low_exposure
        low_turnover = _mean([trimmed_turnover[idx] for idx in low_indices]) if low_indices else 0.0
        high_turnover = _mean([trimmed_turnover[idx] for idx in high_indices]) if high_indices else low_turnover
        exposure_diag[name] = {
            "low": _round(low_exposure, precision),
            "high": _round(high_exposure, precision),
        }
        turnover_diag[name] = {
            "low": _round(low_turnover, precision),
            "high": _round(high_turnover, precision),
        }
        summary_diag[name] = {
            "mean": _round(_mean(column), precision),
            "stdev": _round(_std(column), precision),
        }
    diagnostics["avg_exposure_by_regime"] = exposure_diag
    diagnostics["avg_turnover_by_regime"] = turnover_diag
    diagnostics["regime_feature_summary"] = summary_diag
    return diagnostics


def _mode_diagnostics(
    mode_series: Sequence[str] | None,
    exposures: Sequence[float],
    turnover_by_step: Sequence[float],
    returns: Sequence[float],
    precision: int,
) -> Dict[str, Dict[str, object]]:
    diag = {
        "counts": {},
        "transitions": {},
        "exposure": {},
        "turnover": {},
        "performance": {},
    }
    if not mode_series:
        return diag
    steps = min(len(mode_series), max(len(exposures) - 1, 0), max(len(turnover_by_step) - 1, 0), len(returns))
    if steps <= 0:
        return diag
    counts: Dict[str, int] = {}
    exposure_accum: Dict[str, List[float]] = {}
    turnover_accum: Dict[str, List[float]] = {}
    perf_accum: Dict[str, List[float]] = {}
    transitions: Dict[str, int] = {}
    prev_mode: str | None = None
    for idx in range(steps):
        mode_name = normalize_mode(mode_series[idx])
        counts[mode_name] = counts.get(mode_name, 0) + 1
        exposure_accum.setdefault(mode_name, []).append(float(exposures[idx + 1]))
        turnover_accum.setdefault(mode_name, []).append(float(turnover_by_step[idx + 1]))
        perf_accum.setdefault(mode_name, []).append(float(returns[idx]))
        if prev_mode is not None:
            key = f"{prev_mode}->{mode_name}"
            transitions[key] = transitions.get(key, 0) + 1
        prev_mode = mode_name
    diag["counts"] = {mode: counts[mode] for mode in sorted(counts)}
    diag["transitions"] = {key: transitions[key] for key in sorted(transitions)}
    diag["exposure"] = {
        mode: _round(_mean(values), precision) for mode, values in sorted(exposure_accum.items())
    }
    diag["turnover"] = {
        mode: _round(_mean(values), precision) for mode, values in sorted(turnover_accum.items())
    }
    performance: Dict[str, float] = {}
    for mode, values in perf_accum.items():
        growth = 1.0
        for ret in values:
            growth *= 1.0 + ret
        performance[mode] = _round(growth - 1.0, precision)
    diag["performance"] = {mode: performance[mode] for mode in sorted(performance)}
    return diag


def _stability_metrics(
    trading_metrics: Mapping[str, object],
    mode_diag: Mapping[str, Dict[str, object]],
    precision: int,
    cost_span: float,
) -> Dict[str, float]:
    counts = mode_diag.get("counts") or {}
    total_steps = sum(int(value) for value in counts.values())
    transitions = mode_diag.get("transitions") or {}
    transition_count = sum(int(value) for value in transitions.values())
    denominator = max(total_steps - 1, 1)
    churn_rate = transition_count / denominator if denominator > 0 else 0.0
    return {
        "turnover_std": float(trading_metrics.get("turnover_1d_std", 0.0)),
        "turnover_p95": float(trading_metrics.get("turnover_1d_p95", 0.0)),
        "mode_churn_rate": _round(churn_rate, precision),
        "mode_set_size": float(len(counts)),
        "cost_curve_span": _round(cost_span, precision),
    }


def _simple_returns(account_values: Sequence[float]) -> List[float]:
    returns: List[float] = []
    for idx in range(1, len(account_values)):
        prev = account_values[idx - 1]
        curr = account_values[idx]
        if prev == 0:
            returns.append(0.0)
        else:
            returns.append((curr / prev) - 1.0)
    return returns


def _ann_volatility(returns: Sequence[float], annualization_days: int) -> float:
    if not returns:
        return 0.0
    mean = _mean(returns)
    variance = sum((value - mean) ** 2 for value in returns) / len(returns)
    daily_vol = math.sqrt(variance)
    return daily_vol * math.sqrt(max(annualization_days, 0))


def _max_drawdown(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    peak = float(values[0])
    worst = 0.0
    for value in values:
        val = float(value)
        if val > peak:
            peak = val
        drawdown = (val - peak) / peak if peak else 0.0
        worst = min(worst, drawdown)
    return abs(worst)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean_value = _mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(max(variance, 0.0))


def _fill_weights(entry: Mapping[str, float], symbols: Sequence[str]) -> Dict[str, float]:
    if symbols:
        return {symbol: float(entry.get(symbol, 0.0)) for symbol in symbols}
    ordered: Dict[str, float] = {}
    for key in sorted(entry):
        ordered[str(key)] = float(entry[key])
    return ordered


def _round(value: float, precision: int) -> float:
    rounded = round(float(value), precision)
    if rounded == -0.0:
        return 0.0
    return rounded


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = percentile * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    weight = rank - low
    return ordered[low] + weight * (ordered[high] - ordered[low])


def _cost_sensitivity_curve(
    account_values: Sequence[float],
    transaction_costs: Sequence[float] | None,
    returns: Sequence[float],
    precision: int,
) -> Tuple[Dict[str, float], float]:
    tx = list(transaction_costs or [])
    if not tx or len(account_values) < 2:
        return {}, 0.0
    prev_values = account_values[:-1]
    steps = min(len(prev_values), len(tx))
    if steps == 0:
        return {}, 0.0
    cost_rates = []
    for idx in range(steps):
        prev = prev_values[idx]
        cost_rates.append((tx[idx] / prev) if prev else 0.0)
    gross_returns = [returns[idx] + cost_rates[idx] for idx in range(steps)]
    multipliers = (0.5, 1.0, 1.5)
    curve: Dict[str, float] = OrderedDict()
    for multiplier in multipliers:
        growth = 1.0
        for idx, gross in enumerate(gross_returns):
            net = gross - multiplier * cost_rates[idx]
            growth *= 1.0 + net
        curve[str(multiplier)] = _round(growth - 1.0, precision)
    span = max(curve.values()) - min(curve.values()) if curve else 0.0
    return curve, span


__all__ = [
    "MetricConfig",
    "MetricResult",
    "MetricSchemaEntry",
    "METRIC_SCHEMA",
    "metric_schema_entries",
    "compute_metric_bundle",
]
