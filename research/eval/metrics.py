"""Deterministic evaluation metrics for rollout results."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Dict, List, Mapping, Sequence, Tuple, Literal

_DEFAULT_PRECISION = 10


@dataclass(frozen=True)
class MetricConfig:
    """Configuration controlling deterministic metric computation."""

    annualization_days: int = 252
    min_cagr_periods: int = 252
    risk_free_rate: float = 0.0
    float_precision: int = _DEFAULT_PRECISION
    turnover_tolerance: float = 1e-12


@dataclass(frozen=True)
class MetricResult:
    performance: Dict[str, float | None]
    trading: Dict[str, float]
    safety: Dict[str, float]
    returns: List[float]


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
    MetricSchemaEntry("avg_exposure", "trading", "neutral", "ratio", "Mean total exposure."),
    MetricSchemaEntry("max_concentration", "trading", "lower_is_better", "ratio", "Average max symbol weight."),
    MetricSchemaEntry("hhi_mean", "trading", "lower_is_better", "ratio", "Mean Herfindahl-Hirschman index."),
    MetricSchemaEntry("tx_cost_total", "trading", "lower_is_better", "currency", "Total transaction costs."),
    MetricSchemaEntry("tx_cost_bps", "trading", "lower_is_better", "bps", "Transaction costs in basis points."),
    MetricSchemaEntry("nan_inf_violations", "safety", "lower_is_better", "count", "Number of NaN/Inf values detected."),
    MetricSchemaEntry("action_bounds_violations", "safety", "lower_is_better", "count", "Invalid action magnitudes."),
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
) -> MetricResult:
    """Compute portfolio, trading, and safety metrics."""

    cfg = config or MetricConfig()
    returns = _simple_returns(account_values)
    perf = _performance_metrics(account_values, returns, cfg)
    trading = _trading_metrics(account_values, weights, transaction_costs, symbols or (), cfg)
    safety = _safety_checks(account_values, weights, returns, transaction_costs)
    rounded_returns = [_round(value, cfg.float_precision) for value in returns]
    return MetricResult(performance=perf, trading=trading, safety=safety, returns=rounded_returns)


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
) -> Dict[str, float]:
    symbol_order = tuple(dict.fromkeys(symbols))
    turnover_steps: List[float] = []
    exposures: List[float] = []
    concentrations: List[float] = []
    hhi_values: List[float] = []
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
        turnover_steps.append(total if total > tolerance else 0.0)
    tx_costs = list(transaction_costs or [])
    tx_total = sum(float(value) for value in tx_costs)
    avg_account = _mean(account_values)
    tx_bps = (tx_total / avg_account * 1e4) if avg_account else 0.0
    precision = cfg.float_precision
    return {
        "turnover_1d_mean": _round(_mean(turnover_steps), precision),
        "turnover_1d_median": _round(median(turnover_steps) if turnover_steps else 0.0, precision),
        "avg_exposure": _round(_mean(exposures), precision),
        "max_concentration": _round(_mean(concentrations), precision),
        "hhi_mean": _round(_mean(hhi_values), precision),
        "tx_cost_total": _round(tx_total, precision),
        "tx_cost_bps": _round(tx_bps, precision),
    }


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


__all__ = [
    "MetricConfig",
    "MetricResult",
    "MetricSchemaEntry",
    "METRIC_SCHEMA",
    "metric_schema_entries",
    "compute_metric_bundle",
]
