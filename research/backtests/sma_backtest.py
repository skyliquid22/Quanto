"""Vectorized SMA crossover backtest utilities."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from math import sqrt
from typing import Dict, Iterable, List, Sequence

from research.strategies.sma_crossover import SMAStrategyResult


ANNUALIZATION_DAYS = 252


@dataclass(frozen=True)
class BacktestMetrics:
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float
    number_of_trades: int


@dataclass(frozen=True)
class BacktestResult:
    symbol: str
    timestamps: List
    strategy_returns: List[float]
    buy_and_hold_returns: List[float]
    equity_curve: List[float]
    buy_and_hold_curve: List[float]
    metrics: BacktestMetrics


@dataclass(frozen=True)
class AggregatedBacktest:
    timestamps: List
    strategy_returns: List[float]
    equity_curve: List[float]
    metrics: BacktestMetrics


def run_backtest(strategy: SMAStrategyResult) -> BacktestResult:
    timestamps = strategy.timestamps
    closes = strategy.closes
    if not closes:
        raise ValueError(f"no closes available for symbol {strategy.symbol}")
    returns = _pct_change(closes)
    strategy_returns = [pos * ret for pos, ret in zip(strategy.positions, returns)]
    buy_and_hold_curve = _equity_curve(returns)
    equity_curve = _equity_curve(strategy_returns)
    metrics = _compute_metrics(strategy_returns, equity_curve, strategy.positions)
    return BacktestResult(
        symbol=strategy.symbol,
        timestamps=list(timestamps),
        strategy_returns=strategy_returns,
        buy_and_hold_returns=returns,
        equity_curve=equity_curve,
        buy_and_hold_curve=buy_and_hold_curve,
        metrics=metrics,
    )


def aggregate_results(results: Sequence[BacktestResult]) -> AggregatedBacktest:
    if not results:
        raise ValueError("at least one result is required to aggregate")
    if len(results) == 1:
        single = results[0]
        return AggregatedBacktest(
            timestamps=list(single.timestamps),
            strategy_returns=list(single.strategy_returns),
            equity_curve=list(single.equity_curve),
            metrics=single.metrics,
        )
    first_timestamps = results[0].timestamps
    for result in results[1:]:
        if result.timestamps != first_timestamps:
            raise ValueError("timestamps must match for aggregation")
    aggregate_returns = [
        sum(result.strategy_returns[idx] for result in results) / len(results)
        for idx in range(len(first_timestamps))
    ]
    equity_curve = _equity_curve(aggregate_returns)
    num_trades = sum(result.metrics.number_of_trades for result in results)
    metrics = _compute_metrics(aggregate_returns, equity_curve, None, override_trades=num_trades)
    return AggregatedBacktest(
        timestamps=list(first_timestamps),
        strategy_returns=aggregate_returns,
        equity_curve=equity_curve,
        metrics=metrics,
    )


def _pct_change(closes: Sequence[float]) -> List[float]:
    returns: List[float] = [0.0] * len(closes)
    for idx in range(1, len(closes)):
        prev = closes[idx - 1]
        curr = closes[idx]
        returns[idx] = (curr - prev) / prev if prev else 0.0
    return returns


def _equity_curve(returns: Sequence[float]) -> List[float]:
    curve: List[float] = []
    value = 1.0
    for ret in returns:
        value *= 1.0 + ret
        curve.append(value)
    return curve


def _compute_metrics(
    strategy_returns: Sequence[float],
    equity_curve: Sequence[float],
    positions: Sequence[int] | None,
    *,
    override_trades: int | None = None,
) -> BacktestMetrics:
    total_return = equity_curve[-1] - 1.0 if equity_curve else 0.0
    realized = strategy_returns[1:] if len(strategy_returns) > 1 else []
    periods = max(len(realized), 1)
    annualized_return = (1.0 + total_return) ** (ANNUALIZATION_DAYS / periods) - 1.0 if periods else 0.0
    annualized_volatility = _annualized_volatility(realized)
    sharpe = annualized_return / annualized_volatility if annualized_volatility else 0.0
    max_drawdown = _max_drawdown(equity_curve)
    number_of_trades = (
        override_trades
        if override_trades is not None
        else _count_trades(positions if positions is not None else [])
    )
    return BacktestMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        number_of_trades=number_of_trades,
    )


def _annualized_volatility(returns: Sequence[float]) -> float:
    if not returns:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((ret - mean) ** 2 for ret in returns) / len(returns)
    return sqrt(variance) * sqrt(ANNUALIZATION_DAYS)


def _max_drawdown(equity_curve: Sequence[float]) -> float:
    peak = 1.0
    worst = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak:
            drawdown = (value - peak) / peak
            worst = min(worst, drawdown)
    return abs(worst)


def _count_trades(positions: Sequence[int]) -> int:
    trades = 0
    for idx in range(1, len(positions)):
        if positions[idx] != positions[idx - 1]:
            trades += 1
    return trades


def serialize_metrics(metrics: BacktestMetrics) -> Dict[str, float]:
    return {key: float(value) for key, value in asdict(metrics).items()}


__all__ = [
    "AggregatedBacktest",
    "BacktestMetrics",
    "BacktestResult",
    "aggregate_results",
    "run_backtest",
    "serialize_metrics",
]
