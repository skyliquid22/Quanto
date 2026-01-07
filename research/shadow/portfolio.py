"""Paper portfolio accounting for shadow execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from research.risk import PROJECTION_TOLERANCE, RiskConfig


@dataclass(frozen=True)
class PortfolioUpdate:
    """Result of a deterministic rebalance step."""

    cash: float
    holdings: list[float]
    weights: list[float]
    portfolio_value: float
    turnover: float
    transaction_cost: float


def valuate_portfolio(cash: float, holdings: Sequence[float], prices: Sequence[float]) -> float:
    notional = sum(float(h) * float(p) for h, p in zip(holdings, prices))
    return float(cash) + notional


def weights_from_holdings(
    holdings: Sequence[float],
    prices: Sequence[float],
    portfolio_value: float,
) -> list[float]:
    if portfolio_value <= PROJECTION_TOLERANCE:
        return [0.0 for _ in holdings]
    return [float(holding) * float(price) / portfolio_value for holding, price in zip(holdings, prices)]


def rebalance_portfolio(
    *,
    cash: float,
    holdings: Sequence[float],
    prices: Sequence[float],
    target_weights: Sequence[float],
    transaction_cost_bp: float,
) -> PortfolioUpdate:
    """Apply a deterministic rebalance at the provided prices."""

    if len(holdings) != len(prices) or len(target_weights) != len(prices):
        raise ValueError("holdings, prices, and target_weights must have identical dimensions")
    rate = max(float(transaction_cost_bp), 0.0) / 10_000.0
    prev_value = valuate_portfolio(cash, holdings, prices)
    if prev_value <= PROJECTION_TOLERANCE:
        raise RuntimeError("portfolio value must be positive prior to rebalancing")
    prev_weights = weights_from_holdings(holdings, prices, prev_value)
    trade_notional = [
        float(target_weights[idx]) * prev_value - float(holdings[idx]) * float(prices[idx])
        for idx in range(len(prices))
    ]
    updated_holdings: list[float] = []
    for idx, delta in enumerate(trade_notional):
        price = float(prices[idx])
        if price <= 0:
            if abs(delta) > PROJECTION_TOLERANCE:
                raise ValueError("Cannot trade assets with non-positive prices")
            updated_holdings.append(float(holdings[idx]))
            continue
        updated_holdings.append(float(holdings[idx]) + delta / price)
    cash_after_trade = float(cash) - sum(trade_notional)
    tx_cost = sum(abs(value) for value in trade_notional) * rate
    cash_after_cost = cash_after_trade - tx_cost
    portfolio_value = valuate_portfolio(cash_after_cost, updated_holdings, prices)
    if portfolio_value <= PROJECTION_TOLERANCE:
        raise RuntimeError("portfolio value became non-positive after applying transaction costs")
    weights = weights_from_holdings(updated_holdings, prices, portfolio_value)
    turnover = sum(abs(target_weights[idx] - prev_weights[idx]) for idx in range(len(prices)))
    return PortfolioUpdate(
        cash=float(cash_after_cost),
        holdings=updated_holdings,
        weights=weights,
        portfolio_value=float(portfolio_value),
        turnover=float(turnover),
        transaction_cost=float(tx_cost),
    )


def constraint_diagnostics(
    weights: Sequence[float],
    prev_weights: Sequence[float],
    risk_config: RiskConfig,
) -> dict[str, float]:
    """Per-step constraint diagnostics mirroring evaluation metrics."""

    cfg = risk_config
    long_only = 0
    if cfg.long_only:
        long_only = sum(1 for weight in weights if weight < -PROJECTION_TOLERANCE)
    max_weight = 0
    if cfg.max_weight is not None:
        cap = max(cfg.max_weight, 0.0) + PROJECTION_TOLERANCE
        max_weight = sum(1 for weight in weights if weight > cap)
    exposure_cap = _effective_exposure_cap(cfg)
    exposure_violation = 0
    total_exposure = sum(weights)
    if exposure_cap is not None and total_exposure > exposure_cap + PROJECTION_TOLERANCE:
        exposure_violation = 1
    turnover_violation = 0
    turnover_cap = cfg.max_turnover_1d
    if turnover_cap is not None:
        turnover = sum(abs(weights[idx] - prev_weights[idx]) for idx in range(len(weights)))
        if turnover > turnover_cap + PROJECTION_TOLERANCE:
            turnover_violation = 1
    total = long_only + max_weight + exposure_violation + turnover_violation
    return {
        "constraint_violations_count": float(total),
        "long_only_violation_count": float(long_only),
        "max_weight_violation_count": float(max_weight),
        "exposure_violation_count": float(exposure_violation),
        "turnover_violation_count": float(turnover_violation),
        "total_exposure": float(total_exposure),
    }


def _effective_exposure_cap(cfg: RiskConfig) -> float | None:
    caps: list[float] = []
    if cfg.exposure_cap is not None:
        caps.append(max(0.0, float(cfg.exposure_cap)))
    if cfg.min_cash is not None:
        caps.append(max(0.0, 1.0 - float(cfg.min_cash)))
    if not caps:
        return None
    return min(caps)


__all__ = ["PortfolioUpdate", "constraint_diagnostics", "rebalance_portfolio", "valuate_portfolio", "weights_from_holdings"]
