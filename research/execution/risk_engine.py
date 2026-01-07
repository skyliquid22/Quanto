"""Deterministic execution risk checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from research.execution.orders import TOLERANCE, stable_symbol_weights
from research.execution.types import Order, RiskCheckResult


@dataclass(frozen=True)
class ExecutionRiskConfig:
    """Configuration parameters used by the risk engine."""

    max_gross_exposure: float | None = None
    min_cash_pct: float | None = None
    max_symbol_weight: float | None = None
    max_daily_turnover: float | None = None
    max_active_positions: int | None = None
    max_daily_loss: float | None = None
    max_trailing_drawdown: float | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "ExecutionRiskConfig":
        return cls(
            max_gross_exposure=_maybe_float(payload.get("max_gross_exposure")),
            min_cash_pct=_maybe_float(payload.get("min_cash_pct")),
            max_symbol_weight=_maybe_float(payload.get("max_symbol_weight")),
            max_daily_turnover=_maybe_float(payload.get("max_daily_turnover")),
            max_active_positions=int(payload["max_active_positions"]) if payload.get("max_active_positions") else None,
            max_daily_loss=_maybe_float(payload.get("max_daily_loss")),
            max_trailing_drawdown=_maybe_float(payload.get("max_trailing_drawdown")),
        )


class ExecutionRiskEngine:
    """Applies deterministic hard checks before broker submission."""

    def __init__(self, config: ExecutionRiskConfig) -> None:
        self._config = config

    def evaluate(
        self,
        *,
        orders: Sequence[Order],
        symbol_order: Sequence[str],
        target_weights: Mapping[str, float],
        prev_weights: Sequence[float],
        prices: Mapping[str, float],
        cash: float,
        portfolio_value: float,
        day_start_value: float,
        peak_value: float,
    ) -> RiskCheckResult:
        weights = stable_symbol_weights(target_weights, symbol_order)
        snapshot: dict[str, float] = {}
        halted, halt_reason = self._check_kill_switches(day_start_value, peak_value, portfolio_value, snapshot)
        if halted:
            return self._halted_result(orders, halt_reason, snapshot)
        halted, halt_reason = self._check_global_limits(
            weights,
            prev_weights,
            portfolio_value,
            snapshot,
            symbol_order,
        )
        if halted:
            return self._halted_result(orders, halt_reason, snapshot)
        approved, rejected = self._apply_symbol_limits(orders, weights, prices, cash, portfolio_value, snapshot)
        return RiskCheckResult(
            approved=approved,
            rejected=rejected,
            halted=False,
            halt_reason=None,
            snapshot=snapshot,
        )

    def _check_kill_switches(
        self,
        day_start_value: float,
        peak_value: float,
        portfolio_value: float,
        snapshot: dict[str, float],
    ) -> tuple[bool, str | None]:
        cfg = self._config
        daily_loss = 0.0
        if day_start_value > TOLERANCE:
            daily_loss = max(0.0, (day_start_value - portfolio_value) / day_start_value)
        snapshot["daily_loss"] = float(daily_loss)
        if cfg.max_daily_loss is not None and daily_loss > cfg.max_daily_loss + TOLERANCE:
            return True, "DAILY_LOSS_LIMIT"
        drawdown = 0.0
        if peak_value > TOLERANCE:
            drawdown = max(0.0, (peak_value - portfolio_value) / peak_value)
        snapshot["trailing_drawdown"] = float(drawdown)
        if cfg.max_trailing_drawdown is not None and drawdown > cfg.max_trailing_drawdown + TOLERANCE:
            return True, "TRAILING_DRAWDOWN_LIMIT"
        return False, None

    def _check_global_limits(
        self,
        target_weights: Mapping[str, float],
        prev_weights: Sequence[float],
        portfolio_value: float,
        snapshot: dict[str, float],
        symbol_order: Sequence[str],
    ) -> tuple[bool, str | None]:
        cfg = self._config
        gross_exposure = sum(abs(weight) for weight in target_weights.values())
        snapshot["gross_exposure"] = float(gross_exposure)
        if cfg.max_gross_exposure is not None and gross_exposure > cfg.max_gross_exposure + TOLERANCE:
            return True, "GROSS_EXPOSURE_CAP"
        turnover = 0.0
        for idx, symbol in enumerate(symbol_order):
            prev = prev_weights[idx] if idx < len(prev_weights) else 0.0
            turnover += abs(target_weights.get(symbol, 0.0) - prev)
        snapshot["turnover"] = float(turnover)
        if cfg.max_daily_turnover is not None and turnover > cfg.max_daily_turnover + TOLERANCE:
            return True, "TURNOVER_LIMIT"
        return False, None

    def _apply_symbol_limits(
        self,
        orders: Sequence[Order],
        target_weights: Mapping[str, float],
        prices: Mapping[str, float],
        cash: float,
        portfolio_value: float,
        snapshot: dict[str, float],
    ) -> tuple[list[Order], list[Order]]:
        cfg = self._config
        approved: list[Order] = list(orders)
        rejected: list[Order] = []
        # Max position weight
        if cfg.max_symbol_weight is not None:
            for order in list(approved):
                weight = abs(target_weights.get(order.symbol, 0.0))
                if weight > cfg.max_symbol_weight + TOLERANCE:
                    order.status = "REJECTED"
                    order.reject_reason = "MAX_WEIGHT"
                    approved.remove(order)
                    rejected.append(order)
        # Max active positions
        if cfg.max_active_positions is not None and cfg.max_active_positions > 0:
            ordered_pairs = [(symbol, target_weights.get(symbol, 0.0)) for symbol in target_weights]
            sorted_symbols = sorted(
                ordered_pairs,
                key=lambda item: (-abs(item[1]), item[0]),
            )
            allowed = {symbol for symbol, _ in sorted_symbols[: cfg.max_active_positions]}
            snapshot["active_positions"] = float(sum(1 for symbol, weight in sorted_symbols if abs(weight) > TOLERANCE))
            for order in list(approved):
                if abs(target_weights.get(order.symbol, 0.0)) <= TOLERANCE:
                    continue
                if order.symbol not in allowed:
                    order.status = "REJECTED"
                    order.reject_reason = "MAX_POSITIONS"
                    approved.remove(order)
                    rejected.append(order)
        else:
            snapshot["active_positions"] = float(
                sum(1 for weight in target_weights.values() if abs(weight) > TOLERANCE)
            )
        # Min cash
        if cfg.min_cash_pct is not None:
            min_cash = cfg.min_cash_pct * portfolio_value
            projected_cash = float(cash)
            for order in approved:
                notional = order.qty * float(prices.get(order.symbol, 0.0))
                if order.side == "BUY":
                    projected_cash -= notional
                else:
                    projected_cash += notional
            snapshot["projected_cash"] = projected_cash
            if projected_cash < min_cash - 1e-6:
                for order in approved:
                    order.status = "REJECTED"
                    order.reject_reason = "MIN_CASH"
                rejected.extend(approved)
                return [], rejected
        else:
            snapshot["projected_cash"] = float(cash)
        return approved, rejected

    def _halted_result(
        self,
        orders: Sequence[Order],
        reason: str | None,
        snapshot: dict[str, float],
    ) -> RiskCheckResult:
        rejected: list[Order] = []
        for order in orders:
            order.status = "REJECTED"
            order.reject_reason = reason or "HALTED"
            rejected.append(order)
        return RiskCheckResult(
            approved=[],
            rejected=rejected,
            halted=True,
            halt_reason=reason,
            snapshot=snapshot,
        )


def _maybe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["ExecutionRiskConfig", "ExecutionRiskEngine"]
