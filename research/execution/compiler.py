"""Target weights -> deterministic market orders."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from research.execution.orders import (
    ORDER_SIDE_RANK,
    OrderCandidate,
    deterministic_order_id,
    floor_shares,
    stable_symbol_weights,
)
from research.execution.types import Order


@dataclass(frozen=True)
class OrderCompilerConfig:
    """Configuration for deterministic order compilation."""

    min_notional: float = 1.0


@dataclass
class OrderCompilerDiagnostics:
    """Metadata describing compiler side-effects."""

    min_notional_skipped: list[dict[str, object]] = field(default_factory=list)
    cash_shortfall: list[dict[str, object]] = field(default_factory=list)


@dataclass
class OrderCompilerResult:
    """Compiler output along with diagnostics."""

    orders: list[Order]
    diagnostics: OrderCompilerDiagnostics


class OrderCompiler:
    """Deterministic translation from target weights to market orders."""

    def __init__(
        self,
        *,
        run_id: str,
        symbol_order: Sequence[str],
        config: OrderCompilerConfig | None = None,
    ) -> None:
        self._run_id = run_id
        self._symbol_order = tuple(symbol_order)
        self._config = config or OrderCompilerConfig()

    def compile_orders(
        self,
        *,
        current_cash: float,
        holdings: Sequence[float],
        prices: Mapping[str, float],
        target_weights: Mapping[str, float],
        portfolio_value: float,
        step_index: int,
    ) -> OrderCompilerResult:
        price_map = {symbol: float(prices[symbol]) for symbol in prices}
        weights = stable_symbol_weights(target_weights, self._symbol_order)
        candidates, min_skip = self._build_candidates(holdings, price_map, weights, portfolio_value)
        buys = [order for order in candidates if order.side == "BUY"]
        sells = [order for order in candidates if order.side == "SELL"]
        diagnostics = OrderCompilerDiagnostics(min_notional_skipped=min_skip)
        if buys:
            buys, cash_diag = self._enforce_cash(
                buys,
                current_cash=current_cash,
                sell_proceeds=sum(entry.notional for entry in sells),
            )
            diagnostics.cash_shortfall.extend(cash_diag)
        orders = buys + sells
        materialized = self._materialize_orders(orders, step_index)
        return OrderCompilerResult(orders=materialized, diagnostics=diagnostics)

    def _build_candidates(
        self,
        holdings: Sequence[float],
        prices: Mapping[str, float],
        weights: Mapping[str, float],
        portfolio_value: float,
    ) -> tuple[list[OrderCandidate], list[dict[str, object]]]:
        candidates: list[OrderCandidate] = []
        skipped: list[dict[str, object]] = []
        min_notional = max(self._config.min_notional, 0.0)
        for idx, symbol in enumerate(self._symbol_order):
            price = float(prices.get(symbol, 0.0))
            if price <= 0:
                continue
            target_weight = float(weights.get(symbol, 0.0))
            target_notional = target_weight * portfolio_value
            holding = float(holdings[idx]) if idx < len(holdings) else 0.0
            current_notional = holding * price
            delta_notional = target_notional - current_notional
            qty = floor_shares(delta_notional / price)
            if qty <= 0:
                continue
            notional = qty * price
            if notional < min_notional:
                skipped.append(
                    {
                        "symbol": symbol,
                        "reason": "min_notional",
                        "qty": qty,
                        "notional": notional,
                    }
                )
                continue
            side = "BUY" if delta_notional > 0 else "SELL"
            candidate = OrderCandidate(
                symbol=symbol,
                side=side,
                qty=qty,
                price=price,
                delta_notional=delta_notional,
            )
            candidates.append(candidate)
        candidates.sort(key=lambda entry: (entry.symbol, ORDER_SIDE_RANK.get(entry.side, 9)))
        return candidates, skipped

    def _enforce_cash(
        self,
        buys: list[OrderCandidate],
        *,
        current_cash: float,
        sell_proceeds: float,
    ) -> tuple[list[OrderCandidate], list[dict[str, object]]]:
        available_cash = max(float(current_cash) + float(sell_proceeds), 0.0)
        if available_cash <= 0:
            return [], [{"reason": "insufficient_cash", "symbol": entry.symbol} for entry in buys]
        shortfall: list[dict[str, object]] = []
        ordered = sorted(buys, key=lambda entry: (-entry.notional, entry.symbol))
        approved: list[OrderCandidate] = []
        for entry in ordered:
            max_affordable_qty = int(math.floor((available_cash + 1e-9) / entry.price))
            if max_affordable_qty <= 0:
                shortfall.append({"reason": "insufficient_cash", "symbol": entry.symbol})
                continue
            qty = min(entry.qty, max_affordable_qty)
            available_cash -= qty * entry.price
            if qty <= 0:
                shortfall.append({"reason": "insufficient_cash", "symbol": entry.symbol})
                continue
            approved.append(
                OrderCandidate(
                    symbol=entry.symbol,
                    side=entry.side,
                    qty=qty,
                    price=entry.price,
                    delta_notional=entry.delta_notional,
                )
            )
            if qty < entry.qty:
                shortfall.append({"reason": "partial_fill_due_to_cash", "symbol": entry.symbol})
        approved.sort(key=lambda entry: (entry.symbol, ORDER_SIDE_RANK.get(entry.side, 9)))
        return approved, shortfall

    def _materialize_orders(self, orders: list[OrderCandidate], step_index: int) -> list[Order]:
        materialized: list[Order] = []
        for entry in orders:
            client_order_id = deterministic_order_id(
                run_id=self._run_id,
                step_index=step_index,
                symbol=entry.symbol,
                side=entry.side,
                qty=entry.qty,
            )
            materialized.append(
                Order(
                    client_order_id=client_order_id,
                    symbol=entry.symbol,
                    side=entry.side,  # type: ignore[arg-type]
                    qty=int(entry.qty),
                )
            )
        return materialized


__all__ = [
    "OrderCompiler",
    "OrderCompilerConfig",
    "OrderCompilerDiagnostics",
    "OrderCompilerResult",
]
