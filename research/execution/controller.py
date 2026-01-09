"""Execution controller glue between compiler, risk engine, and broker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from research.execution.broker_base import BrokerAdapter
from research.execution.compiler import OrderCompiler, OrderCompilerConfig
from research.execution.metrics import ExecutionMetricsRecorder
from research.execution.risk_engine import ExecutionRiskConfig, ExecutionRiskEngine
from research.execution.types import ExecutionStepResult, Fill, Order
from research.shadow.portfolio import PortfolioUpdate, valuate_portfolio, weights_from_holdings


@dataclass(frozen=True)
class ExecutionControllerConfig:
    """Container for controller dependencies."""

    run_id: str
    symbol_order: Sequence[str]
    order_config: OrderCompilerConfig | None = None
    risk_config: ExecutionRiskConfig | None = None


class ExecutionController:
    """Deterministic orchestration of compile -> risk check -> broker -> state."""

    def __init__(
        self,
        *,
        broker: BrokerAdapter,
        metrics: ExecutionMetricsRecorder,
        config: ExecutionControllerConfig,
    ) -> None:
        self._broker = broker
        self._metrics = metrics
        self._symbol_order = tuple(config.symbol_order)
        self._symbol_to_index = {symbol: idx for idx, symbol in enumerate(self._symbol_order)}
        self._compiler = OrderCompiler(
            run_id=config.run_id,
            symbol_order=self._symbol_order,
            config=config.order_config,
        )
        risk_config = config.risk_config or ExecutionRiskConfig()
        self._risk_engine = ExecutionRiskEngine(risk_config)

    def process_step(
        self,
        *,
        as_of: str,
        step_index: int,
        cash: float,
        holdings: Sequence[float],
        prices: Mapping[str, float],
        target_weights: Mapping[str, float],
        prev_weights: Sequence[float],
        portfolio_value: float,
        day_start_value: float,
        peak_value: float,
        regime_bucket: str | None = None,
        resume_snapshot: Mapping[str, object] | None = None,
    ) -> ExecutionStepResult:
        price_map = {symbol: float(prices[symbol]) for symbol in prices}
        if hasattr(self._broker, "update_market_data"):
            self._broker.update_market_data(price_map)
        compile_result = self._compiler.compile_orders(
            current_cash=cash,
            holdings=holdings,
            prices=price_map,
            target_weights=target_weights,
            portfolio_value=portfolio_value,
            step_index=step_index,
        )
        compiled_orders = compile_result.orders
        side_lookup = {order.client_order_id: order.side for order in compiled_orders}
        risk_result = self._risk_engine.evaluate(
            orders=compiled_orders,
            symbol_order=self._symbol_order,
            target_weights=target_weights,
            prev_weights=prev_weights,
            prices=price_map,
            cash=cash,
            portfolio_value=portfolio_value,
            day_start_value=day_start_value,
            peak_value=peak_value,
        )
        if risk_result.halted:
            self._metrics.record_halt(risk_result.halt_reason)
            self._metrics.record_step(
                compiled_orders=compiled_orders,
                submitted_orders=[],
                rejected_orders=risk_result.rejected,
                fills=[],
                reference_prices=price_map,
                side_lookup=side_lookup,
                regime_bucket=regime_bucket,
            )
            dummy_update = PortfolioUpdate(
                cash=float(cash),
                holdings=list(holdings),
                weights=list(prev_weights),
                portfolio_value=float(portfolio_value),
                turnover=0.0,
                transaction_cost=0.0,
            )
            return ExecutionStepResult(
                update=dummy_update,
                compiled_orders=compiled_orders,
                orders_submitted=[],
                orders_rejected=risk_result.rejected,
                open_orders=[],
                fills=[],
                risk_snapshot=risk_result.snapshot,
                broker_errors=[],
                halted=True,
                halt_reason=risk_result.halt_reason,
            )
        approved_orders = risk_result.approved
        submission = self._broker.submit_orders(approved_orders, as_of=as_of) if approved_orders else None
        broker_rejects: list[Order] = submission.rejected if submission else []
        submitted_orders = submission.accepted if submission else []
        fills: list[Fill] = []
        if submitted_orders:
            broker_ids = [order.broker_order_id or order.client_order_id for order in submitted_orders]
            fills = self._broker.fetch_fills(broker_ids, as_of=as_of)
        orders_rejected = list(risk_result.rejected) + list(broker_rejects)
        self._metrics.record_step(
            compiled_orders=compiled_orders,
            submitted_orders=submitted_orders,
            rejected_orders=orders_rejected,
            fills=fills,
            reference_prices=price_map,
            side_lookup=side_lookup,
            regime_bucket=regime_bucket,
        )
        total_fees = sum(fill.fees for fill in fills)
        new_cash, new_holdings = self._apply_fills(cash, holdings, fills, side_lookup)
        price_vector = [price_map.get(symbol, 0.0) for symbol in self._symbol_order]
        new_portfolio_value = valuate_portfolio(new_cash, new_holdings, price_vector)
        new_weights = weights_from_holdings(new_holdings, price_vector, new_portfolio_value)
        turnover = sum(abs(new_weights[idx] - prev_weights[idx]) for idx in range(len(self._symbol_order)))
        self._metrics.record_turnover(turnover)
        update = PortfolioUpdate(
            cash=float(new_cash),
            holdings=new_holdings,
            weights=new_weights,
            portfolio_value=float(new_portfolio_value),
            turnover=float(turnover),
            transaction_cost=float(total_fees),
        )
        open_orders = [order for order in submitted_orders if order.status not in {"FILLED", "CANCELED", "REJECTED"}]
        return ExecutionStepResult(
            update=update,
            compiled_orders=compiled_orders,
            orders_submitted=submitted_orders,
            orders_rejected=orders_rejected,
            open_orders=open_orders,
            fills=fills,
            risk_snapshot=risk_result.snapshot,
            broker_errors=submission.errors if submission else [],
            halted=False,
            halt_reason=None,
        )

    def _apply_fills(
        self,
        cash: float,
        holdings: Sequence[float],
        fills: Sequence[Fill],
        side_lookup: Mapping[str, str],
    ) -> tuple[float, list[float]]:
        updated = list(float(value) for value in holdings)
        cash_balance = float(cash)
        for fill in fills:
            side = side_lookup.get(fill.client_order_id)
            if side is None:
                continue
            idx = self._symbol_to_index.get(fill.symbol)
            if idx is None:
                continue
            qty = int(fill.qty)
            if side == "BUY":
                updated[idx] += qty
                cash_balance -= qty * float(fill.price) + float(fill.fees)
            else:
                updated[idx] -= qty
                cash_balance += qty * float(fill.price) - float(fill.fees)
        return cash_balance, updated


__all__ = ["ExecutionController", "ExecutionControllerConfig"]
