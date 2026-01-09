"""Paper execution runner scaffolding."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, TYPE_CHECKING

from research.execution import (
    ExecutionController,
    ExecutionControllerConfig,
    ExecutionMetricsRecorder,
    ExecutionRiskConfig,
)
from research.execution.types import ExecutionStepResult, Order
from research.paper.config import PaperRunConfig, PollingConfig, RiskLimitConfig
from research.paper.reconcile import PaperReconciler
from research.paper.summary import ExecutionGateRunner
from research.shadow.portfolio import PortfolioUpdate, valuate_portfolio, weights_from_holdings
from research.shadow.state_store import StateStore

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance
    from research.shadow.engine import ShadowEngine


def derive_run_id(config: PaperRunConfig) -> str:
    """Derive a deterministic run identifier from config content."""

    payload = {
        "experiment_id": config.experiment_id,
        "execution_mode": config.execution_mode,
        "universe": sorted(config.universe),
        "broker": config.broker.alpaca_base_url,
        "risk_limits": json.dumps(config.to_dict().get("risk_limits"), sort_keys=True),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"paper_{digest[:12]}"


class PaperRunner:
    """Thin wrapper around the shadow engine for paper validation."""

    def __init__(self, config: PaperRunConfig, *, run_id: str | None = None, scheduled_for: str | None = None) -> None:
        self.config = config
        self.base_run_id = derive_run_id(config)
        self.run_id = run_id or self.base_run_id
        self.scheduled_for = scheduled_for
        self.run_dir = Path(config.artifacts.output_root) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config.write_run_config(self.run_dir)

    def build_state_store(self) -> StateStore:
        base_dir = self.run_dir / "state"
        return StateStore(self.config.experiment_id, run_id=self.run_id, base_dir=base_dir)

    def build_execution_options(self) -> Mapping[str, Any]:
        """Translate config knobs into ShadowEngine execution options."""

        return {
            "order_config": {"min_notional": 1.0},
            "risk_overrides": {
                "max_gross_exposure": self.config.risk_limits.max_gross_exposure,
                "max_symbol_weight": self.config.risk_limits.max_symbol_weight,
                "min_cash_pct": self.config.risk_limits.min_cash_pct,
                "max_daily_turnover": self.config.risk_limits.max_turnover,
                "max_trailing_drawdown": self.config.risk_limits.max_drawdown,
            },
            "alpaca": {
                "base_url": self.config.broker.alpaca_base_url,
            },
            "polling": {
                "max_poll_seconds": self.config.polling.max_poll_seconds,
                "poll_interval_seconds": self.config.polling.poll_interval_seconds,
            },
            "reconciliation": {
                "position_tolerance_shares": self.config.reconciliation.position_tolerance_shares,
                "cash_tolerance_usd": self.config.reconciliation.cash_tolerance_usd,
            },
        }


class PaperExecutionController(ExecutionController):
    """Execution controller with broker reconciliation + resume safety."""

    def __init__(
        self,
        *,
        broker,
        metrics: ExecutionMetricsRecorder,
        config: ExecutionControllerConfig,
        reconciler: PaperReconciler,
        polling: PollingConfig,
        gate_runner: ExecutionGateRunner | None = None,
    ) -> None:
        super().__init__(broker=broker, metrics=metrics, config=config)
        self._reconciler = reconciler
        self._polling = polling
        self._gate_runner = gate_runner

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
        resume_snapshot = resume_snapshot or {}
        submitted_ids = set(str(value) for value in resume_snapshot.get("submitted_order_ids", []))
        broker_id_map = {str(key): str(value) for key, value in (resume_snapshot.get("broker_order_map") or {}).items()}
        broker_errors: list[str] = []
        try:
            account_snapshot = self._broker.get_account(as_of=as_of)
            position_snapshots = self._broker.get_positions(as_of=as_of)
        except Exception as exc:  # pragma: no cover - broker failure path
            error = self._broker.normalize_error(exc)
            broker_errors.append(error)
            return self._halt(
                reason=error,
                cash=cash,
                holdings=holdings,
                prev_weights=prev_weights,
                portfolio_value=portfolio_value,
                broker_errors=broker_errors,
            )
        holdings_map = {
            symbol: float(holdings[idx]) if idx < len(holdings) else 0.0
            for idx, symbol in enumerate(self._symbol_order)
        }
        reconcile_result = self._reconciler.reconcile(
            internal_holdings=holdings_map,
            broker_positions=position_snapshots,
            cash=cash,
            account=account_snapshot,
        )
        if not reconcile_result.matched:
            broker_errors.append(reconcile_result.reason or "reconciliation_error")
            halted = self._halt(
                reason=reconcile_result.reason or "reconciliation_error",
                cash=cash,
                holdings=holdings,
                prev_weights=prev_weights,
                portfolio_value=portfolio_value,
                broker_errors=broker_errors,
            )
            halted.account_snapshot = account_snapshot
            halted.position_snapshots = list(position_snapshots)
            return halted

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
            halted = self._halt(
                reason=risk_result.halt_reason or "risk_halt",
                cash=cash,
                holdings=holdings,
                prev_weights=prev_weights,
                portfolio_value=portfolio_value,
                broker_errors=[],
            )
            halted.risk_snapshot = dict(risk_result.snapshot)
            return halted

        approved_orders = risk_result.approved
        previously_submitted = [order for order in approved_orders if order.client_order_id in submitted_ids]
        new_orders = [order for order in approved_orders if order.client_order_id not in submitted_ids]
        submission = self._broker.submit_orders(new_orders, as_of=as_of) if new_orders else None
        broker_rejects = submission.rejected if submission else []
        newly_submitted = submission.accepted if submission else []
        submitted_orders = previously_submitted + newly_submitted
        for order in newly_submitted:
            if order.broker_order_id:
                broker_id_map[order.client_order_id] = order.broker_order_id
            else:
                broker_id_map.setdefault(order.client_order_id, order.client_order_id)
        broker_ids = [
            broker_id_map.get(order.client_order_id) or order.broker_order_id or order.client_order_id
            for order in submitted_orders
        ]
        status_views = []
        if broker_ids:
            try:
                status_views = self._broker.poll_orders(broker_ids, as_of=as_of)
            except Exception as exc:  # pragma: no cover - broker failure path
                broker_errors.append(self._broker.normalize_error(exc))
        status_by_client = {view.client_order_id: view for view in status_views}
        open_orders: list[Order] = []
        for order in submitted_orders:
            view = status_by_client.get(order.client_order_id)
            if view:
                order.status = view.status
                order.broker_order_id = view.broker_order_id
                order.reject_reason = view.reject_reason
            if order.status not in {"FILLED", "CANCELED", "REJECTED"}:
                open_orders.append(order)
        fills = self._broker.fetch_fills(broker_ids, as_of=as_of) if broker_ids else []
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
        gate_report = None
        if self._gate_runner:
            gate_report = self._gate_runner.evaluate(self._metrics.snapshot())
            if gate_report.get("status") == "HALTED":
                self._metrics.record_halt("execution_gate_failure")
        halted = gate_report.get("status") == "HALTED" if gate_report else False
        halt_reason = "execution_gate_failure" if halted else None
        return ExecutionStepResult(
            update=update,
            compiled_orders=compiled_orders,
            orders_submitted=submitted_orders,
            orders_rejected=orders_rejected,
            open_orders=open_orders,
            fills=fills,
            risk_snapshot=dict(risk_result.snapshot),
            broker_errors=broker_errors,
            halted=halted,
            halt_reason=halt_reason,
            client_broker_map=broker_id_map,
            account_snapshot=account_snapshot,
            position_snapshots=list(position_snapshots),
            gate_report=gate_report,
        )

    def _halt(
        self,
        *,
        reason: str | None,
        cash: float,
        holdings: Sequence[float],
        prev_weights: Sequence[float],
        portfolio_value: float,
        broker_errors: Sequence[str],
    ) -> ExecutionStepResult:
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
            compiled_orders=[],
            orders_submitted=[],
            orders_rejected=[],
            open_orders=[],
            fills=[],
            risk_snapshot={},
            broker_errors=list(broker_errors),
            halted=True,
            halt_reason=reason,
            client_broker_map=None,
            account_snapshot=None,
            position_snapshots=None,
            gate_report=None,
        )


def build_paper_execution_controller(
    *,
    broker,
    metrics: ExecutionMetricsRecorder,
    symbol_order: Sequence[str],
    risk_limits: RiskLimitConfig,
    polling: PollingConfig,
    reconciler: PaperReconciler,
    gate_runner: ExecutionGateRunner | None = None,
) -> PaperExecutionController:
    """Convenience helper mirroring ExecutionController construction."""

    controller_config = ExecutionControllerConfig(
        run_id="paper",
        symbol_order=symbol_order,
        order_config=None,
        risk_config=ExecutionRiskConfig(
            max_gross_exposure=risk_limits.max_gross_exposure,
            min_cash_pct=risk_limits.min_cash_pct,
            max_symbol_weight=risk_limits.max_symbol_weight,
            max_daily_turnover=risk_limits.max_turnover,
            max_trailing_drawdown=risk_limits.max_drawdown,
        ),
    )
    return PaperExecutionController(
        broker=broker,
        metrics=metrics,
        config=controller_config,
        reconciler=reconciler,
        polling=polling,
        gate_runner=gate_runner,
    )


__all__ = ["PaperExecutionController", "PaperRunner", "derive_run_id", "build_paper_execution_controller"]
