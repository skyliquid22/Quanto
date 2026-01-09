from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pytest

from research.execution.broker_base import BrokerAdapter
from research.execution import ExecutionControllerConfig, ExecutionMetricsRecorder
from research.execution.risk_engine import ExecutionRiskConfig
from research.execution.types import (
    AccountSnapshot,
    Fill,
    Order,
    OrderStatusView,
    PositionSnapshot,
    SubmissionResult,
)
from research.paper.config import PollingConfig, ReconciliationConfig, load_paper_config
from research.paper.reconcile import PaperReconciler
from research.paper.run import PaperExecutionController
from research.paper.summary import DailySummaryWriter, ExecutionGateRunner, GateThresholds


def test_paper_config_validation(tmp_path, monkeypatch):
    data_root = tmp_path / "quanto_data"
    monkeypatch.setenv("QUANTO_DATA_ROOT", str(data_root))
    promotion_root = data_root / "promotions" / "research"
    promotion_root.mkdir(parents=True, exist_ok=True)
    experiment_id = "exp_paper"
    (promotion_root / f"{experiment_id}.json").write_text("{}", encoding="utf-8")
    config_path = tmp_path / "config.json"
    payload = {
        "experiment_id": experiment_id,
        "execution_mode": "alpaca_paper",
        "universe": ["AAA", "BBB"],
        "broker": {"alpaca_base_url": "https://paper-api.alpaca.markets"},
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    config = load_paper_config(config_path, promotion_root=promotion_root.parent)
    assert config.experiment_id == experiment_id
    payload["broker"]["alpaca_base_url"] = "https://api.alpaca.markets"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_paper_config(config_path, promotion_root=promotion_root.parent)
    payload["broker"]["alpaca_base_url"] = "https://paper-api.alpaca.markets"
    payload["universe"] = ["A", "B", "C", "D", "E", "F"]
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_paper_config(config_path, promotion_root=promotion_root.parent)


def test_paper_idempotent_submission():
    controller = _build_controller()
    prices = {"AAA": 10.0}
    target_weights = {"AAA": 1.0}
    base_kwargs = dict(
        as_of="2024-01-02T00:00:00+00:00",
        step_index=0,
        cash=1_000.0,
        holdings=[0.0],
        prices=prices,
        target_weights=target_weights,
        prev_weights=[0.0],
        portfolio_value=1_000.0,
        day_start_value=1_000.0,
        peak_value=1_000.0,
    )
    first = controller.process_step(**base_kwargs)
    resume_snapshot = {
        "submitted_order_ids": [order.client_order_id for order in first.orders_submitted],
        "broker_order_map": dict(first.client_broker_map or {}),
    }
    second = controller.process_step(**base_kwargs, resume_snapshot=resume_snapshot)
    broker: MockBroker = controller._broker  # type: ignore[attr-defined]
    assert broker.submission_calls == 1
    assert first.halted is False
    assert second.halted is False


def test_reconciliation_behavior():
    reconciler = PaperReconciler(ReconciliationConfig(position_tolerance_shares=0.5, cash_tolerance_usd=2.0))
    holdings = {"AAA": 10.0}
    positions = [PositionSnapshot(symbol="AAA", qty=10.1, avg_price=10.0, market_price=10.0, market_value=101.0)]
    account = AccountSnapshot(as_of="t", equity=1_000.0, cash=1_000.0)
    result = reconciler.reconcile(internal_holdings=holdings, broker_positions=positions, cash=1_000.0, account=account)
    assert result.matched
    positions = [PositionSnapshot(symbol="AAA", qty=12.0, avg_price=10.0, market_price=10.0, market_value=120.0)]
    result = reconciler.reconcile(internal_holdings=holdings, broker_positions=positions, cash=1_000.0, account=account)
    assert not result.matched
    assert "position_mismatch" in (result.reason or "")


def test_daily_artifacts_creation(tmp_path):
    writer = DailySummaryWriter(tmp_path)
    payload = {
        "pnl": 12.3,
        "exposure": 0.4,
        "turnover": 0.1,
        "fees": 0.0,
        "halt_reasons": [],
    }
    json_path, md_path = writer.write("20240101", payload)
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert summary["pnl"] == 12.3
    assert md_path.exists()


def test_gate_runner_halting():
    gate = ExecutionGateRunner(GateThresholds(max_reject_rate=0.0))
    metrics = {"summary": {"reject_rate": 0.5, "execution_halts": 0}}
    report = gate.evaluate(metrics)
    assert report["status"] == "HALTED"


class MockBroker(BrokerAdapter):
    """Deterministic broker stub for controller tests."""

    def __init__(self) -> None:
        self.submission_calls = 0

    def submit_orders(self, orders: Sequence[Order], *, as_of: str) -> SubmissionResult:
        self.submission_calls += 1
        accepted = []
        for order in orders:
            accepted.append(
                Order(
                    client_order_id=order.client_order_id,
                    symbol=order.symbol,
                    side=order.side,
                    qty=order.qty,
                    broker_order_id=f"brok_{order.client_order_id}",
                    status="SUBMITTED",
                )
            )
        return SubmissionResult(accepted=accepted, rejected=[])

    def poll_orders(self, broker_order_ids: Sequence[str], *, as_of: str) -> list[OrderStatusView]:
        views: list[OrderStatusView] = []
        for broker_id in broker_order_ids:
            views.append(
                OrderStatusView(
                    client_order_id=broker_id.split("_", 1)[-1] if "_" in broker_id else broker_id,
                    broker_order_id=broker_id,
                    status="FILLED",
                    filled_qty=100,
                    remaining_qty=0,
                )
            )
        return views

    def fetch_fills(self, broker_order_ids: Sequence[str], *, as_of: str) -> list[Fill]:
        fills: list[Fill] = []
        for broker_id in broker_order_ids:
            fills.append(
                Fill(
                    client_order_id=broker_id.split("_", 1)[-1] if "_" in broker_id else broker_id,
                    symbol="AAA",
                    qty=100,
                    price=10.0,
                    fees=0.0,
                    filled_at=as_of,
                    broker_trade_id=f"fill_{broker_id}",
                )
            )
        return fills

    def get_account(self, *, as_of: str) -> AccountSnapshot:
        return AccountSnapshot(as_of=as_of, equity=1_000.0, cash=1_000.0)

    def get_positions(self, *, as_of: str) -> list[PositionSnapshot]:
        return [PositionSnapshot(symbol="AAA", qty=0.0, avg_price=0.0, market_price=0.0, market_value=0.0)]


def _build_controller() -> PaperExecutionController:
    broker = MockBroker()
    metrics = ExecutionMetricsRecorder()
    risk = ExecutionRiskConfig(max_gross_exposure=1.0, max_symbol_weight=1.0, min_cash_pct=0.0, max_daily_turnover=1.0)
    controller_config = ExecutionControllerConfig(run_id="paper", symbol_order=("AAA",), risk_config=risk)
    reconciler = PaperReconciler(ReconciliationConfig())
    controller = PaperExecutionController(
        broker=broker,
        metrics=metrics,
        config=controller_config,
        reconciler=reconciler,
        polling=PollingConfig(),
        gate_runner=None,
    )
    return controller
