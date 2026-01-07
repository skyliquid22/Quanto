from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from research.execution.compiler import OrderCompiler, OrderCompilerConfig
from research.execution.risk_engine import ExecutionRiskConfig, ExecutionRiskEngine
from research.experiments.registry import ExperimentRegistry
from research.experiments.spec import ExperimentSpec
from research.shadow.data_source import ReplayMarketDataSource
from research.shadow.engine import ShadowEngine
from research.shadow.logging import ShadowLogger
from research.shadow.state_store import StateStore


def test_order_compiler_determinism():
    compiler = OrderCompiler(run_id="run_x", symbol_order=("AAA", "BBB"), config=OrderCompilerConfig(min_notional=0.01))
    orders_a = compiler.compile_orders(
        current_cash=1_000.0,
        holdings=[10.0, 5.0],
        prices={"AAA": 10.0, "BBB": 20.0},
        target_weights={"AAA": 0.6, "BBB": 0.4},
        portfolio_value=2_000.0,
        step_index=3,
    ).orders
    orders_b = compiler.compile_orders(
        current_cash=1_000.0,
        holdings=[10.0, 5.0],
        prices={"AAA": 10.0, "BBB": 20.0},
        target_weights={"AAA": 0.6, "BBB": 0.4},
        portfolio_value=2_000.0,
        step_index=3,
    ).orders
    payload_a = json.dumps([order.to_dict() for order in orders_a], sort_keys=True)
    payload_b = json.dumps([order.to_dict() for order in orders_b], sort_keys=True)
    assert payload_a == payload_b


def test_sim_execution_determinism(shadow_env):
    shadow_env.reset_run_dir()
    run_once = shadow_env.build_engine(execution_mode="sim")
    summary_one = run_once.run()
    state_one = Path(summary_one["state_path"]).read_text(encoding="utf-8")
    log_one = Path(summary_one["log_path"]).read_text(encoding="utf-8")

    shadow_env.reset_run_dir()
    run_twice = shadow_env.build_engine(execution_mode="sim")
    summary_two = run_twice.run()
    state_two = Path(summary_two["state_path"]).read_text(encoding="utf-8")
    log_two = Path(summary_two["log_path"]).read_text(encoding="utf-8")

    assert state_one == state_two
    assert log_one == log_two


def test_execution_resume_safety(shadow_env):
    shadow_env.reset_run_dir()
    baseline = shadow_env.build_engine(execution_mode="sim")
    baseline.run()
    state_baseline = Path(baseline.state_store.state_path).read_text(encoding="utf-8")

    shadow_env.reset_run_dir()
    partial = shadow_env.build_engine(execution_mode="sim")
    partial.step()
    partial.step()
    resumed = shadow_env.build_engine(execution_mode="sim")
    resumed.run()
    final_state = Path(resumed.state_store.state_path).read_text(encoding="utf-8")
    assert state_baseline == final_state

    log_payload = Path(resumed.logger.steps_path).read_text(encoding="utf-8").strip().splitlines()
    order_ids: set[str] = set()
    for line in log_payload:
        entry = json.loads(line)
        for order in entry["execution"]["orders_submitted"]:
            cid = order["client_order_id"]
            assert cid not in order_ids
            order_ids.add(cid)


def test_risk_engine_determinism():
    config = ExecutionRiskConfig(max_gross_exposure=0.5, max_symbol_weight=0.3, min_cash_pct=0.1)
    engine = ExecutionRiskEngine(config)
    orders = shadow_orders()
    target_weights = {"AAA": 0.4, "BBB": 0.2}
    prev_weights = [0.1, 0.1]
    prices = {"AAA": 10.0, "BBB": 20.0}
    result_one = engine.evaluate(
        orders=orders,
        symbol_order=("AAA", "BBB"),
        target_weights=target_weights,
        prev_weights=prev_weights,
        prices=prices,
        cash=1_000.0,
        portfolio_value=2_000.0,
        day_start_value=2_000.0,
        peak_value=2_000.0,
    )
    result_two = engine.evaluate(
        orders=shadow_orders(),
        symbol_order=("AAA", "BBB"),
        target_weights=target_weights,
        prev_weights=prev_weights,
        prices=prices,
        cash=1_000.0,
        portfolio_value=2_000.0,
        day_start_value=2_000.0,
        peak_value=2_000.0,
    )
    assert [order.client_order_id for order in result_one.approved] == [
        order.client_order_id for order in result_two.approved
    ]
    assert [order.reject_reason for order in result_one.rejected] == [
        order.reject_reason for order in result_two.rejected
    ]
    assert result_one.snapshot == result_two.snapshot


@pytest.fixture
def shadow_env(tmp_path, monkeypatch):
    data_root = tmp_path / "quanto_data"
    monkeypatch.setenv("QUANTO_DATA_ROOT", str(data_root))
    symbols = ("AAA", "BBB")
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    for idx, symbol in enumerate(symbols):
        _write_canonical_year(
            data_root,
            symbol,
            year=2020,
            rows=_generate_rows(symbol, start, 12, price_offset=1 + idx),
        )
    spec_payload = {
        "experiment_name": "shadow_exec",
        "symbols": list(symbols),
        "start_date": "2020-01-01",
        "end_date": "2020-01-12",
        "interval": "daily",
        "feature_set": "sma_v1",
        "policy": "sma",
        "policy_params": {"fast_window": 1, "slow_window": 2, "policy_mode": "hard", "sigmoid_scale": 1.0},
        "cost_config": {"transaction_cost_bp": 0.0},
        "risk_config": {"max_weight": 0.7, "exposure_cap": 1.0, "min_cash": 0.0},
        "seed": 3,
    }
    spec = ExperimentSpec.from_mapping(spec_payload)
    registry_root = data_root / "experiments"
    registry = ExperimentRegistry(root=registry_root)
    registry_base = registry_root / spec.experiment_id
    _write_registry_artifacts(registry_base, spec)
    promo_root = data_root / "promotions"
    _write_promotion_record(promo_root, spec.experiment_id, registry_base)
    data_source = ReplayMarketDataSource(
        spec=spec,
        start_date=start.date(),
        end_date=(start + timedelta(days=11)).date(),
        data_root=data_root,
    )
    run_id = _derive_run_id(spec.experiment_id, data_source.window[0], data_source.window[1])
    state_base = data_root / "shadow"
    run_dir = state_base / spec.experiment_id / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)

    class Env:
        def build_engine(self, *, execution_mode: str = "none") -> ShadowEngine:
            logger = ShadowLogger(run_dir)
            state_store = StateStore(spec.experiment_id, run_id=run_id, base_dir=state_base)
            ds = ReplayMarketDataSource(
                spec=spec,
                start_date=start.date(),
                end_date=(start + timedelta(days=11)).date(),
                data_root=data_root,
            )
            return ShadowEngine(
                experiment_id=spec.experiment_id,
                data_source=ds,
                state_store=state_store,
                logger=logger,
                run_id=run_id,
                out_dir=run_dir,
                registry=registry,
                promotion_root=promo_root,
                execution_mode=execution_mode,
                execution_options={
                    "order_config": {"min_notional": 0.0},
                    "risk_overrides": {"max_active_positions": len(symbols)},
                },
            )

        def reset_run_dir(self) -> None:
            if run_dir.exists():
                shutil.rmtree(run_dir)

    env = Env()
    return env


def shadow_orders():
    compiler = OrderCompiler(run_id="risk", symbol_order=("AAA", "BBB"), config=OrderCompilerConfig(min_notional=0.0))
    return compiler.compile_orders(
        current_cash=1_000.0,
        holdings=[0.0, 0.0],
        prices={"AAA": 10.0, "BBB": 20.0},
        target_weights={"AAA": 0.4, "BBB": 0.2},
        portfolio_value=1_000.0,
        step_index=0,
    ).orders


def _write_canonical_year(data_root: Path, symbol: str, *, year: int, rows):
    base = data_root / "canonical" / "equity_ohlcv" / symbol / "daily"
    base.mkdir(parents=True, exist_ok=True)
    shard = base / f"{year}.parquet"
    shard.write_text(json.dumps(list(rows)), encoding="utf-8")


def _generate_rows(symbol: str, start: datetime, num_days: int, price_offset: int):
    rows = []
    for idx in range(num_days):
        timestamp = start + timedelta(days=idx)
        price = price_offset * 10 + idx
        rows.append(
            {
                "symbol": symbol,
                "timestamp": timestamp.isoformat(),
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price,
                "volume": 1_000 + idx,
            }
        )
    return rows


def _write_registry_artifacts(base: Path, spec: ExperimentSpec) -> None:
    (base / "spec").mkdir(parents=True, exist_ok=True)
    (base / "evaluation").mkdir(parents=True, exist_ok=True)
    (base / "logs").mkdir(parents=True, exist_ok=True)
    (base / "promotion").mkdir(parents=True, exist_ok=True)
    (base / "spec" / "experiment_spec.json").write_text(spec.canonical_json, encoding="utf-8")
    metrics_payload = {
        "metadata": {"run_id": spec.experiment_id},
        "performance": {"total_return": 0.1, "cagr": None, "volatility_ann": 0.2, "sharpe": None, "max_drawdown": 0.05},
        "trading": {"turnover_1d_mean": 0.0, "turnover_1d_median": 0.0, "avg_exposure": 1.0, "max_concentration": 0.5},
        "safety": {
            "nan_inf_violations": 0.0,
            "action_bounds_violations": 0.0,
            "constraint_violations_count": 0.0,
            "max_weight_violation_count": 0.0,
            "exposure_violation_count": 0.0,
            "turnover_violation_count": 0.0,
        },
    }
    (base / "evaluation" / "metrics.json").write_text(json.dumps(metrics_payload), encoding="utf-8")
    (base / "logs" / "run_summary.json").write_text(json.dumps({"recorded_at": "2024-01-01T00:00:00Z"}), encoding="utf-8")


def _write_promotion_record(promotion_root: Path, experiment_id: str, registry_base: Path) -> None:
    promo_dir = promotion_root / "candidate"
    promo_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": experiment_id,
        "tier": "candidate",
        "qualification_report_path": str(registry_base / "promotion" / "qualification_report.json"),
        "spec_path": str(registry_base / "spec" / "experiment_spec.json"),
        "metrics_path": str(registry_base / "evaluation" / "metrics.json"),
        "promotion_reason": "tests",
    }
    (registry_base / "promotion" / "qualification_report.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
    (promo_dir / f"{experiment_id}.json").write_text(json.dumps(payload), encoding="utf-8")


def _derive_run_id(experiment_id: str, start: str, end: str) -> str:
    payload = {
        "experiment_id": experiment_id,
        "window_start": start,
        "window_end": end,
        "mode": "replay",
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"replay_{digest[:12]}"
