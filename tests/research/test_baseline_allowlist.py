from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from research.experiments.registry import ExperimentRegistry
from research.experiments.spec import ExperimentSpec
from research.promotion import baseline_allowlist
from research.shadow.data_source import ReplayMarketDataSource
from research.shadow.engine import ShadowEngine
from research.shadow.logging import ShadowLogger
from research.shadow.state_store import StateStore
from tests.research.test_execution_layer import (
    _derive_run_id,
    _generate_rows,
    _write_canonical_year,
    _write_registry_artifacts,
)


@pytest.fixture
def baseline_env(tmp_path, monkeypatch):
    data_root = tmp_path / "quanto_data"
    monkeypatch.setenv("QUANTO_DATA_ROOT", str(data_root))
    symbols = ("AAA", "BBB")
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    for idx, symbol in enumerate(symbols):
        _write_canonical_year(
            data_root,
            symbol,
            year=2020,
            rows=_generate_rows(symbol, start, 8, price_offset=1 + idx),
        )
    spec_payload = {
        "experiment_name": "baseline_exec",
        "symbols": list(symbols),
        "start_date": "2020-01-01",
        "end_date": "2020-01-08",
        "interval": "daily",
        "feature_set": "sma_v1",
        "policy": "sma",
        "policy_params": {"fast_window": 1, "slow_window": 2, "policy_mode": "hard", "sigmoid_scale": 1.0},
        "cost_config": {"transaction_cost_bp": 0.0},
        "risk_config": {"max_weight": 0.7, "exposure_cap": 1.0, "min_cash": 0.0},
        "seed": 5,
    }
    spec = ExperimentSpec.from_mapping(spec_payload)
    registry_root = data_root / "experiments"
    registry = ExperimentRegistry(root=registry_root)
    registry_base = registry_root / spec.experiment_id
    _write_registry_artifacts(registry_base, spec)
    data_source = ReplayMarketDataSource(
        spec=spec,
        start_date=start.date(),
        end_date=(start + timedelta(days=7)).date(),
        data_root=data_root,
    )
    run_id = _derive_run_id(spec.experiment_id, data_source.window[0], data_source.window[1])
    state_root = data_root / "shadow"
    run_dir = state_root / spec.experiment_id / run_id

    class Env:
        def add_allowlist(self, reason: str = "baseline execution metrics") -> Path:
            return baseline_allowlist.add(
                spec.experiment_id,
                reason=reason,
                root=data_root / "baseline_allowlist",
            )

        def add_qualification_allowlist(self, reason: str = "qualification replay") -> Path:
            path = data_root / "qualification_allowlist" / f"{spec.experiment_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "experiment_id": spec.experiment_id,
                "reason": reason,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return path

        def build_engine(
            self,
            *,
            replay_mode: bool = True,
            live_mode: bool = False,
            qualification_replay: bool = False,
            qualification_reason: str | None = None,
        ) -> ShadowEngine:
            state_store = StateStore(
                spec.experiment_id,
                run_id=run_id,
                base_dir=state_root,
            )
            logger = ShadowLogger(run_dir)
            ds = ReplayMarketDataSource(
                spec=spec,
                start_date=start.date(),
                end_date=(start + timedelta(days=7)).date(),
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
                promotion_root=data_root / "promotions",
                replay_mode=replay_mode,
                live_mode=live_mode,
                baseline_allowlist_root=data_root / "baseline_allowlist",
                qualification_allowlist_root=data_root / "qualification_allowlist",
                qualification_replay_allowed=qualification_replay,
                qualification_allow_reason=qualification_reason,
            )

        def reset(self) -> None:
            if run_dir.exists():
                for path in run_dir.glob("*"):
                    if path.is_file():
                        path.unlink()

    return Env()


def test_allowlisted_baseline_can_replay(baseline_env):
    baseline_env.add_allowlist()
    engine = baseline_env.build_engine()
    summary = engine.run(max_steps=2)
    assert summary["unpromoted_execution_allowed"] is True
    assert summary["unpromoted_execution_reason"] == "baseline execution metrics"
    state_payload = json.loads(Path(summary["state_path"]).read_text(encoding="utf-8"))
    assert state_payload["unpromoted_execution_allowed"] is True
    assert state_payload["baseline_allowlist_path"]


def test_unallowlisted_baseline_fails(baseline_env):
    with pytest.raises(RuntimeError, match="not promoted"):
        baseline_env.build_engine()


def test_allowlist_does_not_allow_live_execution(baseline_env):
    baseline_env.add_allowlist()
    with pytest.raises(RuntimeError, match="Baseline allowlist only permits replay-mode execution"):
        baseline_env.build_engine(replay_mode=False, live_mode=True)


def test_qualification_allowlist_enables_replay(baseline_env):
    baseline_env.add_qualification_allowlist()
    engine = baseline_env.build_engine()
    summary = engine.run(max_steps=1)
    assert summary["unpromoted_allow_source"] == "qualification_allowlist"
    state = json.loads(Path(summary["state_path"]).read_text(encoding="utf-8"))
    assert state["qualification_allowlist_path"]


def test_cli_flag_enables_replay(baseline_env):
    engine = baseline_env.build_engine(qualification_replay=True, qualification_reason="cli test")
    summary = engine.run(max_steps=1)
    assert summary["unpromoted_allow_source"] == "qualification_cli"
    assert summary["unpromoted_execution_reason"] == "cli test"


def test_qualification_flag_disallowed_for_live(baseline_env):
    with pytest.raises(RuntimeError, match="not promoted"):
        baseline_env.build_engine(replay_mode=False, qualification_replay=True)
