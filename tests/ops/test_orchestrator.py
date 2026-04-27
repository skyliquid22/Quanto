from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from research.ops.config import BackoffPolicyConfig, OpsConfig, PaperOpsConfig
from research.ops.service import PaperRunOrchestrator
from research.paper.config import (
    ArtifactConfig,
    BrokerConfig,
    PaperRunConfig,
    PollingConfig,
    ReconciliationConfig,
    RiskLimitConfig,
)


def _stub_execute_fn(run_id: str, scheduled_for: object, lifecycle: object) -> dict:
    """Stub execute_fn that returns zero metrics without touching Alpaca or the registry."""
    return {
        "run_dir": "",
        "metrics": {"pnl": 0.0, "exposure": 0.0, "turnover": 0.0, "fees": 0.0},
        "run_id": run_id,
    }


def test_orchestrator_writes_summary(tmp_path, monkeypatch):
    data_root = tmp_path / "quanto_data"
    monkeypatch.setenv("QUANTO_DATA_ROOT", str(data_root))
    artifact_root = data_root / "paper" / "exp_ops" / "runs"
    config = PaperRunConfig(
        experiment_id="exp_ops",
        execution_mode="alpaca_paper",
        universe=["AAA", "BBB"],
        broker=BrokerConfig(),
        risk_limits=RiskLimitConfig(),
        polling=PollingConfig(),
        reconciliation=ReconciliationConfig(),
        artifacts=ArtifactConfig(output_root=artifact_root),
    )
    ops_config = OpsConfig(
        paper_trading=PaperOpsConfig(
            cron="0 0 * * *",
            timezone="UTC",
            grace_minutes=15,
            expect_trades=False,
            backoff=BackoffPolicyConfig(initial_seconds=0.1, max_seconds=0.1, multiplier=1.0, max_attempts=1),
        )
    )
    orchestrator = PaperRunOrchestrator(
        paper_config=config,
        ops_config=ops_config,
        execute_fn=_stub_execute_fn,
    )
    now = datetime(2024, 1, 2, 0, 5, tzinfo=timezone.utc)
    report = orchestrator.run(now=now)
    assert report.status == "COMPLETED"
    assert report.summary_json is not None and report.summary_json.exists()
    payload = json.loads(report.summary_json.read_text(encoding="utf-8"))
    assert payload["run_id"] == report.run_id
    assert payload["pnl"] == 0.0
