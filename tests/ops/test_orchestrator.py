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


class StubRunner:
    """Minimal runner used for orchestrator tests."""

    def __init__(self, config: PaperRunConfig, run_id: str, **_: object) -> None:
        self.config = config
        self.run_id = run_id
        self.run_dir = Path(config.artifacts.output_root) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)


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
        runner_factory=StubRunner,
    )
    now = datetime(2024, 1, 2, 0, 5, tzinfo=timezone.utc)
    report = orchestrator.run(now=now)
    assert report.status == "COMPLETED"
    assert report.summary_json is not None and report.summary_json.exists()
    payload = json.loads(report.summary_json.read_text(encoding="utf-8"))
    assert payload["run_id"] == report.run_id
    assert payload["pnl"] == 0.0
