from __future__ import annotations

from datetime import datetime, timezone

from research.ops.alerts import AlertEmitter
from research.ops.config import BackoffPolicyConfig, PaperOpsConfig
from research.ops.lifecycle import RunLifecycleTracker
from research.ops.runtime import BrokerRetryableError, RunExecutor
from research.ops.scheduler import PaperRunScheduler


def test_run_executor_backoff_and_recovery(tmp_path, monkeypatch):
    data_root = tmp_path / "quanto_data"
    alerts_root = tmp_path / "alerts_root"
    monkeypatch.setenv("QUANTO_DATA_ROOT", str(data_root))
    tracker = RunLifecycleTracker("exp_rt", "run_alpha")
    scheduler = PaperRunScheduler("exp_rt", PaperOpsConfig(cron="0 13 * * 1-5"))
    scheduled_for = datetime(2024, 1, 2, 13, 0, tzinfo=timezone.utc)
    scheduler.mark_active("run_alpha", scheduled_for)
    sleep_calls: list[float] = []

    def fake_sleep(value: float) -> None:
        sleep_calls.append(value)

    attempts = {"count": 0}

    def run_callable() -> dict[str, object]:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise BrokerRetryableError("alpaca_down")
        run_dir = tmp_path / "runs" / "run_alpha"
        run_dir.mkdir(parents=True, exist_ok=True)
        return {"run_dir": str(run_dir), "metrics": {"pnl": 1.0}}

    executor = RunExecutor(
        experiment_id="exp_rt",
        run_id="run_alpha",
        lifecycle=tracker,
        scheduler=scheduler,
        alert_emitter=AlertEmitter(base_dir=alerts_root),
        backoff_config=BackoffPolicyConfig(initial_seconds=1, max_seconds=4, multiplier=2, max_attempts=3),
        sleep_fn=fake_sleep,
    )
    result = executor.execute(run_callable)
    assert result["metrics"]["pnl"] == 1.0
    assert sleep_calls == [1.0, 2.0]
    assert tracker.current_state == "COMPLETED"
