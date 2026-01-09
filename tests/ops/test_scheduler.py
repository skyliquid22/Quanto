from __future__ import annotations

from datetime import datetime, timezone

from research.ops.config import PaperOpsConfig
from research.ops.scheduler import PaperRunScheduler


def test_scheduler_detects_missed_runs(tmp_path, monkeypatch):
    data_root = tmp_path / "quanto_data"
    monkeypatch.setenv("QUANTO_DATA_ROOT", str(data_root))
    config = PaperOpsConfig(cron="0 13 * * 1-5", timezone="UTC", grace_minutes=5)
    scheduler = PaperRunScheduler("exp_sched", config)
    base_run_id = "paper_base"
    now = datetime(2024, 1, 2, 13, 2, tzinfo=timezone.utc)
    decision = scheduler.evaluate(base_run_id=base_run_id, now=now)
    assert decision.due
    assert decision.run_id is not None
    scheduler.mark_active(decision.run_id, decision.scheduled_for)
    scheduler.mark_terminal(decision.run_id, "COMPLETED")
    later = datetime(2024, 1, 3, 14, 0, tzinfo=timezone.utc)
    decision_late = scheduler.evaluate(base_run_id=base_run_id, now=later)
    assert decision_late.missed
    assert decision_late.missed[0].reason == "window_expired"
