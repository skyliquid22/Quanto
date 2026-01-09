from __future__ import annotations

import json
from datetime import datetime, timezone

from research.ops.lifecycle import RunLifecycleTracker


def test_lifecycle_transitions(tmp_path, monkeypatch):
    data_root = tmp_path / "quanto_data"
    monkeypatch.setenv("QUANTO_DATA_ROOT", str(data_root))
    tracker = RunLifecycleTracker("exp_1", "run_abc")
    tracker.record_transition("SCHEDULED", scheduled_for="2024-01-01T00:00:00+00:00")
    tracker.record_transition("STARTING")
    tracker.record_transition("RUNNING")
    tracker.record_transition("COMPLETED", recorded_at=datetime.now(timezone.utc).isoformat())
    payload = json.loads(tracker.state_path.read_text(encoding="utf-8"))
    assert payload["current_state"] == "COMPLETED"
    assert len(payload["transitions"]) == 4
    assert tracker.is_terminal()
