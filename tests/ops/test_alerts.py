from __future__ import annotations

import json
from datetime import datetime, timezone

from research.ops.alerts import AlertEmitter


def test_alert_emission(tmp_path, monkeypatch):
    data_root = tmp_path / "quanto_data"
    monkeypatch.setenv("QUANTO_DATA_ROOT", str(data_root))
    emitter = AlertEmitter()
    ts = datetime(2024, 1, 2, tzinfo=timezone.utc)
    path = emitter.emit(
        severity="hard",
        kind="run_failed",
        message="Run failed",
        experiment_id="exp_alert",
        run_id="run_1",
        context={"reason": "test"},
        timestamp=ts,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload[0]["severity"] == "hard"
    assert payload[0]["context"]["reason"] == "test"
