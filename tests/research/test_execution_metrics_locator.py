from __future__ import annotations

import json
from pathlib import Path

from research.promotion.execution_metrics_locator import locate_execution_metrics


def test_locator_finds_shadow_metrics(tmp_path: Path):
    registry_root = tmp_path / "experiments"
    shadow_root = tmp_path / "shadow"
    exp_id = "exp_shadow"
    run_dir = shadow_root / exp_id / "run_a"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "steps.jsonl").write_text("{}", encoding="utf-8")
    exec_metrics = logs_dir / "execution_metrics.json"
    exec_metrics.write_text(json.dumps({"summary": {"fill_rate": 1.0}}, indent=2), encoding="utf-8")

    result = locate_execution_metrics(exp_id, registry_root=registry_root, shadow_root=shadow_root)
    assert result.found is True
    assert result.source == "shadow"
    assert result.execution_metrics_path == exec_metrics
    assert str(exec_metrics) in result.attempted


def test_locator_missing_records_attempted(tmp_path: Path):
    registry_root = tmp_path / "experiments"
    shadow_root = tmp_path / "shadow"
    exp_id = "exp_missing"

    result = locate_execution_metrics(exp_id, registry_root=registry_root, shadow_root=shadow_root)
    assert result.found is False
    assert not result.metrics_path
    assert not result.execution_metrics_path
    assert result.attempted == ()
