from __future__ import annotations

import json
from pathlib import Path

import scripts.run_v1_release_candidate as runner
from tests.integration.v1_release_candidate_helpers import (
    make_args,
    patch_run_experiment,
    patch_run_qualification,
    patch_shadow,
    snapshot_dir,
    write_spec,
)


def test_v1_release_candidate_determinism(tmp_path, monkeypatch):
    data_root = tmp_path / "quanto_data"
    spec_path = write_spec(tmp_path / "spec.json")
    args = make_args(spec_path, data_root)
    patch_run_experiment(monkeypatch)
    patch_run_qualification(monkeypatch, passed=True)
    shadow_calls = patch_shadow(monkeypatch)

    summary_one = runner.run_v1_release(args)
    release_dir = Path(summary_one["release_dir"])
    snapshot_one = snapshot_dir(release_dir)
    report_payload = json.loads((release_dir / "report" / "v1_release_report.json").read_text(encoding="utf-8"))
    assert report_payload["v1_ready"] is True
    assert report_payload["shadow"]["steps_executed"] == 3
    assert shadow_calls, "shadow replay should execute when qualification passes"

    summary_two = runner.run_v1_release(args)
    assert summary_one["run_id"] == summary_two["run_id"]
    release_dir_again = Path(summary_two["release_dir"])
    snapshot_two = snapshot_dir(release_dir_again)
    assert snapshot_one == snapshot_two
