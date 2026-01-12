from __future__ import annotations

import json
from pathlib import Path

import scripts.run_v1_release_candidate as runner
from tests.integration.v1_release_candidate_helpers import (
    make_args,
    patch_run_experiment,
    patch_run_qualification,
    patch_shadow,
    write_spec,
)


def test_v1_release_candidate_gate_failure(tmp_path, monkeypatch):
    data_root = tmp_path / "quanto_data"
    spec_path = write_spec(tmp_path / "spec.json")
    args = make_args(spec_path, data_root)
    patch_run_experiment(monkeypatch)
    patch_run_qualification(monkeypatch, passed=False)
    shadow_calls = patch_shadow(monkeypatch)

    summary = runner.run_v1_release(args)
    release_dir = Path(summary["release_dir"])
    report_payload = json.loads((release_dir / "report" / "v1_release_report.json").read_text(encoding="utf-8"))
    assert report_payload["v1_ready"] is False
    assert report_payload["shadow"]["status"] == "skipped"
    assert not shadow_calls, "shadow replay must be skipped when qualification fails"
    assert not (release_dir / "shadow").exists()
    assert not (release_dir / "report" / "promotion_record.json").exists()
