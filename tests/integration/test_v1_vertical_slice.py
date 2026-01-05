from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import hashlib

from infra.normalization.lineage import compute_file_hash

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_v1_slice.py"
CONFIG_PATH = PROJECT_ROOT / "tests" / "fixtures" / "configs" / "v1_slice.json"


def test_v1_vertical_slice_deterministic(tmp_path):
    data_root = tmp_path / "slice_data"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    cmd = [sys.executable, str(SCRIPT_PATH), "--config", str(CONFIG_PATH), "--data-root", str(data_root)]

    subprocess.check_call(cmd, cwd=PROJECT_ROOT, env=env)
    report_path = data_root / "monitoring" / "reports" / "v1_slice_report.json"
    plot_path = data_root / "monitoring" / "plots" / "v1_slice_equity.png"
    first_report_bytes = report_path.read_bytes()
    first_plot_bytes = plot_path.read_bytes()

    subprocess.check_call(cmd, cwd=PROJECT_ROOT, env=env)
    assert report_path.read_bytes() == first_report_bytes
    assert plot_path.read_bytes() == first_plot_bytes

    payload = json.loads(first_report_bytes)
    _assert_embedded_hashes(payload, data_root, report_path, plot_path)


def _assert_embedded_hashes(payload: dict, data_root: Path, report_path: Path, plot_path: Path) -> None:
    for rel_path, recorded in payload["hashes"]["canonical_files"].items():
        target = data_root / rel_path
        assert target.exists(), f"missing canonical file {target}"
        assert compute_file_hash(target) == recorded

    for rel_path, recorded in payload["hashes"]["canonical_manifests"].items():
        target = data_root / rel_path
        assert target.exists(), f"missing manifest {target}"
        assert compute_file_hash(target) == recorded

    assert compute_file_hash(plot_path) == payload["hashes"]["plot_png"]
    expected_report_hash = _compute_report_hash(payload)
    assert expected_report_hash == payload["hashes"]["report_json"]


def _compute_report_hash(payload: dict) -> str:
    clone = json.loads(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    clone["hashes"]["report_json"] = ""
    canonical_bytes = json.dumps(clone, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical_bytes).hexdigest()}"
