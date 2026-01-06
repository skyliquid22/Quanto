from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import hashlib

from infra.normalization.lineage import compute_file_hash
from research.datasets.canonical_equity_loader import load_canonical_equity
from research.strategies.sma_crossover import SMAStrategyConfig, run_sma_crossover

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DATA_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "canonical_data_root"


def test_sma_crossover_deterministic(tmp_path):
    data_root = _prepare_data_root(tmp_path)
    env = _build_env(data_root)

    args = [
        sys.executable,
        "-m",
        "scripts.run_sma_crossover",
        "--symbols",
        "AAPL,MSFT",
        "--start-date",
        "2023-01-03",
        "--end-date",
        "2023-01-06",
        "--fast",
        "2",
        "--slow",
        "3",
        "--data-root",
        str(data_root),
    ]

    first_paths = _run_cli(args, env)
    report_path = Path(first_paths["report"])
    plot_path = Path(first_paths["plot"])
    first_report_bytes = report_path.read_bytes()
    first_plot_bytes = plot_path.read_bytes()

    second_paths = _run_cli(args, env)
    assert first_paths == second_paths
    assert report_path.read_bytes() == first_report_bytes
    assert plot_path.read_bytes() == first_plot_bytes

    payload = json.loads(first_report_bytes.decode("utf-8"))
    _assert_hashes(payload, report_path, plot_path, data_root)
    _assert_positions_shift(data_root)


def _prepare_data_root(tmp_path: Path) -> Path:
    dest = tmp_path / "quanto_data"
    shutil.copytree(FIXTURE_DATA_ROOT, dest)
    return dest


def _build_env(data_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["QUANTO_DATA_ROOT"] = str(data_root)
    return env


def _run_cli(args, env):
    output = subprocess.check_output(args, cwd=PROJECT_ROOT, env=env)
    text = output.decode("utf-8").strip().splitlines()[-1]
    return json.loads(text)


def _assert_hashes(payload: dict, report_path: Path, plot_path: Path, data_root: Path) -> None:
    for rel_path, recorded in payload["hashes"]["canonical_files"].items():
        target = data_root / rel_path
        assert target.exists(), f"missing canonical file {target}"
        assert compute_file_hash(target) == recorded
    assert compute_file_hash(plot_path) == payload["hashes"]["plot_png"]
    expected_report_hash = _compute_report_hash(payload)
    assert payload["hashes"]["report_json"] == expected_report_hash


def _compute_report_hash(payload: dict) -> str:
    clone = json.loads(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    clone["hashes"]["report_json"] = ""
    canonical_bytes = json.dumps(clone, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical_bytes).hexdigest()}"


def _assert_positions_shift(data_root: Path) -> None:
    slices, _ = load_canonical_equity(
        ["AAPL"],
        start_date="2023-01-03",
        end_date="2023-01-06",
        data_root=data_root,
    )
    config = SMAStrategyConfig(fast_window=2, slow_window=3)
    strategy = run_sma_crossover("AAPL", slices["AAPL"].timestamps, slices["AAPL"].closes, config)
    assert strategy.positions[0] == 0
    assert strategy.positions[1] == strategy.signal[0]
    first_signal_idx = next(idx for idx, value in enumerate(strategy.signal) if value == 1)
    assert strategy.positions[first_signal_idx] == 0
    assert first_signal_idx + 1 < len(strategy.positions)
    assert strategy.positions[first_signal_idx + 1] == 1
