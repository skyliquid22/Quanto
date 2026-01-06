from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys

from infra.normalization.lineage import compute_file_hash
from research.datasets.canonical_equity_loader import load_canonical_equity
from research.strategies.sma_crossover import SMAStrategyConfig, run_sma_crossover

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SLICE_CONFIG = PROJECT_ROOT / "tests" / "fixtures" / "configs" / "v1_slice.json"


def test_sma_finrl_rollout_determinism(tmp_path):
    data_root = tmp_path / "rollout_data"
    slice_config = _extend_slice_config(tmp_path)
    env = _build_env(data_root)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "scripts.run_v1_slice",
            "--config",
            str(slice_config),
            "--data-root",
            str(data_root),
            "--offline",
        ],
        cwd=PROJECT_ROOT,
        env=env,
    )

    args = [
        sys.executable,
        "-m",
        "scripts.run_sma_finrl_rollout",
        "--symbol",
        "AAPL",
        "--start-date",
        "2023-01-03",
        "--end-date",
        "2023-01-10",
        "--fast",
        "2",
        "--slow",
        "3",
        "--transaction-cost-bp",
        "2.0",
        "--data-root",
        str(data_root),
    ]

    first_paths = _run_cli(args, env)
    second_paths = _run_cli(args, env)
    assert first_paths == second_paths

    report_path = Path(first_paths["report"])
    plot_path = Path(first_paths["plot"])
    assert report_path.read_bytes() == Path(second_paths["report"]).read_bytes()
    assert plot_path.read_bytes() == Path(second_paths["plot"]).read_bytes()

    payload = json.loads(report_path.read_text())
    _assert_hashes(payload, report_path, plot_path, data_root)
    _assert_weights_clipped(payload)
    _assert_no_lookahead(payload, data_root)


def _extend_slice_config(tmp_path: Path) -> Path:
    config = json.loads(SLICE_CONFIG.read_text())
    config["offline_ingestion"]["end_date"] = "2023-01-10"
    config_path = tmp_path / "slice_config.json"
    config_path.write_text(json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return config_path


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
    import hashlib

    return f"sha256:{hashlib.sha256(canonical_bytes).hexdigest()}"


def _assert_weights_clipped(payload: dict) -> None:
    weights = payload["series"]["weights"]
    assert all(0.0 <= value <= 1.0 for value in weights)
    assert weights[0] == 0.0


def _assert_no_lookahead(payload: dict, data_root: Path) -> None:
    timestamps = payload["series"]["timestamps"]
    account_values = payload["series"]["account_value"]
    weights = payload["series"]["weights"]
    steps = payload["steps"]
    assert len(steps) + 1 == len(account_values)
    first_step = steps[0]
    assert first_step["timestamp"] == timestamps[0]
    slices, _ = load_canonical_equity(
        ["AAPL"],
        start_date=payload["parameters"]["start_date"],
        end_date=payload["parameters"]["end_date"],
        data_root=data_root,
    )
    slice_data = slices["AAPL"]
    strategy = run_sma_crossover(
        "AAPL",
        slice_data.timestamps,
        slice_data.closes,
        SMAStrategyConfig(
            fast_window=payload["parameters"]["fast_window"],
            slow_window=payload["parameters"]["slow_window"],
        ),
    )
    usable_rows = [
        idx
        for idx, fast in enumerate(strategy.fast_sma)
        if fast is not None and strategy.slow_sma[idx] is not None
    ]
    first_idx = usable_rows[0]
    next_idx = first_idx + 1
    close_now = strategy.closes[first_idx]
    close_next = strategy.closes[next_idx]
    pct_return = (close_next - close_now) / close_now
    cost_paid = first_step["cost_paid"]
    initial_value = account_values[0]
    assert math.isclose(weights[1], first_step["weight_realized"], rel_tol=0, abs_tol=0)
    assert math.isclose(first_step["weight_target"], first_step["weight_realized"], rel_tol=0, abs_tol=0)
    expected_value = (initial_value - cost_paid) * (1.0 + weights[1] * pct_return)
    assert math.isclose(account_values[1], expected_value, rel_tol=1e-12, abs_tol=1e-9)
    assert math.isclose(first_step["price_close"], close_now, rel_tol=1e-12, abs_tol=1e-9)
