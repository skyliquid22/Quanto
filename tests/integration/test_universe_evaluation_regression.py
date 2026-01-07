from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UTC = timezone.utc


def test_universe_evaluation_regression(tmp_path):
    data_root = tmp_path / "quanto_data"
    _prepare_canonical_payload(data_root)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    args = [
        sys.executable,
        "-m",
        "scripts.evaluate_agent",
        "--symbols",
        "AAPL,MSFT",
        "--start-date",
        "2023-01-04",
        "--end-date",
        "2023-01-10",
        "--feature-set",
        "sma_v1",
        "--policy",
        "equal_weight",
        "--fast-window",
        "2",
        "--slow-window",
        "3",
        "--transaction-cost-bp",
        "0.5",
        "--data-root",
        str(data_root),
        "--out-dir",
        str(tmp_path / "artifacts"),
    ]
    first = _run_cli(args, env)
    second = _run_cli(args, env)
    assert first == second

    metrics_path = Path(first["metrics_path"])
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    second_payload = json.loads(Path(second["metrics_path"]).read_text(encoding="utf-8"))
    assert payload == second_payload

    _assert_payload_integrity(payload)


def _run_cli(args, env):
    output = subprocess.check_output(args, cwd=PROJECT_ROOT, env=env)
    text = output.decode("utf-8").strip().splitlines()[-1]
    return json.loads(text)


def _prepare_canonical_payload(data_root: Path) -> None:
    prices = {
        "AAPL": [150.0, 151.5, 152.2, 152.9, 153.4, 154.2, 155.0],
        "MSFT": [240.0, 241.0, 241.6, 242.2, 242.8, 243.5, 244.1],
    }
    for symbol, closes in prices.items():
        rows = []
        for idx, close in enumerate(closes, start=1):
            ts = datetime(2023, 1, idx + 2, 16, tzinfo=UTC)
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": ts.isoformat(),
                    "open": close - 0.5,
                    "high": close + 0.5,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1_000_000 + idx,
                }
            )
        shard = data_root / "canonical" / "equity_ohlcv" / symbol / "daily" / "2023.parquet"
        shard.parent.mkdir(parents=True, exist_ok=True)
        shard.write_text(json.dumps(rows, separators=(",", ":")), encoding="utf-8")


def _assert_payload_integrity(payload: dict) -> None:
    metadata = payload["metadata"]
    assert metadata["symbols"] == ["AAPL", "MSFT"]
    assert metadata["feature_set"] == "sma_v1"
    performance = payload["performance"]
    assert performance["total_return"] is not None
    assert performance["max_drawdown"] <= 1.0
    trading = payload["trading"]
    assert 0.0 <= trading["turnover_1d_mean"] <= 2.0
    assert trading["hhi_mean"] <= 1.0
    safety = payload["safety"]
    assert safety["nan_inf_violations"] == 0.0
    assert safety["action_bounds_violations"] == 0.0
    series = payload["series"]
    timestamps = series["timestamps"]
    returns = series["returns"]
    assert len(timestamps) >= 2
    assert len(returns) == len(timestamps) - 1
    assert len(series["transaction_costs"]) == len(returns)
    if isinstance(series["weights"], dict):
        for values in series["weights"].values():
            assert len(values) == len(timestamps)
            assert all(math.isfinite(value) for value in values)
    else:
        assert len(series["weights"]) == len(timestamps)
    assert payload["inputs_used"]
