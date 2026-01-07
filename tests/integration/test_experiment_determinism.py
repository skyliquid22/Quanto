from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from research.experiments import ExperimentRegistry, ExperimentSpec, run_experiment

UTC = timezone.utc


def _write_yearly_file(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _build_rows(symbol: str, closes: list[float]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, close in enumerate(closes, start=1):
        ts = datetime(2023, 1, idx + 2, 16, tzinfo=UTC)
        rows.append(
            {
                "symbol": symbol,
                "timestamp": ts.isoformat(),
                "open": close - 0.5,
                "high": close + 0.5,
                "low": close - 0.75,
                "close": close,
                "volume": 1_000_000 + idx,
            }
        )
    return rows


def test_experiment_determinism(tmp_path: Path):
    data_root = tmp_path / "quanto_data"
    registry_root = tmp_path / "experiments"
    symbols = ["AAPL", "MSFT"]
    samples = {
        "AAPL": [150.0, 150.5, 151.2, 152.0, 152.7, 153.1],
        "MSFT": [240.0, 240.2, 240.5, 241.0, 241.5, 241.9],
    }
    for symbol in symbols:
        yearly_path = data_root / "canonical" / "equity_ohlcv" / symbol / "daily" / "2023.parquet"
        _write_yearly_file(yearly_path, _build_rows(symbol, samples[symbol]))

    spec_payload = {
        "experiment_name": "sma_test",
        "symbols": symbols,
        "start_date": "2023-01-04",
        "end_date": "2023-01-09",
        "interval": "daily",
        "feature_set": "sma_v1",
        "policy": "sma",
        "policy_params": {
            "fast_window": 2,
            "slow_window": 3,
            "policy_mode": "sigmoid",
            "sigmoid_scale": 3.0,
        },
        "cost_config": {"transaction_cost_bp": 1.0},
        "seed": 17,
        "notes": "integration sanity",
    }
    spec = ExperimentSpec.from_mapping(spec_payload)
    registry = ExperimentRegistry(root=registry_root)

    first = run_experiment(spec, registry=registry, force=True, data_root=data_root)
    payload_one = json.loads(first.metrics_path.read_text())

    second = run_experiment(spec, registry=registry, force=True, data_root=data_root)
    payload_two = json.loads(second.metrics_path.read_text())

    assert first.experiment_id == second.experiment_id
    assert payload_one == payload_two
    assert payload_two["series"]["timestamps"] == payload_one["series"]["timestamps"]
    assert payload_two["series"]["weights"] == payload_one["series"]["weights"]
    assert payload_two["performance"] == payload_one["performance"]
