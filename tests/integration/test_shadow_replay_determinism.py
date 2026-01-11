from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from research.experiments.spec import ExperimentSpec

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_shadow_replay_determinism(tmp_path, monkeypatch):
    data_root = tmp_path / "quanto_data"
    monkeypatch.setenv("QUANTO_DATA_ROOT", str(data_root))
    symbols = ("AAA", "BBB")
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    num_days = 12
    for idx, symbol in enumerate(symbols):
        _write_canonical_year(
            data_root,
            symbol,
            year=2020,
            rows=_generate_rows(symbol, start, num_days, price_offset=1 + idx),
        )
    spec_payload = {
        "experiment_name": "shadow_integration",
        "symbols": list(symbols),
        "start_date": "2020-01-01",
        "end_date": "2020-01-12",
        "interval": "daily",
        "feature_set": "sma_v1",
        "policy": "sma",
        "policy_params": {
            "fast_window": 1,
            "slow_window": 2,
            "policy_mode": "hard",
            "sigmoid_scale": 1.0,
        },
        "cost_config": {"transaction_cost_bp": 1.0},
        "risk_config": {"max_weight": 1.0, "exposure_cap": 1.0, "min_cash": 0.0},
        "seed": 7,
    }
    spec = ExperimentSpec.from_mapping(spec_payload)
    registry_base = data_root / "experiments" / spec.experiment_id
    _write_registry_artifacts(registry_base, spec)
    _write_promotion_record(data_root, spec.experiment_id, registry_base)

    cmd = [
        sys.executable,
        "scripts/run_shadow.py",
        "--experiment-id",
        spec.experiment_id,
        "--replay",
        "--start-date",
        "2020-01-01",
        "--end-date",
        "2020-01-10",
        "--max-steps",
        "6",
        "--reset",
        "--registry-root",
        str(data_root / "experiments"),
        "--promotion-root",
        str(data_root / "promotions"),
    ]
    run_id = _derive_run_id(spec.experiment_id, "2020-01-01", "2020-01-10")
    run_dir = data_root / "shadow" / spec.experiment_id / run_id
    state_path = run_dir / "state.json"
    log_path = run_dir / "logs" / "steps.jsonl"
    run_one = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert run_one.returncode == 0, run_one.stdout + run_one.stderr
    assert state_path.exists()
    assert log_path.exists()
    first_state = state_path.read_text(encoding="utf-8")
    first_logs = log_path.read_text(encoding="utf-8")

    run_two = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert run_two.returncode == 0, run_two.stdout + run_two.stderr
    second_state = state_path.read_text(encoding="utf-8")
    second_logs = log_path.read_text(encoding="utf-8")
    assert first_state == second_state
    assert first_logs == second_logs

    first_values = _extract_portfolio_values(first_logs)
    second_values = _extract_portfolio_values(second_logs)
    assert first_values == second_values

    cmd_sim = cmd + ["--execution-mode", "sim", "--reset"]
    sim_run = subprocess.run(cmd_sim, cwd=REPO_ROOT, capture_output=True, text=True)
    assert sim_run.returncode == 0, sim_run.stdout + sim_run.stderr
    metrics_path = run_dir / "execution_metrics.json"
    assert metrics_path.exists()
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    fill_rate = metrics_payload.get("summary", {}).get("fill_rate")
    assert fill_rate is not None and fill_rate > 0.0


def _write_canonical_year(data_root: Path, symbol: str, *, year: int, rows: Iterable[dict]) -> None:
    base = data_root / "canonical" / "equity_ohlcv" / symbol / "daily"
    base.mkdir(parents=True, exist_ok=True)
    shard = base / f"{year}.parquet"
    shard.write_text(json.dumps(list(rows)), encoding="utf-8")


def _generate_rows(symbol: str, start: datetime, num_days: int, price_offset: int) -> list[dict]:
    rows = []
    for idx in range(num_days):
        timestamp = start + timedelta(days=idx)
        price = price_offset * 10 + idx
        rows.append(
            {
                "symbol": symbol,
                "timestamp": timestamp.isoformat(),
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price,
                "volume": 1_000 + idx,
            }
        )
    return rows


def _write_registry_artifacts(base: Path, spec: ExperimentSpec) -> None:
    (base / "spec").mkdir(parents=True, exist_ok=True)
    (base / "evaluation").mkdir(parents=True, exist_ok=True)
    (base / "logs").mkdir(parents=True, exist_ok=True)
    (base / "promotion").mkdir(parents=True, exist_ok=True)
    (base / "spec" / "experiment_spec.json").write_text(spec.canonical_json, encoding="utf-8")
    metrics_payload = {
        "metadata": {"run_id": spec.experiment_id},
        "performance": {"total_return": 0.1, "cagr": None, "volatility_ann": 0.2, "sharpe": None, "max_drawdown": 0.05},
        "trading": {"turnover_1d_mean": 0.0, "turnover_1d_median": 0.0, "avg_exposure": 1.0, "max_concentration": 0.5},
        "safety": {
            "nan_inf_violations": 0.0,
            "action_bounds_violations": 0.0,
            "constraint_violations_count": 0.0,
            "max_weight_violation_count": 0.0,
            "exposure_violation_count": 0.0,
            "turnover_violation_count": 0.0,
        },
    }
    (base / "evaluation" / "metrics.json").write_text(json.dumps(metrics_payload), encoding="utf-8")
    (base / "logs" / "run_summary.json").write_text(json.dumps({"recorded_at": "2024-01-01T00:00:00Z"}), encoding="utf-8")


def _write_promotion_record(data_root: Path, experiment_id: str, registry_base: Path) -> None:
    promo_dir = data_root / "promotions" / "candidate"
    promo_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": experiment_id,
        "tier": "candidate",
        "qualification_report_path": str(registry_base / "promotion" / "qualification_report.json"),
        "spec_path": str(registry_base / "spec" / "experiment_spec.json"),
        "metrics_path": str(registry_base / "evaluation" / "metrics.json"),
        "promotion_reason": "integration-test",
    }
    (registry_base / "promotion" / "qualification_report.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
    (promo_dir / f"{experiment_id}.json").write_text(json.dumps(payload), encoding="utf-8")


def _derive_run_id(experiment_id: str, start: str, end: str) -> str:
    payload = {
        "experiment_id": experiment_id,
        "window_start": start,
        "window_end": end,
        "mode": "replay",
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"replay_{digest[:12]}"


def _extract_portfolio_values(log_text: str) -> list[float]:
    values: list[float] = []
    for line in log_text.strip().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        values.append(float(payload["portfolio_value"]))
    return values
