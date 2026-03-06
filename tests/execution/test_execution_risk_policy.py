from __future__ import annotations

import json
from pathlib import Path

from research.execution.risk_engine import ExecutionRiskConfig


def test_from_policy_payload_maps_limits() -> None:
    payload = {
        "exposure": {"max_gross_exposure": 1.2, "max_positions": 7},
        "position_limits": {"max_position_pct_nav": 0.15},
        "liquidity": {"max_daily_turnover": 0.4},
        "loss_controls": {"max_daily_loss": -0.03, "max_drawdown": -0.2},
    }
    cfg = ExecutionRiskConfig.from_policy_payload(payload)
    assert cfg.max_gross_exposure == 1.2
    assert cfg.max_active_positions == 7
    assert cfg.max_symbol_weight == 0.15
    assert cfg.max_daily_turnover == 0.4
    assert cfg.max_daily_loss == 0.03
    assert cfg.max_trailing_drawdown == 0.2


def test_from_policy_file_json(tmp_path: Path) -> None:
    payload = {
        "exposure": {"max_gross_exposure": 0.8, "max_positions": 3},
        "position_limits": {"max_position_pct_nav": 0.05},
        "liquidity": {"max_daily_turnover": 0.25},
        "loss_controls": {"max_daily_loss": -0.01, "max_drawdown": -0.05},
    }
    policy_path = tmp_path / "risk_policy.json"
    policy_path.write_text(json.dumps(payload), encoding="utf-8")
    cfg = ExecutionRiskConfig.from_policy_file(policy_path)
    assert cfg.max_gross_exposure == 0.8
    assert cfg.max_active_positions == 3
    assert cfg.max_symbol_weight == 0.05
    assert cfg.max_daily_turnover == 0.25
    assert cfg.max_daily_loss == 0.01
    assert cfg.max_trailing_drawdown == 0.05
