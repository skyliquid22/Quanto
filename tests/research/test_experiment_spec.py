from __future__ import annotations

import json
import pytest

from research.experiments.spec import ExperimentSpec


def _base_spec() -> dict[str, object]:
    return {
        "experiment_name": "sma_baseline",
        "symbols": ["msft", "aapl"],
        "start_date": "2023-01-01",
        "end_date": "2023-01-05",
        "interval": "daily",
        "feature_set": "sma_v1",
        "policy": "sma",
        "policy_params": {
            "slow_window": 5,
            "fast_window": 2,
            "policy_mode": "sigmoid",
            "sigmoid_scale": 3.0,
        },
        "cost_config": {"transaction_cost_bp": 1.0},
        "seed": 7,
        "notes": "first spec",
    }


def test_experiment_spec_id_deterministic():
    left = _base_spec()
    right = json.loads(json.dumps(_base_spec(), sort_keys=True))
    spec_a = ExperimentSpec.from_mapping(left)
    spec_b = ExperimentSpec.from_mapping(right)
    assert spec_a.experiment_id == spec_b.experiment_id
    assert spec_a.canonical_json == spec_b.canonical_json


def test_experiment_spec_id_changes_on_field_update():
    spec = _base_spec()
    spec_alt = dict(spec)
    spec_alt["end_date"] = "2023-02-01"
    spec_a = ExperimentSpec.from_mapping(spec)
    spec_b = ExperimentSpec.from_mapping(spec_alt)
    assert spec_a.experiment_id != spec_b.experiment_id


def test_experiment_spec_id_changes_on_risk_update():
    spec = _base_spec()
    spec_with_risk = dict(spec)
    spec_with_risk["risk_config"] = {"max_weight": 0.2}
    spec_a = ExperimentSpec.from_mapping(spec)
    spec_b = ExperimentSpec.from_mapping(spec_with_risk)
    assert spec_a.experiment_id != spec_b.experiment_id


def test_missing_required_field_fails_fast():
    spec = _base_spec()
    spec.pop("experiment_name")
    with pytest.raises(ValueError):
        ExperimentSpec.from_mapping(spec)
