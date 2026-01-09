from __future__ import annotations

import pytest

from research.experiments.spec import ExperimentSpec
from research.experiments.sweep import SweepSpec, expand_sweep


def _base_spec_payload() -> dict[str, object]:
    return {
        "experiment_name": "sweep_test",
        "symbols": ["AAPL"],
        "start_date": "2023-01-01",
        "end_date": "2023-01-05",
        "interval": "daily",
        "feature_set": "sma_v1",
        "policy": "sma",
        "policy_params": {
            "fast_window": 2,
            "slow_window": 4,
        },
        "cost_config": {"transaction_cost_bp": 1.0},
        "seed": 17,
    }


def test_expand_sweep_is_deterministic():
    base_spec = ExperimentSpec.from_mapping(_base_spec_payload())
    sweep_payload = {
        "sweep_name": "determinism_check",
        "base_experiment_spec": base_spec.to_dict(),
        "sweep_dimensions": {
            "feature_set": ["sma_v1", "sma_alt"],
            "cost_config.tx_cost_bps": [0.0, 5.0],
            "seed": [1, 2],
        },
    }
    sweep_spec = SweepSpec.from_mapping(sweep_payload)
    first_expansion = expand_sweep(sweep_spec)
    second_expansion = expand_sweep(sweep_spec)

    assert [spec.experiment_id for spec in first_expansion] == [
        spec.experiment_id for spec in second_expansion
    ]

    expected_order = [
        ("sma_v1", 0.0, 1),
        ("sma_v1", 0.0, 2),
        ("sma_alt", 0.0, 1),
        ("sma_alt", 0.0, 2),
        ("sma_v1", 5.0, 1),
        ("sma_v1", 5.0, 2),
        ("sma_alt", 5.0, 1),
        ("sma_alt", 5.0, 2),
    ]
    realized_order = [
        (spec.feature_set, spec.cost_config.transaction_cost_bp, spec.seed)
        for spec in first_expansion
    ]
    assert realized_order == expected_order


def test_invalid_dimension_path_rejected():
    base_spec = ExperimentSpec.from_mapping(_base_spec_payload())
    sweep_payload = {
        "sweep_name": "invalid_dimensions",
        "base_experiment_spec": base_spec.to_dict(),
        "sweep_dimensions": {
            "nonexistent.field": [1],
        },
    }
    with pytest.raises(ValueError) as excinfo:
        SweepSpec.from_mapping(sweep_payload)
    assert "nonexistent.field" in str(excinfo.value)


def test_seed_grid_expands_linear_sequence():
    base_spec = ExperimentSpec.from_mapping(_base_spec_payload())
    sweep_payload = {
        "sweep_name": "seed_grid",
        "base_experiment_spec": base_spec.to_dict(),
        "sweep_dimensions": {
            "seed": {"grid": {"start": 0, "count": 3, "step": 2}},
        },
    }
    sweep_spec = SweepSpec.from_mapping(sweep_payload)
    expansions = expand_sweep(sweep_spec)
    assert [spec.seed for spec in expansions] == [0, 2, 4]
