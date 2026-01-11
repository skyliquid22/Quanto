from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.shadow.ppo_policy import PpoShadowPolicy, resolve_ppo_checkpoint_path


class _StubModel:
    def __init__(self, output):
        self._output = output
        self.calls = []

    def predict(self, observation, deterministic=True):
        self.calls.append(tuple(observation.tolist()))
        return self._output, None


def test_policy_act_clips_and_matches_symbol_count():
    model = _StubModel([1.2, -0.5])
    policy = PpoShadowPolicy(model=model, num_assets=2)
    result = policy.act([0.0, 0.5])
    assert result == [1.0, 0.0]
    assert len(model.calls) == 1


def test_policy_broadcasts_scalar_action():
    model = _StubModel([0.25])
    policy = PpoShadowPolicy(model=model, num_assets=3)
    assert policy.act([0.1, 0.2]) == [0.25, 0.25, 0.25]


def test_from_checkpoint_uses_loader(monkeypatch, tmp_path):
    checkpoint = tmp_path / "ppo_model.zip"
    checkpoint.write_text("stub", encoding="utf-8")
    model = _StubModel([0.1])

    def fake_loader(path: Path):
        assert path == checkpoint
        return model

    monkeypatch.setattr("research.shadow.ppo_policy._load_ppo_model", fake_loader)
    policy = PpoShadowPolicy.from_checkpoint(checkpoint, num_assets=1)
    assert policy.act([0.0]) == pytest.approx([0.1])


def test_resolve_checkpoint_prefers_exact_file(tmp_path):
    experiment_id = "abc123"
    training_dir = tmp_path / "runs" / "training"
    training_dir.mkdir(parents=True)
    exact = training_dir / f"ppo_{experiment_id}.zip"
    exact.write_text("model", encoding="utf-8")
    fallback = training_dir / "ppo_other.zip"
    fallback.write_text("other", encoding="utf-8")
    resolved = resolve_ppo_checkpoint_path(tmp_path, experiment_id)
    assert resolved == exact


def test_resolve_checkpoint_uses_summary_name(tmp_path):
    experiment_id = "missing"
    training_dir = tmp_path / "runs" / "training"
    training_dir.mkdir(parents=True)
    custom = training_dir / "custom_model.zip"
    custom.write_text("model", encoding="utf-8")
    summary = training_dir / "training_summary.json"
    summary_payload = {"artifacts": {"model": "models/custom_model.zip"}}
    summary.write_text(json.dumps(summary_payload), encoding="utf-8")
    resolved = resolve_ppo_checkpoint_path(tmp_path, experiment_id)
    assert resolved == custom


def test_resolve_checkpoint_raises_when_missing(tmp_path):
    experiment_id = "nope"
    training_dir = tmp_path / "runs" / "training"
    training_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        resolve_ppo_checkpoint_path(tmp_path, experiment_id)
