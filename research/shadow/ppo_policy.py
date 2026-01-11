"""Utilities for executing PPO policies inside the shadow engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, Tuple

import json

import numpy as np


def _load_ppo_model(checkpoint: Path):
    try:  # pragma: no cover - exercised via patched loader in tests
        from stable_baselines3 import PPO  # type: ignore
    except Exception as exc:  # pragma: no cover - deterministic error message
        raise RuntimeError("stable_baselines3 is required to replay PPO experiments") from exc
    return PPO.load(str(checkpoint), device="cpu")  # type: ignore[no-any-return]


@dataclass
class PpoShadowPolicy:
    """Thin wrapper around a trained PPO model for deterministic inference."""

    model: Any
    num_assets: int
    action_clip: Tuple[float, float] = (0.0, 1.0)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Path,
        *,
        num_assets: int,
        action_clip: Tuple[float, float] = (0.0, 1.0),
    ) -> "PpoShadowPolicy":
        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"PPO checkpoint not found: {checkpoint}")
        model = _load_ppo_model(checkpoint)
        return cls(model=model, num_assets=num_assets, action_clip=action_clip)

    def act(self, observation: Sequence[float]) -> list[float]:
        obs_array = np.asarray(observation, dtype=np.float32).reshape(-1)
        action, _ = self.model.predict(obs_array, deterministic=True)
        vector = np.asarray(action, dtype=np.float32).reshape(-1)
        if vector.size == 0:
            raise ValueError("PPO model returned an empty action vector")
        if vector.size == 1 and self.num_assets > 1:
            vector = np.repeat(vector, self.num_assets)
        if vector.size != self.num_assets:
            raise ValueError(
                f"PPO action dimension {vector.size} does not match symbol count {self.num_assets}"
            )
        low, high = self.action_clip
        clipped = np.clip(vector, low, high)
        return [float(value) for value in clipped]


def resolve_ppo_checkpoint_path(record_root: Path, experiment_id: str) -> Path:
    """Locate the PPO checkpoint within an experiment registry entry."""

    training_dir = Path(record_root) / "runs" / "training"
    if not training_dir.exists():
        raise FileNotFoundError(f"PPO training artifacts not found: {training_dir}")
    expected = training_dir / f"ppo_{experiment_id}.zip"
    if expected.exists():
        return expected
    summary_path = training_dir / "training_summary.json"
    if summary_path.exists():
        try:
            payload = summary_path.read_text(encoding="utf-8")
            summary = json.loads(payload)
        except json.JSONDecodeError:
            summary = {}
        artifact_name = (summary.get("artifacts") or {}).get("model")
        if artifact_name:
            candidate = training_dir / Path(artifact_name).name
            if candidate.exists():
                return candidate
    matches = sorted(training_dir.glob("ppo_*.zip"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"PPO checkpoint not found under {training_dir}. "
        "Ensure run_experiment copied the model artifact into runs/training/."
    )


__all__ = ["PpoShadowPolicy", "resolve_ppo_checkpoint_path"]
