"""Atomic persistence for shadow execution state."""

from __future__ import annotations

import json
from pathlib import Path

from infra.paths import get_data_root
from research.shadow.schema import ShadowState


class StateStore:
    """Filesystem-backed state persistence with deterministic serialization."""

    def __init__(
        self,
        experiment_id: str,
        *,
        run_id: str | None = None,
        base_dir: Path | None = None,
    ) -> None:
        root = Path(base_dir) if base_dir else get_data_root() / "shadow"
        destination = root / experiment_id
        if run_id:
            destination = destination / run_id
        self._state_dir = destination
        self._state_path = destination / "state.json"

    @property
    def state_path(self) -> Path:
        return self._state_path

    @property
    def state_dir(self) -> Path:
        return self._state_dir

    def load(self) -> ShadowState | None:
        if not self._state_path.exists():
            return None
        payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        return ShadowState.from_dict(payload)

    def save(self, state: ShadowState) -> None:
        self._state_dir.mkdir(parents=True, exist_ok=True)
        temp_path = self._state_path.with_suffix(".tmp")
        serialized = json.dumps(state.to_dict(), sort_keys=True, indent=2)
        temp_path.write_text(serialized, encoding="utf-8")
        temp_path.replace(self._state_path)


__all__ = ["StateStore"]
