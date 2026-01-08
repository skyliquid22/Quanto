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
        destination: Path | None = None,
    ) -> None:
        if destination is not None:
            resolved = Path(destination)
        else:
            root = Path(base_dir) if base_dir else get_data_root() / "shadow"
            resolved = root / experiment_id
            if run_id:
                resolved = resolved / run_id
        self._state_dir = resolved
        self._state_path = resolved / "state.json"

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

    @staticmethod
    def resume_snapshot(state: ShadowState) -> dict[str, object]:
        """Extract the portions of state required for resume-safe execution."""

        return {
            "submitted_order_ids": list(state.submitted_order_ids),
            "open_orders": [dict(entry) for entry in state.open_orders],
            "broker_order_map": dict(state.broker_order_map),
            "last_completed_step_ts": state.last_completed_step_ts,
            "last_broker_sync": state.last_broker_sync,
        }


__all__ = ["StateStore"]
