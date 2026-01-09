"""Run lifecycle persistence for paper trading operations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from infra.paths import get_data_root

RUN_STATES = {
    "SCHEDULED",
    "STARTING",
    "RUNNING",
    "COMPLETED",
    "HALTED",
    "FAILED",
    "MISSED",
}
TERMINAL_STATES = {"COMPLETED", "HALTED", "FAILED", "MISSED"}


def _now_iso(ts: datetime | None = None) -> str:
    value = ts or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


@dataclass
class RunState:
    """In-memory representation of run lifecycle state."""

    experiment_id: str
    run_id: str
    transitions: list[Mapping[str, Any]]
    metadata: dict[str, Any]
    current_state: str | None = None


class RunLifecycleTracker:
    """Persists lifecycle transitions for scheduled paper runs."""

    def __init__(self, experiment_id: str, run_id: str, *, data_root: Path | None = None) -> None:
        base = Path(data_root) if data_root else get_data_root()
        self.run_dir = base / "paper" / experiment_id / "runs" / run_id
        self.state_path = self.run_dir / "state.json"
        self._state = self._load(experiment_id, run_id)

    @property
    def current_state(self) -> str | None:
        return self._state.current_state

    @property
    def transitions(self) -> list[Mapping[str, Any]]:
        return list(self._state.transitions)

    def is_terminal(self) -> bool:
        return self.current_state in TERMINAL_STATES

    def record_transition(
        self,
        state: str,
        *,
        timestamp: datetime | str | None = None,
        **details: Any,
    ) -> None:
        if state not in RUN_STATES:
            raise ValueError(f"Unsupported run lifecycle state '{state}'")
        if isinstance(timestamp, str):
            ts = timestamp
        else:
            ts = _now_iso(timestamp)
        entry = {"state": state, "timestamp": ts, "details": dict(details)}
        self._state.transitions.append(entry)
        self._state.current_state = state
        self._persist()

    def update_metadata(self, **fields: Any) -> None:
        self._state.metadata.update(fields)
        self._persist()

    def halt_reasons(self) -> list[str]:
        reasons: list[str] = []
        for entry in self._state.transitions:
            if entry.get("state") == "HALTED":
                reason = entry.get("details", {}).get("reason")
                if reason:
                    reasons.append(str(reason))
        return reasons

    def _load(self, experiment_id: str, run_id: str) -> RunState:
        if self.state_path.exists():
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            transitions = list(payload.get("transitions", []))
            metadata = dict(payload.get("metadata", {}))
            current = payload.get("current_state")
            return RunState(
                experiment_id=experiment_id,
                run_id=run_id,
                transitions=transitions,
                metadata=metadata,
                current_state=current,
            )
        return RunState(
            experiment_id=experiment_id,
            run_id=run_id,
            transitions=[],
            metadata={},
            current_state=None,
        )

    def _persist(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "experiment_id": self._state.experiment_id,
            "run_id": self._state.run_id,
            "current_state": self._state.current_state,
            "transitions": self._state.transitions,
            "metadata": self._state.metadata,
        }
        temp_path = self.state_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(self.state_path)


__all__ = ["RUN_STATES", "TERMINAL_STATES", "RunLifecycleTracker"]
