"""Scheduling + metadata tracking for paper trading runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from infra.paths import get_data_root
from research.ops.config import PaperOpsConfig
from research.ops.lifecycle import RunLifecycleTracker, TERMINAL_STATES

try:  # Python 3.9+
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - zoneinfo always available in Py3.10
    ZoneInfo = None


@dataclass(frozen=True)
class MissedRun:
    """Represents a scheduled run window that was never started."""

    run_id: str
    scheduled_for: datetime
    reason: str


@dataclass(frozen=True)
class ScheduleDecision:
    """Output of the scheduler for a given evaluation cycle."""

    run_id: str | None
    scheduled_for: datetime | None
    due: bool
    resume_run_id: str | None
    missed: list[MissedRun]


class CronField:
    """Parses and evaluates a single cron component."""

    def __init__(self, expr: str, *, minimum: int, maximum: int, allow_7: bool = False) -> None:
        values, wildcard = self._parse(expr, minimum=minimum, maximum=maximum, allow_7=allow_7)
        self.values = values
        self.is_wildcard = wildcard

    def _parse(
        self,
        expr: str,
        *,
        minimum: int,
        maximum: int,
        allow_7: bool,
    ) -> tuple[set[int], bool]:
        expr = expr.strip()
        if not expr:
            raise ValueError("Cron expressions cannot contain empty fields.")
        wildcard = expr == "*"
        tokens = expr.split(",")
        values: set[int] = set()
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            step = 1
            if "/" in token:
                token, step_str = token.split("/")
                step = max(1, int(step_str))
            if token == "*" or not token:
                start, end = minimum, maximum
            elif "-" in token:
                start_str, end_str = token.split("-")
                start = int(start_str)
                end = int(end_str)
            else:
                start = end = int(token)
            start = max(minimum, start)
            end = min(maximum, end)
            for value in range(start, end + 1, step):
                if allow_7 and value == 7:
                    values.add(0)
                else:
                    values.add(value)
        if not values:
            values = set(range(minimum, maximum + 1))
            wildcard = True
        return values, wildcard

    def matches(self, value: int) -> bool:
        return value in self.values


class CronSchedule:
    """Very small cron evaluator covering minute/hour/day/month/dow fields."""

    def __init__(self, expr: str) -> None:
        fields = expr.split()
        if len(fields) != 5:
            raise ValueError("Cron expression must have 5 fields (m h dom mon dow).")
        self._minute = CronField(fields[0], minimum=0, maximum=59)
        self._hour = CronField(fields[1], minimum=0, maximum=23)
        self._dom = CronField(fields[2], minimum=1, maximum=31)
        self._month = CronField(fields[3], minimum=1, maximum=12)
        self._dow = CronField(fields[4], minimum=0, maximum=6, allow_7=True)

    def matches(self, timestamp: datetime) -> bool:
        dom_match = self._dom.matches(timestamp.day)
        dow = (timestamp.weekday() + 1) % 7  # Sunday=0
        dow_match = self._dow.matches(dow)
        if self._dom.is_wildcard and self._dow.is_wildcard:
            day_match = True
        elif self._dom.is_wildcard:
            day_match = dow_match
        elif self._dow.is_wildcard:
            day_match = dom_match
        else:
            day_match = dom_match or dow_match
        return (
            self._minute.matches(timestamp.minute)
            and self._hour.matches(timestamp.hour)
            and self._month.matches(timestamp.month)
            and day_match
        )

    def previous(self, timestamp: datetime, *, limit_minutes: int = 60 * 24 * 370) -> datetime | None:
        current = timestamp.replace(second=0, microsecond=0)
        for _ in range(limit_minutes):
            if self.matches(current):
                return current
            current -= timedelta(minutes=1)
        return None

    def slots_since(self, last_slot: datetime | None, now: datetime, *, limit: int = 366) -> list[datetime]:
        cursor = now
        slots: list[datetime] = []
        iterations = 0
        while iterations < limit:
            slot = self.previous(cursor)
            if slot is None:
                break
            if last_slot and slot <= last_slot:
                break
            slots.append(slot)
            cursor = slot - timedelta(minutes=1)
            iterations += 1
            if last_slot is None:
                break
        return sorted(slots)


class PaperRunScheduler:
    """Scheduler + metadata persistence for paper trading runs."""

    def __init__(
        self,
        experiment_id: str,
        schedule_config: PaperOpsConfig,
        *,
        data_root: Path | None = None,
    ) -> None:
        self.experiment_id = experiment_id
        self._config = schedule_config
        self._cron = CronSchedule(schedule_config.cron)
        tz_name = schedule_config.timezone or "UTC"
        if ZoneInfo is not None:
            try:
                self._tz = ZoneInfo(tz_name)
            except Exception:  # pragma: no cover - fallback path
                self._tz = timezone.utc
        else:  # pragma: no cover - zoneinfo always available
            self._tz = timezone.utc
        self._grace = timedelta(minutes=schedule_config.grace_minutes or 0)
        self._data_root = Path(data_root) if data_root else get_data_root()
        self._metadata_path = self._data_root / "paper" / experiment_id / "schedule.json"
        self._metadata = self._load_metadata()

    def evaluate(self, *, base_run_id: str, now: datetime | None = None) -> ScheduleDecision:
        """Evaluate scheduling status for the current instant."""

        horizon = (now or datetime.now(timezone.utc)).astimezone(self._tz)
        missed: list[MissedRun] = []
        active_run_id = self._metadata.get("active_run_id")
        if active_run_id:
            tracker = RunLifecycleTracker(self.experiment_id, active_run_id, data_root=self._data_root)
            if tracker.current_state in TERMINAL_STATES:
                self._clear_active()
            else:
                scheduled_for = self._parse_time(self._metadata.get("active_slot")) or horizon
                return ScheduleDecision(
                    run_id=active_run_id,
                    scheduled_for=scheduled_for,
                    due=True,
                    resume_run_id=active_run_id,
                    missed=[],
                )
        last_recorded = self._parse_time(self._metadata.get("last_recorded_slot"))
        slots = self._cron.slots_since(last_recorded, horizon)
        due_slot: datetime | None = None
        for slot in slots:
            if horizon - slot > self._grace:
                run_id = self._slot_run_id(base_run_id, slot)
                missed.append(MissedRun(run_id=run_id, scheduled_for=slot, reason="window_expired"))
                self._metadata["last_recorded_slot"] = slot.isoformat()
                self._append_history(run_id, "MISSED", slot, horizon, reason="window_expired")
            else:
                due_slot = slot
                break
        self._metadata["last_evaluated_at"] = horizon.isoformat()
        self._save_metadata()
        run_id = self._slot_run_id(base_run_id, due_slot) if due_slot else None
        return ScheduleDecision(
            run_id=run_id,
            scheduled_for=due_slot,
            due=bool(due_slot),
            resume_run_id=None,
            missed=missed,
        )

    def mark_active(self, run_id: str, scheduled_for: datetime) -> None:
        scheduled = scheduled_for.astimezone(self._tz).isoformat()
        self._metadata["active_run_id"] = run_id
        self._metadata["active_slot"] = scheduled
        self._metadata["last_recorded_slot"] = scheduled
        self._append_history(run_id, "STARTING", scheduled_for, datetime.now(timezone.utc))
        self._save_metadata()

    def mark_terminal(self, run_id: str, state: str) -> None:
        if self._metadata.get("active_run_id") == run_id:
            self._metadata["active_run_id"] = None
            self._metadata["active_slot"] = None
        recorded_at = datetime.now(timezone.utc)
        scheduled_for = self._parse_time(self._metadata.get("last_recorded_slot"))
        self._append_history(run_id, state, scheduled_for, recorded_at)
        self._save_metadata()

    def _slot_run_id(self, base_run_id: str, slot: datetime | None) -> str | None:
        if slot is None:
            return None
        suffix = slot.strftime("%Y%m%d%H%M")
        return f"{base_run_id}_{suffix}"

    def _parse_time(self, value: Any) -> datetime | None:
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        try:
            parsed = datetime.fromisoformat(str(value))
        except Exception:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=self._tz)
        return parsed

    def _load_metadata(self) -> dict[str, Any]:
        if self._metadata_path.exists():
            return json.loads(self._metadata_path.read_text(encoding="utf-8"))
        return {
            "experiment_id": self.experiment_id,
            "last_recorded_slot": None,
            "active_run_id": None,
            "active_slot": None,
            "history": [],
        }

    def _save_metadata(self) -> None:
        self._metadata_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._metadata_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(self._metadata, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(self._metadata_path)

    def _append_history(
        self,
        run_id: str,
        state: str,
        scheduled_for: datetime | None,
        recorded_at: datetime,
        *,
        reason: str | None = None,
    ) -> None:
        history = list(self._metadata.get("history") or [])
        entry = {
            "run_id": run_id,
            "state": state,
            "scheduled_for": scheduled_for.isoformat() if isinstance(scheduled_for, datetime) else scheduled_for,
            "recorded_at": recorded_at.astimezone(timezone.utc).isoformat(),
        }
        if reason:
            entry["reason"] = reason
        history.append(entry)
        self._metadata["history"] = history[-50:]

    def _clear_active(self) -> None:
        self._metadata["active_run_id"] = None
        self._metadata["active_slot"] = None
        self._save_metadata()


__all__ = ["CronSchedule", "MissedRun", "PaperRunScheduler", "ScheduleDecision"]
