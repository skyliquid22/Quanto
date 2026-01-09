"""Filesystem alert emission for paper ops."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from infra.paths import get_data_root


class AlertEmitter:
    """Writes alert payloads to the monitoring root."""

    def __init__(self, base_dir: Path | None = None) -> None:
        root = Path(base_dir) if base_dir else get_data_root() / "monitoring" / "alerts"
        root.mkdir(parents=True, exist_ok=True)
        self._root = root

    def emit(
        self,
        *,
        severity: str,
        kind: str,
        message: str,
        experiment_id: str,
        run_id: str | None = None,
        context: Mapping[str, Any] | None = None,
        timestamp: datetime | None = None,
        runbook_url: str | None = None,
    ) -> Path:
        normalized = severity.lower()
        if normalized not in {"hard", "soft"}:
            raise ValueError("severity must be 'hard' or 'soft'")
        ts = timestamp or datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        entry = {
            "timestamp": ts.isoformat(),
            "severity": normalized,
            "kind": kind,
            "message": message,
            "experiment_id": experiment_id,
            "run_id": run_id,
            "context": dict(context or {}),
        }
        if runbook_url:
            entry["runbook_url"] = runbook_url
        date_key = ts.strftime("%Y%m%d")
        alert_path = self._root / f"{date_key}.json"
        if alert_path.exists():
            existing = json.loads(alert_path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        else:
            existing = []
        existing.append(entry)
        alert_path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")
        return alert_path


__all__ = ["AlertEmitter"]
