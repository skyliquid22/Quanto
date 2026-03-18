"""Execution trace emitter for show_traces lineage tracking."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def emit_trace(
    initializer: str,
    task_title: str,
    run_id: str,
    status: str,
    error: str | None = None,
    duration_s: float | None = None,
    artifacts: list[str] | None = None,
    log_path: str = "monitoring/logs/task_ledger.jsonl",
) -> dict[str, Any]:
    """Emit a JSONL trace record to the task ledger.

    Args:
        initializer: The entity that initiated the task.
        task_title: Human-readable task title.
        run_id: Unique run identifier.
        status: Final status, either 'succeeded' or 'failed'.
        error: Optional error message if failed.
        duration_s: Optional execution duration in seconds.
        artifacts: Optional list of output artifact paths.
        log_path: Path to the JSONL log file.

    Returns:
        The emitted trace record.
    """
    trace = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "task_id": str(uuid.uuid4()),
        "initializer": initializer,
        "task_title": task_title,
        "run_id": run_id,
        "status": status,
        "error": error,
        "duration_s": duration_s,
        "artifacts": artifacts or [],
    }

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trace) + "\n")

    return trace
