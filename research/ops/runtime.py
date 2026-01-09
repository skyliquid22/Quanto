"""Run execution helpers (retries/backoff/alerts)."""

from __future__ import annotations

import time
from typing import Any, Callable

from research.ops.alerts import AlertEmitter
from research.ops.config import BackoffPolicyConfig
from research.ops.lifecycle import RunLifecycleTracker
from research.ops.scheduler import PaperRunScheduler


class BrokerRetryableError(RuntimeError):
    """Raised when the broker is temporarily unavailable and the run should retry."""


class RunHaltError(RuntimeError):
    """Raised when the run must halt immediately (e.g., hard gate failure)."""


class RunExecutor:
    """Executes a run callable with lifecycle accounting and backoff."""

    def __init__(
        self,
        *,
        experiment_id: str,
        run_id: str,
        lifecycle: RunLifecycleTracker,
        scheduler: PaperRunScheduler,
        alert_emitter: AlertEmitter,
        backoff_config: BackoffPolicyConfig,
        runbook_url: str | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self._experiment_id = experiment_id
        self._run_id = run_id
        self._lifecycle = lifecycle
        self._scheduler = scheduler
        self._alert_emitter = alert_emitter
        self._backoff = backoff_config
        self._runbook_url = runbook_url
        self._sleep = sleep_fn or time.sleep

    def execute(self, run_callable: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        attempts = max(1, int(self._backoff.max_attempts))
        delay = max(0.0, float(self._backoff.initial_seconds))
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            self._lifecycle.record_transition("STARTING", attempt=attempt)
            self._lifecycle.record_transition("RUNNING", attempt=attempt)
            try:
                result = run_callable()
            except BrokerRetryableError as exc:
                last_error = exc
                self._lifecycle.record_transition("HALTED", attempt=attempt, reason=str(exc), classification="retryable")
                if attempt >= attempts:
                    self._finalize("FAILED", reason=str(exc))
                    raise
                self._sleep(delay)
                delay = min(delay * max(1.0, float(self._backoff.multiplier)), float(self._backoff.max_seconds))
                continue
            except RunHaltError as exc:
                last_error = exc
                self._lifecycle.record_transition("HALTED", attempt=attempt, reason=str(exc))
                self._finalize("HALTED", reason=str(exc))
                raise
            except Exception as exc:  # pragma: no cover - defensive guard
                last_error = exc
                self._lifecycle.record_transition("FAILED", attempt=attempt, reason=str(exc))
                self._finalize("FAILED", reason=str(exc))
                raise
            else:
                self._lifecycle.record_transition("COMPLETED", attempt=attempt)
                self._finalize("COMPLETED")
                return result or {}
        if last_error:
            raise last_error
        raise RuntimeError("Run executor exhausted attempts without producing a result.")

    def _finalize(self, state: str, *, reason: str | None = None) -> None:
        self._scheduler.mark_terminal(self._run_id, state)
        if state in {"FAILED", "HALTED"}:
            message = f"run {self._run_id} {state.lower()}"
            if reason:
                message = f"{message}: {reason}"
            self._alert_emitter.emit(
                severity="hard",
                kind=f"run_{state.lower()}",
                message=message,
                experiment_id=self._experiment_id,
                run_id=self._run_id,
                context={"state": state, "reason": reason} if reason else {"state": state},
                runbook_url=self._runbook_url,
            )


__all__ = ["BrokerRetryableError", "RunExecutor", "RunHaltError"]
