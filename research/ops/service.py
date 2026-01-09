"""Service orchestration for scheduled paper runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from research.ops.alerts import AlertEmitter
from research.ops.config import OpsConfig
from research.ops.lifecycle import RunLifecycleTracker
from research.ops.runtime import RunExecutor
from research.ops.scheduler import MissedRun, PaperRunScheduler
from research.paper.config import PaperRunConfig
from research.paper.run import PaperRunner, derive_run_id
from research.paper.summary import DailySummaryWriter


def _ensure_datetime(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


@dataclass(frozen=True)
class RunReport:
    """Return payload for orchestrator executions."""

    status: str
    run_id: str | None
    summary_json: Path | None
    summary_markdown: Path | None
    details: dict[str, Any]


class PaperRunOrchestrator:
    """Coordinates scheduling, lifecycle, alerts, and summaries."""

    def __init__(
        self,
        *,
        paper_config: PaperRunConfig,
        ops_config: OpsConfig,
        runner_factory: Callable[..., PaperRunner] | None = None,
        alert_emitter: AlertEmitter | None = None,
        scheduler: PaperRunScheduler | None = None,
        data_root: Path | None = None,
    ) -> None:
        self._paper_config = paper_config
        self._ops_config = ops_config
        self._runner_factory = runner_factory or PaperRunner
        self._alerts = alert_emitter or AlertEmitter()
        self._scheduler = scheduler or PaperRunScheduler(
            paper_config.experiment_id,
            ops_config.paper_trading,
            data_root=data_root,
        )

    def run(self, *, now: datetime | None = None) -> RunReport:
        current_time = _ensure_datetime(now)
        base_run_id = derive_run_id(self._paper_config)
        decision = self._scheduler.evaluate(base_run_id=base_run_id, now=current_time)
        for missed in decision.missed:
            self._handle_missed(missed, current_time)
        if not decision.due or decision.run_id is None or decision.scheduled_for is None:
            return RunReport(
                status="IDLE",
                run_id=None,
                summary_json=None,
                summary_markdown=None,
                details={"missed_runs": [missed.run_id for missed in decision.missed]},
            )
        lifecycle = RunLifecycleTracker(self._paper_config.experiment_id, decision.run_id)
        if lifecycle.current_state is None or lifecycle.current_state == "SCHEDULED":
            lifecycle.record_transition(
                "SCHEDULED",
                scheduled_for=decision.scheduled_for.isoformat(),
            )
        lifecycle.update_metadata(scheduled_for=decision.scheduled_for.isoformat())
        self._scheduler.mark_active(decision.run_id, decision.scheduled_for)
        executor = RunExecutor(
            experiment_id=self._paper_config.experiment_id,
            run_id=decision.run_id,
            lifecycle=lifecycle,
            scheduler=self._scheduler,
            alert_emitter=self._alerts,
            backoff_config=self._ops_config.paper_trading.backoff,
            runbook_url=self._ops_config.paper_trading.runbook_url,
        )
        run_result = executor.execute(
            lambda: self._execute_run(decision.run_id, decision.scheduled_for, lifecycle)
        )
        summaries = self._write_summary(
            lifecycle,
            run_result,
            decision.run_id,
            decision.scheduled_for,
            current_time,
        )
        self._maybe_emit_soft_alerts(summaries)
        return RunReport(
            status="COMPLETED",
            run_id=decision.run_id,
            summary_json=summaries.get("json"),
            summary_markdown=summaries.get("markdown"),
            details={"run_dir": str(run_result.get("run_dir")), "payload": summaries.get("payload")},
        )

    def _handle_missed(self, missed: MissedRun, now: datetime) -> None:
        tracker = RunLifecycleTracker(self._paper_config.experiment_id, missed.run_id)
        tracker.record_transition("SCHEDULED", scheduled_for=missed.scheduled_for.isoformat())
        tracker.record_transition("MISSED", recorded_at=now.isoformat(), reason=missed.reason)
        self._alerts.emit(
            severity="hard",
            kind="run_missed",
            message=f"Missed paper run {missed.run_id} scheduled at {missed.scheduled_for.isoformat()}",
            experiment_id=self._paper_config.experiment_id,
            run_id=missed.run_id,
            context={"reason": missed.reason},
            runbook_url=self._ops_config.paper_trading.runbook_url,
        )

    def _execute_run(
        self,
        run_id: str,
        scheduled_for: datetime,
        lifecycle: RunLifecycleTracker,
    ) -> dict[str, Any]:
        runner = self._runner_factory(
            self._paper_config,
            run_id=run_id,
            scheduled_for=scheduled_for.isoformat(),
        )
        lifecycle.update_metadata(run_dir=str(runner.run_dir))
        # TODO: replace placeholders with real execution metrics once brokers expose them.
        metrics = {
            "pnl": 0.0,
            "exposure": 0.0,
            "turnover": 0.0,
            "fees": 0.0,
        }
        return {"run_dir": str(runner.run_dir), "metrics": metrics, "run_id": run_id}

    def _write_summary(
        self,
        lifecycle: RunLifecycleTracker,
        run_result: dict[str, Any],
        run_id: str,
        scheduled_for: datetime,
        completed_at: datetime,
    ) -> dict[str, Any]:
        run_dir = Path(str(run_result.get("run_dir")))
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics = run_result.get("metrics") or {}
        payload = {
            "pnl": float(metrics.get("pnl", 0.0)),
            "exposure": float(metrics.get("exposure", 0.0)),
            "turnover": float(metrics.get("turnover", 0.0)),
            "fees": float(metrics.get("fees", 0.0)),
            "halt_reasons": lifecycle.halt_reasons(),
            "run_id": run_id,
            "experiment_id": self._paper_config.experiment_id,
            "scheduled_for": scheduled_for.isoformat(),
            "completed_at": completed_at.isoformat(),
            "status": lifecycle.current_state,
        }
        writer = DailySummaryWriter(run_dir)
        date_key = completed_at.strftime("%Y%m%d")
        json_path, md_path = writer.write(date_key, payload)
        return {"json": json_path, "markdown": md_path, "payload": payload}

    def _maybe_emit_soft_alerts(self, summary: dict[str, Any]) -> None:
        expect_trades = self._ops_config.paper_trading.expect_trades
        payload = summary.get("payload") or {}
        turnover = float(payload.get("turnover", 0.0))
        if expect_trades and turnover == 0.0:
            self._alerts.emit(
                severity="soft",
                kind="no_trades",
                message=f"No trades executed for run {payload.get('run_id')}",
                experiment_id=self._paper_config.experiment_id,
                run_id=str(payload.get("run_id")),
                context={"summary": payload},
                runbook_url=self._ops_config.paper_trading.runbook_url,
            )


__all__ = ["PaperRunOrchestrator", "RunReport"]
