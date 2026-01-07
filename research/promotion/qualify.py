"""Qualification pipeline orchestration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from research.experiments.registry import ExperimentRegistry
from research.experiments.regression import RegressionGateRule
from research.promotion.criteria import QualificationCriteria
from research.promotion.report import (
    DEFAULT_PROMOTION_ROOT,
    QualificationReport,
    write_qualification_report,
)


@dataclass(frozen=True)
class QualificationResult:
    """Return value for qualification pipelines."""

    report: QualificationReport
    report_path: Path


def run_qualification(
    experiment_id: str,
    baseline_identifier: str,
    *,
    registry: ExperimentRegistry | None = None,
    gate_rules: Sequence[RegressionGateRule] | None = None,
    sweep_name: str | None = None,
    criteria: QualificationCriteria | None = None,
) -> QualificationResult:
    """Run the qualification pipeline and persist a deterministic report."""

    registry = registry or ExperimentRegistry()
    candidate_record = registry.get(experiment_id)
    baseline_record = registry.resolve_identifier(baseline_identifier)
    criteria = criteria or QualificationCriteria()
    evaluation = criteria.evaluate(
        candidate_record,
        baseline_record,
        registry=registry,
        gate_rules=gate_rules,
        sweep_name=sweep_name,
    )
    report = QualificationReport(
        experiment_id=candidate_record.experiment_id,
        baseline_experiment_id=baseline_record.experiment_id,
        sweep_name=sweep_name,
        passed=evaluation.passed,
        failed_hard=evaluation.failed_hard,
        failed_soft=evaluation.failed_soft,
        metrics_snapshot=evaluation.metrics_snapshot,
        gate_summary=evaluation.gate_summary,
        sweep_summary=evaluation.sweep_summary,
    )
    path = write_qualification_report(candidate_record, report)
    return QualificationResult(report=report, report_path=path)


__all__ = ["QualificationResult", "run_qualification"]


def is_experiment_promoted(
    experiment_id: str,
    *,
    promotion_root: Path | None = None,
    tier: str | None = None,
) -> bool:
    """Return True when an immutable promotion record exists for experiment_id."""

    if not experiment_id:
        raise ValueError("experiment_id must be provided to check promotion status.")
    base = Path(promotion_root) if promotion_root else DEFAULT_PROMOTION_ROOT
    tiers = (tier,) if tier else ("research", "candidate", "production")
    for label in tiers:
        if not label:
            continue
        record_path = base / label / f"{experiment_id}.json"
        if record_path.exists():
            return True
    return False


__all__.append("is_experiment_promoted")
