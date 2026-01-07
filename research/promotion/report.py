"""Qualification report and promotion record helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from research.experiments.registry import ExperimentRecord

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMOTION_ROOT = PROJECT_ROOT / ".quanto_data" / "promotions"


@dataclass(frozen=True)
class QualificationReport:
    """Structured payload persisted for each qualification run."""

    experiment_id: str
    baseline_experiment_id: str
    passed: bool
    failed_hard: Sequence[str]
    failed_soft: Sequence[str]
    metrics_snapshot: Mapping[str, Any]
    gate_summary: Mapping[str, Any]
    sweep_name: str | None = None
    sweep_summary: Mapping[str, Any] | None = None

    def to_dict(self) -> Mapping[str, Any]:
        payload = {
            "experiment_id": self.experiment_id,
            "baseline_experiment_id": self.baseline_experiment_id,
            "passed": bool(self.passed),
            "failed_hard": list(self.failed_hard),
            "failed_soft": list(self.failed_soft),
            "metrics_snapshot": self.metrics_snapshot,
            "gate_summary": self.gate_summary,
        }
        if self.sweep_name:
            payload["sweep_name"] = self.sweep_name
        if self.sweep_summary is not None:
            payload["sweep_summary"] = self.sweep_summary
        return payload


def write_qualification_report(record: ExperimentRecord, report: QualificationReport) -> Path:
    """Persist the qualification report under the experiment promotion directory."""

    record.promotion_dir.mkdir(parents=True, exist_ok=True)
    path = record.promotion_dir / "qualification_report.json"
    payload = report.to_dict()
    path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return path


@dataclass(frozen=True)
class PromotionRecord:
    """Immutable promotion approval record."""

    experiment_id: str
    tier: str
    qualification_report_path: str
    spec_path: str
    metrics_path: str
    promotion_reason: str

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "tier": self.tier,
            "qualification_report_path": self.qualification_report_path,
            "spec_path": self.spec_path,
            "metrics_path": self.metrics_path,
            "promotion_reason": self.promotion_reason,
        }


def write_promotion_record(record: PromotionRecord, *, root: Path | None = None) -> Path:
    """Write an immutable promotion record, raising if conflicting payloads exist."""

    base = Path(root) if root else DEFAULT_PROMOTION_ROOT
    tier_dir = base / record.tier
    tier_dir.mkdir(parents=True, exist_ok=True)
    path = tier_dir / f"{record.experiment_id}.json"
    payload = record.to_dict()
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"Promotion record already exists with different payload: {path}")
        return path
    path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return path


__all__ = ["DEFAULT_PROMOTION_ROOT", "PromotionRecord", "QualificationReport", "write_promotion_record", "write_qualification_report"]
