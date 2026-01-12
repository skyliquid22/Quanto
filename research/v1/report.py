"""Utilities for building deterministic v1 release reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from research.experiments.spec import ExperimentSpec
from research.promotion.report import QualificationReport


def build_release_report(
    *,
    run_id: str,
    spec: ExperimentSpec,
    baseline_id: str,
    qualification: QualificationReport,
    qualification_report_path: Path,
    shadow_summary: Mapping[str, Any] | None,
    promotion_record_path: Path | None,
    promotion_tier: str | None,
    release_dir: Path,
) -> dict[str, Any]:
    """Assemble the canonical release report payload."""

    reward_version = _reward_version(spec)
    qualification_payload = qualification.to_dict()
    qualification_payload["report_path"] = str(qualification_report_path)
    shadow_payload = _shadow_payload(shadow_summary)
    promotion_payload = {
        "tier": promotion_tier if promotion_record_path else None,
        "record_path": str(promotion_record_path) if promotion_record_path else None,
    }
    artifacts = {
        "experiment_dir": str(release_dir / "experiment"),
        "shadow_dir": str(release_dir / "shadow") if shadow_summary else None,
        "report_dir": str(release_dir / "report"),
    }
    ready = bool(
        qualification.passed
        and promotion_record_path is not None
        and shadow_summary is not None
        and shadow_summary.get("completed")
    )
    return {
        "run_id": run_id,
        "experiment_id": spec.experiment_id,
        "baseline_id": baseline_id,
        "feature_set": spec.feature_set,
        "regime_feature_set": spec.regime_feature_set,
        "policy": spec.policy,
        "reward_version": reward_version,
        "qualification": qualification_payload,
        "regression": qualification_payload.get("gate_summary"),
        "shadow": shadow_payload,
        "promotion": promotion_payload,
        "artifacts": artifacts,
        "release_dir": str(release_dir),
        "v1_ready": ready,
    }


def write_release_report(report_dir: Path, payload: Mapping[str, Any]) -> Path:
    """Persist the release report as JSON under report_dir."""

    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "v1_release_report.json"
    path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return path


def _reward_version(spec: ExperimentSpec) -> str | None:
    version = spec.policy_params.get("reward_version")
    if version is None:
        return None
    normalized = str(version).strip()
    return normalized or None


def _shadow_payload(summary: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if summary is None:
        return {"status": "skipped", "reason": "qualification_failed"}
    ordered = {key: summary[key] for key in sorted(summary)}
    return ordered


__all__ = ["build_release_report", "write_release_report"]
