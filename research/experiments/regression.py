"""Regression gate evaluation built on top of experiment comparisons."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

from research.eval.metrics import MetricSchemaEntry, metric_schema_entries
from research.experiments.comparator import ComparisonResult, MetricComparison

_TOLERANCE = 1e-9


class GateType(str, Enum):
    HARD = "hard"
    SOFT = "soft"


@dataclass(frozen=True)
class RegressionGateRule:
    gate_id: str
    metric_id: str
    gate_type: GateType
    max_degradation_pct: float | None = None
    max_value: float | None = None
    description: str | None = None


@dataclass(frozen=True)
class GateEvaluation:
    gate_id: str
    metric_id: str
    gate_type: str
    status: str
    message: str
    observed: Dict[str, float | None]
    thresholds: Dict[str, float | None]


@dataclass(frozen=True)
class RegressionGateReport:
    overall_status: str
    hard_failures: int
    soft_warnings: int
    evaluations: List[GateEvaluation]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status,
            "hard_failures": self.hard_failures,
            "soft_warnings": self.soft_warnings,
            "evaluations": [asdict(entry) for entry in self.evaluations],
        }


def evaluate_gates(
    comparison: ComparisonResult,
    gate_rules: Sequence[RegressionGateRule],
) -> RegressionGateReport:
    """Evaluate configured gates and summarize pass/fail state."""

    schema_map = {entry.metric_id: entry for entry in metric_schema_entries()}
    metric_map = {entry.metric_id: entry for entry in comparison.metrics}

    evaluations: List[GateEvaluation] = []
    hard_failures = 0
    soft_warnings = 0

    for rule in gate_rules:
        schema_entry = schema_map.get(rule.metric_id)
        if schema_entry is None:
            raise ValueError(f"Unknown metric_id '{rule.metric_id}' in gate '{rule.gate_id}'.")
        metric_entry = metric_map.get(rule.metric_id)
        evaluation = _evaluate_rule(rule, schema_entry, metric_entry)
        evaluations.append(evaluation)
        if evaluation.status == "fail":
            hard_failures += 1
        elif evaluation.status == "warn":
            soft_warnings += 1

    overall_status = "fail" if hard_failures else "pass"
    return RegressionGateReport(
        overall_status=overall_status,
        hard_failures=hard_failures,
        soft_warnings=soft_warnings,
        evaluations=evaluations,
    )


def load_gate_rules(config_path: str | Path | None) -> List[RegressionGateRule]:
    """Load gate rules from YAML/JSON; fall back to defaults when not provided."""

    if not config_path:
        return default_gate_rules()
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Gate configuration not found: {path}")
    text = path.read_text(encoding="utf-8")
    data: Mapping[str, Any] | None = None
    if yaml is not None:
        data = yaml.safe_load(text)  # type: ignore[assignment]
    if data is None:
        data = json.loads(text)
    gates = data.get("gates")
    if not isinstance(gates, list):
        raise ValueError("Gate configuration must include a 'gates' list.")
    rules: List[RegressionGateRule] = []
    for idx, entry in enumerate(gates):
        if not isinstance(entry, Mapping):
            raise ValueError("Each gate entry must be a mapping.")
        metric_id = str(entry.get("metric") or "")
        if not metric_id:
            raise ValueError("Gate entries must include 'metric'.")
        gate_id = str(entry.get("id") or entry.get("name") or f"gate_{idx}")
        gate_type_token = str(entry.get("type") or GateType.HARD.value).lower()
        if gate_type_token not in (GateType.HARD.value, GateType.SOFT.value):
            raise ValueError(f"Gate '{gate_id}' has invalid type '{gate_type_token}'.")
        pct_value = entry.get("max_degradation_pct")
        if pct_value is None:
            pct_value = entry.get("threshold_pct")
        if pct_value is not None:
            pct_value = float(pct_value)
        max_value = entry.get("max_value")
        if max_value is not None:
            max_value = float(max_value)
        if pct_value is None and max_value is None:
            raise ValueError(f"Gate '{gate_id}' must set max_degradation_pct or max_value.")
        description = entry.get("description")
        rules.append(
            RegressionGateRule(
                gate_id=gate_id,
                metric_id=metric_id,
                gate_type=GateType(gate_type_token),
                max_degradation_pct=pct_value,
                max_value=max_value,
                description=str(description) if description is not None else None,
            )
        )
    return rules


def default_gate_rules() -> List[RegressionGateRule]:
    """Default regression gates for baseline enforcement."""

    return [
        RegressionGateRule(
            gate_id="sharpe_guard",
            metric_id="performance.sharpe",
            gate_type=GateType.HARD,
            max_degradation_pct=5.0,
            description="Sharpe must not decrease by more than 5%.",
        ),
        RegressionGateRule(
            gate_id="drawdown_guard",
            metric_id="performance.max_drawdown",
            gate_type=GateType.SOFT,
            max_degradation_pct=10.0,
            description="Max drawdown must not increase by more than 10%.",
        ),
        RegressionGateRule(
            gate_id="turnover_guard",
            metric_id="trading.turnover_1d_mean",
            gate_type=GateType.SOFT,
            max_degradation_pct=15.0,
            description="Turnover must not increase by more than 15%.",
        ),
        RegressionGateRule(
            gate_id="nan_inf_guard",
            metric_id="safety.nan_inf_violations",
            gate_type=GateType.HARD,
            max_value=0.0,
            description="NaN/Inf violations must remain zero.",
        ),
    ]


def _evaluate_rule(
    rule: RegressionGateRule,
    schema_entry: MetricSchemaEntry,
    metric_entry: MetricComparison | None,
) -> GateEvaluation:
    observed: Dict[str, float | None] = {}
    thresholds: Dict[str, float | None] = {
        "max_degradation_pct": rule.max_degradation_pct,
        "max_value": rule.max_value,
    }
    message = "within thresholds"
    violated = False

    if metric_entry is None:
        message = "Metric not present in comparison results."
        violated = True
    elif metric_entry.reason:
        message = f"Metric unavailable: {metric_entry.reason}"
        violated = True
        observed.update(
            {
                "candidate": metric_entry.candidate_value,
                "baseline": metric_entry.baseline_value,
                "delta": metric_entry.delta,
                "delta_pct": metric_entry.delta_pct,
            }
        )
    else:
        observed.update(
            {
                "candidate": metric_entry.candidate_value,
                "baseline": metric_entry.baseline_value,
                "delta": metric_entry.delta,
                "delta_pct": metric_entry.delta_pct,
            }
        )
        violation_msgs: List[str] = []
        if rule.max_degradation_pct is not None:
            violation, degrade_pct = _relative_violation(
                schema_entry.direction,
                metric_entry.candidate_value or 0.0,
                metric_entry.baseline_value or 0.0,
                rule.max_degradation_pct,
            )
            observed["degradation_pct"] = degrade_pct
            if violation:
                violated = True
                pct_text = (
                    f"{_round(degrade_pct, 4)}%" if degrade_pct is not None else "an unknown amount"
                )
                violation_msgs.append(
                    f"{schema_entry.metric_id} degraded by "
                    f"{pct_text} (limit {rule.max_degradation_pct}%)."
                )
        if rule.max_value is not None and metric_entry.candidate_value is not None:
            if metric_entry.candidate_value > rule.max_value + _TOLERANCE:
                violated = True
                violation_msgs.append(
                    f"{schema_entry.metric_id} = {metric_entry.candidate_value} exceeds {rule.max_value}."
                )
        if violation_msgs:
            message = " ".join(violation_msgs)

    status = "pass"
    if violated:
        status = "fail" if rule.gate_type is GateType.HARD else "warn"
    if not violated and rule.description:
        message = rule.description

    return GateEvaluation(
        gate_id=rule.gate_id,
        metric_id=rule.metric_id,
        gate_type=rule.gate_type.value,
        status=status,
        message=message,
        observed=observed,
        thresholds=thresholds,
    )


def _relative_violation(
    direction: str,
    candidate_value: float,
    baseline_value: float,
    threshold_pct: float,
) -> tuple[bool, float | None]:
    degrade = _degradation_amount(direction, candidate_value, baseline_value)
    allowed = abs(baseline_value) * (threshold_pct / 100.0)
    allowed += _TOLERANCE
    if degrade <= allowed:
        return False, _degradation_pct(degrade, baseline_value)
    return True, _degradation_pct(degrade, baseline_value)


def _degradation_amount(direction: str, candidate_value: float, baseline_value: float) -> float:
    delta = candidate_value - baseline_value
    if direction == "lower_is_better":
        return max(0.0, delta)
    # Treat neutral like higher_is_better to avoid silent skips.
    return max(0.0, -delta)


def _degradation_pct(degrade_amount: float, baseline_value: float) -> float | None:
    denom = abs(baseline_value)
    if denom <= _TOLERANCE:
        if degrade_amount <= _TOLERANCE:
            return 0.0
        return None
    value = (degrade_amount / denom) * 100.0
    return _round(value, 6)


def _round(value: float, precision: int) -> float:
    rounded = round(float(value), precision)
    if rounded == -0.0:
        return 0.0
    return rounded


__all__ = [
    "RegressionGateReport",
    "RegressionGateRule",
    "GateEvaluation",
    "GateType",
    "default_gate_rules",
    "load_gate_rules",
    "evaluate_gates",
]
