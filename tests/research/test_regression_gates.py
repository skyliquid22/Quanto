from __future__ import annotations

from research.experiments.comparator import ComparisonResult, ComparisonSummary, MetricComparison
from research.experiments.regression import GateType, RegressionGateRule, evaluate_gates


def _comparison(metrics: list[MetricComparison]) -> ComparisonResult:
    summary = ComparisonSummary(
        total_metrics=len(metrics),
        compared_metrics=len(metrics),
        num_improved=0,
        num_regressed=0,
        num_unchanged=0,
        num_missing=0,
    )
    return ComparisonResult(
        candidate_experiment_id="candidate",
        baseline_experiment_id="baseline",
        metadata={},
        metrics=metrics,
        summary=summary,
        missing_metrics=[],
        regime_deltas={},
        execution_metrics={},
    )


def _metric(
    *,
    metric_id: str,
    category: str,
    name: str,
    directionality: str,
    candidate: float | None,
    baseline: float | None,
    delta: float | None,
    delta_pct: float | None,
    status: str,
) -> MetricComparison:
    return MetricComparison(
        metric_id=metric_id,
        category=category,
        name=name,
        directionality=directionality,
        candidate_value=candidate,
        baseline_value=baseline,
        delta=delta,
        delta_pct=delta_pct,
        status=status,
    )


def test_sharpe_regression_gate_fails_when_degradation_exceeds_threshold() -> None:
    rule = RegressionGateRule(
        gate_id="sharpe_guard",
        metric_id="performance.sharpe",
        gate_type=GateType.HARD,
        max_degradation_pct=5.0,
    )
    metrics = [
        _metric(
            metric_id="performance.sharpe",
            category="performance",
            name="sharpe",
            directionality="higher_is_better",
            candidate=0.94,
            baseline=1.0,
            delta=-0.06,
            delta_pct=-6.0,
            status="regressed",
        )
    ]
    report = evaluate_gates(_comparison(metrics), [rule])
    assert report.overall_status == "fail"
    assert report.evaluations[0].status == "fail"


def test_sharpe_gate_passes_within_threshold() -> None:
    rule = RegressionGateRule(
        gate_id="sharpe_guard",
        metric_id="performance.sharpe",
        gate_type=GateType.HARD,
        max_degradation_pct=5.0,
    )
    metrics = [
        _metric(
            metric_id="performance.sharpe",
            category="performance",
            name="sharpe",
            directionality="higher_is_better",
            candidate=0.96,
            baseline=1.0,
            delta=-0.04,
            delta_pct=-4.0,
            status="regressed",
        )
    ]
    report = evaluate_gates(_comparison(metrics), [rule])
    assert report.overall_status == "pass"
    assert report.evaluations[0].status == "pass"


def test_sharpe_gate_handles_zero_baseline_as_hard_failure() -> None:
    rule = RegressionGateRule(
        gate_id="sharpe_guard",
        metric_id="performance.sharpe",
        gate_type=GateType.HARD,
        max_degradation_pct=5.0,
    )
    metrics = [
        _metric(
            metric_id="performance.sharpe",
            category="performance",
            name="sharpe",
            directionality="higher_is_better",
            candidate=-0.1,
            baseline=0.0,
            delta=-0.1,
            delta_pct=None,
            status="regressed",
        )
    ]
    report = evaluate_gates(_comparison(metrics), [rule])
    evaluation = report.evaluations[0]
    assert evaluation.status == "fail"
    assert evaluation.observed.get("degradation_pct") is None


def test_max_value_gate_fails_when_exceeded() -> None:
    rule = RegressionGateRule(
        gate_id="nan_inf_guard",
        metric_id="safety.nan_inf_violations",
        gate_type=GateType.HARD,
        max_value=0.0,
    )
    metrics = [
        _metric(
            metric_id="safety.nan_inf_violations",
            category="safety",
            name="nan_inf_violations",
            directionality="lower_is_better",
            candidate=1.0,
            baseline=0.0,
            delta=1.0,
            delta_pct=None,
            status="regressed",
        )
    ]
    report = evaluate_gates(_comparison(metrics), [rule])
    assert report.overall_status == "fail"
    assert report.evaluations[0].status == "fail"


def test_missing_metric_marks_gate_as_failed() -> None:
    rule = RegressionGateRule(
        gate_id="sharpe_guard",
        metric_id="performance.sharpe",
        gate_type=GateType.SOFT,
        max_degradation_pct=5.0,
    )
    report = evaluate_gates(_comparison([]), [rule])
    assert report.evaluations[0].status == "warn"
