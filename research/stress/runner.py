"""Stress test runner: orchestrates scenario execution and gate evaluation.

Usage:
    config = StressTestConfig.from_file("configs/stress/my_test.yml")
    spec   = ExperimentSpec.from_file(config.experiment_file)
    result = StressTestRunner(config, spec, out_dir=Path("runs/stress/my_test")).run()

The runner executes each scenario × seed combination independently, collecting
portfolio metrics from the ShadowEngine and evaluating all configured gates.

Design notes:
  - Experiments must already be registered and have evaluation artifacts (i.e.
    training was completed before the stress test is run). The runner uses the
    default ExperimentRegistry unless an override is provided.
  - The engine runs in qualification_replay_allowed mode, bypassing the normal
    promotion requirement. Stress tests are explicitly pre-promotion.
  - Scenario exceptions are caught and recorded as "error" status rather than
    crashing the entire run, so all scenarios always produce a result.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from research.experiments.registry import ExperimentRegistry
from research.experiments.spec import ExperimentSpec
from research.shadow.data_source import ReplayMarketDataSource
from research.shadow.engine import ShadowEngine
from research.shadow.logging import ShadowLogger
from research.shadow.state_store import StateStore
from research.stress.config import ScenarioSpec, StressTestConfig
from research.stress.data_source import StressDataSource
from research.stress.gates import StressGateEvaluator
from research.stress.regime_windows import find_regime_windows
from research.stress.results import (
    ScenarioRunResult,
    StressTestResult,
    aggregate_results,
)


def _derive_run_id(experiment_id: str, scenario_name: str, seed: int) -> str:
    payload = {
        "experiment_id": experiment_id,
        "scenario": scenario_name or "",
        "seed": seed,
        "mode": "stress",
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"stress_{digest[:12]}"


def _aggregate_gate_status(gate_results: list) -> str:
    statuses = {g.status for g in gate_results}
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "pass"


class StressTestRunner:
    """Runs all scenario × seed combinations defined in a StressTestConfig.

    Args:
        config: Parsed StressTestConfig from YAML.
        spec: ExperimentSpec for the agent under test.
        out_dir: Root directory for all stress test run artifacts.
        registry: ExperimentRegistry containing the agent's training artifacts.
                  Defaults to the project-wide default registry.
        data_root: Optional override for the data root path.
    """

    def __init__(
        self,
        config: StressTestConfig,
        spec: ExperimentSpec,
        out_dir: Path,
        registry: ExperimentRegistry | None = None,
        data_root: Path | None = None,
    ) -> None:
        self._config = config
        self._spec = spec
        self._out_dir = Path(out_dir)
        self._registry = registry or ExperimentRegistry()
        self._data_root = Path(data_root) if data_root else None
        self._gate_evaluator = StressGateEvaluator(config.gates)

    def run(self) -> StressTestResult:
        """Execute all scenarios and return the aggregated result."""
        results: list[ScenarioRunResult] = []
        for scenario in self._config.scenarios:
            for seed in self._config.seeds:
                result = self._run_one(scenario, seed)
                results.append(result)
        return aggregate_results(results, self._config.name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_one(self, scenario: ScenarioSpec, seed: int) -> ScenarioRunResult:
        scenario_name = scenario.name or scenario.type
        run_id = _derive_run_id(self._spec.experiment_id, scenario_name, seed)
        run_dir = self._out_dir / _sanitize(scenario_name) / f"seed_{seed}"

        if run_dir.exists():
            shutil.rmtree(run_dir)

        try:
            data_source = self._build_data_source(scenario)
            engine = self._build_engine(data_source, run_id, run_dir)
            engine.run()
            gate_results = self._gate_evaluator.evaluate(run_dir)
            status = _aggregate_gate_status(gate_results)
        except Exception as exc:  # noqa: BLE001
            return ScenarioRunResult(
                scenario_name=scenario_name,
                seed=seed,
                status="error",
                gate_results=[],
                run_dir=run_dir,
                error=str(exc),
            )

        return ScenarioRunResult(
            scenario_name=scenario_name,
            seed=seed,
            status=status,
            gate_results=gate_results,
            run_dir=run_dir,
        )

    def _build_data_source(self, scenario: ScenarioSpec) -> ReplayMarketDataSource | StressDataSource:
        if scenario.type == "historical_regime":
            return self._build_historical_source(scenario)
        if scenario.type == "synthetic":
            return self._build_synthetic_source(scenario)
        raise ValueError(f"Unknown scenario type: '{scenario.type}'")

    def _build_historical_source(self, scenario: ScenarioSpec) -> ReplayMarketDataSource:
        hist_spec = scenario.historical_regime
        if hist_spec is None:
            raise ValueError(f"Scenario '{scenario.name}' is missing historical_regime spec.")

        # Build a full-window source to extract the regime calendar.
        full_source = ReplayMarketDataSource(
            spec=self._spec,
            start_date=self._spec.start_date,
            end_date=self._spec.end_date,
            data_root=self._data_root,
        )
        windows = find_regime_windows(
            calendar=full_source.calendar(),
            regime_series=full_source.regime_series,
            feature_names=full_source.regime_feature_names,
            hist_spec=hist_spec,
        )
        # Take the longest qualifying window.
        start, end = windows[0]
        return ReplayMarketDataSource(
            spec=self._spec,
            start_date=start,
            end_date=end,
            data_root=self._data_root,
        )

    def _build_synthetic_source(self, scenario: ScenarioSpec) -> StressDataSource:
        syn_spec = scenario.synthetic
        if syn_spec is None:
            raise ValueError(f"Scenario '{scenario.name}' is missing synthetic spec.")
        base = ReplayMarketDataSource(
            spec=self._spec,
            start_date=self._spec.start_date,
            end_date=self._spec.end_date,
            data_root=self._data_root,
        )
        return StressDataSource(base, syn_spec)

    def _build_engine(
        self,
        data_source: ReplayMarketDataSource | StressDataSource,
        run_id: str,
        run_dir: Path,
    ) -> ShadowEngine:
        run_dir.mkdir(parents=True, exist_ok=True)
        logger = ShadowLogger(run_dir)
        state_store = StateStore(
            self._spec.experiment_id,
            run_id=run_id,
            destination=run_dir,
        )
        return ShadowEngine(
            experiment_id=self._spec.experiment_id,
            data_source=data_source,
            state_store=state_store,
            logger=logger,
            run_id=run_id,
            out_dir=run_dir,
            registry=self._registry,
            execution_mode="none",
            replay_mode=True,
            qualification_replay_allowed=True,
            qualification_allow_reason="stress_test",
        )


def _sanitize(name: str) -> str:
    """Convert a scenario name to a safe directory component."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_") or "scenario"


__all__ = ["StressTestRunner"]
