"""Research stress testing module."""

from .config import (
    GateConfig,
    HistoricalRegimeSpec,
    ScenarioSpec,
    StressTestConfig,
    SyntheticSpec,
    load_stress_config,
)
from .data_source import StressDataSource
from .gates import StressGateEvaluator
from .regime_windows import find_regime_windows
from .results import (
    GateResult,
    ScenarioRunResult,
    StressTestResult,
    aggregate_results,
)
from .runner import StressTestRunner

__all__ = [
    "GateConfig",
    "HistoricalRegimeSpec",
    "ScenarioSpec",
    "StressDataSource",
    "StressGateEvaluator",
    "StressTestConfig",
    "StressTestRunner",
    "StressTestResult",
    "ScenarioRunResult",
    "GateResult",
    "SyntheticSpec",
    "aggregate_results",
    "find_regime_windows",
    "load_stress_config",
]
