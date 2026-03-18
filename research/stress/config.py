"""Stress testing configuration loader for scenario-based stress tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Union

from infra.paths import get_repo_root

try:  # Optional dependency.
    import yaml  # type: ignore
except Exception:  # pragma: no cover - PyYAML is optional
    yaml = None


@dataclass(frozen=True)
class HistoricalRegimeSpec:
    """Historical regime scenario specification.

    Replays historical market regimes for stress testing.

    Attributes:
        regime_version: Version identifier for the regime classification scheme.
        regime_filter: Optional filter for specific regime types (e.g., 'high_vol', 'crash').
        min_period_days: Minimum number of days required in the historical period.
    """

    regime_version: str
    regime_filter: str | None = None
    min_period_days: int = 60


@dataclass(frozen=True)
class SyntheticSpec:
    """Synthetic stress scenario specification.

    Applies synthetic mutations to simulate stress conditions.

    Attributes:
        mutation: Type of mutation to apply ('shock', 'drift', 'volatility_spike').
        magnitude: Absolute magnitude of the shock (for 'shock' mutation).
        factor: Multiplicative factor (for 'volatility_spike' mutation).
        duration_days: Number of days to apply the stress condition.
        inject_at: Date or event marker to inject the stress at.
    """

    mutation: Literal["shock", "drift", "volatility_spike"]
    magnitude: float | None = None
    factor: float | None = None
    duration_days: int = 20
    inject_at: str | None = None


@dataclass(frozen=True)
class ScenarioSpec:
    """Stress test scenario specification.

    A scenario can be either historical_regime or synthetic type.

    Attributes:
        type: Scenario type - 'historical_regime' or 'synthetic'.
        historical_regime: Historical regime specification (when type is 'historical_regime').
        synthetic: Synthetic specification (when type is 'synthetic').
        name: Optional name for this scenario.
    """

    type: Literal["historical_regime", "synthetic"]
    historical_regime: HistoricalRegimeSpec | None = None
    synthetic: SyntheticSpec | None = None
    name: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScenarioSpec:
        """Create a ScenarioSpec from a dictionary."""
        scenario_type = data.get("type")
        if scenario_type not in {"historical_regime", "synthetic"}:
            raise ValueError(f"Invalid scenario type: {scenario_type}")

        historical_regime = None
        synthetic = None

        if scenario_type == "historical_regime":
            hist_data = data.get("historical_regime", {})
            historical_regime = HistoricalRegimeSpec(
                regime_version=str(hist_data.get("regime_version", "default")),
                regime_filter=hist_data.get("regime_filter"),
                min_period_days=int(hist_data.get("min_period_days", 60)),
            )
        elif scenario_type == "synthetic":
            syn_data = data.get("synthetic", {})
            synthetic = SyntheticSpec(
                mutation=syn_data.get("mutation", "shock"),
                magnitude=syn_data.get("magnitude"),
                factor=syn_data.get("factor"),
                duration_days=int(syn_data.get("duration_days", 20)),
                inject_at=syn_data.get("inject_at"),
            )

        return cls(
            type=scenario_type,
            historical_regime=historical_regime,
            synthetic=synthetic,
            name=data.get("name"),
        )


@dataclass(frozen=True)
class GateConfig:
    """Gate configuration for stress testing.

    Attributes:
        enabled: Whether this gate is enabled.
        thresholds: Dictionary of threshold configurations.
    """

    enabled: bool = True
    thresholds: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StressTestConfig:
    """Stress test configuration.

    Top-level configuration for stress testing scenarios.

    Attributes:
        name: Name of the stress test configuration.
        experiment_id: Experiment identifier for tracking results.
        experiment_file: Path to experiment configuration file.
        scenarios: List of stress test scenarios to run.
        gates: Dictionary of gate configurations.
        seeds: List of random seeds for reproducibility.
    """

    name: str
    experiment_id: str | None = None
    experiment_file: str | None = None
    scenarios: list[ScenarioSpec] = field(default_factory=list)
    gates: dict[str, GateConfig] = field(default_factory=dict)
    seeds: list[int] = field(default_factory=lambda: [42])

    @classmethod
    def from_file(cls, path: Path | str) -> StressTestConfig:
        """Load stress test configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            StressTestConfig instance populated from the YAML file.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            RuntimeError: If PyYAML is not installed.
            ValueError: If the configuration is invalid.
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Stress test configuration not found at {config_path}")

        if yaml is None:
            raise RuntimeError("PyYAML is required to parse YAML configs, but it is not installed.")

        text = config_path.read_text(encoding="utf-8")
        payload = yaml.safe_load(text)

        if not isinstance(payload, dict):
            raise ValueError("Stress test config must be a YAML mapping.")

        # Parse scenarios
        scenarios = []
        for scenario_data in payload.get("scenarios", []):
            scenarios.append(ScenarioSpec.from_dict(scenario_data))

        # Parse gates
        gates = {}
        gates_data = payload.get("gates", {})
        for gate_name, gate_config in gates_data.items():
            gate_dict = gate_config if isinstance(gate_config, dict) else {}
            gates[gate_name] = GateConfig(
                enabled=gate_dict.get("enabled", True),
                thresholds=gate_dict.get("thresholds", {}),
            )

        # Parse seeds
        seeds = payload.get("seeds", [42])
        if isinstance(seeds, list):
            seeds = [int(s) for s in seeds]

        return cls(
            name=str(payload.get("name", "stress_test")),
            experiment_id=payload.get("experiment_id"),
            experiment_file=payload.get("experiment_file"),
            scenarios=scenarios,
            gates=gates,
            seeds=seeds,
        )


def load_stress_config(path: Path | str | None = None) -> StressTestConfig:
    """Load stress test configuration from disk.

    Args:
        path: Optional path to the configuration file. If not provided,
              loads from the default location at configs/stress/default_stress_test.yml.

    Returns:
        StressTestConfig instance populated from the YAML file.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        RuntimeError: If PyYAML is not installed.
        ValueError: If the configuration is invalid.
    """
    config_path = Path(path) if path else get_repo_root() / "configs" / "stress" / "default_stress_test.yml"
    return StressTestConfig.from_file(config_path)


__all__ = [
    "GateConfig",
    "HistoricalRegimeSpec",
    "ScenarioSpec",
    "StressTestConfig",
    "SyntheticSpec",
    "load_stress_config",
]
