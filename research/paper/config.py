"""Configuration helpers for paper execution runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from infra.paths import get_data_root
from research.promotion.qualify import is_experiment_promoted

try:  # Optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback when PyYAML missing
    yaml = None


def _default_output_root(experiment_id: str) -> Path:
    return get_data_root() / "paper" / experiment_id / "runs"


@dataclass(frozen=True)
class BrokerConfig:
    """Broker connectivity block."""

    alpaca_base_url: str = "https://paper-api.alpaca.markets"


@dataclass(frozen=True)
class RiskLimitConfig:
    """Risk guardrails enforced during paper validation."""

    max_position_notional: float = 5_000.0
    max_gross_exposure: float = 1.0
    max_turnover: float = 0.25
    max_drawdown: float = 0.05
    max_symbol_weight: float = 0.25
    min_cash_pct: float = 0.1


@dataclass(frozen=True)
class PollingConfig:
    """Order polling cadence."""

    max_poll_seconds: float = 30.0
    poll_interval_seconds: float = 2.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "PollingConfig":
        if not payload:
            return cls()
        return cls(
            max_poll_seconds=float(payload.get("max_poll_seconds", cls.max_poll_seconds)),
            poll_interval_seconds=float(payload.get("poll_interval_seconds", cls.poll_interval_seconds)),
        )


@dataclass(frozen=True)
class ReconciliationConfig:
    """Position/cash reconciliation tolerances."""

    position_tolerance_shares: float = 0.01
    cash_tolerance_usd: float = 5.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "ReconciliationConfig":
        if not payload:
            return cls()
        return cls(
            position_tolerance_shares=float(payload.get("position_tolerance_shares", cls.position_tolerance_shares)),
            cash_tolerance_usd=float(payload.get("cash_tolerance_usd", cls.cash_tolerance_usd)),
        )


@dataclass(frozen=True)
class ArtifactConfig:
    """Filesystem layout for paper run artifacts."""

    output_root: Path

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None, experiment_id: str) -> "ArtifactConfig":
        if payload and payload.get("output_root"):
            root = Path(str(payload["output_root"]))
        else:
            root = _default_output_root(experiment_id)
        return cls(output_root=root)


@dataclass(frozen=True)
class ScheduleConfig:
    """Optional daily scheduling hints."""

    enabled: bool = False
    time_of_day: str | None = None
    timezone: str = "UTC"


@dataclass
class PaperRunConfig:
    """Strongly typed paper execution configuration."""

    experiment_id: str
    execution_mode: str
    universe: list[str]
    broker: BrokerConfig
    risk_limits: RiskLimitConfig
    polling: PollingConfig
    reconciliation: ReconciliationConfig
    artifacts: ArtifactConfig
    schedule: ScheduleConfig | None = None
    allow_large_universe: bool = False
    raw_payload: dict[str, Any] = field(repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        if self.execution_mode != "alpaca_paper":
            raise ValueError("Paper runs must specify execution_mode='alpaca_paper'.")
        universe_size = len(self.universe)
        if universe_size < 2:
            raise ValueError("Paper validation requires at least two symbols in the universe.")
        if universe_size > 5 and not self.allow_large_universe:
            raise ValueError("Paper validation universe must be <=5 symbols unless allow_large_universe is true.")
        base_url = (self.broker.alpaca_base_url or "").lower()
        if "paper" not in base_url:
            raise ValueError("Alpaca base URL must point to the paper environment.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "execution_mode": self.execution_mode,
            "universe": list(self.universe),
            "broker": {"alpaca_base_url": self.broker.alpaca_base_url},
            "risk_limits": {
                "max_position_notional": self.risk_limits.max_position_notional,
                "max_gross_exposure": self.risk_limits.max_gross_exposure,
                "max_turnover": self.risk_limits.max_turnover,
                "max_drawdown": self.risk_limits.max_drawdown,
                "max_symbol_weight": self.risk_limits.max_symbol_weight,
                "min_cash_pct": self.risk_limits.min_cash_pct,
            },
            "polling": {
                "max_poll_seconds": self.polling.max_poll_seconds,
                "poll_interval_seconds": self.polling.poll_interval_seconds,
            },
            "reconciliation": {
                "position_tolerance_shares": self.reconciliation.position_tolerance_shares,
                "cash_tolerance_usd": self.reconciliation.cash_tolerance_usd,
            },
            "artifacts": {
                "output_root": str(self.artifacts.output_root),
            },
            "schedule": None
            if self.schedule is None
            else {
                "enabled": self.schedule.enabled,
                "time_of_day": self.schedule.time_of_day,
                "timezone": self.schedule.timezone,
            },
            "allow_large_universe": bool(self.allow_large_universe),
        }

    def write_run_config(self, run_dir: Path) -> Path:
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "run_config.json"
        payload = self.raw_payload or self.to_dict()
        path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        return path


def _load_mapping(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to parse YAML configs, but it is not installed.")
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, Mapping):
        raise ValueError("Paper run config must map keys to values.")
    return data


def load_paper_config(path: Path, *, promotion_root: Path | None = None) -> PaperRunConfig:
    """Load and validate a paper execution configuration."""

    payload = dict(_load_mapping(path))
    experiment_id = str(payload.get("experiment_id") or "")
    if not experiment_id:
        raise ValueError("experiment_id is required in paper config.")
    if not is_experiment_promoted(experiment_id, promotion_root=promotion_root):
        raise RuntimeError(f"Experiment '{experiment_id}' is not promoted; promotion record required for paper runs.")
    universe = [str(symbol) for symbol in payload.get("universe", [])]
    broker_block = payload.get("broker") or {}
    risk_block = payload.get("risk_limits") or {}
    schedule_block = payload.get("schedule")
    config = PaperRunConfig(
        experiment_id=experiment_id,
        execution_mode=str(payload.get("execution_mode") or "alpaca_paper"),
        universe=universe,
        broker=BrokerConfig(alpaca_base_url=str(broker_block.get("alpaca_base_url", BrokerConfig.alpaca_base_url))),
        risk_limits=RiskLimitConfig(
            max_position_notional=float(risk_block.get("max_position_notional", RiskLimitConfig.max_position_notional)),
            max_gross_exposure=float(risk_block.get("max_gross_exposure", RiskLimitConfig.max_gross_exposure)),
            max_turnover=float(risk_block.get("max_turnover", RiskLimitConfig.max_turnover)),
            max_drawdown=float(risk_block.get("max_drawdown", RiskLimitConfig.max_drawdown)),
            max_symbol_weight=float(risk_block.get("max_symbol_weight", RiskLimitConfig.max_symbol_weight)),
            min_cash_pct=float(risk_block.get("min_cash_pct", RiskLimitConfig.min_cash_pct)),
        ),
        polling=PollingConfig.from_mapping(payload.get("polling")),
        reconciliation=ReconciliationConfig.from_mapping(payload.get("reconciliation")),
        artifacts=ArtifactConfig.from_mapping(payload.get("artifacts"), experiment_id),
        schedule=ScheduleConfig(
            enabled=bool(schedule_block.get("enabled", False)) if schedule_block else False,
            time_of_day=schedule_block.get("time_of_day") if schedule_block else None,
            timezone=schedule_block.get("timezone", "UTC") if schedule_block else "UTC",
        )
        if schedule_block
        else None,
        allow_large_universe=bool(payload.get("allow_large_universe", False)),
        raw_payload=json.loads(json.dumps(payload, default=str)),
    )
    return config


__all__ = [
    "ArtifactConfig",
    "BrokerConfig",
    "PaperRunConfig",
    "PollingConfig",
    "ReconciliationConfig",
    "RiskLimitConfig",
    "ScheduleConfig",
    "load_paper_config",
]
