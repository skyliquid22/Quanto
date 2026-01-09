"""Operations configuration loader for scheduling + alerting."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from infra.paths import get_repo_root

try:  # Optional dependency.
    import yaml  # type: ignore
except Exception:  # pragma: no cover - PyYAML is optional
    yaml = None


@dataclass(frozen=True)
class BackoffPolicyConfig:
    """Retry/backoff policy for broker/transient failures."""

    initial_seconds: float = 30.0
    max_seconds: float = 600.0
    multiplier: float = 2.0
    max_attempts: int = 3


@dataclass(frozen=True)
class AlertingConfig:
    """Alert routing configuration."""

    hard_channels: list[str] = field(default_factory=list)
    soft_channels: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PaperOpsConfig:
    """Operations knobs for paper trading."""

    cron: str = "0 13 * * 1-5"
    timezone: str = "UTC"
    grace_minutes: int = 30
    expect_trades: bool = False
    runbook_url: str | None = None
    backoff: BackoffPolicyConfig = field(default_factory=BackoffPolicyConfig)
    alerts: AlertingConfig = field(default_factory=AlertingConfig)


@dataclass(frozen=True)
class OpsConfig:
    """Top-level ops configuration surface."""

    paper_trading: PaperOpsConfig


def _load_mapping(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to parse YAML configs, but it is not installed.")
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, Mapping):
        raise ValueError("Ops config must map keys to values.")
    return payload


def load_ops_config(path: Path | None = None) -> OpsConfig:
    """Load ops configuration from disk."""

    config_path = Path(path) if path else get_repo_root() / "configs" / "ops.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Ops configuration not found at {config_path}")
    payload = dict(_load_mapping(config_path))
    paper_block = payload.get("paper_trading") or {}
    backoff_block = paper_block.get("backoff") or {}
    alerts_block = paper_block.get("alerts") or {}
    paper_config = PaperOpsConfig(
        cron=str(paper_block.get("cron", PaperOpsConfig.cron)),
        timezone=str(paper_block.get("timezone", PaperOpsConfig.timezone)),
        grace_minutes=int(paper_block.get("grace_minutes", PaperOpsConfig.grace_minutes)),
        expect_trades=bool(paper_block.get("expect_trades", PaperOpsConfig.expect_trades)),
        runbook_url=str(paper_block["runbook_url"]) if paper_block.get("runbook_url") else None,
        backoff=BackoffPolicyConfig(
            initial_seconds=float(backoff_block.get("initial_seconds", BackoffPolicyConfig.initial_seconds)),
            max_seconds=float(backoff_block.get("max_seconds", BackoffPolicyConfig.max_seconds)),
            multiplier=float(backoff_block.get("multiplier", BackoffPolicyConfig.multiplier)),
            max_attempts=int(backoff_block.get("max_attempts", BackoffPolicyConfig.max_attempts)),
        ),
        alerts=AlertingConfig(
            hard_channels=[str(value) for value in alerts_block.get("hard_channels", [])],
            soft_channels=[str(value) for value in alerts_block.get("soft_channels", [])],
        ),
    )
    return OpsConfig(paper_trading=paper_config)


__all__ = [
    "AlertingConfig",
    "BackoffPolicyConfig",
    "OpsConfig",
    "PaperOpsConfig",
    "load_ops_config",
]
