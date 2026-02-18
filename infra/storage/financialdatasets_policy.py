"""Financial Datasets raw storage policy loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

from infra.paths import get_repo_root

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


@dataclass(frozen=True)
class DomainPolicy:
    policy: str
    date_priority: tuple[str, ...] = ()
    date_kind: str = "date"
    dedup_keys: tuple[str, ...] = ()
    expected_columns: tuple[str, ...] = ()


def load_financialdatasets_policies(config_path: Path | str | None = None) -> Dict[str, DomainPolicy]:
    path = Path(config_path) if config_path is not None else get_repo_root() / "configs" / "data_sources.yml"
    text = path.read_text()
    if path.suffix.lower() in {".yml", ".yaml"}:
        if not yaml:
            raise RuntimeError("PyYAML must be installed to parse data_sources.yml")
        data = yaml.safe_load(text)
    else:
        import json

        data = json.loads(text)
    return extract_financialdatasets_policies(data)


def extract_financialdatasets_policies(config: Mapping[str, Any]) -> Dict[str, DomainPolicy]:
    vendors = config.get("vendors", []) if isinstance(config, Mapping) else []
    for entry in vendors:
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("name", "")).lower() != "financialdatasets":
            continue
        policy_block = entry.get("raw_storage_policy", {}) or {}
        policies: Dict[str, DomainPolicy] = {}
        for domain, raw in policy_block.items():
            if isinstance(raw, str):
                policies[str(domain)] = DomainPolicy(policy=str(raw))
                continue
            if not isinstance(raw, Mapping):
                continue
            policies[str(domain)] = DomainPolicy(
                policy=str(raw.get("policy", "")),
                date_priority=tuple(raw.get("date_priority", []) or []),
                date_kind=str(raw.get("date_kind", "date") or "date"),
                dedup_keys=tuple(raw.get("dedup_keys", []) or []),
                expected_columns=tuple(raw.get("expected_columns", []) or []),
            )
        return policies
    return {}


__all__ = ["DomainPolicy", "extract_financialdatasets_policies", "load_financialdatasets_policies"]
