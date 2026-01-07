"""Declarative experiment specification with deterministic identity."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

try:  # pragma: no cover - yaml optional in some envs
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


@dataclass(frozen=True)
class CostConfig:
    """Transaction cost configuration captured in an experiment spec."""

    transaction_cost_bp: float
    slippage_bp: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"transaction_cost_bp": float(self.transaction_cost_bp)}
        if self.slippage_bp is not None:
            payload["slippage_bp"] = float(self.slippage_bp)
        return payload


@dataclass(frozen=True)
class ExperimentSpec:
    """Immutable experiment specification with canonical serialization."""

    experiment_name: str
    symbols: tuple[str, ...]
    start_date: date
    end_date: date
    interval: str
    feature_set: str
    policy: str
    policy_params: Mapping[str, Any]
    cost_config: CostConfig
    seed: int
    notes: str | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> ExperimentSpec:
        spec_path = Path(path)
        if not spec_path.exists():
            raise FileNotFoundError(f"Experiment spec not found: {path}")
        text = spec_path.read_text(encoding="utf-8")
        data = _parse_spec_text(text)
        if not isinstance(data, Mapping):
            raise ValueError("Experiment spec must be a mapping.")
        return cls.from_mapping(data)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ExperimentSpec:
        experiment_name = _require_str(payload, "experiment_name")
        symbols = _normalize_symbols(payload)
        start = _parse_date(payload.get("start_date"), "start_date")
        end = _parse_date(payload.get("end_date"), "end_date")
        if end < start:
            raise ValueError("end_date cannot be earlier than start_date")
        interval = _require_str(payload, "interval").strip().lower()
        if interval != "daily":
            raise ValueError("Only the daily interval is supported in v1.")
        feature_set = _require_str(payload, "feature_set").strip()
        if not feature_set:
            raise ValueError("feature_set must be provided")
        policy = _require_str(payload, "policy").strip().lower()
        if policy not in {"equal_weight", "sma", "ppo"}:
            raise ValueError("policy must be one of: equal_weight, sma, ppo")
        policy_params = _normalize_mapping(payload.get("policy_params") or {})
        cost_config = _build_cost_config(payload.get("cost_config"))
        seed = _coerce_int(payload.get("seed"), "seed")
        notes = payload.get("notes")
        normalized_notes = None
        if notes is not None:
            notes_str = str(notes).strip()
            normalized_notes = notes_str if notes_str else None
        return cls(
            experiment_name=experiment_name,
            symbols=symbols,
            start_date=start,
            end_date=end,
            interval=interval,
            feature_set=feature_set,
            policy=policy,
            policy_params=policy_params,
            cost_config=cost_config,
            seed=seed,
            notes=normalized_notes,
        )

    @property
    def canonical_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "experiment_name": self.experiment_name,
            "symbols": list(self.symbols),
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "interval": self.interval,
            "feature_set": self.feature_set,
            "policy": self.policy,
            "policy_params": _canonicalize(self.policy_params),
            "cost_config": self.cost_config.to_dict(),
            "seed": int(self.seed),
            "notes": self.notes or "",
        }
        return payload

    @property
    def canonical_json(self) -> str:
        return json.dumps(self.canonical_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @property
    def experiment_id(self) -> str:
        digest = json.dumps(self.canonical_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
        return _sha256_hex(digest)

    def to_dict(self) -> Dict[str, Any]:
        payload = dict(self.canonical_dict)
        if self.notes is None:
            payload.pop("notes", None)
        return payload


def _parse_spec_text(text: str) -> Any:
    if yaml is not None:
        loaded = yaml.safe_load(text)
        if loaded is not None:
            return loaded
    return json.loads(text)


def _require_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if value is None:
        raise ValueError(f"{key} must be provided in the experiment spec.")
    string_value = str(value).strip()
    if not string_value:
        raise ValueError(f"{key} must be a non-empty string.")
    return string_value


def _coerce_int(value: Any, label: str) -> int:
    if value is None:
        raise ValueError(f"{label} must be provided in the experiment spec.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"{label} must be an integer") from exc


def _parse_date(value: Any, label: str) -> date:
    if not value:
        raise ValueError(f"{label} must be provided in the experiment spec.")
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"{label} must be an ISO date (YYYY-MM-DD).") from exc


def _normalize_symbols(payload: Mapping[str, Any]) -> tuple[str, ...]:
    explicit = payload.get("symbols")
    fallback = payload.get("symbol")
    symbols: Sequence[str] | None
    if explicit:
        if not isinstance(explicit, Sequence) or isinstance(explicit, (str, bytes)):
            raise ValueError("symbols must be a list of strings.")
        symbols = explicit
    elif fallback:
        symbols = [fallback]
    else:
        raise ValueError("Either symbol or symbols must be provided.")
    normalized = []
    seen = set()
    for value in symbols:
        token = str(value).strip().upper()
        if not token:
            continue
        if token in seen:
            continue
        seen.add(token)
        normalized.append(token)
    if not normalized:
        raise ValueError("At least one symbol must be provided.")
    normalized.sort()
    return tuple(normalized)


def _normalize_mapping(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("policy_params must be a mapping when provided.")
    normalized: Dict[str, Any] = {}
    for key, entry in value.items():
        normalized[str(key)] = _coerce_json(entry)
    return normalized


def _coerce_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _coerce_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce_json(item) for item in value]
    if isinstance(value, tuple):
        return [_coerce_json(item) for item in value]
    if isinstance(value, (date,)):
        return value.isoformat()
    return value


def _build_cost_config(payload: Any) -> CostConfig:
    if not isinstance(payload, Mapping):
        raise ValueError("cost_config must be provided as a mapping.")
    if "transaction_cost_bp" not in payload:
        raise ValueError("cost_config.transaction_cost_bp is required.")
    transaction_cost = float(payload["transaction_cost_bp"])
    if transaction_cost < 0:
        raise ValueError("transaction_cost_bp must be non-negative.")
    slippage = payload.get("slippage_bp")
    slippage_value = float(slippage) if slippage is not None else None
    return CostConfig(transaction_cost_bp=transaction_cost, slippage_bp=slippage_value)


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(entry) for entry in value]
    return value


def _sha256_hex(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


__all__ = ["CostConfig", "ExperimentSpec"]
