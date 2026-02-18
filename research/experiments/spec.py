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

from research.risk import RiskConfig
from research.features.feature_registry import normalize_regime_feature_set_name
from research.execution.execution_simulator import ExecutionSimConfig, resolve_execution_sim_config
from research.hierarchy.modes import ensure_mode_inventory, normalize_mode


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
class EvaluationSplitConfig:
    """Optional evaluation split configuration for in/out-of-sample windows."""

    train_ratio: float
    test_ratio: float
    test_window_months: int | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_ratio": float(self.train_ratio),
            "test_ratio": float(self.test_ratio),
            "test_window_months": None if self.test_window_months is None else int(self.test_window_months),
        }


def _default_evaluation_split() -> EvaluationSplitConfig:
    return EvaluationSplitConfig(train_ratio=0.8, test_ratio=0.2, test_window_months=None)


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
    evaluation_split: EvaluationSplitConfig | None = None
    risk_config: RiskConfig = RiskConfig()
    execution_sim: ExecutionSimConfig | None = None
    notes: str | None = None
    regime_feature_set: str | None = None
    regime_labeling: str | None = None
    hierarchy_enabled: bool = False
    controller_config: Mapping[str, Any] | None = None
    allocator_by_mode: Mapping[str, Mapping[str, Any]] | None = None

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
        if policy not in {"equal_weight", "sma", "ppo"} and not policy.startswith("sac"):
            raise ValueError("policy must be one of: equal_weight, sma, ppo, sac_*")
        policy_params = _normalize_mapping(payload.get("policy_params") or {})
        cost_config = _build_cost_config(payload.get("cost_config"))
        evaluation_split = _build_evaluation_split(payload.get("evaluation_split"))
        risk_config = _build_risk_config(payload.get("risk_config"))
        execution_sim = _build_execution_sim(payload.get("execution_sim"))
        seed = _coerce_int(payload.get("seed"), "seed")
        notes = payload.get("notes")
        normalized_notes = None
        if notes is not None:
            notes_str = str(notes).strip()
            normalized_notes = notes_str if notes_str else None
        regime_feature_set = payload.get("regime_feature_set")
        normalized_regime = None
        if regime_feature_set:
            normalized_regime = normalize_regime_feature_set_name(str(regime_feature_set))
        regime_labeling = _normalize_regime_labeling(payload.get("regime_labeling") or payload.get("regime_labeling_version"))
        hierarchy_enabled = bool(payload.get("hierarchy_enabled", False))
        controller_config = _normalize_mapping(payload.get("controller_config")) or None
        allocator_by_mode = _normalize_allocator_mapping(payload.get("allocator_by_mode")) or None
        if hierarchy_enabled:
            if not normalized_regime:
                raise ValueError("hierarchy_enabled requires regime_feature_set to be provided")
            if not controller_config:
                raise ValueError("controller_config must be provided when hierarchy_enabled")
            if not allocator_by_mode:
                raise ValueError("allocator_by_mode must be provided when hierarchy_enabled")
            _validate_controller_config(controller_config)
            _validate_allocator_configs(allocator_by_mode)
        else:
            if controller_config or allocator_by_mode:
                raise ValueError("controller_config and allocator_by_mode require hierarchy_enabled=True")
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
            risk_config=risk_config,
            execution_sim=execution_sim,
            seed=seed,
            evaluation_split=evaluation_split,
            notes=normalized_notes,
            regime_feature_set=normalized_regime,
            regime_labeling=regime_labeling,
            hierarchy_enabled=hierarchy_enabled,
            controller_config=controller_config,
            allocator_by_mode=allocator_by_mode,
        )

    @property
    def canonical_dict(self) -> Dict[str, Any]:
        evaluation_split = self.evaluation_split or _default_evaluation_split()
        payload: Dict[str, Any] = {
            "experiment_name": self.experiment_name,
            "symbols": list(self.symbols),
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "interval": self.interval,
            "feature_set": self.feature_set,
            "regime_feature_set": self.regime_feature_set or "",
            "policy": self.policy,
            "policy_params": _canonicalize(self.policy_params),
            "cost_config": self.cost_config.to_dict(),
            "risk_config": self.risk_config.to_dict(),
            "execution_sim": (self.execution_sim or ExecutionSimConfig()).to_dict(),
            "seed": int(self.seed),
            "evaluation_split": evaluation_split.to_dict(),
            "notes": self.notes or "",
            "hierarchy_enabled": bool(self.hierarchy_enabled),
            "controller_config": _canonicalize(self.controller_config or {}),
            "allocator_by_mode": _canonicalize(self.allocator_by_mode or {}),
        }
        if self.regime_labeling is not None:
            payload["regime_labeling"] = self.regime_labeling
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
        evaluation_split = self.evaluation_split or _default_evaluation_split()
        payload = dict(self.canonical_dict)
        if self.notes is None:
            payload.pop("notes", None)
        if self.regime_feature_set is None:
            payload.pop("regime_feature_set", None)
        if self.regime_labeling is None:
            payload.pop("regime_labeling", None)
        payload["evaluation_split"] = evaluation_split.to_dict()
        if self.execution_sim is None:
            payload.pop("execution_sim", None)
        if not self.hierarchy_enabled:
            payload.pop("hierarchy_enabled", None)
            payload.pop("controller_config", None)
            payload.pop("allocator_by_mode", None)
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


def _normalize_regime_labeling(value: object | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized not in {"v1", "v2"}:
        raise ValueError("regime_labeling must be one of: v1, v2")
    return normalized


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


def _normalize_allocator_mapping(value: Any) -> Dict[str, Dict[str, Any]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("allocator_by_mode must be a mapping")
    normalized: Dict[str, Dict[str, Any]] = {}
    for key, entry in value.items():
        normalized[str(key)] = _normalize_mapping(entry)
    return normalized


def _validate_controller_config(config: Mapping[str, Any]) -> None:
    freq = str(config.get("update_frequency", "")).strip().lower()
    if freq not in {"weekly", "monthly", "every_k_steps"}:
        raise ValueError("controller_config.update_frequency must be weekly, monthly, or every_k_steps")
    min_hold = int(config.get("min_hold_steps", 0))
    if min_hold < 0:
        raise ValueError("controller_config.min_hold_steps must be non-negative")
    for key in ("vol_threshold_high", "trend_threshold_high", "dispersion_threshold_high"):
        if key not in config:
            raise ValueError(f"controller_config.{key} must be provided")
        float(config[key])
    fallback = normalize_mode(config.get("fallback_mode", "neutral"))
    if freq == "every_k_steps":
        k_value = config.get("k", config.get("every_k_steps"))
        if k_value is None:
            raise ValueError("controller_config.k must be provided for every_k_steps frequency")
        if int(k_value) <= 0:
            raise ValueError("controller_config.k must be positive")
    config["fallback_mode"] = fallback


def _validate_allocator_configs(configs: Mapping[str, Mapping[str, Any]]) -> None:
    ensure_mode_inventory(configs.keys())
    for mode, entry in configs.items():
        if "type" not in entry:
            raise ValueError(f"allocator config for mode '{mode}' must include a type")
        if not str(entry["type"]).strip():
            raise ValueError(f"allocator config for mode '{mode}' must include a type")


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


def _build_evaluation_split(payload: Any) -> EvaluationSplitConfig | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError("evaluation_split must be provided as a mapping when present.")
    train_ratio = float(payload.get("train_ratio", 0.8))
    test_ratio = float(payload.get("test_ratio", 0.2))
    if abs((train_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("evaluation_split ratios must sum to 1.0.")
    if train_ratio <= 0 or test_ratio <= 0:
        raise ValueError("evaluation_split ratios must be positive.")
    test_window_months = payload.get("test_window_months")
    resolved_months = None
    if test_window_months is not None:
        resolved_months = int(test_window_months)
        if resolved_months not in {1, 3, 4, 6, 12}:
            raise ValueError("evaluation_split.test_window_months must be one of: 1, 3, 4, 6, 12.")
    return EvaluationSplitConfig(
        train_ratio=train_ratio,
        test_ratio=test_ratio,
        test_window_months=resolved_months,
    )


def _build_execution_sim(payload: Any) -> ExecutionSimConfig | None:
    if payload is None:
        return None
    if isinstance(payload, ExecutionSimConfig):
        return payload
    if not isinstance(payload, Mapping):
        raise ValueError("execution_sim must be provided as a mapping when present.")
    return resolve_execution_sim_config(payload)


def _build_risk_config(payload: Any) -> RiskConfig:
    if payload is None:
        return RiskConfig()
    if not isinstance(payload, Mapping):
        raise ValueError("risk_config must be provided as a mapping when present.")
    params: Dict[str, Any] = {}
    if "long_only" in payload:
        params["long_only"] = bool(payload["long_only"])
    if "max_weight" in payload:
        params["max_weight"] = None if payload["max_weight"] is None else float(payload["max_weight"])
    if "exposure_cap" in payload:
        params["exposure_cap"] = None if payload["exposure_cap"] is None else float(payload["exposure_cap"])
    if "min_cash" in payload:
        params["min_cash"] = None if payload["min_cash"] is None else float(payload["min_cash"])
    if "max_turnover_1d" in payload:
        params["max_turnover_1d"] = (
            None if payload["max_turnover_1d"] is None else float(payload["max_turnover_1d"])
        )
    return RiskConfig(**params)


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(entry) for entry in value]
    return value


def _sha256_hex(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


__all__ = ["CostConfig", "EvaluationSplitConfig", "ExecutionSimConfig", "ExperimentSpec", "RiskConfig"]
