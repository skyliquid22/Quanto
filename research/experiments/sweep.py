"""Experiment sweep specification and deterministic expansion."""

from __future__ import annotations

import json
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

try:  # pragma: no cover - yaml optional in some envs
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

from research.experiments.spec import ExperimentSpec


@dataclass(frozen=True)
class SweepDimension:
    """Single dimension definition for a sweep."""

    field_path: str
    values: tuple[Any, ...]


@dataclass(frozen=True)
class SweepSpec:
    """Declarative specification defining a family of experiments."""

    sweep_name: str
    base_spec: ExperimentSpec
    dimensions: tuple[SweepDimension, ...]

    @classmethod
    def from_file(cls, path: str | Path) -> SweepSpec:
        sweep_path = Path(path)
        if not sweep_path.exists():
            raise FileNotFoundError(f"Sweep spec not found: {path}")
        text = sweep_path.read_text(encoding="utf-8")
        payload = _parse_payload(text)
        if not isinstance(payload, Mapping):
            raise ValueError("Sweep specification must be a mapping.")
        return cls.from_mapping(payload, base_dir=sweep_path.parent)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, base_dir: Path | None = None) -> SweepSpec:
        sweep_name = _require_str(payload.get("sweep_name"), "sweep_name")
        base_spec_payload = payload.get("base_experiment_spec")
        if base_spec_payload is None:
            raise ValueError("Sweep spec must include base_experiment_spec.")
        base_spec = _parse_base_spec(base_spec_payload, base_dir=base_dir)
        dimensions_payload = payload.get("sweep_dimensions")
        if not isinstance(dimensions_payload, Mapping) or not dimensions_payload:
            raise ValueError("sweep_dimensions must be a non-empty mapping.")
        dimension_entries: List[SweepDimension] = []
        for field_path, values in dimensions_payload.items():
            normalized_path = _require_str(field_path, "sweep_dimensions key")
            normalized_values = _normalize_values(values)
            dimension_entries.append(
                SweepDimension(field_path=normalized_path, values=tuple(normalized_values))
            )
        base_payload: Dict[str, Any] = json.loads(base_spec.canonical_json)
        _validate_dimensions(base_payload, dimension_entries)
        return cls(
            sweep_name=sweep_name,
            base_spec=base_spec,
            dimensions=tuple(dimension_entries),
        )

    @property
    def dimension_names(self) -> tuple[str, ...]:
        return tuple(d.field_path for d in self.dimensions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sweep_name": self.sweep_name,
            "base_experiment_spec": self.base_spec.to_dict(),
            "sweep_dimensions": OrderedDict(
                (dimension.field_path, list(dimension.values)) for dimension in self.dimensions
            ),
        }


@dataclass(frozen=True)
class ExpandedExperiment:
    """Resolved experiment spec paired with its sweep dimension values."""

    spec: ExperimentSpec
    dimension_values: Dict[str, Any]


def expand_sweep(sweep_spec: SweepSpec) -> List[ExperimentSpec]:
    """Expand a sweep specification into individual experiment specs."""

    return [entry.spec for entry in expand_sweep_entries(sweep_spec)]


def expand_sweep_entries(sweep_spec: SweepSpec) -> List[ExpandedExperiment]:
    """Return expanded experiments with associated sweep dimension assignments."""

    base_payload = json.loads(sweep_spec.base_spec.canonical_json)
    if not sweep_spec.dimensions:
        return [ExpandedExperiment(spec=sweep_spec.base_spec, dimension_values={})]

    entries: List[ExpandedExperiment] = []
    dimension_order: List[SweepDimension] = []
    for entry in sweep_spec.dimensions:
        if isinstance(entry, SweepDimension):
            dimension_order.append(entry)
        elif isinstance(entry, Mapping):
            dimension_order.append(
                SweepDimension(
                    field_path=str(entry.get("field_path")),
                    values=tuple(_normalize_values(entry.get("values", []))),
                )
            )
        else:
            raise TypeError("Sweep dimensions must be SweepDimension instances.")
    dimension_order.sort(key=lambda d: d.field_path)
    assignment = OrderedDict()

    def _recurse(idx: int, payload: Dict[str, Any]) -> None:
        if idx >= len(dimension_order):
            spec = ExperimentSpec.from_mapping(payload)
            entries.append(ExpandedExperiment(spec=spec, dimension_values=dict(assignment)))
            return
        dimension = dimension_order[idx]
        for value in dimension.values:
            next_payload = deepcopy(payload)
            _assign_dimension(next_payload, dimension.field_path, value)
            assignment[dimension.field_path] = value
            _recurse(idx + 1, next_payload)
            assignment.popitem()

    _recurse(0, base_payload)
    return entries


def _parse_payload(text: str) -> Any:
    if yaml is not None:  # pragma: no cover - depends on optional yaml
        loaded = yaml.safe_load(text)
        if loaded is not None:
            return loaded
    return json.loads(text)


def _parse_base_spec(payload: Any, *, base_dir: Path | None) -> ExperimentSpec:
    if isinstance(payload, ExperimentSpec):
        return payload
    if isinstance(payload, Mapping):
        return ExperimentSpec.from_mapping(payload)
    if isinstance(payload, str):
        spec_path = Path(payload)
        if not spec_path.is_absolute() and base_dir is not None:
            spec_path = (base_dir / spec_path).resolve()
        return ExperimentSpec.from_file(spec_path)
    raise TypeError("base_experiment_spec must be a path, mapping, or ExperimentSpec instance.")


def _require_str(value: Any, label: str) -> str:
    if value is None:
        raise ValueError(f"{label} must be provided in the sweep spec.")
    token = str(value).strip()
    if not token:
        raise ValueError(f"{label} must be a non-empty string.")
    return token


def _normalize_values(values: Any) -> Sequence[Any]:
    if isinstance(values, Mapping):
        if "grid" in values:
            return _grid_values(values["grid"])
        if "values" in values:
            values = values["values"]
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        normalized = list(values)
        if not normalized:
            raise ValueError("Sweep dimension lists must contain at least one value.")
        return normalized
    raise ValueError("Sweep dimension values must be provided as an array or grid.")


def _grid_values(payload: Mapping[str, Any]) -> Sequence[float]:
    if not isinstance(payload, Mapping):
        raise ValueError("grid specification must be a mapping")
    start = _require_number(payload.get("start"), "grid.start")
    count = int(_require_number(payload.get("count"), "grid.count"))
    if count <= 0:
        raise ValueError("grid.count must be positive")
    step_value = payload.get("step")
    stop_value = payload.get("stop")
    if step_value is None and stop_value is None:
        step_value = 1
    if step_value is None:
        stop = _require_number(stop_value, "grid.stop")
        step_value = (stop - start) / max(count - 1, 1)
    step = float(step_value)
    return [float(start + idx * step) for idx in range(count)]


def _require_number(value: Any, label: str) -> float:
    if value is None:
        raise ValueError(f"{label} must be provided")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"{label} must be numeric") from exc


def _validate_dimensions(base_payload: Mapping[str, Any], dimensions: Sequence[SweepDimension]) -> None:
    for dimension in dimensions:
        parts = dimension.field_path.split(".")
        if not parts:
            raise ValueError("Sweep dimension paths must not be empty.")
        node: Any = base_payload
        path_history: List[str] = []
        for idx, raw_part in enumerate(parts):
            part = _normalize_key(path_history, raw_part)
            is_last = idx == len(parts) - 1
            if idx == 0 and part not in node:
                raise ValueError(f"Sweep dimension '{dimension.field_path}' does not match the base spec.")
            if not isinstance(node, Mapping):
                raise ValueError(f"Sweep dimension '{dimension.field_path}' cannot traverse non-mapping fields.")
            if part not in node and not is_last:
                raise ValueError(
                    f"Sweep dimension '{dimension.field_path}' references missing intermediate field '{part}'."
                )
            node = node.get(part, {})
            path_history.append(part)


def _assign_dimension(payload: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    node: Dict[str, Any] = payload
    traversed: List[str] = []
    for idx, raw_part in enumerate(parts):
        part = _normalize_key(traversed, raw_part)
        is_last = idx == len(parts) - 1
        if is_last:
            node[part] = value
            return
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
        traversed.append(part)


def _normalize_key(parent_keys: List[str], key: str) -> str:
    if parent_keys == ["cost_config"] and key == "tx_cost_bps":
        return "transaction_cost_bp"
    return key


__all__ = ["SweepSpec", "SweepDimension", "ExpandedExperiment", "expand_sweep", "expand_sweep_entries"]
