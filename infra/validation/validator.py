"""Validation engine enforcing canonical schemas."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timezone
import math
import numbers
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .canonical_schemas import CanonicalSchema, FieldSpec, get_canonical_schema
from .manifest import DEFAULT_MANIFEST_DIR, build_manifest, persist_manifest


UTC = timezone.utc


class ValidationError(Exception):
    """Raised when a validation run contains invalid records."""

    def __init__(
        self,
        message: str,
        *,
        manifest: Dict[str, Any],
        errors: Sequence[Dict[str, Any]],
        validated_records: Sequence[Dict[str, Any]],
    ) -> None:
        super().__init__(message)
        self.manifest = manifest
        self.errors = list(errors)
        self.validated_records = list(validated_records)


class RecordValidationError(Exception):
    """Internal error used to capture per-record validation failures."""


class Validator:
    """Validator tied to a specific schema and vendor context."""

    def __init__(
        self,
        schema: CanonicalSchema,
        *,
        source_vendor: str,
        run_id: str,
        config: Dict[str, Any] | None = None,
    ) -> None:
        self.schema = schema
        self.source_vendor = source_vendor
        self.run_id = run_id
        self.config = config or {}
        self.allow_naive_timestamps = bool(self.config.get("allow_naive_timestamps", False))
        self.allow_extra_fields = bool(self.config.get("allow_extra_fields", False))
        self.manifest_base_path = Path(self.config.get("manifest_base_path", DEFAULT_MANIFEST_DIR))
        self.input_file_hashes = list(self.config.get("input_file_hashes", []))
        self.creation_timestamp = self._resolve_creation_timestamp(self.config.get("creation_timestamp"))

    def validate(self, records: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]]):
        if records is None:
            raise ValueError("records must be an iterable of mappings")
        record_list = list(records)
        errors: List[Dict[str, Any]] = []
        validated: List[Dict[str, Any]] = []
        seen_keys: set[Tuple[Any, ...]] = set()

        for index, record in enumerate(record_list):
            try:
                normalized = self._validate_record(record, index)
                unique_key = tuple(normalized[field] for field in self.schema.uniqueness_key)
                if unique_key in seen_keys:
                    raise RecordValidationError(
                        f"duplicate canonical key {self.schema.uniqueness_key} observed"
                    )
                seen_keys.add(unique_key)
                validated.append(normalized)
            except RecordValidationError as exc:
                errors.append({"index": index, "error": str(exc)})

        manifest = build_manifest(
            domain=self.schema.domain,
            schema_version=self.schema.schema_version,
            source_vendor=self.source_vendor,
            run_id=self.run_id,
            input_file_hashes=self.input_file_hashes,
            total_records=len(record_list),
            valid_records=len(validated),
            invalid_records=len(errors),
            validation_status="passed" if not errors else "failed",
            creation_timestamp=self.creation_timestamp,
            errors=errors,
        )
        manifest_path = persist_manifest(manifest, self.manifest_base_path)
        manifest["manifest_path"] = str(manifest_path)

        if errors:
            raise ValidationError(
                f"Validation failed for domain {self.schema.domain}",
                manifest=manifest,
                errors=errors,
                validated_records=validated,
            )

        return validated, manifest

    def _validate_record(self, record: Mapping[str, Any], index: int) -> Dict[str, Any]:
        if not isinstance(record, Mapping):
            raise RecordValidationError("Each record must be a mapping")

        normalized: Dict[str, Any] = dict(record) if self.allow_extra_fields else {}
        record_keys = set(record.keys())
        expected_keys = set(self.schema.field_specs.keys())
        extra_keys = record_keys - expected_keys
        if extra_keys and not self.allow_extra_fields:
            raise RecordValidationError(f"Unexpected field(s) {sorted(extra_keys)} present")

        for field_name, spec in self.schema.field_specs.items():
            if field_name not in record:
                if spec.required:
                    raise RecordValidationError(f"Missing required field '{field_name}'")
                continue

            value = record[field_name]
            normalized[field_name] = self._normalize_value(field_name, value, spec)

        if normalized.get("source_vendor") and normalized["source_vendor"] != self.source_vendor:
            raise RecordValidationError(
                f"Record source_vendor '{normalized['source_vendor']}' does not match "
                f"validator context '{self.source_vendor}'"
            )

        return normalized

    def _normalize_value(self, field_name: str, value: Any, spec: FieldSpec) -> Any:
        if spec.is_timestamp:
            return self._normalize_timestamp(field_name, value)
        if spec.is_date:
            return self._normalize_date(field_name, value, spec)

        if not self._matches_dtype(value, spec.dtype):
            raise RecordValidationError(
                f"Field '{field_name}' expected types {spec.dtype} but received {type(value)}"
            )

        if spec.numeric_constraint:
            self._enforce_numeric_constraint(field_name, value, spec.numeric_constraint)

        return value

    def _normalize_timestamp(self, field_name: str, value: Any) -> datetime:
        dt = self._coerce_datetime(value, field_name)
        if dt.tzinfo is None:
            if not self.allow_naive_timestamps:
                raise RecordValidationError(
                    f"Naive timestamp found for '{field_name}' and configuration disallows it"
                )
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)

    def _normalize_date(self, field_name: str, value: Any, spec: FieldSpec) -> Any:
        if isinstance(value, datetime):
            coerced = value.astimezone(UTC).date()
        elif isinstance(value, date):
            coerced = value
        elif isinstance(value, str):
            try:
                coerced = date.fromisoformat(value)
            except ValueError as exc:  # pragma: no cover - defensive
                raise RecordValidationError(
                    f"Field '{field_name}' must be ISO formatted YYYY-MM-DD"
                ) from exc
        else:
            raise RecordValidationError(f"Field '{field_name}' must be a date value")

        return coerced.isoformat() if spec.as_iso_date_str else coerced

    def _coerce_datetime(self, value: Any, field_name: str) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            text = value.strip()
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                return datetime.fromisoformat(text)
            except ValueError as exc:
                raise RecordValidationError(
                    f"Field '{field_name}' must be ISO datetime text"
                ) from exc
        raise RecordValidationError(f"Field '{field_name}' must be a datetime value")

    def _matches_dtype(self, value: Any, dtypes: Tuple[type, ...]) -> bool:
        for dtype in dtypes:
            if dtype is numbers.Real:
                if self._is_real_number(value):
                    return True
            elif isinstance(value, dtype):
                return True
        return False

    def _is_real_number(self, value: Any) -> bool:
        return isinstance(value, numbers.Real) and not isinstance(value, bool)

    def _enforce_numeric_constraint(self, field_name: str, value: Any, constraint: str) -> None:
        if not self._is_real_number(value):
            raise RecordValidationError(f"Field '{field_name}' must be numeric")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise RecordValidationError(f"Field '{field_name}' must be finite")
        if constraint == "positive" and not numeric_value > 0:
            raise RecordValidationError(f"Field '{field_name}' must be positive")
        if constraint == "non_negative" and numeric_value < 0:
            raise RecordValidationError(f"Field '{field_name}' must be non-negative")

    def _resolve_creation_timestamp(self, candidate: Any | None) -> str:
        if candidate is None:
            candidate = datetime.now(timezone.utc)
        if isinstance(candidate, str):
            return candidate
        if isinstance(candidate, datetime):
            dt = candidate if candidate.tzinfo else candidate.replace(tzinfo=UTC)
            return dt.astimezone(UTC).isoformat()
        raise TypeError("creation_timestamp must be a datetime or ISO string")


def validate_records(
    domain: str,
    records: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    *,
    source_vendor: str,
    run_id: str,
    config: Dict[str, Any] | None = None,
):
    """Public API for validating records against the canonical schema."""

    if not isinstance(source_vendor, str) or not source_vendor:
        raise ValueError("source_vendor must be a non-empty string")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty string")

    schema = get_canonical_schema(domain)
    validator = Validator(schema, source_vendor=source_vendor, run_id=run_id, config=config)
    return validator.validate(records)


__all__ = ["validate_records", "ValidationError", "Validator"]
