"""Public validation API."""

from .validator import ValidationError, validate_records

__all__ = ["validate_records", "ValidationError"]
