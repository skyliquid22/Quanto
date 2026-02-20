"""Shared timestamp coercion utility.

Consolidates the various _coerce_timestamp / _coerce_datetime functions that
were duplicated across adapters with incompatible epoch handling.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

UTC = timezone.utc

# Threshold: values above this are assumed to be epoch milliseconds when
# epoch_unit="auto".  1e12 ms ≈ 2001-09-09, so any modern epoch-ms value
# exceeds it while epoch-seconds values (up to ~1.7e9 in 2024) stay below.
_MS_THRESHOLD = 1e12


def coerce_timestamp(
    value: Any,
    *,
    epoch_unit: str = "auto",
    index: int | None = None,
) -> datetime:
    """Convert *value* to a timezone-aware UTC datetime.

    Parameters
    ----------
    value:
        The value to convert.  Accepted types:

        * ``datetime`` – returned as-is (made tz-aware if naive).
        * ``date`` – interpreted as midnight UTC.
        * ``int | float`` – interpreted as an epoch timestamp; the
          *epoch_unit* parameter controls the unit.
        * ``str`` – parsed via ``datetime.fromisoformat`` with ``Z``-suffix
          handling.  A bare ``YYYY-MM-DD`` string is treated as midnight UTC.
        * Objects with a ``to_pydatetime()`` method (e.g. pandas Timestamp)
          are converted first.

    epoch_unit:
        How to interpret numeric values.

        * ``"auto"`` (default) – values > 1 × 10¹² are treated as
          milliseconds; smaller values as seconds.
        * ``"s"`` – always treat as epoch seconds.
        * ``"ms"`` – always treat as epoch milliseconds.

    index:
        Optional positional index for richer error messages when processing
        sequences of records.

    Returns
    -------
    datetime
        A timezone-aware ``datetime`` in UTC.

    Raises
    ------
    TypeError | ValueError
        If *value* cannot be interpreted as a timestamp.
    """
    # pandas Timestamp / similar wrappers
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)

    if isinstance(value, (int, float)):
        seconds = _epoch_to_seconds(float(value), epoch_unit)
        return datetime.fromtimestamp(seconds, tz=UTC)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            _raise(ValueError, "timestamp string cannot be empty", index)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            # Bare YYYY-MM-DD (10 chars) → midnight UTC
            if len(text) == 10:
                parsed = datetime.fromisoformat(text + "T00:00:00+00:00")
            else:
                _raise(ValueError, f"Invalid timestamp '{value}'", index)
                return None  # unreachable, helps type-checkers
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)

    _raise(TypeError, f"Unsupported timestamp type {type(value)}", index)
    return None  # unreachable


def _epoch_to_seconds(value: float, epoch_unit: str) -> float:
    if epoch_unit == "s":
        return value
    if epoch_unit == "ms":
        return value / 1000
    if epoch_unit == "auto":
        return value / 1000 if value > _MS_THRESHOLD else value
    raise ValueError(f"Unknown epoch_unit '{epoch_unit}'; expected 'auto', 's', or 'ms'")


def _raise(
    exc_type: type[Exception],
    message: str,
    index: int | None,
) -> None:
    position = f" at index {index}" if index is not None else ""
    raise exc_type(f"{message}{position}")
