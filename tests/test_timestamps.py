"""Tests for infra.timestamps shared coercion utility."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from infra.timestamps import coerce_timestamp

UTC = timezone.utc


# --- datetime passthrough ---

class TestDatetimeInput:
    def test_aware_datetime_returned_as_utc(self):
        dt = datetime(2024, 6, 15, 12, 0, tzinfo=UTC)
        assert coerce_timestamp(dt) == dt

    def test_naive_datetime_gets_utc(self):
        naive = datetime(2024, 6, 15, 12, 0)
        result = coerce_timestamp(naive)
        assert result == datetime(2024, 6, 15, 12, 0, tzinfo=UTC)
        assert result.tzinfo is UTC

    def test_non_utc_datetime_converted(self):
        from datetime import timedelta
        est = timezone(timedelta(hours=-5))
        dt = datetime(2024, 6, 15, 12, 0, tzinfo=est)
        result = coerce_timestamp(dt)
        assert result == datetime(2024, 6, 15, 17, 0, tzinfo=UTC)


# --- date input ---

class TestDateInput:
    def test_date_becomes_midnight_utc(self):
        d = date(2024, 6, 15)
        result = coerce_timestamp(d)
        assert result == datetime(2024, 6, 15, 0, 0, tzinfo=UTC)


# --- numeric (epoch) input ---

class TestNumericInput:
    def test_epoch_seconds_explicit(self):
        # 2024-01-01T00:00:00Z
        ts = 1704067200
        result = coerce_timestamp(ts, epoch_unit="s")
        assert result == datetime(2024, 1, 1, 0, 0, tzinfo=UTC)

    def test_epoch_ms_explicit(self):
        ts = 1704067200000
        result = coerce_timestamp(ts, epoch_unit="ms")
        assert result == datetime(2024, 1, 1, 0, 0, tzinfo=UTC)

    def test_auto_detects_ms(self):
        ts = 1704067200000  # > 1e12, so auto treats as ms
        result = coerce_timestamp(ts, epoch_unit="auto")
        assert result == datetime(2024, 1, 1, 0, 0, tzinfo=UTC)

    def test_auto_detects_seconds(self):
        ts = 1704067200  # < 1e12, so auto treats as seconds
        result = coerce_timestamp(ts, epoch_unit="auto")
        assert result == datetime(2024, 1, 1, 0, 0, tzinfo=UTC)

    def test_default_is_auto(self):
        ts = 1704067200000
        result = coerce_timestamp(ts)
        assert result == datetime(2024, 1, 1, 0, 0, tzinfo=UTC)

    def test_float_epoch(self):
        ts = 1704067200.5
        result = coerce_timestamp(ts, epoch_unit="s")
        assert result.year == 2024
        assert result.microsecond == 500000


# --- string input ---

class TestStringInput:
    def test_iso_string(self):
        result = coerce_timestamp("2024-06-15T12:00:00+00:00")
        assert result == datetime(2024, 6, 15, 12, 0, tzinfo=UTC)

    def test_z_suffix(self):
        result = coerce_timestamp("2024-06-15T12:00:00Z")
        assert result == datetime(2024, 6, 15, 12, 0, tzinfo=UTC)

    def test_naive_iso_string_gets_utc(self):
        result = coerce_timestamp("2024-06-15T12:00:00")
        assert result == datetime(2024, 6, 15, 12, 0, tzinfo=UTC)

    def test_bare_date_string(self):
        result = coerce_timestamp("2024-06-15")
        assert result == datetime(2024, 6, 15, 0, 0, tzinfo=UTC)

    def test_whitespace_stripped(self):
        result = coerce_timestamp("  2024-06-15T12:00:00Z  ")
        assert result == datetime(2024, 6, 15, 12, 0, tzinfo=UTC)

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="empty"):
            coerce_timestamp("")

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Invalid"):
            coerce_timestamp("not-a-date")


# --- pandas Timestamp-like ---

class TestPandasLike:
    def test_to_pydatetime_called(self):
        class FakeTimestamp:
            def to_pydatetime(self):
                return datetime(2024, 6, 15, 12, 0, tzinfo=UTC)

        result = coerce_timestamp(FakeTimestamp())
        assert result == datetime(2024, 6, 15, 12, 0, tzinfo=UTC)


# --- error handling ---

class TestErrors:
    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported"):
            coerce_timestamp([1, 2, 3])

    def test_index_in_error_message(self):
        with pytest.raises(TypeError, match="at index 5"):
            coerce_timestamp([1, 2, 3], index=5)

    def test_invalid_epoch_unit(self):
        with pytest.raises(ValueError, match="Unknown epoch_unit"):
            coerce_timestamp(1234, epoch_unit="ns")


# --- epoch_unit consistency: the bug we're fixing ---

class TestEpochConsistency:
    """Verify that explicit epoch_unit prevents the old ambiguity."""

    def test_same_value_different_units(self):
        val = 1704067200
        as_seconds = coerce_timestamp(val, epoch_unit="s")
        as_ms = coerce_timestamp(val, epoch_unit="ms")
        # Treating as seconds: 2024-01-01
        assert as_seconds == datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
        # Treating as ms: ~1970-01-20 (much earlier)
        assert as_ms.year == 1970

    def test_polygon_uses_auto(self):
        # Polygon sends epoch-ms, which are > 1e12
        polygon_ts = 1704067200000
        result = coerce_timestamp(polygon_ts, epoch_unit="auto")
        assert result == datetime(2024, 1, 1, 0, 0, tzinfo=UTC)

    def test_ivolatility_uses_seconds(self):
        ivol_ts = 1704067200
        result = coerce_timestamp(ivol_ts, epoch_unit="s")
        assert result == datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
