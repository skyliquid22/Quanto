"""Normalize iVolatility bulk options surface payloads into Quanto schema."""

from __future__ import annotations

from datetime import date as date_type, datetime
import math
from typing import Any, Iterable, Mapping, Sequence


OPT_OI_COLUMNS = (
    "OPT:OI:CALL",
    "OPT:OI:PUT",
    "OPT:OI:TOTAL",
    "OPT:OI:CALL_PUT_RATIO",
)
OPT_VOLUME_COLUMNS = (
    "OPT:VOL:CALL",
    "OPT:VOL:PUT",
    "OPT:VOL:TOTAL",
    "OPT:VOL:CALL_PUT_RATIO",
)
OPT_IVX_COLUMNS = (
    "OPT:IVX:30",
    "OPT:IVX:60",
    "OPT:IVX:90",
    "OPT:IVX:120",
    "OPT:IVX:150",
    "OPT:IVX:180",
    "OPT:IVX:TERM_SLOPE_30_90",
    "OPT:IVX:TERM_SLOPE_30_180",
)
OPT_IVR_COLUMNS = (
    "OPT:IVR:30",
    "OPT:IVR:60",
    "OPT:IVR:90",
    "OPT:IVR:120",
    "OPT:IVR:150",
    "OPT:IVR:180",
)
OPT_IVP_COLUMNS = (
    "OPT:IVP:30",
    "OPT:IVP:60",
    "OPT:IVP:90",
    "OPT:IVP:120",
    "OPT:IVP:150",
    "OPT:IVP:180",
)
OPT_HV_COLUMNS = (
    "OPT:HV:10",
    "OPT:HV:20",
    "OPT:HV:30",
    "OPT:HV:60",
    "OPT:HV:90",
    "OPT:HV:120",
    "OPT:HV:150",
    "OPT:HV:180",
)
OPT_COVERAGE_COLUMNS = (
    "OPT:COVERAGE:HAS_OI",
    "OPT:COVERAGE:HAS_OPT_VOLUME",
    "OPT:COVERAGE:HAS_IVX",
    "OPT:COVERAGE:HAS_IVR",
    "OPT:COVERAGE:HAS_IVP",
    "OPT:COVERAGE:HAS_HV",
    "OPT:COVERAGE:ROW_VALID",
)
NORMALIZED_OPTIONS_SURFACE_COLUMNS: tuple[str, ...] = (
    "symbol",
    "date",
    *OPT_OI_COLUMNS,
    *OPT_VOLUME_COLUMNS,
    *OPT_IVX_COLUMNS,
    *OPT_IVR_COLUMNS,
    *OPT_IVP_COLUMNS,
    *OPT_HV_COLUMNS,
    *OPT_COVERAGE_COLUMNS,
)
OPT_SURFACE_VALUE_COLUMNS: tuple[str, ...] = NORMALIZED_OPTIONS_SURFACE_COLUMNS[2:-len(OPT_COVERAGE_COLUMNS)]
OPT_SURFACE_ALL_OPT_COLUMNS: tuple[str, ...] = tuple(
    column for column in NORMALIZED_OPTIONS_SURFACE_COLUMNS if column.startswith("OPT:")
)

_OI_FIELD_CANDIDATES = {
    "OPT:OI:CALL": ("call_open_interest", "callOpenInterest", "callOI", "oiCall", "openInterestCall"),
    "OPT:OI:PUT": ("put_open_interest", "putOpenInterest", "putOI", "oiPut", "openInterestPut"),
    "OPT:OI:TOTAL": ("total_open_interest", "totalOpenInterest", "openInterestTotal", "oiTotal"),
}
_VOLUME_FIELD_CANDIDATES = {
    "OPT:VOL:CALL": ("call_volume", "callVolume", "volumeCall", "optVolumeCall"),
    "OPT:VOL:PUT": ("put_volume", "putVolume", "volumePut", "optVolumePut"),
    "OPT:VOL:TOTAL": ("total_volume", "totalVolume", "volumeTotal", "optVolumeTotal"),
}
_IVX_FIELD_CANDIDATES = {
    "OPT:IVX:30": ("ivx_30", "ivx30", "IVX30"),
    "OPT:IVX:60": ("ivx_60", "ivx60", "IVX60"),
    "OPT:IVX:90": ("ivx_90", "ivx90", "IVX90"),
    "OPT:IVX:120": ("ivx_120", "ivx120", "IVX120"),
    "OPT:IVX:150": ("ivx_150", "ivx150", "IVX150"),
    "OPT:IVX:180": ("ivx_180", "ivx180", "IVX180"),
}
_IVR_FIELD_CANDIDATES = {
    "OPT:IVR:30": ("ivr_30", "ivr30", "IVR30"),
    "OPT:IVR:60": ("ivr_60", "ivr60", "IVR60"),
    "OPT:IVR:90": ("ivr_90", "ivr90", "IVR90"),
    "OPT:IVR:120": ("ivr_120", "ivr120", "IVR120"),
    "OPT:IVR:150": ("ivr_150", "ivr150", "IVR150"),
    "OPT:IVR:180": ("ivr_180", "ivr180", "IVR180"),
}
_IVP_FIELD_CANDIDATES = {
    "OPT:IVP:30": ("ivp_30", "ivp30", "IVP30"),
    "OPT:IVP:60": ("ivp_60", "ivp60", "IVP60"),
    "OPT:IVP:90": ("ivp_90", "ivp90", "IVP90"),
    "OPT:IVP:120": ("ivp_120", "ivp120", "IVP120"),
    "OPT:IVP:150": ("ivp_150", "ivp150", "IVP150"),
    "OPT:IVP:180": ("ivp_180", "ivp180", "IVP180"),
}
_HV_FIELD_CANDIDATES = {
    "OPT:HV:10": ("hv_10", "hv10", "HV10"),
    "OPT:HV:20": ("hv_20", "hv20", "HV20"),
    "OPT:HV:30": ("hv_30", "hv30", "HV30"),
    "OPT:HV:60": ("hv_60", "hv60", "HV60"),
    "OPT:HV:90": ("hv_90", "hv90", "HV90"),
    "OPT:HV:120": ("hv_120", "hv120", "HV120"),
    "OPT:HV:150": ("hv_150", "hv150", "HV150"),
    "OPT:HV:180": ("hv_180", "hv180", "HV180"),
}

_SYMBOL_FIELDS = ("symbol", "underlying_symbol", "underlyingSymbol", "ticker", "root")
_DATE_FIELDS = ("date", "pricingDate", "as_of", "asOf", "tradeDate", "tradingDate")
_VALID_VOL_RANGE = (0.0, 5.0)  # expressed in decimals (0-500%)


def normalize_ivol_surface(
    records: Iterable[Mapping[str, Any]],
    *,
    allowed_symbols: Sequence[str] | None = None,
    start_date: date_type | str | None = None,
    end_date: date_type | str | None = None,
    vol_unit: str = "percent",
) -> list[dict[str, Any]]:
    """Normalize arbitrary iVolatility surface payloads."""

    symbol_filter = {sym.strip().upper() for sym in (allowed_symbols or []) if sym}
    start = _coerce_date_optional(start_date)
    end = _coerce_date_optional(end_date)
    vol_mode = _normalize_vol_unit(vol_unit)

    normalized: list[dict[str, Any]] = []
    for entry in records:
        if not isinstance(entry, Mapping):
            continue
        symbol = _extract_symbol(entry)
        if not symbol:
            continue
        if symbol_filter and symbol not in symbol_filter:
            continue
        trade_date = _extract_trade_date(entry)
        if trade_date is None:
            continue
        if start and trade_date < start:
            continue
        if end and trade_date > end:
            continue
        row = _build_row_template(symbol, trade_date)
        invalid = False

        _populate_group(entry, row, _OI_FIELD_CANDIDATES)
        _populate_group(entry, row, _VOLUME_FIELD_CANDIDATES)
        _populate_group(entry, row, _IVX_FIELD_CANDIDATES)
        _populate_group(entry, row, _IVR_FIELD_CANDIDATES)
        _populate_group(entry, row, _IVP_FIELD_CANDIDATES)
        _populate_group(entry, row, _HV_FIELD_CANDIDATES)

        for name in ("OPT:OI:CALL", "OPT:OI:PUT", "OPT:OI:TOTAL"):
            value = row[name]
            if _is_number(value) and value < 0:
                row[name] = math.nan
                invalid = True
        for name in ("OPT:VOL:CALL", "OPT:VOL:PUT", "OPT:VOL:TOTAL"):
            value = row[name]
            if _is_number(value) and value < 0:
                row[name] = math.nan
                invalid = True

        row["OPT:OI:TOTAL"] = _resolve_total(row["OPT:OI:TOTAL"], row["OPT:OI:CALL"], row["OPT:OI:PUT"])
        row["OPT:VOL:TOTAL"] = _resolve_total(row["OPT:VOL:TOTAL"], row["OPT:VOL:CALL"], row["OPT:VOL:PUT"])
        row["OPT:OI:CALL_PUT_RATIO"] = _safe_ratio(row["OPT:OI:CALL"], row["OPT:OI:PUT"])
        row["OPT:VOL:CALL_PUT_RATIO"] = _safe_ratio(row["OPT:VOL:CALL"], row["OPT:VOL:PUT"])
        row["OPT:IVX:TERM_SLOPE_30_90"] = _compute_slope(row["OPT:IVX:30"], row["OPT:IVX:90"])
        row["OPT:IVX:TERM_SLOPE_30_180"] = _compute_slope(row["OPT:IVX:30"], row["OPT:IVX:180"])

        for column in list(OPT_IVX_COLUMNS) + list(OPT_HV_COLUMNS):
            value = row[column]
            if not _is_number(value):
                continue
            vol_value = _normalize_vol_value(value, vol_mode)
            if vol_value is None:
                row[column] = math.nan
                invalid = True
            else:
                row[column] = vol_value

        _update_coverage_flags(row)
        row["OPT:COVERAGE:ROW_VALID"] = bool(row["OPT:COVERAGE:ROW_VALID"]) and not invalid
        normalized.append(row)

    normalized.sort(key=lambda payload: (payload["symbol"], payload["date"]))
    return normalized


def _build_row_template(symbol: str, trade_date: date_type) -> dict[str, Any]:
    row: dict[str, Any] = {"symbol": symbol, "date": trade_date.isoformat()}
    for column in OPT_SURFACE_VALUE_COLUMNS:
        row.setdefault(column, math.nan)
    for flag in OPT_COVERAGE_COLUMNS:
        row[flag] = False
    row["OPT:COVERAGE:ROW_VALID"] = True
    return row


def _populate_group(entry: Mapping[str, Any], row: Mapping[str, Any], mapping: Mapping[str, Sequence[str]]) -> None:
    for column, fields in mapping.items():
        current = row[column]
        if _is_number(current):
            continue
        row[column] = _extract_float(entry, fields)


def _extract_symbol(record: Mapping[str, Any]) -> str:
    for field in _SYMBOL_FIELDS:
        value = record.get(field)
        if value is None:
            continue
        text = str(value).strip().upper()
        if text:
            return text
    return ""


def _extract_trade_date(record: Mapping[str, Any]) -> date_type | None:
    for field in _DATE_FIELDS:
        value = record.get(field)
        if value in (None, ""):
            continue
        try:
            parsed = _coerce_date(value)
        except (ValueError, TypeError):
            continue
        return parsed
    return None


def _extract_float(record: Mapping[str, Any], candidates: Sequence[str]) -> float:
    for candidate in candidates:
        if candidate not in record:
            continue
        value = record[candidate]
        if value in (None, ""):
            continue
        try:
            text = str(value).replace(",", "")
            return float(text)
        except (ValueError, TypeError):
            continue
    return math.nan


def _coerce_date(value: Any) -> date_type:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date_type):
        return value
    text = str(value).strip()
    if not text:
        raise ValueError("date string cannot be empty")
    return datetime.fromisoformat(text.replace("Z", "+00:00")).date()


def _coerce_date_optional(value: Any | None) -> date_type | None:
    if value is None:
        return None
    return _coerce_date(value)


def _resolve_total(total: Any, call_value: Any, put_value: Any) -> float:
    if _is_number(total):
        return float(total)
    if _is_number(call_value) or _is_number(put_value):
        return float((call_value if _is_number(call_value) else 0.0) + (put_value if _is_number(put_value) else 0.0))
    return math.nan


def _safe_ratio(numerator: Any, denominator: Any) -> float:
    if not _is_number(numerator) or not _is_number(denominator):
        return math.nan
    denom = float(denominator)
    if denom == 0.0:
        return math.nan
    return float(numerator) / denom


def _compute_slope(start_value: Any, end_value: Any) -> float:
    if not _is_number(start_value) or not _is_number(end_value):
        return math.nan
    return float(end_value) - float(start_value)


def _normalize_vol_unit(vol_unit: str) -> str:
    mode = (vol_unit or "percent").strip().lower()
    if mode not in {"percent", "decimal"}:
        raise ValueError("vol_unit must be 'percent' or 'decimal'")
    return mode


def _normalize_vol_value(value: float, mode: str) -> float | None:
    number = float(value)
    if mode == "percent":
        number = number / 100.0
    if not math.isfinite(number):
        return None
    if number < _VALID_VOL_RANGE[0] or number > _VALID_VOL_RANGE[1]:
        return None
    return number


def _is_number(value: Any) -> bool:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return not math.isnan(float(value))
    return False


def _update_coverage_flags(row: Mapping[str, Any]) -> None:
    row["OPT:COVERAGE:HAS_OI"] = _has_any(row, OPT_OI_COLUMNS[:3])
    row["OPT:COVERAGE:HAS_OPT_VOLUME"] = _has_any(row, OPT_VOLUME_COLUMNS[:3])
    row["OPT:COVERAGE:HAS_IVX"] = _has_any(row, OPT_IVX_COLUMNS[:-2])
    row["OPT:COVERAGE:HAS_IVR"] = _has_any(row, OPT_IVR_COLUMNS)
    row["OPT:COVERAGE:HAS_IVP"] = _has_any(row, OPT_IVP_COLUMNS)
    row["OPT:COVERAGE:HAS_HV"] = _has_any(row, OPT_HV_COLUMNS)
    row["OPT:COVERAGE:ROW_VALID"] = any(
        (
            row["OPT:COVERAGE:HAS_OI"],
            row["OPT:COVERAGE:HAS_OPT_VOLUME"],
            row["OPT:COVERAGE:HAS_IVX"],
            row["OPT:COVERAGE:HAS_IVR"],
            row["OPT:COVERAGE:HAS_IVP"],
            row["OPT:COVERAGE:HAS_HV"],
        )
    )


def _has_any(row: Mapping[str, Any], columns: Sequence[str]) -> bool:
    for column in columns:
        if _is_number(row.get(column)):
            return True
    return False


__all__ = [
    "NORMALIZED_OPTIONS_SURFACE_COLUMNS",
    "OPT_SURFACE_ALL_OPT_COLUMNS",
    "OPT_COVERAGE_COLUMNS",
    "normalize_ivol_surface",
]

