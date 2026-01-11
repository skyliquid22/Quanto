"""Feature registry controlling deterministic observation ordering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence, Tuple
import logging

import numpy as np

from infra.paths import get_data_root

try:  # pragma: no cover - pandas optional in some environments
    import pandas as pd  # type: ignore
except Exception as exc:  # pragma: no cover
    pd = None
    _PANDAS_ERROR: Exception | None = exc
else:  # pragma: no cover
    _PANDAS_ERROR = None

from research.datasets.canonical_options_loader import CanonicalOptionData, load_canonical_options
from research.features.equity_xsec_features_v1 import EQUITY_XSEC_OBSERVATION_COLUMNS
from research.features.options_features_v1 import OPTION_FEATURE_COLUMNS, compute_options_features
from research.features.regime_features_v1 import REGIME_FEATURE_COLUMNS, compute_regime_features
from research.regime import RegimeState
from research.strategies.sma_crossover import SMAStrategyResult
from research.features.core_features_v1 import CORE_V1_OBSERVATION_COLUMNS, compute_core_features_v1


LOGGER = logging.getLogger(__name__)

SMA_OBSERVATION_COLUMNS = ("close", "sma_fast", "sma_slow", "sma_diff", "sma_signal")
OPTIONS_OBSERVATION_COLUMNS = ("close",) + OPTION_FEATURE_COLUMNS
SMA_PLUS_OPTIONS_COLUMNS = SMA_OBSERVATION_COLUMNS + tuple(name for name in OPTION_FEATURE_COLUMNS)
SMA_PLUS_XSEC_OBSERVATION_COLUMNS = SMA_OBSERVATION_COLUMNS + tuple(
    column for column in EQUITY_XSEC_OBSERVATION_COLUMNS if column != "close"
)


def _merge_observation_columns(
    primary: Sequence[str], secondary: Sequence[str], *, warning_label: str | None = None
) -> Tuple[str, ...]:
    seen: set[str] = set()
    combined: List[str] = []
    duplicates: List[str] = []
    for column in primary:
        if column not in seen:
            combined.append(column)
            seen.add(column)
    for column in secondary:
        if column in seen:
            duplicates.append(column)
            continue
        combined.append(column)
        seen.add(column)
    if duplicates and warning_label:
        LOGGER.warning(
            "%s dropping duplicate columns from secondary feature set: %s",
            warning_label,
            ", ".join(sorted(set(duplicates))),
        )
    return tuple(combined)


CORE_V1_XSEC_OBSERVATION_COLUMNS = _merge_observation_columns(
    CORE_V1_OBSERVATION_COLUMNS,
    tuple(column for column in EQUITY_XSEC_OBSERVATION_COLUMNS if column != "close"),
    warning_label="core_v1_xsec",
)


@dataclass(frozen=True)
class FeatureSetResult:
    """Outcome of building a specific feature set."""

    frame: "pd.DataFrame"
    observation_columns: Tuple[str, ...]
    feature_set: str
    inputs_used: Dict[str, str]


@dataclass(frozen=True)
class UniverseFeaturePanel:
    """Aligned multi-symbol observations for universe environments."""

    rows: List[Dict[str, object]]
    symbol_order: Tuple[str, ...]
    observation_columns: Tuple[str, ...]


@dataclass(frozen=True)
class _FeatureSetSpec:
    name: str
    observation_columns: Tuple[str, ...]
    requires_options: bool
    builder: Callable[["pd.DataFrame", CanonicalOptionData | None], "pd.DataFrame"]


_UNIVERSE_FEATURE_OBSERVATION_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "equity_xsec_v1": EQUITY_XSEC_OBSERVATION_COLUMNS,
    "sma_plus_xsec_v1": SMA_PLUS_XSEC_OBSERVATION_COLUMNS,
    "sma_plus_regime_v1": SMA_OBSERVATION_COLUMNS + REGIME_FEATURE_COLUMNS,
    "xsec_plus_regime_v1": EQUITY_XSEC_OBSERVATION_COLUMNS + REGIME_FEATURE_COLUMNS,
    "core_v1_regime": CORE_V1_OBSERVATION_COLUMNS + REGIME_FEATURE_COLUMNS,
    "core_v1_xsec": CORE_V1_XSEC_OBSERVATION_COLUMNS,
    "core_v1_xsec_regime": CORE_V1_XSEC_OBSERVATION_COLUMNS + REGIME_FEATURE_COLUMNS,
}
_REGIME_FEATURE_OBSERVATION_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "regime_v1": REGIME_FEATURE_COLUMNS,
}

_FEATURE_SET_DEFAULT_REGIME: Dict[str, str] = {
    "sma_plus_regime_v1": "regime_v1",
    "xsec_plus_regime_v1": "regime_v1",
    "core_v1_regime": "regime_v1",
    "core_v1_xsec_regime": "regime_v1",
}


def strategy_to_feature_frame(strategy: SMAStrategyResult) -> "pd.DataFrame":
    """Convert SMA strategy outputs into a canonical equity feature frame."""

    _ensure_pandas_available()
    rows = []
    for idx, timestamp in enumerate(strategy.timestamps):
        fast = strategy.fast_sma[idx]
        slow = strategy.slow_sma[idx]
        if fast is None or slow is None:
            continue
        rows.append(
            {
                "timestamp": pd.to_datetime(timestamp, utc=True),
                "close": float(strategy.closes[idx]),
                "sma_fast": float(fast),
                "sma_slow": float(slow),
                "sma_diff": float(fast - slow),
                "sma_signal": float(strategy.signal[idx]),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["timestamp", "close", *SMA_OBSERVATION_COLUMNS[1:]])
    frame = pd.DataFrame(rows)
    frame.sort_values("timestamp", inplace=True, kind="mergesort")
    frame.reset_index(drop=True, inplace=True)
    return frame


def build_features(
    feature_set: str,
    equity_df: "pd.DataFrame",
    *,
    underlying_symbol: str,
    start_date: date | str,
    end_date: date | str,
    data_root: Path | None = None,
    options_data: CanonicalOptionData | None = None,
) -> FeatureSetResult:
    """Build the requested feature set and return the deterministic column order."""

    _ensure_pandas_available()
    normalized_name = normalize_feature_set_name(feature_set)
    if normalized_name in _UNIVERSE_FEATURE_OBSERVATION_COLUMNS:
        raise ValueError(f"Feature set '{feature_set}' requires multi-symbol universe context")
    if normalized_name not in _FEATURE_REGISTRY:
        raise ValueError(f"Unknown feature set '{feature_set}'")
    spec = _FEATURE_REGISTRY[normalized_name]
    prepared = _prepare_equity_frame(equity_df)

    feature_hashes: Dict[str, str] = {}
    options_payload = options_data
    if spec.requires_options:
        options_payload, feature_hashes = _load_options_payload(
            underlying_symbol,
            start_date,
            end_date,
            data_root=data_root,
            options_data=options_data,
        )
    frame = spec.builder(prepared.copy(), options_payload)
    frame.sort_values("timestamp", inplace=True, kind="mergesort")
    frame.reset_index(drop=True, inplace=True)
    frame = _order_columns(frame, spec.observation_columns)
    _validate_columns(frame, spec.observation_columns)
    return FeatureSetResult(
        frame=frame,
        observation_columns=spec.observation_columns,
        feature_set=spec.name,
        inputs_used=feature_hashes,
    )


def normalize_feature_set_name(feature_set: str) -> str:
    name = str(feature_set or "").strip().lower()
    if not name:
        raise ValueError("feature_set must be provided")
    return name


def normalize_regime_feature_set_name(regime_feature_set: str) -> str:
    name = normalize_feature_set_name(regime_feature_set)
    if name not in _REGIME_FEATURE_OBSERVATION_COLUMNS:
        raise ValueError(f"Unknown regime feature set '{regime_feature_set}'")
    return name


def is_regime_feature_set(regime_feature_set: str | None) -> bool:
    if not regime_feature_set:
        return False
    normalized = normalize_feature_set_name(regime_feature_set)
    return normalized in _REGIME_FEATURE_OBSERVATION_COLUMNS


def default_regime_for_feature_set(feature_set: str) -> str | None:
    if not feature_set:
        return None
    normalized = normalize_feature_set_name(feature_set)
    return _FEATURE_SET_DEFAULT_REGIME.get(normalized)


def is_universe_feature_set(feature_set: str) -> bool:
    normalized = normalize_feature_set_name(feature_set)
    return normalized in _UNIVERSE_FEATURE_OBSERVATION_COLUMNS


def observation_columns_for_feature_set(feature_set: str) -> Tuple[str, ...]:
    normalized = normalize_feature_set_name(feature_set)
    if normalized in _FEATURE_REGISTRY:
        return _FEATURE_REGISTRY[normalized].observation_columns
    if normalized in _UNIVERSE_FEATURE_OBSERVATION_COLUMNS:
        return _UNIVERSE_FEATURE_OBSERVATION_COLUMNS[normalized]
    if normalized in _REGIME_FEATURE_OBSERVATION_COLUMNS:
        return _REGIME_FEATURE_OBSERVATION_COLUMNS[normalized]
    raise ValueError(f"Unknown feature set '{feature_set}'")


def _prepare_equity_frame(equity_df: "pd.DataFrame") -> "pd.DataFrame":
    if "timestamp" not in equity_df or "close" not in equity_df:
        raise ValueError("equity_df must include timestamp and close columns")
    frame = equity_df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["close"] = frame["close"].astype(float)
    frame.sort_values("timestamp", inplace=True, kind="mergesort")
    frame.reset_index(drop=True, inplace=True)
    if "sma_diff" not in frame and {"sma_fast", "sma_slow"}.issubset(frame.columns):
        frame["sma_diff"] = frame["sma_fast"].astype(float) - frame["sma_slow"].astype(float)
    if "sma_signal" not in frame:
        frame["sma_signal"] = 0.0
    return frame


def _sma_only_builder(equity_df: "pd.DataFrame", _: CanonicalOptionData | None) -> "pd.DataFrame":
    return equity_df


def _options_only_builder(equity_df: "pd.DataFrame", options: CanonicalOptionData | None) -> "pd.DataFrame":
    if options is None:
        raise ValueError("options data is required for options feature sets")
    merged = compute_options_features(
        equity_df,
        options.reference,
        options.open_interest,
        options.ohlcv,
    )
    return merged


def _core_v1_builder(equity_df: "pd.DataFrame", _: CanonicalOptionData | None) -> "pd.DataFrame":
    return compute_core_features_v1(equity_df)



def _load_options_payload(
    symbol: str,
    start: date | str,
    end: date | str,
    *,
    data_root: Path | None,
    options_data: CanonicalOptionData | None,
) -> Tuple[CanonicalOptionData, Dict[str, str]]:
    if options_data is not None:
        return options_data, {}
    root = Path(data_root) if data_root else get_data_root()
    payload, hashes = load_canonical_options(symbol, start, end, data_root=root)
    return payload, hashes


def _validate_columns(frame: "pd.DataFrame", observation_columns: Sequence[str]) -> None:
    missing = [column for column in observation_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Feature set missing required columns: {missing}")


def _order_columns(frame: "pd.DataFrame", observation_columns: Sequence[str]) -> "pd.DataFrame":
    columns = list(frame.columns)
    if "timestamp" not in columns:
        raise ValueError("Feature frames must include timestamp columns")
    ordered = ["timestamp"]
    for column in observation_columns:
        if column not in ordered:
            ordered.append(column)
    remainder = [column for column in columns if column not in ordered]
    return frame[ordered + remainder]


def _ensure_pandas_available() -> None:
    if pd is None:
        raise RuntimeError("pandas is required for feature registry operations") from _PANDAS_ERROR


def _normalize_regime_feature_set(name: str | None) -> str | None:
    if not name:
        return None
    return normalize_regime_feature_set_name(name)


def build_universe_feature_panel(
    feature_results: Mapping[str, FeatureSetResult],
    *,
    symbol_order: Sequence[str] | None = None,
    calendar: "pd.DatetimeIndex | Sequence[object] | None" = None,
    forward_fill_limit: int = 3,
    fill_value: float = 0.0,
    regime_feature_set: str | None = None,
) -> UniverseFeaturePanel:
    """Align feature frames for multiple symbols on a shared calendar."""

    _ensure_pandas_available()
    if not feature_results:
        raise ValueError("feature_results cannot be empty")
    order = tuple(dict.fromkeys((symbol_order or sorted(feature_results.keys()))))
    if len(order) != len(set(order)):
        raise ValueError("symbol_order must not contain duplicates")
    missing = [symbol for symbol in order if symbol not in feature_results]
    if missing:
        raise ValueError(f"feature_results missing symbols: {missing}")
    normalized_regime = _normalize_regime_feature_set(regime_feature_set)
    if normalized_regime and len(order) < 2:
        raise ValueError("Regime features require at least two symbols")
    base_columns: Tuple[str, ...] | None = None
    union_index: "pd.DatetimeIndex | None" = None
    if calendar is not None:
        union_index = pd.DatetimeIndex(pd.to_datetime(calendar, utc=True))
        union_index = pd.DatetimeIndex(union_index.sort_values().unique(), tz=union_index.tz)
    per_symbol_frames: Dict[str, "pd.DataFrame"] = {}
    for symbol in order:
        result = feature_results[symbol]
        if base_columns is None:
            base_columns = result.observation_columns
        elif base_columns != result.observation_columns:
            raise ValueError("All feature sets must expose identical observation columns in universe mode")
        frame = result.frame.copy()
        if frame.empty:
            continue
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame.set_index("timestamp", inplace=True)
        if union_index is None:
            union_index = frame.index
        else:
            union_index = union_index.union(frame.index)
        per_symbol_frames[symbol] = frame
    if union_index is None or union_index.empty:
        raise ValueError("No overlapping timestamps found across feature frames")
    union_index = pd.DatetimeIndex(union_index.sort_values(), tz=union_index.tz)
    normalized_frames: Dict[str, "pd.DataFrame"] = {}
    ffill_limit = max(0, int(forward_fill_limit))
    for symbol in order:
        frame = per_symbol_frames.get(symbol)
        if frame is None or frame.empty:
            aligned = pd.DataFrame(index=union_index, columns=base_columns or [])
        else:
            aligned = frame.reindex(union_index)
        if ffill_limit > 0:
            aligned = aligned.ffill(limit=ffill_limit)
        normalized_frames[symbol] = aligned
    valid_mask = pd.Series(True, index=union_index)
    for aligned in normalized_frames.values():
        if "close" not in aligned.columns:
            raise ValueError("Feature frames must include 'close' columns for valuation")
        valid_mask &= aligned["close"].notna()
    valid_index = union_index[valid_mask.to_numpy()]
    if len(valid_index) < 2:
        raise ValueError("Universe alignment requires at least two overlapping timestamps")
    regime_columns: Tuple[str, ...] = tuple()
    regime_states: Dict[pd.Timestamp, RegimeState] = {}
    if normalized_regime:
        regime_columns = _REGIME_FEATURE_OBSERVATION_COLUMNS[normalized_regime]
        close_panel = pd.DataFrame(index=valid_index)
        for symbol in order:
            aligned = normalized_frames[symbol]
            close_series = aligned.loc[valid_index, "close"].astype(float)
            close_panel[symbol] = close_series
        regime_frame = compute_regime_features(close_panel)
        for timestamp in valid_index:
            values = [float(regime_frame.at[timestamp, column]) for column in regime_columns]
            regime_states[timestamp] = RegimeState(features=np.asarray(values, dtype="float64"), feature_names=regime_columns)
    rows: List[Dict[str, object]] = []
    fill_number = float(fill_value)
    for timestamp in valid_index:
        panel: Dict[str, Dict[str, float]] = {}
        for symbol in order:
            aligned = normalized_frames[symbol]
            features: Dict[str, float] = {}
            for column in base_columns or ():
                value = aligned.at[timestamp, column] if timestamp in aligned.index else float("nan")
                if pd.isna(value):
                    value = fill_number
                features[column] = float(value)
            panel[symbol] = features
        payload: Dict[str, object] = {"timestamp": timestamp.to_pydatetime(), "panel": panel}
        if regime_states:
            payload["regime_state"] = regime_states[timestamp]
        rows.append(payload)
    observation_columns = base_columns or tuple()
    if regime_columns:
        observation_columns = observation_columns + regime_columns
    return UniverseFeaturePanel(rows=rows, symbol_order=order, observation_columns=observation_columns)


_FEATURE_REGISTRY: Dict[str, _FeatureSetSpec] = {
    "sma_v1": _FeatureSetSpec(
        name="sma_v1",
        observation_columns=SMA_OBSERVATION_COLUMNS,
        requires_options=False,
        builder=_sma_only_builder,
    ),
    "sma_universe_v1": _FeatureSetSpec(
        name="sma_universe_v1",
        observation_columns=SMA_OBSERVATION_COLUMNS,
        requires_options=False,
        builder=_sma_only_builder,
    ),
    "options_v1": _FeatureSetSpec(
        name="options_v1",
        observation_columns=OPTIONS_OBSERVATION_COLUMNS,
        requires_options=True,
        builder=_options_only_builder,
    ),
    "sma_plus_options_v1": _FeatureSetSpec(
        name="sma_plus_options_v1",
        observation_columns=SMA_PLUS_OPTIONS_COLUMNS,
        requires_options=True,
        builder=_options_only_builder,
    ),
    "core_v1": _FeatureSetSpec(
        name="core_v1",
        observation_columns=CORE_V1_OBSERVATION_COLUMNS,
        requires_options=False,
        builder=_core_v1_builder,
    ),

}


__all__ = [
    "FeatureSetResult",
    "UniverseFeaturePanel",
    "build_features",
    "build_universe_feature_panel",
    "strategy_to_feature_frame",
    "SMA_OBSERVATION_COLUMNS",
    "OPTIONS_OBSERVATION_COLUMNS",
    "SMA_PLUS_XSEC_OBSERVATION_COLUMNS",
    "EQUITY_XSEC_OBSERVATION_COLUMNS",
    "CORE_V1_XSEC_OBSERVATION_COLUMNS",
    "normalize_feature_set_name",
    "normalize_regime_feature_set_name",
    "is_universe_feature_set",
    "is_regime_feature_set",
    "default_regime_for_feature_set",
    "observation_columns_for_feature_set",
]
