"""Feature registry controlling deterministic observation ordering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Dict, Mapping, Sequence, Tuple

from infra.paths import get_data_root

try:  # pragma: no cover - pandas optional in some environments
    import pandas as pd  # type: ignore
except Exception as exc:  # pragma: no cover
    pd = None
    _PANDAS_ERROR: Exception | None = exc
else:  # pragma: no cover
    _PANDAS_ERROR = None

from research.datasets.canonical_options_loader import CanonicalOptionData, load_canonical_options
from research.features.options_features_v1 import OPTION_FEATURE_COLUMNS, compute_options_features
from research.strategies.sma_crossover import SMAStrategyResult

SMA_OBSERVATION_COLUMNS = ("close", "sma_fast", "sma_slow", "sma_diff", "sma_signal")
OPTIONS_OBSERVATION_COLUMNS = ("close",) + OPTION_FEATURE_COLUMNS
SMA_PLUS_OPTIONS_COLUMNS = SMA_OBSERVATION_COLUMNS + tuple(name for name in OPTION_FEATURE_COLUMNS)


@dataclass(frozen=True)
class FeatureSetResult:
    """Outcome of building a specific feature set."""

    frame: "pd.DataFrame"
    observation_columns: Tuple[str, ...]
    feature_set: str
    inputs_used: Dict[str, str]


@dataclass(frozen=True)
class _FeatureSetSpec:
    name: str
    observation_columns: Tuple[str, ...]
    requires_options: bool
    builder: Callable[["pd.DataFrame", CanonicalOptionData | None], "pd.DataFrame"]


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
    normalized_name = str(feature_set).strip().lower()
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
    _validate_columns(frame, spec.observation_columns)
    return FeatureSetResult(
        frame=frame,
        observation_columns=spec.observation_columns,
        feature_set=spec.name,
        inputs_used=feature_hashes,
    )


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


def _ensure_pandas_available() -> None:
    if pd is None:
        raise RuntimeError("pandas is required for feature registry operations") from _PANDAS_ERROR


_FEATURE_REGISTRY: Dict[str, _FeatureSetSpec] = {
    "sma_v1": _FeatureSetSpec(
        name="sma_v1",
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
}


__all__ = ["FeatureSetResult", "build_features", "strategy_to_feature_frame", "SMA_OBSERVATION_COLUMNS", "OPTIONS_OBSERVATION_COLUMNS"]
