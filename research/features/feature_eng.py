"""Feature engineering helpers for canonical equity datasets."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, Mapping, Sequence

from research.datasets.canonical_equity_loader import CanonicalEquitySlice
from research.datasets.options_surface_loader import load_options_surface
from research.features.equity_xsec_features_v1 import build_equity_xsec_feature_frames
from research.features.feature_registry import (
    CORE_V1_XSEC_OBSERVATION_COLUMNS,
    CORE_V1_XSEC_REGIME_OPTS_COLUMNS,
    EQUITY_XSEC_OBSERVATION_COLUMNS,
    FeatureSetResult,
    SMA_PLUS_XSEC_OBSERVATION_COLUMNS,
    build_features,
    normalize_feature_set_name,
    normalize_regime_feature_set_name,
    strategy_to_feature_frame,
)
from research.regime.universe import PRIMARY_REGIME_UNIVERSE
from research.features.sets.opts_surface_v1 import attach_surface_columns
from research.features.sets.options_surface_v1 import (
    OPTIONS_SURFACE_V1_COLUMNS,
    compute_previous_session_dates,
)
from research.strategies.sma_crossover import SMAStrategyConfig, run_sma_crossover


def build_sma_feature_result(
    slice_data: CanonicalEquitySlice,
    *,
    fast_window: int,
    slow_window: int,
    feature_set: str = "sma_v1",
    start_date: date | None = None,
    end_date: date | None = None,
    data_root: Path | None = None,
) -> FeatureSetResult:
    """Generate SMA-aligned feature frames for a canonical slice."""

    if fast_window <= 0 or slow_window <= 0:
        raise ValueError("SMA windows must be positive")
    if fast_window >= slow_window:
        raise ValueError("fast_window must be strictly less than slow_window")
    if slice_data.frame.empty:
        raise ValueError("slice_data must contain at least two rows to build SMA features")

    timestamps = slice_data.timestamps
    closes = slice_data.closes
    if len(timestamps) != len(closes) or len(timestamps) < 2:
        raise ValueError("slice_data must provide at least two aligned timestamps and close prices")

    normalized_feature_set = normalize_feature_set_name(feature_set)
    if normalized_feature_set == "core_v1":
        equity_df = slice_data.frame.reset_index()
        equity_df.rename(columns={"index": "timestamp"}, inplace=True)
        start = start_date or timestamps[0].date()
        end = end_date or timestamps[-1].date()
        return build_features(
            "core_v1",
            equity_df,
            underlying_symbol=slice_data.symbol,
            start_date=start,
            end_date=end,
            data_root=data_root,
        )

    config = SMAStrategyConfig(fast_window=fast_window, slow_window=slow_window)
    strategy = run_sma_crossover(slice_data.symbol, timestamps, closes, config)
    frame = strategy_to_feature_frame(strategy)
    if len(frame) < 2:
        raise ValueError("Not enough SMA-aligned rows to build features")

    start = start_date or timestamps[0].date()
    end = end_date or timestamps[-1].date()
    return build_features(
        feature_set,
        frame,
        underlying_symbol=slice_data.symbol,
        start_date=start,
        end_date=end,
        data_root=data_root,
    )


def build_universe_feature_results(
    feature_set: str,
    slices: Mapping[str, CanonicalEquitySlice],
    *,
    symbol_order: Sequence[str] | None = None,
    start_date: date,
    end_date: date,
    sma_config: SMAStrategyConfig | None = None,
    data_root: Path | None = None,
) -> Dict[str, FeatureSetResult]:
    """Build multi-symbol feature sets that require cross-sectional context."""

    normalized = normalize_feature_set_name(feature_set)
    order = tuple(dict.fromkeys(symbol_order or sorted(slices.keys())))
    if len(order) < 2:
        raise ValueError("Universe feature sets require at least two symbols")
    if normalized not in {
        "equity_xsec_v1",
        "sma_plus_xsec_v1",
        "core_v1_regime",
        "core_v1_xsec",
        "core_v1_xsec_regime",
        "core_v1_xsec_regime_opts_v1",
        "core_v1_xsec_regime_opts_v1_lag1",
    }:
        raise ValueError(f"Unsupported universe feature set '{feature_set}'")

    frames = build_equity_xsec_feature_frames(
        slices,
        start_date=start_date,
        end_date=end_date,
        symbol_order=order,
    )
    results: Dict[str, FeatureSetResult] = {}

    if normalized == "equity_xsec_v1":
        for symbol in order:
            frame = frames.get(symbol)
            if frame is None or frame.empty:
                raise ValueError(f"No cross-sectional rows available for {symbol}")
            results[symbol] = FeatureSetResult(
                frame=frame,
                observation_columns=EQUITY_XSEC_OBSERVATION_COLUMNS,
                feature_set="equity_xsec_v1",
                inputs_used={},
            )
        return results

    if sma_config is None and normalized == "sma_plus_xsec_v1":
        raise ValueError("sma_config is required for sma_plus_xsec_v1 features")
    if normalized == "core_v1_regime":
        results: Dict[str, FeatureSetResult] = {}
        for symbol in order:
            slice_data = slices.get(symbol)
            if slice_data is None:
                raise ValueError(f"Missing canonical slice for symbol {symbol}")
            equity_df = slice_data.frame.reset_index()
            feature_result = build_features(
                "core_v1",
                equity_df,
                underlying_symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                data_root=data_root,
            )
            results[symbol] = FeatureSetResult(
                frame=feature_result.frame,
                observation_columns=feature_result.observation_columns,
                feature_set="core_v1_regime",
                inputs_used=feature_result.inputs_used,
            )
        return results

    if normalized in {"core_v1_xsec", "core_v1_xsec_regime"}:
        results: Dict[str, FeatureSetResult] = {}
        for symbol in order:
            slice_data = slices.get(symbol)
            if slice_data is None:
                raise ValueError(f"Missing canonical slice for symbol {symbol}")
            equity_df = slice_data.frame.reset_index()
            feature_result = build_features(
                "core_v1",
                equity_df,
                underlying_symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                data_root=data_root,
            )
            xsec_frame = frames.get(symbol)
            if xsec_frame is None or xsec_frame.empty:
                raise ValueError(f"No cross-sectional rows available for {symbol}")
            merged = feature_result.frame.merge(xsec_frame.drop(columns=["close"], errors="ignore"), on="timestamp", how="inner")
            merged.sort_values("timestamp", inplace=True, kind="mergesort")
            merged.reset_index(drop=True, inplace=True)
            merged = merged.fillna(0.0)
            missing = [column for column in CORE_V1_XSEC_OBSERVATION_COLUMNS if column not in merged.columns]
            for column in missing:
                merged[column] = 0.0
            merged = merged[["timestamp", *CORE_V1_XSEC_OBSERVATION_COLUMNS]]
            if len(merged) < 2:
                raise ValueError(f"Not enough overlapping rows to build {normalized} for {symbol}")
            results[symbol] = FeatureSetResult(
                frame=merged,
                observation_columns=CORE_V1_XSEC_OBSERVATION_COLUMNS,
                feature_set=normalized,
                inputs_used=dict(feature_result.inputs_used),
            )
        return results

    if normalized in {"core_v1_xsec_regime_opts_v1", "core_v1_xsec_regime_opts_v1_lag1"}:
        surface_slices, surface_hashes = load_options_surface(order, start_date, end_date, data_root=data_root)
        per_symbol_hashes: Dict[str, Dict[str, str]] = {symbol: {} for symbol in order}
        for rel_path, digest in surface_hashes.items():
            path_obj = Path(rel_path)
            parts = [part.upper() for part in path_obj.parts]
            for symbol in order:
                if symbol in parts:
                    per_symbol_hashes[symbol][rel_path] = digest
        results: Dict[str, FeatureSetResult] = {}
        for symbol in order:
            slice_data = slices.get(symbol)
            if slice_data is None:
                raise ValueError(f"Missing canonical slice for symbol {symbol}")
            equity_df = slice_data.frame.reset_index()
            feature_result = build_features(
                "core_v1",
                equity_df,
                underlying_symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                data_root=data_root,
            )
            xsec_frame = frames.get(symbol)
            if xsec_frame is None or xsec_frame.empty:
                raise ValueError(f"No cross-sectional rows available for {symbol}")
            merged = feature_result.frame.merge(xsec_frame.drop(columns=["close"], errors="ignore"), on="timestamp", how="inner")
            merged.sort_values("timestamp", inplace=True, kind="mergesort")
            merged.reset_index(drop=True, inplace=True)
            merged = merged.fillna(0.0)
            missing = [column for column in CORE_V1_XSEC_OBSERVATION_COLUMNS if column not in merged.columns]
            for column in missing:
                merged[column] = 0.0
            merged = merged[["timestamp", *CORE_V1_XSEC_OBSERVATION_COLUMNS]]
            if len(merged) < 2:
                raise ValueError(f"Not enough overlapping rows to build core_v1_xsec_regime_opts_v1 for {symbol}")
            surface_slice = surface_slices.get(symbol)
            surface_dates = None
            if normalized == "core_v1_xsec_regime_opts_v1_lag1":
                surface_dates = compute_previous_session_dates(merged["timestamp"])
            enriched = attach_surface_columns(merged, surface_slice, surface_dates=surface_dates)
            for column in OPTIONS_SURFACE_V1_COLUMNS:
                if column not in enriched.columns:
                    enriched[column] = float("nan")
            enriched = enriched[["timestamp", *CORE_V1_XSEC_OBSERVATION_COLUMNS, *OPTIONS_SURFACE_V1_COLUMNS]]
            inputs_used = dict(feature_result.inputs_used)
            inputs_used.update(per_symbol_hashes.get(symbol, {}))
            results[symbol] = FeatureSetResult(
                frame=enriched,
                observation_columns=CORE_V1_XSEC_REGIME_OPTS_COLUMNS,
                feature_set=normalized,
                inputs_used=inputs_used,
            )
        return results

    if sma_config is None:
        raise ValueError("sma_config is required for sma_plus_xsec_v1 features")
    for symbol in order:
        slice_data = slices.get(symbol)
        if slice_data is None:
            raise ValueError(f"Missing canonical slice for symbol {symbol}")
        base_result = build_sma_feature_result(
            slice_data,
            fast_window=sma_config.fast_window,
            slow_window=sma_config.slow_window,
            feature_set="sma_universe_v1",
            start_date=start_date,
            end_date=end_date,
            data_root=data_root,
        )
        cross_frame = frames[symbol].drop(columns=["close"])
        merged = base_result.frame.merge(cross_frame, on="timestamp", how="inner")
        merged.sort_values("timestamp", inplace=True, kind="mergesort")
        merged.reset_index(drop=True, inplace=True)
        merged = merged.fillna(0.0)
        missing = [column for column in SMA_PLUS_XSEC_OBSERVATION_COLUMNS if column not in merged.columns]
        for column in missing:
            merged[column] = 0.0
        merged = merged[["timestamp", *SMA_PLUS_XSEC_OBSERVATION_COLUMNS]]
        if len(merged) < 2:
            raise ValueError(f"Not enough overlapping rows to build sma_plus_xsec_v1 for {symbol}")
        results[symbol] = FeatureSetResult(
            frame=merged,
            observation_columns=SMA_PLUS_XSEC_OBSERVATION_COLUMNS,
            feature_set="sma_plus_xsec_v1",
            inputs_used=dict(base_result.inputs_used),
        )
    return results


def resolve_regime_metadata(regime_feature_set: str | None) -> Dict[str, object] | None:
    """Describe regime feature inputs for downstream metadata artifacts."""

    if not regime_feature_set:
        return None
    normalized = normalize_regime_feature_set_name(regime_feature_set)
    payload: Dict[str, object] = {"feature_set": normalized}
    if normalized == "regime_v1_1":
        payload["universe"] = list(PRIMARY_REGIME_UNIVERSE)
    return payload


__all__ = ["build_sma_feature_result", "build_universe_feature_results", "resolve_regime_metadata"]
