"""Market data sources powering the shadow execution engine."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd

from infra.paths import get_data_root
from research.datasets.canonical_equity_loader import build_union_calendar, load_canonical_equity
from research.experiments.spec import ExperimentSpec
from research.features.feature_eng import build_sma_feature_result, build_universe_feature_results
from research.features.feature_registry import (
    FeatureSetResult,
    build_universe_feature_panel,
    default_regime_for_feature_set,
    is_universe_feature_set,
    normalize_feature_set_name,
)
from research.strategies.sma_crossover import SMAStrategyConfig


class MarketDataSource:
    """Abstract data stream used by the shadow engine."""

    def calendar(self) -> list[pd.Timestamp]:  # pragma: no cover - interface
        raise NotImplementedError

    def snapshot(self, as_of: pd.Timestamp) -> dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


class ReplayMarketDataSource(MarketDataSource):
    """Deterministic data source backed by canonical replay windows."""

    def __init__(
        self,
        *,
        spec: ExperimentSpec,
        start_date: date,
        end_date: date,
        data_root: Path | None = None,
    ) -> None:
        if end_date < start_date:
            raise ValueError("end_date must be on or after start_date for replay data sources.")
        self._spec = spec
        self._start_date = start_date
        self._end_date = end_date
        self._data_root = Path(data_root) if data_root else get_data_root()
        self._rows: List[dict[str, Any]] = []
        self._calendar: List[pd.Timestamp] = []
        self._symbol_order: tuple[str, ...] = tuple()
        self._observation_columns: tuple[str, ...] = tuple()
        self._regime_feature_names: tuple[str, ...] = tuple()
        self._regime_series: List[tuple[float, ...]] = []
        self._materialize_rows()

    @property
    def symbol_order(self) -> tuple[str, ...]:
        return self._symbol_order

    @property
    def observation_columns(self) -> tuple[str, ...]:
        return self._observation_columns

    @property
    def regime_feature_names(self) -> tuple[str, ...]:
        return self._regime_feature_names

    @property
    def regime_series(self) -> Sequence[tuple[float, ...]]:
        return tuple(self._regime_series)

    @property
    def window(self) -> tuple[str, str]:
        return self._start_date.isoformat(), self._end_date.isoformat()

    def calendar(self) -> list[pd.Timestamp]:
        return list(self._calendar)

    def snapshot(self, as_of: pd.Timestamp) -> dict[str, Any]:
        key = _normalize_timestamp(as_of)
        if key not in self._index:
            raise KeyError(f"No snapshot available for {key.isoformat()}")
        entry = self._index[key]
        panel = {symbol: dict(features) for symbol, features in entry["panel"].items()}
        return {
            "as_of": key,
            "symbols": self._symbol_order,
            "panel": panel,
            "regime_features": entry["regime_features"],
            "regime_feature_names": self._regime_feature_names,
            "observation_columns": self._observation_columns,
        }

    def _materialize_rows(self) -> None:
        slices, _ = load_canonical_equity(
            self._spec.symbols,
            self._start_date,
            self._end_date,
            data_root=self._data_root,
            interval=self._spec.interval,
        )
        feature_results = self._build_feature_results(slices)
        calendar = build_union_calendar(slices, start_date=self._start_date, end_date=self._end_date)
        normalized_feature_set = normalize_feature_set_name(self._spec.feature_set)
        regime_for_panel = self._spec.regime_feature_set or default_regime_for_feature_set(normalized_feature_set)
        panel = build_universe_feature_panel(
            feature_results,
            symbol_order=self._spec.symbols,
            calendar=calendar,
            forward_fill_limit=3,
            regime_feature_set=regime_for_panel,
            data_root=self._data_root,
        )
        self._symbol_order = tuple(panel.symbol_order)
        self._observation_columns = tuple(panel.observation_columns)
        rows: List[dict[str, Any]] = []
        regime_series: List[tuple[float, ...]] = []
        regime_names: tuple[str, ...] = tuple()
        for row in panel.rows:
            timestamp = _normalize_timestamp(row["timestamp"])
            panel_payload = {
                symbol: {column: float(features[column]) for column in self._observation_columns if column in features}
                for symbol, features in row["panel"].items()
            }
            regime_state = row.get("regime_state")
            regime_values: tuple[float, ...] = tuple()
            if regime_state is not None:
                feature_names = tuple(getattr(regime_state, "feature_names", ()))
                values = tuple(float(value) for value in getattr(regime_state, "features", ()))
                regime_values = values
                if feature_names and not regime_names:
                    regime_names = feature_names
            rows.append(
                {
                    "timestamp": timestamp,
                    "panel": panel_payload,
                    "regime_features": regime_values,
                }
            )
            regime_series.append(regime_values)
        if not rows:
            raise RuntimeError("Replay window did not produce any aligned snapshots")
        self._rows = rows
        self._calendar = [entry["timestamp"] for entry in rows]
        self._regime_feature_names = regime_names
        self._regime_series = regime_series
        self._index: Dict[pd.Timestamp, dict[str, Any]] = {entry["timestamp"]: entry for entry in rows}

    def _build_feature_results(
        self,
        slices: Mapping[str, Any],
    ) -> Mapping[str, FeatureSetResult]:
        spec = self._spec
        normalized_feature_set = normalize_feature_set_name(spec.feature_set)
        windows = _resolve_sma_config(spec)
        start = self._start_date
        end = self._end_date
        if len(spec.symbols) > 1 and is_universe_feature_set(normalized_feature_set):
            return build_universe_feature_results(
                normalized_feature_set,
                slices,
                symbol_order=spec.symbols,
                start_date=start,
                end_date=end,
                sma_config=windows,
                data_root=self._data_root,
            )
        results: Dict[str, FeatureSetResult] = {}
        for symbol in spec.symbols:
            slice_data = slices.get(symbol)
            if slice_data is None:
                raise ValueError(f"Canonical slice missing for symbol {symbol}")
            result = build_sma_feature_result(
                slice_data,
                fast_window=windows.fast_window,
                slow_window=windows.slow_window,
                feature_set=spec.feature_set,
                start_date=start,
                end_date=end,
                data_root=self._data_root,
            )
            results[symbol] = result
        return results


def _resolve_sma_config(spec: ExperimentSpec) -> SMAStrategyConfig:
    params = dict(spec.policy_params)
    fast = int(params.get("fast_window", 20))
    slow = int(params.get("slow_window", 50))
    if fast <= 0 or slow <= 0:
        raise ValueError("SMA windows must be positive when building replay data sources")
    if fast >= slow:
        raise ValueError("fast_window must be strictly less than slow_window")
    return SMAStrategyConfig(fast_window=fast, slow_window=slow)


def _normalize_timestamp(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


__all__ = ["MarketDataSource", "ReplayMarketDataSource"]
