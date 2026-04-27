"""Live market data source backed by the Alpaca REST API."""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, TYPE_CHECKING

import httpx

from research.datasets.canonical_equity_loader import build_union_calendar
from research.experiments.spec import ExperimentSpec
from research.features.feature_eng import build_sma_feature_result, build_universe_feature_results
from research.features.feature_registry import (
    FeatureSetResult,
    build_universe_feature_panel,
    default_regime_for_feature_set,
    is_universe_feature_set,
    normalize_feature_set_name,
)
from research.features.feature_eng import SMAStrategyConfig
from research.shadow.data_source import MarketDataSource, _normalize_timestamp, _resolve_sma_config

if TYPE_CHECKING:
    import pandas as pd

_ALPACA_DATA_BASE = "https://data.alpaca.markets"
_FEEDS = ("sip", "iex")
# Calendar days to look back when fetching bars for feature warmup.
# 60 trading days ≈ 90 calendar days; use 95 for a small buffer.
_LOOKBACK_CALENDAR_DAYS = 95


def _auth_headers(api_key: str, secret_key: str) -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }


def _fetch_bars(
    symbol: str,
    start: date,
    end: date,
    api_key: str,
    secret_key: str,
    *,
    client: httpx.Client,
) -> list[dict[str, Any]]:
    """Fetch daily OHLCV bars for *symbol* from Alpaca, SIP feed with IEX fallback."""

    params: dict[str, Any] = {
        "timeframe": "1Day",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "adjustment": "split",
        "limit": 1000,
    }
    headers = _auth_headers(api_key, secret_key)
    url = f"{_ALPACA_DATA_BASE}/v2/stocks/{symbol}/bars"
    bars: list[dict[str, Any]] = []
    for feed in _FEEDS:
        params["feed"] = feed
        page_token: str | None = None
        feed_bars: list[dict[str, Any]] = []
        try:
            while True:
                if page_token:
                    params["page_token"] = page_token
                elif "page_token" in params:
                    del params["page_token"]
                response = client.get(url, params=params, headers=headers, timeout=30.0)
                if response.status_code == 403 and feed == "sip":
                    # No SIP subscription — fall through to IEX.
                    break
                response.raise_for_status()
                payload = response.json()
                feed_bars.extend(payload.get("bars") or [])
                page_token = payload.get("next_page_token")
                if not page_token:
                    break
            else:
                # Inner while completed without break → success.
                bars = feed_bars
                break
        except httpx.HTTPStatusError:
            if feed == _FEEDS[-1]:
                raise
            continue

    return bars


def _bars_to_equity_df(symbol: str, bars: list[dict[str, Any]]) -> "pd.DataFrame":
    """Convert Alpaca bar dicts to the canonical equity DataFrame format."""

    import pandas as pd

    rows = []
    for bar in bars:
        rows.append(
            {
                "timestamp": pd.Timestamp(bar["t"]).tz_convert("UTC"),
                "open": float(bar["o"]),
                "high": float(bar["h"]),
                "low": float(bar["l"]),
                "close": float(bar["c"]),
                "volume": float(bar["v"]),
                "symbol": symbol,
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.sort_values("timestamp", inplace=True, kind="mergesort")
        frame.reset_index(drop=True, inplace=True)
    return frame


class CanonicalEquitySliceFacade:
    """Thin wrapper around a DataFrame that satisfies the slice interface used by feature builders."""

    def __init__(self, symbol: str, frame: "pd.DataFrame") -> None:
        self.symbol = symbol
        self.frame = frame
        self.file_paths: list[Path] = []


class AlpacaLiveDataSource(MarketDataSource):
    """Live data source that fetches recent OHLCV bars from the Alpaca REST API.

    Provides the same interface as ``ReplayMarketDataSource`` so it can be
    dropped in to ``ShadowEngine`` with ``live_mode=True``.

    The calendar exposed to the engine spans from *run_start_date* to today
    (inclusive of both).  Feature computation uses an additional 60-trading-day
    warm-up window fetched from Alpaca that is NOT included in the calendar,
    ensuring rolling indicators (SMA, vol) are fully initialised on day one.

    Args:
        spec: Experiment specification (symbols, feature_set, policy_params, …).
        run_start_date: The date of the first paper run invocation.  The
            calendar will span ``[run_start_date, today]``.  Pass
            ``date.today()`` on the very first invocation and persist it so
            future invocations can pass the same value.
        api_key: Alpaca API key (defaults to ``ALPACA_API_KEY`` env var).
        secret_key: Alpaca secret key (defaults to ``ALPACA_SECRET_KEY`` env var).
        lookback_trading_days: Number of trading days of warm-up bars to fetch
            before *run_start_date* for feature initialisation (default 60).
        data_root: Optional override for the Quanto data root (used by some
            regime feature builders that need canonical data).
    """

    def __init__(
        self,
        *,
        spec: ExperimentSpec,
        run_start_date: date,
        api_key: str | None = None,
        secret_key: str | None = None,
        lookback_trading_days: int = 60,
        data_root: Path | None = None,
    ) -> None:
        self._spec = spec
        self._run_start_date = run_start_date
        self._api_key = api_key or os.environ.get("ALPACA_API_KEY") or ""
        self._secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY") or ""
        if not self._api_key or not self._secret_key:
            raise RuntimeError(
                "AlpacaLiveDataSource requires ALPACA_API_KEY and ALPACA_SECRET_KEY."
            )
        self._lookback_trading_days = max(1, int(lookback_trading_days))
        self._data_root = Path(data_root) if data_root else None

        self._rows: List[dict[str, Any]] = []
        self._calendar: List["pd.Timestamp"] = []
        self._symbol_order: tuple[str, ...] = tuple()
        self._observation_columns: tuple[str, ...] = tuple()
        self._regime_feature_names: tuple[str, ...] = tuple()
        self._regime_series: List[tuple[float, ...]] = []
        self._index: Dict["pd.Timestamp", dict[str, Any]] = {}

        self._materialize()

    # ------------------------------------------------------------------
    # MarketDataSource interface
    # ------------------------------------------------------------------

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

    def calendar(self) -> list["pd.Timestamp"]:
        return list(self._calendar)

    def snapshot(self, as_of: "pd.Timestamp") -> dict[str, Any]:
        key = _normalize_timestamp(as_of)
        if key not in self._index:
            raise KeyError(f"No live snapshot available for {key.isoformat()}")
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

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _materialize(self) -> None:
        import pandas as pd

        today = date.today()
        # Warmup window: extra calendar days to cover ~lookback_trading_days trading days.
        # Trading days ≈ 5/7 of calendar days; over-fetch to be safe.
        warmup_calendar_days = int(self._lookback_trading_days * 7 / 5) + 10
        fetch_start = self._run_start_date - timedelta(days=warmup_calendar_days)
        fetch_end = today

        slices = self._fetch_all_bars(fetch_start, fetch_end)
        feature_results = self._build_feature_results(slices, fetch_start, fetch_end)

        # Build the full panel (includes warmup dates — we trim later).
        all_timestamps = [
            ts
            for sym_slice in slices.values()
            if not sym_slice.frame.empty
            for ts in pd.to_datetime(sym_slice.frame["timestamp"], utc=True)
        ]
        if not all_timestamps:
            raise RuntimeError("AlpacaLiveDataSource: Alpaca returned no bars for the requested window.")

        union_cal = sorted(set(_normalize_timestamp(ts) for ts in all_timestamps))
        normalized_feature_set = normalize_feature_set_name(self._spec.feature_set)
        regime_for_panel = self._spec.regime_feature_set or default_regime_for_feature_set(normalized_feature_set)

        panel = build_universe_feature_panel(
            feature_results,
            symbol_order=self._spec.symbols,
            calendar=union_cal,
            forward_fill_limit=3,
            regime_feature_set=regime_for_panel,
            data_root=self._data_root,
        )

        self._symbol_order = tuple(panel.symbol_order)
        self._observation_columns = tuple(panel.observation_columns)

        # Parse all rows, track regime metadata, then trim to run window.
        run_start_ts = _normalize_timestamp(pd.Timestamp(self._run_start_date))
        rows: List[dict[str, Any]] = []
        regime_series: List[tuple[float, ...]] = []
        regime_names: tuple[str, ...] = tuple()

        for row in panel.rows:
            timestamp = _normalize_timestamp(row["timestamp"])
            # Only include rows on or after run_start_date in the exposed calendar.
            if timestamp < run_start_ts:
                continue
            panel_payload = {
                symbol: {
                    col: float(features[col])
                    for col in self._observation_columns
                    if col in features
                }
                for symbol, features in row["panel"].items()
            }
            regime_state = row.get("regime_state")
            regime_values: tuple[float, ...] = tuple()
            if regime_state is not None:
                feature_names = tuple(getattr(regime_state, "feature_names", ()))
                values = tuple(float(v) for v in getattr(regime_state, "features", ()))
                regime_values = values
                if feature_names and not regime_names:
                    regime_names = feature_names
            rows.append({"timestamp": timestamp, "panel": panel_payload, "regime_features": regime_values})
            regime_series.append(regime_values)

        if not rows:
            raise RuntimeError(
                f"AlpacaLiveDataSource: no data on or after run_start_date={self._run_start_date}. "
                "Verify that Alpaca has bars for the requested symbols and date range."
            )

        self._rows = rows
        self._calendar = [entry["timestamp"] for entry in rows]
        self._regime_feature_names = regime_names
        self._regime_series = regime_series
        self._index = {entry["timestamp"]: entry for entry in rows}

    def _fetch_all_bars(
        self, fetch_start: date, fetch_end: date
    ) -> Dict[str, CanonicalEquitySliceFacade]:
        slices: Dict[str, CanonicalEquitySliceFacade] = {}
        with httpx.Client() as client:
            for symbol in self._spec.symbols:
                bars = _fetch_bars(
                    symbol,
                    fetch_start,
                    fetch_end,
                    self._api_key,
                    self._secret_key,
                    client=client,
                )
                frame = _bars_to_equity_df(symbol, bars)
                slices[symbol] = CanonicalEquitySliceFacade(symbol=symbol, frame=frame)
        return slices

    def _build_feature_results(
        self,
        slices: Mapping[str, CanonicalEquitySliceFacade],
        start: date,
        end: date,
    ) -> Dict[str, FeatureSetResult]:
        spec = self._spec
        normalized_feature_set = normalize_feature_set_name(spec.feature_set)
        windows = _resolve_sma_config(spec)

        if len(spec.symbols) > 1 and is_universe_feature_set(normalized_feature_set):
            return build_universe_feature_results(
                normalized_feature_set,
                slices,  # type: ignore[arg-type]
                symbol_order=spec.symbols,
                start_date=start,
                end_date=end,
                sma_config=windows,
                data_root=self._data_root,
            )

        results: Dict[str, FeatureSetResult] = {}
        for symbol in spec.symbols:
            sym_slice = slices.get(symbol)
            if sym_slice is None:
                raise ValueError(f"No bars fetched for symbol {symbol}")
            result = build_sma_feature_result(
                sym_slice,  # type: ignore[arg-type]
                fast_window=windows.fast_window,
                slow_window=windows.slow_window,
                feature_set=spec.feature_set,
                start_date=start,
                end_date=end,
                data_root=self._data_root,
            )
            results[symbol] = result
        return results


__all__ = ["AlpacaLiveDataSource"]
