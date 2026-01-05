"""Dynamic routing logic for ingestion requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from .adapters import (
    EquityIngestionRequest,
    FundamentalsIngestionRequest,
    OptionReferenceIngestionRequest,
    OptionTimeseriesIngestionRequest,
)

Mode = Literal["rest", "flat_file"]


@dataclass
class RouterConfig:
    rest_max_days: int = 31
    rest_max_symbols: int = 50
    rest_max_total_bars: int = 10_000
    flat_file_min_bars: int = 20_000
    force_mode: Mode | None = None

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> "RouterConfig":
        if not config:
            return cls()
        return cls(
            rest_max_days=int(config.get("rest_max_days", cls.rest_max_days)),
            rest_max_symbols=int(config.get("rest_max_symbols", cls.rest_max_symbols)),
            rest_max_total_bars=int(config.get("rest_max_total_bars", cls.rest_max_total_bars)),
            flat_file_min_bars=int(config.get("flat_file_min_bars", cls.flat_file_min_bars)),
            force_mode=config.get("force_mode"),
        )


class IngestionRouter:
    """Routes ingestion requests between REST and flat-file modes."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = RouterConfig.from_mapping(config)

    def route_equity_ohlcv(self, request: EquityIngestionRequest) -> Mode:
        return self._route_timeseries(
            total_days=request.total_days,
            total_symbols=request.total_symbols,
            has_flat_files=bool(request.flat_file_uris),
        )

    def route_option_contract_reference(self, request: OptionReferenceIngestionRequest) -> Mode:
        cfg = self.config
        if cfg.force_mode:
            return cfg.force_mode

        total_underlyings = request.total_underlyings
        if request.flat_file_uris and total_underlyings >= cfg.rest_max_symbols:
            return "flat_file"
        if total_underlyings > cfg.rest_max_symbols:
            return "flat_file"
        return "rest"

    def route_option_contract_ohlcv(self, request: OptionTimeseriesIngestionRequest) -> Mode:
        return self._route_timeseries(
            total_days=request.total_days,
            total_symbols=request.total_symbols,
            has_flat_files=bool(request.flat_file_uris),
        )

    def route_option_open_interest(self, request: OptionTimeseriesIngestionRequest) -> Mode:
        return self._route_timeseries(
            total_days=request.total_days,
            total_symbols=request.total_symbols,
            has_flat_files=bool(request.flat_file_uris),
        )

    def route_fundamentals(self, request: FundamentalsIngestionRequest) -> Mode:
        cfg = self.config
        if cfg.force_mode:
            return cfg.force_mode
        if request.flat_file_uris:
            return "flat_file"
        if request.total_symbols > cfg.rest_max_symbols:
            return "flat_file"
        return "rest"

    def _route_timeseries(self, *, total_days: int, total_symbols: int, has_flat_files: bool) -> Mode:
        cfg = self.config
        if cfg.force_mode:
            return cfg.force_mode

        est_bars = total_days * total_symbols
        if has_flat_files and est_bars >= cfg.flat_file_min_bars:
            return "flat_file"
        if total_days > cfg.rest_max_days:
            return "flat_file"
        if total_symbols > cfg.rest_max_symbols:
            return "flat_file"
        if est_bars > cfg.rest_max_total_bars:
            return "flat_file"
        return "rest"


__all__ = ["IngestionRouter", "RouterConfig"]
