"""Dynamic routing logic for ingestion requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Mapping, Tuple, Type

from .adapters import (
    EquityIngestionRequest,
    IvolatilityEquityAdapter,
    IvolatilityOptionsAdapter,
    IvolatilityOptionsSurfaceAdapter,
    FundamentalsIngestionRequest,
    OptionReferenceIngestionRequest,
    OptionTimeseriesIngestionRequest,
    PolygonEquityAdapter,
    PolygonFundamentalsAdapter,
    PolygonOptionsAdapter,
)

Mode = Literal["rest", "flat_file"]
Domain = Literal[
    "equity_ohlcv",
    "option_contract_reference",
    "option_contract_ohlcv",
    "option_open_interest",
    "fundamentals",
    "options_surface_v1",
]


@dataclass(frozen=True)
class AdapterRoute:
    """Resolved adapter target for a specific domain/vendor/mode tuple."""

    domain: Domain
    vendor: str
    mode: Mode
    adapter: Type[Any]

    @property
    def adapter_name(self) -> str:
        return self.adapter.__name__


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
        self._adapter_registry: Dict[Tuple[str, str, Mode], AdapterRoute] = self._build_default_registry()

    def register_adapter(self, domain: Domain, vendor: str, mode: Mode, adapter: Type[Any]) -> None:
        key = (domain, vendor.lower(), mode)
        self._adapter_registry[key] = AdapterRoute(domain=domain, vendor=vendor.lower(), mode=mode, adapter=adapter)

    def resolve_vendor_adapter(self, domain: str, vendor: str, mode: Mode) -> AdapterRoute:
        normalized_vendor = vendor.lower()
        key = (domain, normalized_vendor, mode)
        try:
            return self._adapter_registry[key]
        except KeyError as exc:
            raise ValueError(
                f"No adapter registered for domain={domain}, vendor={vendor}, mode={mode}. "
                "Confirm that the run config specifies a supported combination."
            ) from exc

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

    def route_options_surface_v1(self) -> Mode:
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

    def _build_default_registry(self) -> Dict[Tuple[str, str, Mode], AdapterRoute]:
        registry: Dict[Tuple[str, str, Mode], AdapterRoute] = {}
        for mode in ("rest", "flat_file"):
            registry[("equity_ohlcv", "polygon", mode)] = AdapterRoute(
                domain="equity_ohlcv",
                vendor="polygon",
                mode=mode,  # type: ignore[arg-type]
                adapter=PolygonEquityAdapter,
            )
        registry[("equity_ohlcv", "ivolatility", "rest")] = AdapterRoute(
            domain="equity_ohlcv",
            vendor="ivolatility",
            mode="rest",
            adapter=IvolatilityEquityAdapter,
        )
        for mode in ("rest", "flat_file"):
            registry[("fundamentals", "polygon", mode)] = AdapterRoute(
                domain="fundamentals",
                vendor="polygon",
                mode=mode,  # type: ignore[arg-type]
                adapter=PolygonFundamentalsAdapter,
            )
        registry[("option_contract_reference", "polygon", "rest")] = AdapterRoute(
            domain="option_contract_reference",
            vendor="polygon",
            mode="rest",
            adapter=PolygonOptionsAdapter,
        )
        registry[("option_contract_reference", "polygon", "flat_file")] = AdapterRoute(
            domain="option_contract_reference",
            vendor="polygon",
            mode="flat_file",
            adapter=PolygonOptionsAdapter,
        )
        registry[("option_contract_reference", "ivolatility", "rest")] = AdapterRoute(
            domain="option_contract_reference",
            vendor="ivolatility",
            mode="rest",
            adapter=IvolatilityOptionsAdapter,
        )
        registry[("option_contract_ohlcv", "polygon", "rest")] = AdapterRoute(
            domain="option_contract_ohlcv",
            vendor="polygon",
            mode="rest",
            adapter=PolygonOptionsAdapter,
        )
        registry[("option_contract_ohlcv", "polygon", "flat_file")] = AdapterRoute(
            domain="option_contract_ohlcv",
            vendor="polygon",
            mode="flat_file",
            adapter=PolygonOptionsAdapter,
        )
        registry[("option_contract_ohlcv", "ivolatility", "rest")] = AdapterRoute(
            domain="option_contract_ohlcv",
            vendor="ivolatility",
            mode="rest",
            adapter=IvolatilityOptionsAdapter,
        )
        registry[("option_open_interest", "polygon", "rest")] = AdapterRoute(
            domain="option_open_interest",
            vendor="polygon",
            mode="rest",
            adapter=PolygonOptionsAdapter,
        )
        registry[("option_open_interest", "polygon", "flat_file")] = AdapterRoute(
            domain="option_open_interest",
            vendor="polygon",
            mode="flat_file",
            adapter=PolygonOptionsAdapter,
        )
        registry[("option_open_interest", "ivolatility", "rest")] = AdapterRoute(
            domain="option_open_interest",
            vendor="ivolatility",
            mode="rest",
            adapter=IvolatilityOptionsAdapter,
        )
        registry[("options_surface_v1", "ivolatility", "rest")] = AdapterRoute(
            domain="options_surface_v1",
            vendor="ivolatility",
            mode="rest",
            adapter=IvolatilityOptionsSurfaceAdapter,
        )
        return registry


__all__ = ["AdapterRoute", "IngestionRouter", "RouterConfig"]
