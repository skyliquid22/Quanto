"""Shadow execution state schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass
class ShadowState:
    """JSON-serializable snapshot of the paper portfolio."""

    experiment_id: str
    symbols: list[str]
    current_step: int
    current_date: str
    cash: float
    holdings: list[float]
    last_weights: list[float]
    last_mode: str | None
    portfolio_value: float
    run_id: str
    last_raw_action: list[float] | None = None
    last_turnover: float | None = None
    open_orders: list[dict[str, object]] = field(default_factory=list)
    last_broker_sync: str | None = None
    halted: bool = False
    halt_reason: str | None = None
    daily_start_value: float = 0.0
    peak_portfolio_value: float = 0.0
    submitted_order_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "symbols": list(self.symbols),
            "current_step": int(self.current_step),
            "current_date": self.current_date,
            "cash": float(self.cash),
            "holdings": [float(value) for value in self.holdings],
            "last_weights": [float(value) for value in self.last_weights],
            "last_mode": self.last_mode,
            "portfolio_value": float(self.portfolio_value),
            "run_id": self.run_id,
            "last_raw_action": None if self.last_raw_action is None else [float(v) for v in self.last_raw_action],
            "last_turnover": None if self.last_turnover is None else float(self.last_turnover),
            "open_orders": [dict(entry) for entry in self.open_orders],
            "last_broker_sync": self.last_broker_sync,
            "halted": bool(self.halted),
            "halt_reason": self.halt_reason,
            "daily_start_value": float(self.daily_start_value),
            "peak_portfolio_value": float(self.peak_portfolio_value),
            "submitted_order_ids": list(self.submitted_order_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ShadowState":
        return cls(
            experiment_id=str(payload["experiment_id"]),
            symbols=[str(symbol) for symbol in payload.get("symbols", [])],
            current_step=int(payload.get("current_step", 0)),
            current_date=str(payload.get("current_date", "")),
            cash=float(payload.get("cash", 0.0)),
            holdings=[float(value) for value in payload.get("holdings", [])],
            last_weights=[float(value) for value in payload.get("last_weights", [])],
            last_mode=payload.get("last_mode"),
            portfolio_value=float(payload.get("portfolio_value", 0.0)),
            run_id=str(payload.get("run_id") or ""),
            last_raw_action=[float(value) for value in payload.get("last_raw_action", [])]
            if payload.get("last_raw_action") is not None
            else None,
            last_turnover=float(payload["last_turnover"]) if payload.get("last_turnover") is not None else None,
            open_orders=[dict(entry) for entry in payload.get("open_orders", [])],
            last_broker_sync=payload.get("last_broker_sync"),
            halted=bool(payload.get("halted", False)),
            halt_reason=payload.get("halt_reason"),
            daily_start_value=float(payload.get("daily_start_value", 0.0)),
            peak_portfolio_value=float(payload.get("peak_portfolio_value", payload.get("portfolio_value", 0.0))),
            submitted_order_ids=[str(entry) for entry in payload.get("submitted_order_ids", [])],
        )


def initial_state(
    *,
    experiment_id: str,
    symbols: Sequence[str],
    start_date: str,
    run_id: str,
    initial_cash: float,
) -> ShadowState:
    ordered = [str(symbol) for symbol in symbols]
    width = len(ordered)
    zeros = [0.0 for _ in range(width)]
    return ShadowState(
        experiment_id=experiment_id,
        symbols=ordered,
        current_step=0,
        current_date=start_date,
        cash=float(initial_cash),
        holdings=zeros.copy(),
        last_weights=zeros.copy(),
        last_mode=None,
        portfolio_value=float(initial_cash),
        run_id=run_id,
        last_raw_action=zeros.copy(),
        last_turnover=0.0,
        open_orders=[],
        last_broker_sync=None,
        halted=False,
        halt_reason=None,
        daily_start_value=float(initial_cash),
        peak_portfolio_value=float(initial_cash),
        submitted_order_ids=[],
    )


__all__ = ["ShadowState", "initial_state"]
