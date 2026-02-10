from __future__ import annotations

from research.execution.execution_simulator import (
    ExecutionSimConfig,
    ExecutionSimulator,
    compute_effective_band,
)


def _price(open_p: float, high: float, low: float, close: float) -> dict[str, float]:
    return {"open": open_p, "high": high, "low": low, "close": close}


def test_effective_band_shrinks_range() -> None:
    high, low = compute_effective_band(110.0, 90.0, range_shrink_pct=0.15, slippage_bps=0.0)
    assert round(high, 2) == 108.5
    assert round(low, 2) == 91.5


def test_limit_fill_uses_next_bar_only() -> None:
    config = ExecutionSimConfig(enabled=True, range_shrink_pct=0.0, slippage_bps=0.0)
    sim = ExecutionSimulator(["AAPL"], config)
    result = sim.resolve_step(
        prev_weights=[0.0],
        target_weights=[1.0],
        prev_value=10_000.0,
        price_panel={"AAPL": _price(100, 110, 97, 100)},
        next_price_panel={"AAPL": _price(101, 105, 99, 102)},
        execution_action={"order_type": "limit", "limit_offset": 0.02},
    )
    assert result.executed_weights == (0.0,)
    assert result.missed_fill_ratio > 0.0


def test_limit_fill_executes_when_next_bar_hits() -> None:
    config = ExecutionSimConfig(enabled=True, range_shrink_pct=0.0, slippage_bps=0.0)
    sim = ExecutionSimulator(["AAPL"], config)
    result = sim.resolve_step(
        prev_weights=[0.0],
        target_weights=[1.0],
        prev_value=10_000.0,
        price_panel={"AAPL": _price(100, 110, 97, 100)},
        next_price_panel={"AAPL": _price(101, 105, 97, 102)},
        execution_action={"order_type": "limit", "limit_offset": 0.02},
    )
    assert result.executed_weights == (1.0,)
    assert result.missed_fill_ratio == 0.0


def test_trailing_stop_triggers_on_next_low() -> None:
    config = ExecutionSimConfig(enabled=True, range_shrink_pct=0.0, slippage_bps=0.0)
    sim = ExecutionSimulator(["AAPL"], config)
    result = sim.resolve_step(
        prev_weights=[1.0],
        target_weights=[0.0],
        prev_value=10_000.0,
        price_panel={"AAPL": _price(100, 110, 100, 105)},
        next_price_panel={"AAPL": _price(103, 106, 98, 104)},
        execution_action={"order_type": "trailing_stop", "trailing_distance": 0.1},
    )
    assert result.executed_weights == (0.0,)


def test_determinism_for_fixed_inputs() -> None:
    config = ExecutionSimConfig(enabled=True, range_shrink_pct=0.0, slippage_bps=0.0)
    sim_a = ExecutionSimulator(["AAPL"], config)
    sim_b = ExecutionSimulator(["AAPL"], config)
    kwargs = dict(
        prev_weights=[0.2],
        target_weights=[0.6],
        prev_value=10_000.0,
        price_panel={"AAPL": _price(100, 105, 95, 102)},
        next_price_panel={"AAPL": _price(101, 106, 96, 103)},
        execution_action={"order_type": "market"},
    )
    result_a = sim_a.resolve_step(**kwargs)
    result_b = sim_b.resolve_step(**kwargs)
    assert result_a.executed_weights == result_b.executed_weights
    assert result_a.execution_slippage_bps == result_b.execution_slippage_bps
