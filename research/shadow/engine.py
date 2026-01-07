"""Shadow execution engine orchestrating forward-only paper portfolios."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from research.experiments.registry import ExperimentRecord, ExperimentRegistry
from research.hierarchy.allocator_registry import build_allocator
from research.hierarchy.controller import ControllerConfig, ModeController
from research.hierarchy.policy_wrapper import HierarchicalPolicy
from research.policies.sma_weight_policy import SMAWeightPolicy, SMAWeightPolicyConfig
from research.promotion.qualify import is_experiment_promoted
from research.risk import RiskConfig, project_weights
from research.shadow.data_source import MarketDataSource
from research.shadow.logging import ShadowLogger
from research.shadow.portfolio import (
    PortfolioUpdate,
    constraint_diagnostics,
    rebalance_portfolio,
    valuate_portfolio,
    weights_from_holdings,
)
from research.shadow.schema import ShadowState, initial_state
from research.shadow.state_store import StateStore


DEFAULT_INITIAL_CASH = 10_000.0


class ShadowEngine:
    """Forward-only execution harness with deterministic persistence."""

    def __init__(
        self,
        *,
        experiment_id: str,
        data_source: MarketDataSource,
        state_store: StateStore,
        logger: ShadowLogger,
        run_id: str,
        out_dir: Path,
        registry: ExperimentRegistry | None = None,
        promotion_root: Path | None = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.data_source = data_source
        self.state_store = state_store
        self.logger = logger
        self.run_id = run_id
        self.out_dir = Path(out_dir)
        self.registry = registry or ExperimentRegistry()
        self._promotion_root = promotion_root
        self._record: ExperimentRecord
        self._spec = None
        self._state: ShadowState | None = None
        self._policy = None
        self._sma_policy: SMAWeightPolicy | None = None
        self._hierarchical_policy: HierarchicalPolicy | None = None
        self._calendar = self.data_source.calendar()
        if not self._calendar:
            raise ValueError("Replay calendar is empty; nothing to execute.")
        self._symbol_order = self._resolve_symbol_order()
        self._observation_columns = self._resolve_observation_columns()
        self._regime_names = self._resolve_regime_names()
        self._base_feature_columns = self._split_feature_columns(self._observation_columns, self._regime_names)
        self._dates_array = np.asarray(self._calendar, dtype=object)
        self._regime_matrix = self._resolve_regime_matrix()
        self._load_spec()
        self._validate_promotion()
        self._initialize_policy()

    def step(self) -> dict[str, Any]:
        state = self._ensure_state()
        if state.current_step >= len(self._calendar):
            raise StopIteration("Replay range exhausted.")
        as_of = self._calendar[state.current_step]
        snapshot = self.data_source.snapshot(as_of)
        prices = self._extract_prices(snapshot)
        portfolio_value = valuate_portfolio(state.cash, state.holdings, prices)
        prev_weights = weights_from_holdings(state.holdings, prices, portfolio_value)
        state.portfolio_value = portfolio_value
        mode = state.last_mode
        raw_action, target_weights, mode = self._determine_policy_outputs(snapshot, prev_weights, mode, state.current_step)
        update = rebalance_portfolio(
            cash=state.cash,
            holdings=state.holdings,
            prices=prices,
            target_weights=target_weights,
            transaction_cost_bp=self._spec.cost_config.transaction_cost_bp,
        )
        diag = constraint_diagnostics(update.weights, prev_weights, self._spec.risk_config)
        state.cash = update.cash
        state.holdings = update.holdings
        state.last_weights = update.weights
        state.last_raw_action = raw_action
        state.last_turnover = update.turnover
        state.last_mode = mode
        state.portfolio_value = update.portfolio_value
        state.current_step += 1
        if state.current_step < len(self._calendar):
            state.current_date = self._calendar[state.current_step].date().isoformat()
        else:
            state.current_date = self._calendar[-1].date().isoformat()
        record = self._build_log_record(
            step_index=state.current_step - 1,
            snapshot=snapshot,
            raw_action=raw_action,
            update=update,
            mode=mode,
            diagnostics=diag,
        )
        self.logger.append(record)
        self.state_store.save(state)
        return record

    def run(self, *, max_steps: int | None = None) -> dict[str, Any]:
        steps = 0
        while max_steps is None or steps < max_steps:
            try:
                self.step()
            except StopIteration:
                break
            steps += 1
        state = self._ensure_state()
        completed = state.current_step >= len(self._calendar)
        summary = {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "steps_executed": steps,
            "completed": completed,
            "state_path": str(self.state_store.state_path),
            "log_path": str(self.logger.steps_path),
        }
        self.logger.write_summary(summary)
        return summary

    def _load_spec(self) -> None:
        record, spec = self.registry.resolve_with_spec(self.experiment_id)
        self._record = record
        self._spec = spec

    def _validate_promotion(self) -> None:
        if not is_experiment_promoted(self.experiment_id, promotion_root=self._promotion_root):
            raise RuntimeError(f"Experiment '{self.experiment_id}' is not promoted; shadow execution is disabled.")

    def _ensure_state(self) -> ShadowState:
        if self._state is not None:
            return self._state
        state = self.state_store.load()
        if state is None:
            first_date = self._calendar[0].date().isoformat()
            state = initial_state(
                experiment_id=self.experiment_id,
                symbols=self._symbol_order,
                start_date=first_date,
                run_id=self.run_id,
                initial_cash=DEFAULT_INITIAL_CASH,
            )
        self._validate_state(state)
        self._state = state
        return state

    def _validate_state(self, state: ShadowState) -> None:
        if state.experiment_id != self.experiment_id:
            raise RuntimeError("State experiment_id mismatch.")
        if state.run_id != self.run_id:
            raise RuntimeError("State run_id mismatch.")
        if tuple(state.symbols) != self._symbol_order:
            raise RuntimeError("State symbol ordering differs from replay source.")
        if len(state.holdings) != len(self._symbol_order):
            raise RuntimeError("State holdings dimension mismatch.")
        if state.current_step < 0 or state.current_step > len(self._calendar):
            raise RuntimeError("State current_step out of bounds.")

    def _initialize_policy(self) -> None:
        if self._spec.hierarchy_enabled:
            self._hierarchical_policy = self._build_hierarchical_policy()
            return
        policy = self._spec.policy
        if policy == "sma":
            params = self._spec.policy_params
            mode = str(params.get("policy_mode", "hard"))
            sigmoid = float(params.get("sigmoid_scale", 5.0))
            self._sma_policy = SMAWeightPolicy(SMAWeightPolicyConfig(mode=mode, sigmoid_scale=sigmoid))
        elif policy == "equal_weight":
            self._sma_policy = None
        elif policy == "ppo":
            raise RuntimeError("PPO policies are not supported in shadow execution v1.")

    def _build_hierarchical_policy(self) -> HierarchicalPolicy:
        spec = self._spec
        if not spec.controller_config or not spec.allocator_by_mode:
            raise ValueError("Hierarchical experiments require controller and allocator configs.")
        controller_cfg = ControllerConfig.from_mapping(spec.controller_config)
        controller = ModeController(config=controller_cfg)
        allocators = {
            mode: build_allocator(config, num_assets=len(self._symbol_order))
            for mode, config in spec.allocator_by_mode.items()
        }
        policy = HierarchicalPolicy(controller, allocators, spec.risk_config)
        policy.reset()
        return policy

    def _determine_policy_outputs(
        self,
        snapshot: Mapping[str, Any],
        prev_weights: Sequence[float],
        prev_mode: str | None,
        step_index: int,
    ) -> tuple[list[float], list[float], str | None]:
        if self._hierarchical_policy is not None:
            obs = self._build_observation(snapshot, prev_weights)
            context = {
                "panel": snapshot["panel"],
                "symbol_order": self._symbol_order,
                "num_assets": len(self._symbol_order),
                "timestamp": snapshot["as_of"],
            }
            self._hierarchical_policy.set_allocator_context(context)
            regime = self._regime_matrix if self._regime_matrix.size else None
            weights, mode, raw_action = self._hierarchical_policy.step(
                obs,
                step_index,
                self._dates_array,
                regime,
                prev_weights,
                prev_mode,
            )
            return (
                [float(value) for value in raw_action.tolist()],
                [float(value) for value in weights.tolist()],
                str(mode) if mode is not None else None,
            )
        if self._spec.policy == "equal_weight":
            per_asset = 1.0 / len(self._symbol_order)
            raw = [per_asset for _ in self._symbol_order]
        elif self._spec.policy == "sma":
            if self._sma_policy is None:
                raise RuntimeError("SMA policy not initialized.")
            raw = [float(self._sma_policy.decide(snapshot["panel"][symbol])) for symbol in self._symbol_order]
        else:
            raise RuntimeError(f"Unsupported policy '{self._spec.policy}' for shadow execution.")
        projected = project_weights(raw, prev_weights, self._spec.risk_config)
        return (
            [float(value) for value in raw],
            [float(value) for value in projected],
            None,
        )

    def _build_observation(self, snapshot: Mapping[str, Any], prev_weights: Sequence[float]) -> tuple[float, ...]:
        panel = snapshot["panel"]
        regime_values = tuple(float(value) for value in snapshot.get("regime_features") or ())
        values: List[float] = []
        for symbol in self._symbol_order:
            features = panel[symbol]
            for column in self._base_feature_columns:
                if column not in features:
                    raise ValueError(f"Snapshot missing required feature '{column}' for symbol '{symbol}'")
                values.append(float(features[column]))
            if regime_values:
                values.extend(regime_values)
        values.extend(float(value) for value in prev_weights)
        return tuple(values)

    def _build_log_record(
        self,
        *,
        step_index: int,
        snapshot: Mapping[str, Any],
        raw_action: Sequence[float],
        update: PortfolioUpdate,
        mode: str | None,
        diagnostics: Mapping[str, float],
    ) -> Dict[str, Any]:
        return {
            "step": int(step_index),
            "as_of": snapshot["as_of"].isoformat(),
            "symbols": list(self._symbol_order),
            "raw_action": [float(value) for value in raw_action],
            "weights": [float(value) for value in update.weights],
            "mode": mode,
            "turnover": float(update.turnover),
            "tx_cost": float(update.transaction_cost),
            "portfolio_value": float(update.portfolio_value),
            "cash": float(update.cash),
            "holdings": [float(value) for value in update.holdings],
            "constraint_diagnostics": {key: float(diagnostics[key]) for key in sorted(diagnostics)},
        }

    def _resolve_symbol_order(self) -> tuple[str, ...]:
        if hasattr(self.data_source, "symbol_order"):
            order = tuple(getattr(self.data_source, "symbol_order"))
            if order:
                return order
        snapshot = self.data_source.snapshot(self._calendar[0])
        return tuple(snapshot.get("symbols") or [])

    def _resolve_observation_columns(self) -> tuple[str, ...]:
        if hasattr(self.data_source, "observation_columns"):
            columns = tuple(getattr(self.data_source, "observation_columns"))
            if columns:
                return columns
        snapshot = self.data_source.snapshot(self._calendar[0])
        return tuple(snapshot.get("observation_columns") or ())

    def _resolve_regime_names(self) -> tuple[str, ...]:
        if hasattr(self.data_source, "regime_feature_names"):
            names = tuple(getattr(self.data_source, "regime_feature_names"))
            if names:
                return names
        snapshot = self.data_source.snapshot(self._calendar[0])
        names = snapshot.get("regime_feature_names") or ()
        return tuple(names)

    def _split_feature_columns(
        self,
        observation_columns: Sequence[str],
        regime_names: Sequence[str],
    ) -> tuple[str, ...]:
        if not regime_names:
            return tuple(observation_columns)
        if len(observation_columns) <= len(regime_names):
            raise ValueError("Observation columns do not leave room for per-symbol features.")
        base = tuple(observation_columns[: len(observation_columns) - len(regime_names)])
        suffix = tuple(observation_columns[len(base) :])
        if tuple(suffix) != tuple(regime_names):
            raise ValueError("Regime feature ordering mismatch between observation columns and replay data.")
        return base

    def _resolve_regime_matrix(self) -> np.ndarray:
        series = getattr(self.data_source, "regime_series", ())
        if not series:
            return np.asarray([], dtype=float)
        return np.asarray(series, dtype=float)

    def _extract_prices(self, snapshot: Mapping[str, Any]) -> List[float]:
        prices: List[float] = []
        panel = snapshot["panel"]
        for symbol in self._symbol_order:
            features = panel.get(symbol) or {}
            price = features.get("close")
            if price is None:
                raise ValueError(f"Snapshot missing close price for symbol '{symbol}'")
            prices.append(float(price))
        return prices


__all__ = ["DEFAULT_INITIAL_CASH", "ShadowEngine"]
