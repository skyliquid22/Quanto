"""Shadow execution engine orchestrating forward-only paper portfolios."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from research.execution import (
    ExecutionController,
    ExecutionControllerConfig,
    ExecutionMetricsRecorder,
    ExecutionRiskConfig,
    OrderCompilerConfig,
    SimBrokerAdapter,
    SimBrokerConfig,
)
from research.experiments.registry import ExperimentRecord, ExperimentRegistry
from research.hierarchy.allocator_registry import build_allocator
from research.hierarchy.controller import ControllerConfig, ModeController
from research.hierarchy.policy_wrapper import HierarchicalPolicy
from research.policies.sma_weight_policy import SMAWeightPolicy, SMAWeightPolicyConfig
from research.promotion.qualify import is_experiment_promoted
from research.paper.config import PollingConfig, ReconciliationConfig
from research.paper.reconcile import PaperReconciler
from research.paper.run import PaperExecutionController
from research.paper.summary import ExecutionGateRunner
from research.risk import RiskConfig, project_weights
from research.promotion import baseline_allowlist
from research.shadow.data_source import MarketDataSource
from research.shadow.logging import ShadowLogger
from research.shadow.ppo_policy import PpoShadowPolicy, resolve_ppo_checkpoint_path
from research.shadow.portfolio import (
    PortfolioUpdate,
    constraint_diagnostics,
    rebalance_portfolio,
    valuate_portfolio,
    weights_from_holdings,
)
from research.shadow.schema import ShadowState, initial_state
from research.shadow.state_store import StateStore
from research.execution.alpaca_broker import AlpacaBrokerAdapter, AlpacaBrokerConfig


DEFAULT_INITIAL_CASH = 10_000.0


def _stable_timestamp(seed: str) -> str:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    seconds = int.from_bytes(digest[:6], "big")
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    window = 5 * 365 * 24 * 60 * 60
    return (base + timedelta(seconds=seconds % window)).isoformat()


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
        execution_mode: str = "none",
        execution_options: Mapping[str, Any] | None = None,
        replay_mode: bool = False,
        live_mode: bool = False,
        baseline_allowlist_root: Path | None = None,
        qualification_allowlist_root: Path | None = None,
        qualification_replay_allowed: bool = False,
        qualification_allow_reason: str | None = None,
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
        self._ppo_policy: PpoShadowPolicy | None = None
        self._calendar = self.data_source.calendar()
        if not self._calendar:
            raise ValueError("Replay calendar is empty; nothing to execute.")
        self._symbol_order = self._resolve_symbol_order()
        self._observation_columns = self._resolve_observation_columns()
        self._regime_names = self._resolve_regime_names()
        self._base_feature_columns = self._split_feature_columns(self._observation_columns, self._regime_names)
        self._dates_array = np.asarray(self._calendar, dtype=object)
        self._regime_matrix = self._resolve_regime_matrix()
        self._regime_buckets = self._compute_regime_buckets()
        self._execution_mode = execution_mode or "none"
        self._execution_options = dict(execution_options or {})
        self._execution_controller: ExecutionController | None = None
        self._execution_metrics: ExecutionMetricsRecorder | None = ExecutionMetricsRecorder()
        self._execution_broker = None
        self._execution_metrics_path: Path | None = self.out_dir / "execution_metrics.json"
        self._replay_mode = bool(replay_mode)
        self._live_mode = bool(live_mode)
        self._baseline_allowlist_root = Path(baseline_allowlist_root) if baseline_allowlist_root else None
        self._qualification_allowlist_root = (
            Path(qualification_allowlist_root) if qualification_allowlist_root else None
        )
        self._qualification_replay_allowed = bool(qualification_replay_allowed)
        self._qualification_allow_reason = qualification_allow_reason
        self._allowlist_metadata: Dict[str, str] | None = None
        self._load_spec()
        self._validate_promotion()
        self._initialize_policy()
        self._initialize_execution()

    def step(self) -> dict[str, Any]:
        state = self._ensure_state()
        if state.current_step >= len(self._calendar):
            raise StopIteration("Replay range exhausted.")
        if self._execution_controller and state.halted:
            raise RuntimeError(f"Execution halted previously: {state.halt_reason or 'HALTED'}")
        as_of = self._calendar[state.current_step]
        step_identity = self._derive_step_identity(as_of)
        snapshot = self.data_source.snapshot(as_of)
        prices = self._extract_prices(snapshot)
        portfolio_value = valuate_portfolio(state.cash, state.holdings, prices)
        prev_weights = weights_from_holdings(state.holdings, prices, portfolio_value)
        state.portfolio_value = portfolio_value
        day_start_value = state.daily_start_value or portfolio_value
        state.peak_portfolio_value = max(state.peak_portfolio_value, portfolio_value, day_start_value)
        mode = state.last_mode
        raw_action, target_weights, mode = self._determine_policy_outputs(snapshot, prev_weights, mode, state.current_step)
        target_weight_map = {symbol: float(target_weights[idx]) for idx, symbol in enumerate(self._symbol_order)}
        prices_by_symbol = {symbol: float(prices[idx]) for idx, symbol in enumerate(self._symbol_order)}
        execution_details: dict[str, Any] = self._empty_execution_log()
        if self._execution_controller is None:
            update = rebalance_portfolio(
                cash=state.cash,
                holdings=state.holdings,
                prices=prices,
                target_weights=target_weights,
                transaction_cost_bp=self._spec.cost_config.transaction_cost_bp,
            )
            state.open_orders = []
            state.last_broker_sync = None
            state.halted = False
            state.halt_reason = None
        else:
            resume_snapshot = StateStore.resume_snapshot(state)
            result = self._execution_controller.process_step(
                as_of=as_of.isoformat(),
                step_index=state.current_step,
                cash=state.cash,
                holdings=state.holdings,
                prices=prices_by_symbol,
                target_weights=target_weight_map,
                prev_weights=prev_weights,
                portfolio_value=portfolio_value,
                day_start_value=day_start_value,
                peak_value=state.peak_portfolio_value,
                regime_bucket=self._regime_bucket_for_step(state.current_step),
                resume_snapshot=resume_snapshot,
            )
            update = result.update
            execution_details = self._summarize_execution(result, step_identity=step_identity)
            self._update_state_from_execution(state, result, as_of)
        diag = constraint_diagnostics(update.weights, prev_weights, self._spec.risk_config)
        if self._execution_controller is None and self._execution_metrics is not None:
            self._execution_metrics.record_turnover(update.turnover)
        state.cash = update.cash
        state.holdings = update.holdings
        state.last_weights = update.weights
        state.last_raw_action = raw_action
        state.last_turnover = update.turnover
        state.last_mode = mode
        state.portfolio_value = update.portfolio_value
        state.daily_start_value = update.portfolio_value
        state.peak_portfolio_value = max(state.peak_portfolio_value, update.portfolio_value)
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
            target_weights=target_weight_map,
            execution=execution_details,
            step_identity=step_identity,
            state=state,
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
            if self._execution_controller is not None:
                state = self._ensure_state()
                if state.halted:
                    break
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
        summary["unpromoted_execution_allowed"] = state.unpromoted_execution_allowed
        summary["unpromoted_execution_reason"] = state.unpromoted_execution_reason
        summary["baseline_allowlist_path"] = state.baseline_allowlist_path
        summary["qualification_allowlist_path"] = state.qualification_allowlist_path
        summary["unpromoted_allow_source"] = state.unpromoted_allow_source
        summary["unpromoted_allow_timestamp"] = state.unpromoted_allow_timestamp
        metrics_path = self._write_execution_metrics()
        if metrics_path is not None:
            summary["execution_metrics_path"] = str(metrics_path)
        self.logger.write_summary(summary)
        return summary

    def _load_spec(self) -> None:
        record, spec = self.registry.resolve_with_spec(self.experiment_id)
        self._record = record
        self._spec = spec

    def _validate_promotion(self) -> None:
        self._allowlist_metadata = None
        promoted = is_experiment_promoted(self.experiment_id, promotion_root=self._promotion_root)
        if promoted:
            return
        allow_entry = baseline_allowlist.load_entry(
            self.experiment_id,
            root=self._baseline_allowlist_root,
        )
        if (
            allow_entry is not None
            and self._replay_mode
            and not self._live_mode
        ):
            timestamp = str(
                allow_entry.payload.get("created_at")
                or _stable_timestamp(f"baseline_allowlist:{self.experiment_id}:{self.run_id}")
            )
            self._allowlist_metadata = {
                "path": str(allow_entry.path),
                "reason": str(allow_entry.payload.get("reason") or "baseline_allowlist"),
                "source": "baseline_allowlist",
                "timestamp": timestamp,
            }
            return
        if allow_entry is not None and not self._replay_mode:
            raise RuntimeError(
                "Baseline allowlist only permits replay-mode execution; live execution remains disabled."
            )
        qual_entry = self._load_qualification_allow_entry()
        if (
            qual_entry is not None
            and self._replay_mode
            and not self._live_mode
        ):
            self._allowlist_metadata = qual_entry
            self._allowlist_metadata.setdefault(
                "timestamp",
                _stable_timestamp(f"qualification_allowlist:{self.experiment_id}:{self.run_id}"),
            )
            return
        if (
            self._qualification_replay_allowed
            and self._replay_mode
            and not self._live_mode
        ):
            self._allowlist_metadata = {
                "path": None,
                "reason": self._qualification_allow_reason or "qualification_cli",
                "source": "qualification_cli",
                "timestamp": _stable_timestamp(f"qualification_cli:{self.experiment_id}:{self.run_id}"),
            }
            return
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
        if self._allowlist_metadata:
            state.unpromoted_execution_allowed = True
            state.unpromoted_execution_reason = self._allowlist_metadata.get("reason")
            state.unpromoted_allow_source = self._allowlist_metadata.get("source")
            state.unpromoted_allow_timestamp = self._allowlist_metadata.get("timestamp")
            source = self._allowlist_metadata.get("source")
            if source == "baseline_allowlist":
                state.baseline_allowlist_path = self._allowlist_metadata.get("path")
            elif source in {"qualification_allowlist", "qualification_cli"}:
                state.qualification_allowlist_path = self._allowlist_metadata.get("path")
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

    def _load_qualification_allow_entry(self) -> Dict[str, str] | None:
        if not self._qualification_allowlist_root:
            return None
        path = self._qualification_allowlist_root / f"{self.experiment_id}.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        reason = str(payload.get("reason") or "qualification_allowlist")
        metadata = {
            "path": str(path),
            "reason": reason,
            "source": "qualification_allowlist",
        }
        created_at = payload.get("created_at")
        if isinstance(created_at, str) and created_at:
            metadata["timestamp"] = created_at
        else:
            metadata["timestamp"] = _stable_timestamp(f"qualification_allowlist:{experiment_id}")
        return metadata

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
            checkpoint = resolve_ppo_checkpoint_path(self._record.root, self.experiment_id)
            self._ppo_policy = PpoShadowPolicy.from_checkpoint(
                checkpoint,
                num_assets=len(self._symbol_order),
            )
        else:
            raise RuntimeError(f"Unsupported policy '{policy}' for shadow execution.")

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
        elif self._spec.policy == "ppo":
            if self._ppo_policy is None:
                raise RuntimeError("PPO policy not initialized.")
            obs = self._build_observation(snapshot, prev_weights)
            raw = self._ppo_policy.act(obs)
        else:
            raise RuntimeError(f"Unsupported policy '{self._spec.policy}' for shadow execution.")
        projected = project_weights(raw, prev_weights, self._spec.risk_config)
        return (
            [float(value) for value in raw],
            [float(value) for value in projected],
            None,
        )

    def _initialize_execution(self) -> None:
        mode = (self._execution_mode or "none").lower()
        if mode == "none":
            return
        if mode not in {"sim", "alpaca_paper"}:
            raise ValueError(f"Unsupported execution_mode '{self._execution_mode}'")
        order_config = self._build_order_config()
        risk_config = self._build_execution_risk_config()
        controller_kwargs: dict[str, object] = {}
        controller_class = ExecutionController
        if mode == "sim":
            sim_payload = self._execution_options.get("sim_broker") or {}
            broker = SimBrokerAdapter(
                SimBrokerConfig(
                    slippage_bps=float(sim_payload.get("slippage_bps", 0.0)),
                    fee_bps=float(sim_payload.get("fee_bps", 0.0)),
                    fee_per_order=float(sim_payload.get("fee_per_order", 0.0)),
                )
            )
        else:
            alpaca_payload = self._execution_options.get("alpaca") or {}
            broker_config = None
            if alpaca_payload:
                broker_config = AlpacaBrokerConfig(
                    api_key=str(alpaca_payload["api_key"]),
                    secret_key=str(alpaca_payload["secret_key"]),
                    base_url=str(alpaca_payload.get("base_url", "https://paper-api.alpaca.markets")),
                )
            broker = AlpacaBrokerAdapter(config=broker_config)
            controller_class = PaperExecutionController
            recon_cfg = ReconciliationConfig.from_mapping(self._execution_options.get("reconciliation"))
            polling_cfg = PollingConfig.from_mapping(self._execution_options.get("polling"))
            controller_kwargs = {
                "reconciler": PaperReconciler(recon_cfg),
                "polling": polling_cfg,
                "gate_runner": ExecutionGateRunner(),
            }
        controller_config = ExecutionControllerConfig(
            run_id=self.run_id,
            symbol_order=self._symbol_order,
            order_config=order_config,
            risk_config=risk_config,
        )
        self._execution_controller = controller_class(
            broker=broker,
            metrics=self._execution_metrics,
            config=controller_config,
            **controller_kwargs,
        )
        self._execution_broker = broker
        self._execution_metrics_path = self.out_dir / "execution_metrics.json"

    def _build_order_config(self) -> OrderCompilerConfig:
        payload = self._execution_options.get("order_config") or {}
        min_notional = float(payload.get("min_notional", 1.0))
        return OrderCompilerConfig(min_notional=max(min_notional, 0.0))

    def _build_execution_risk_config(self) -> ExecutionRiskConfig:
        spec_cfg = self._spec.risk_config or RiskConfig()
        base = ExecutionRiskConfig(
            max_gross_exposure=spec_cfg.exposure_cap,
            min_cash_pct=spec_cfg.min_cash,
            max_symbol_weight=spec_cfg.max_weight,
            max_daily_turnover=spec_cfg.max_turnover_1d,
            max_active_positions=len(self._symbol_order),
        )
        overrides_payload = self._execution_options.get("risk_overrides") or {}
        if not overrides_payload:
            return base
        overrides = ExecutionRiskConfig.from_mapping(overrides_payload)
        return ExecutionRiskConfig(
            max_gross_exposure=_override(overrides.max_gross_exposure, base.max_gross_exposure),
            min_cash_pct=_override(overrides.min_cash_pct, base.min_cash_pct),
            max_symbol_weight=_override(overrides.max_symbol_weight, base.max_symbol_weight),
            max_daily_turnover=_override(overrides.max_daily_turnover, base.max_daily_turnover),
            max_active_positions=int(overrides.max_active_positions or base.max_active_positions),
            max_daily_loss=_override(overrides.max_daily_loss, base.max_daily_loss),
            max_trailing_drawdown=_override(overrides.max_trailing_drawdown, base.max_trailing_drawdown),
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
        target_weights: Mapping[str, float],
        execution: Mapping[str, Any],
        step_identity: str | None = None,
        state: ShadowState,
    ) -> Dict[str, Any]:
        return {
            "step": int(step_index),
            "as_of": snapshot["as_of"].isoformat(),
            "step_identity": step_identity,
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
            "target_weights": {symbol: float(target_weights.get(symbol, 0.0)) for symbol in self._symbol_order},
            "execution": dict(execution),
            "unpromoted_execution_allowed": state.unpromoted_execution_allowed,
            "unpromoted_execution_reason": state.unpromoted_execution_reason,
            "baseline_allowlist_path": state.baseline_allowlist_path,
        }

    def _derive_step_identity(self, as_of: Any) -> str:
        timestamp = as_of.isoformat() if hasattr(as_of, "isoformat") else str(as_of)
        payload = {
            "experiment_id": self.experiment_id,
            "timestamp": timestamp,
            "universe": list(self._symbol_order),
            "execution_mode": self._execution_mode,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return digest[:16]

    def _empty_execution_log(self) -> dict[str, Any]:
        return {
            "mode": self._execution_mode,
            "orders_compiled": [],
            "orders_submitted": [],
            "orders_rejected": [],
            "open_orders": [],
            "fills": [],
            "risk_snapshot": {},
            "broker_errors": [],
            "halted": False,
            "halt_reason": None,
            "account_snapshot": None,
            "position_snapshots": [],
            "gate_report": {},
            "step_identity": None,
        }

    def _summarize_execution(self, result: "ExecutionStepResult", *, step_identity: str | None = None) -> dict[str, Any]:
        account_snapshot = _account_snapshot_to_dict(result.account_snapshot)
        position_snapshots = [_position_snapshot_to_dict(entry) for entry in (result.position_snapshots or [])]
        return {
            "mode": self._execution_mode,
            "orders_compiled": [order.to_dict() for order in result.compiled_orders],
            "orders_submitted": [order.to_dict() for order in result.orders_submitted],
            "orders_rejected": [order.to_dict() for order in result.orders_rejected],
            "open_orders": [order.to_dict() for order in result.open_orders],
            "fills": [fill.to_dict() for fill in result.fills],
            "risk_snapshot": dict(result.risk_snapshot),
            "broker_errors": list(result.broker_errors),
            "halted": result.halted,
            "halt_reason": result.halt_reason,
            "account_snapshot": account_snapshot,
            "position_snapshots": position_snapshots,
            "gate_report": dict(result.gate_report or {}),
            "step_identity": step_identity,
        }

    def _update_state_from_execution(self, state: ShadowState, result: "ExecutionStepResult", as_of: Any) -> None:
        state.open_orders = [order.to_dict() for order in result.open_orders]
        if result.account_snapshot and getattr(result.account_snapshot, "as_of", None):
            state.last_broker_sync = result.account_snapshot.as_of
        else:
            state.last_broker_sync = as_of.isoformat()
        state.halted = bool(result.halted)
        state.halt_reason = result.halt_reason
        existing = set(state.submitted_order_ids)
        for order in result.orders_submitted:
            existing.add(order.client_order_id)
        state.submitted_order_ids = sorted(existing)
        if result.client_broker_map:
            merged = dict(state.broker_order_map)
            merged.update(result.client_broker_map)
            state.broker_order_map = merged
        state.last_completed_step_ts = as_of.isoformat()

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

    def _compute_regime_buckets(self) -> tuple[str, ...] | None:
        if not self._regime_names or not self._regime_matrix.size:
            return None
        try:
            signal_index = self._regime_names.index("market_vol_20d")
        except ValueError:
            return None
        values = []
        for idx in range(min(len(self._calendar), len(self._regime_matrix))):
            row = self._regime_matrix[idx]
            value = float(row[signal_index]) if signal_index < len(row) else 0.0
            values.append(value)
        if not values:
            return None
        q33 = _quantile(values, 0.33)
        q66 = _quantile(values, 0.66)
        return tuple(_bucket_label(value, q33, q66) for value in values)

    def _regime_bucket_for_step(self, step_index: int) -> str | None:
        buckets = self._regime_buckets
        if not buckets or step_index >= len(buckets):
            return None
        return buckets[step_index]

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

    def _write_execution_metrics(self) -> Path | None:
        if self._execution_metrics is None or self._execution_metrics_path is None:
            return None
        path = self._execution_metrics.write(self._execution_metrics_path)
        metrics_json = self.out_dir / "metrics.json"
        if metrics_json.exists():
            payload = json.loads(metrics_json.read_text(encoding="utf-8"))
            payload["execution"] = self._execution_metrics.snapshot()
            metrics_json.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        return path


def _override(value: float | int | None, fallback: float | int | None) -> float | int | None:
    return value if value is not None else fallback


def _account_snapshot_to_dict(snapshot: Any) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    return {
        "as_of": getattr(snapshot, "as_of", None),
        "equity": float(getattr(snapshot, "equity", 0.0)),
        "cash": float(getattr(snapshot, "cash", 0.0)),
        "buying_power": float(getattr(snapshot, "buying_power", 0.0)) if getattr(snapshot, "buying_power", None) is not None else None,
        "currency": getattr(snapshot, "currency", "USD"),
    }


def _position_snapshot_to_dict(snapshot: Any) -> dict[str, Any]:
    return {
        "symbol": getattr(snapshot, "symbol", None),
        "qty": float(getattr(snapshot, "qty", 0.0)),
        "avg_price": float(getattr(snapshot, "avg_price", 0.0)),
        "market_price": float(getattr(snapshot, "market_price", 0.0)),
        "market_value": float(getattr(snapshot, "market_value", 0.0)),
    }


def _bucket_label(value: float, q33: float, q66: float) -> str:
    if value <= q33:
        return "low_vol"
    if value <= q66:
        return "mid_vol"
    return "high_vol"


def _quantile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = percentile * (len(ordered) - 1)
    low_index = int(np.floor(rank))
    high_index = int(np.ceil(rank))
    low_value = ordered[low_index]
    high_value = ordered[high_index]
    if low_index == high_index:
        return low_value
    weight = rank - low_index
    return low_value + weight * (high_value - low_value)


__all__ = ["DEFAULT_INITIAL_CASH", "ShadowEngine"]
