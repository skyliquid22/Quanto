"""Continuous target-weight trading environment for deterministic rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from research.regime import RegimeState
from research.risk import RiskConfig, project_weights


@dataclass(frozen=True)
class SignalWeightEnvConfig:
    """Configuration controlling portfolio mechanics."""

    initial_cash: float = 10_000.0
    transaction_cost_bp: float = 1.0
    action_clip: Tuple[float, float] = (0.0, 1.0)
    risk_config: RiskConfig = RiskConfig()

    def __post_init__(self) -> None:
        low, high = self.action_clip
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if self.transaction_cost_bp < 0:
            raise ValueError("transaction_cost_bp must be non-negative")
        if high < low:
            raise ValueError("action_clip upper bound cannot be less than lower bound")
        if not isinstance(self.risk_config, RiskConfig):
            raise TypeError("risk_config must be a RiskConfig instance")


class SignalWeightTradingEnv:
    """Minimal RL-style environment using canonical features as observations."""

    _default_feature_columns = ("close", "sma_fast", "sma_slow", "sma_diff")

    def __init__(
        self,
        rows: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
        config: SignalWeightEnvConfig | None = None,
        observation_columns: Sequence[str] | None = None,
    ) -> None:
        self.config = config or SignalWeightEnvConfig()
        feature_columns = tuple(str(col).strip() for col in (observation_columns or self._default_feature_columns))
        if not feature_columns:
            raise ValueError("observation_columns must include at least one feature")
        if "close" not in feature_columns:
            raise ValueError("observation_columns must include 'close' so the environment can value positions")
        materialized_rows = _materialize_rows(rows)
        entries = [self._coerce_row(row, feature_columns) for row in materialized_rows]
        entries.sort(key=lambda item: item["timestamp"])
        if len(entries) < 2:
            raise ValueError("at least two rows are required to run the environment")
        self._rows: List[MutableMapping[str, object]] = entries
        regime_names = self._infer_regime_feature_names(entries)
        base_columns = self._split_feature_columns(feature_columns, regime_names)
        self._feature_columns = base_columns
        self._regime_feature_columns = regime_names
        self._symbol_order = self._determine_symbol_order()
        if not self._symbol_order:
            raise ValueError("at least one symbol is required to run the environment")
        self._num_assets = len(self._symbol_order)
        self._portfolio_value: float = self.config.initial_cash
        self._current_weights: List[float] = [0.0 for _ in range(self._num_assets)]
        self._step_index: int = 0
        self._done: bool = False
        self._observation_columns = self._build_observation_headers()
        self._mode_timeline: List[str | None] = []

    @property
    def observation_columns(self) -> Tuple[str, ...]:
        return self._observation_columns

    @property
    def current_row(self) -> Mapping[str, object]:
        return self._rows[self._step_index]

    @property
    def portfolio_value(self) -> float:
        return self._portfolio_value

    @property
    def current_weight(self) -> float:
        return self._current_weights[0] if self._current_weights else 0.0

    @property
    def current_weights(self) -> Tuple[float, ...]:
        return tuple(self._current_weights)

    @property
    def symbols(self) -> Tuple[str, ...]:
        return tuple(self._symbol_order)

    @property
    def num_assets(self) -> int:
        return self._num_assets

    def reset(self) -> Tuple[float, ...]:
        self._portfolio_value = self.config.initial_cash
        self._current_weights = [0.0 for _ in range(self._num_assets)]
        self._step_index = 0
        self._done = False
        self._mode_timeline = []
        return self._build_observation()

    def step(
        self,
        action: float | Sequence[float],
        *,
        mode: str | None = None,
    ) -> Tuple[Tuple[float, ...], float, bool, Mapping[str, object]]:
        if self._done:
            raise RuntimeError("cannot step once the environment is done; call reset() first")

        row = self._rows[self._step_index]
        next_row = self._rows[self._step_index + 1]
        prev_value = self._portfolio_value

        action_vector = self._prepare_action(action)
        target_weights = self._project_action(action_vector)
        prev_weights = list(self._current_weights)
        trade_notional = [(target_weights[idx] - prev_weights[idx]) * prev_value for idx in range(self._num_assets)]
        cost_rate = self.config.transaction_cost_bp / 10_000.0
        cost_paid = sum(abs(value) * cost_rate for value in trade_notional)
        post_cost_value = prev_value - cost_paid
        pct_returns = self._compute_returns(row, next_row)
        weighted_return = sum(target_weights[idx] * pct_returns[idx] for idx in range(self._num_assets))
        next_value = post_cost_value * (1.0 + weighted_return)
        if next_value <= 0:
            raise RuntimeError("portfolio value became non-positive; check inputs")

        reward = math.log(next_value / prev_value) if prev_value > 0 else 0.0
        self._portfolio_value = next_value
        self._current_weights = list(target_weights)
        self._step_index += 1
        done = self._step_index >= len(self._rows) - 1
        self._done = done

        price_payload = self._format_price_payload(row)
        target_payload = self._format_weight_payload(target_weights)
        raw_payload = self._format_weight_payload(action_vector)
        info = {
            "timestamp": row["timestamp"],
            "price_close": price_payload,
            "raw_action": raw_payload,
            "weights": target_payload,
            "weight_target": target_payload,
            "weight_realized": target_payload,
            "portfolio_value": next_value,
            "cost_paid": cost_paid,
            "reward": reward,
        }
        regime_values = self._regime_vector(row)
        if regime_values:
            info["regime_features"] = regime_values
            info["regime_state"] = row.get("regime_state")
        if mode is not None:
            info["mode"] = str(mode)
        self._mode_timeline.append(mode)
        return self._build_observation(), reward, done, info

    @property
    def mode_timeline(self) -> Tuple[str | None, ...]:
        return tuple(self._mode_timeline)

    def _prepare_action(self, value: float | Sequence[float]) -> List[float]:
        low, high = self.config.action_clip
        if isinstance(value, (list, tuple)):
            raw = [float(v) for v in value]
        else:
            raw = [float(value)]
        if len(raw) not in (1, self._num_assets):
            raise ValueError(f"Action dimension {len(raw)} does not match expected assets {self._num_assets}")
        if len(raw) == 1 and self._num_assets > 1:
            raw = raw * self._num_assets
        return [max(low, min(high, entry)) for entry in raw]

    def _project_action(self, vector: Sequence[float]) -> List[float]:
        prev = self._current_weights
        projected = project_weights(vector, prev, self.config.risk_config)
        if len(projected) != self._num_assets:
            raise ValueError("Projected weight vector does not match environment assets")
        return projected

    def _compute_returns(self, row: Mapping[str, object], next_row: Mapping[str, object]) -> List[float]:
        returns: List[float] = []
        for symbol in self._symbol_order:
            panel_now = row["panel"][symbol]
            panel_next = next_row["panel"][symbol]
            price_now = float(panel_now["close"])
            price_next = float(panel_next["close"])
            pct_return = (price_next - price_now) / price_now if price_now else 0.0
            returns.append(pct_return)
        return returns

    def _build_observation(self) -> Tuple[float, ...]:
        row = self._rows[self._step_index]
        values: List[float] = []
        regime_values = self._regime_vector(row)
        for symbol in self._symbol_order:
            features = row["panel"][symbol]
            for column in self._feature_columns:
                value = features.get(column)
                if value is None:
                    raise ValueError(f"Row missing required observation column '{column}' for symbol '{symbol}'")
                values.append(float(value))
        if regime_values:
            values.extend(regime_values)
        values.extend(self._current_weights)
        return tuple(values)

    def _build_observation_headers(self) -> Tuple[str, ...]:
        if self._num_assets == 1:
            headers: List[str] = list(self._feature_columns)
            headers.extend(f"REGIME:{column}" for column in self._regime_feature_columns)
            headers.append("prev_weight")
            return tuple(headers)
        headers: List[str] = []
        for symbol in self._symbol_order:
            headers.extend(f"{symbol}:{column}" for column in self._feature_columns)
        headers.extend(f"REGIME:{column}" for column in self._regime_feature_columns)
        headers.extend(f"{symbol}:prev_weight" for symbol in self._symbol_order)
        return tuple(headers)

    def _determine_symbol_order(self) -> Tuple[str, ...]:
        if not self._rows:
            return tuple()
        first = self._rows[0]
        panel = first.get("panel")
        if not isinstance(panel, dict) or not panel:
            raise ValueError("Rows must include per-symbol feature panels")
        order = tuple(sorted(panel.keys()))
        for row in self._rows:
            row_panel = row.get("panel") or {}
            missing = set(order) - set(row_panel.keys())
            if missing:
                raise ValueError(f"Row missing panel data for symbols: {sorted(missing)}")
            ordered_panel = {symbol: row_panel[symbol] for symbol in order}
            row["panel"] = ordered_panel
            if len(order) == 1:
                row["symbol"] = order[0]
                row.update({column: ordered_panel[order[0]][column] for column in self._feature_columns})
        return order

    def _format_weight_payload(self, weights: Sequence[float]) -> object:
        if self._num_assets == 1:
            return float(weights[0])
        return {symbol: float(weights[idx]) for idx, symbol in enumerate(self._symbol_order)}

    def _format_price_payload(self, row: Mapping[str, object]) -> object:
        if self._num_assets == 1:
            return float(row["panel"][self._symbol_order[0]]["close"])
        return {symbol: float(row["panel"][symbol]["close"]) for symbol in self._symbol_order}

    def _infer_regime_feature_names(self, rows: Sequence[Mapping[str, object]]) -> Tuple[str, ...]:
        for row in rows:
            state = row.get("regime_state")
            if isinstance(state, RegimeState):
                return tuple(state.feature_names)
        return tuple()

    def _split_feature_columns(self, columns: Tuple[str, ...], regime_names: Tuple[str, ...]) -> Tuple[str, ...]:
        if not regime_names:
            return columns
        if len(columns) <= len(regime_names):
            raise ValueError("Regime features require at least one dedicated per-symbol feature column")
        base = columns[: len(columns) - len(regime_names)]
        suffix = columns[len(base) :]
        if tuple(suffix) != regime_names:
            raise ValueError("Regime feature ordering must match observation column suffix")
        return base

    def _regime_vector(self, row: Mapping[str, object]) -> Tuple[float, ...]:
        if not self._regime_feature_columns:
            return tuple()
        state = row.get("regime_state")
        if not isinstance(state, RegimeState):
            raise ValueError("Rows are missing regime_state despite regime features being enabled")
        if tuple(state.feature_names) != self._regime_feature_columns:
            raise ValueError("Regime feature names changed during rollout")
        return tuple(float(value) for value in state.features)

    @staticmethod
    def _coerce_row(raw: Mapping[str, object], feature_columns: Sequence[str]) -> MutableMapping[str, object]:
        timestamp = raw.get("timestamp")
        if not isinstance(timestamp, datetime):
            raise TypeError("rows must include datetime timestamps")
        normalized: MutableMapping[str, object] = dict(raw)
        normalized["timestamp"] = timestamp
        panel = raw.get("panel")
        if isinstance(panel, Mapping):
            normalized_panel: MutableMapping[str, Dict[str, float]] = {}
            for symbol, values in panel.items():
                if not isinstance(values, Mapping):
                    raise ValueError("panel entries must be mappings of feature columns")
                clean_symbol = str(symbol)
                per_symbol: Dict[str, float] = {}
                for column in feature_columns:
                    if column == "close" and column not in values:
                        raise ValueError("panel rows must include close prices for every symbol")
                    per_symbol[column] = float(values.get(column, 0.0))
                normalized_panel[clean_symbol] = per_symbol
            if not normalized_panel:
                raise ValueError("panel rows must include at least one symbol")
            normalized["panel"] = normalized_panel
            return normalized

        symbol = str(raw.get("symbol") or raw.get("ticker") or "asset").strip()
        if not symbol:
            symbol = "asset"
        per_symbol: Dict[str, float] = {}
        for column in feature_columns:
            if column == "symbol":
                continue
            value = raw.get(column)
            if value is None:
                raise ValueError(f"Row missing required observation column '{column}'")
            per_symbol[column] = float(value)
        normalized["panel"] = {symbol: per_symbol}
        normalized["symbol"] = symbol
        normalized["close"] = float(per_symbol["close"])
        if "sma_diff" not in normalized and "sma_fast" in normalized and "sma_slow" in normalized:
            normalized["sma_diff"] = float(normalized["sma_fast"]) - float(normalized["sma_slow"])
        if "sma_signal" not in normalized:
            normalized["sma_signal"] = float(raw.get("sma_signal", 0.0))
        return normalized


def _materialize_rows(rows: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]]) -> List[Mapping[str, object]]:
    if _looks_like_dataframe(rows):
        records = rows.to_dict("records")  # type: ignore[call-arg]
        return list(records)
    return list(rows)


def _looks_like_dataframe(candidate: object) -> bool:
    return (
        candidate is not None
        and hasattr(candidate, "to_dict")
        and hasattr(candidate, "columns")
        and candidate.__class__.__name__ == "DataFrame"
    )


__all__ = ["SignalWeightEnvConfig", "SignalWeightTradingEnv"]
