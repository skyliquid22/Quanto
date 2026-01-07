from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.datasets.canonical_equity_loader import verify_coverage
from scripts import run_sma_crossover
from scripts import run_sma_finrl_rollout as rollout


UTC = timezone.utc


class DummySlice:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.timestamps = [
            datetime(2023, 1, 3, tzinfo=UTC),
            datetime(2023, 1, 4, tzinfo=UTC),
            datetime(2023, 1, 5, tzinfo=UTC),
        ]
        self.closes = [100.0, 101.0, 102.0]
        self.rows = [{"symbol": symbol, "timestamp": ts.isoformat()} for ts in self.timestamps]


def _touch_canonical(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[]", encoding="utf-8")


def test_universe_bootstrap_creates_missing_shards_and_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    data_root = tmp_path / "quanto_data"
    existing_aapl = data_root / "canonical" / "equity_ohlcv" / "AAPL" / "daily" / "2022.parquet"
    existing_msft = data_root / "canonical" / "equity_ohlcv" / "MSFT" / "daily" / "2023.parquet"
    _touch_canonical(existing_aapl)
    _touch_canonical(existing_msft)

    built_pairs: list[tuple[str, int]] = []

    def fake_build_missing(*, symbols, years, config_path, raw_root, data_root, run_id):
        coverage = verify_coverage(symbols, f"{min(years)}-01-01", f"{max(years)}-12-31", data_root=data_root)
        requested_years = set(years)
        for symbol in symbols:
            for year in coverage["missing_by_symbol"].get(symbol, []):
                if year not in requested_years:
                    continue
                built_pairs.append((symbol, year))
                shard = Path(data_root) / "canonical" / "equity_ohlcv" / symbol / "daily" / f"{year}.parquet"
                _touch_canonical(shard)

    monkeypatch.setattr(rollout, "build_missing_equity_ohlcv_canonical", fake_build_missing)

    def fake_maybe_run_live_bootstrap(**kwargs):
        return rollout.BootstrapMetadata(mode="none")

    monkeypatch.setattr(run_sma_crossover, "maybe_run_live_bootstrap", fake_maybe_run_live_bootstrap)

    def fake_load_canonical_equity(symbols, start, end, data_root, interval="daily"):
        slices = {symbol: DummySlice(symbol) for symbol in symbols}
        hashes = {f"canonical/equity_ohlcv/{symbol}/daily/{start.year}.parquet": f"sha256:{symbol.lower()}" for symbol in symbols}
        return slices, hashes

    monkeypatch.setattr(run_sma_crossover, "load_canonical_equity", fake_load_canonical_equity)

    def fake_run_sma(symbol, timestamps, closes, config):
        return SimpleNamespace(
            timestamps=timestamps,
            closes=closes,
            fast_sma=[close * 0.9 for close in closes],
            slow_sma=[close for close in closes],
            signal=[1 for _ in closes],
            positions=[0 for _ in closes],
        )

    monkeypatch.setattr(run_sma_crossover, "run_sma_crossover", fake_run_sma)

    def fake_backtest(strategy_result):
        return SimpleNamespace(
            equity_curve=[1.0, 1.1, 1.2],
            buy_and_hold_curve=[1.0, 1.05, 1.1],
            metrics={"return": 0.1},
        )

    monkeypatch.setattr(run_sma_crossover, "run_backtest", fake_backtest)

    def fake_aggregate(results):
        return SimpleNamespace(metrics={"return": 0.1}, equity_curve=[1.0, 1.1])

    monkeypatch.setattr(run_sma_crossover, "aggregate_results", fake_aggregate)
    monkeypatch.setattr(run_sma_crossover, "serialize_metrics", lambda metrics: metrics)

    def fake_render_equity_curves(path: Path, curves):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("plot", encoding="utf-8")
        return path

    monkeypatch.setattr(run_sma_crossover, "render_equity_curves", fake_render_equity_curves)

    def fake_write_report(payload, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"run_id": payload["run_id"]}), encoding="utf-8")

    monkeypatch.setattr(run_sma_crossover, "write_report", fake_write_report)

    args = SimpleNamespace(
        symbols="AAPL,MSFT",
        start_date="2022-01-03",
        end_date="2023-01-05",
        fast=5,
        slow=10,
        run_id="universe_test",
        data_root=str(data_root),
        vendor="polygon",
        ingest_mode="rest",
        canonical_domain="equity_ohlcv",
        force_ingest=False,
        force_canonical_build=False,
    )
    monkeypatch.setattr(run_sma_crossover, "parse_args", lambda: args)

    result = run_sma_crossover.main()
    assert result == 0
    captured = capsys.readouterr()
    assert "[universe-bootstrap] Missing canonical shards: AAPL:2023, MSFT:2022" in captured.out
    assert "[universe-bootstrap] Building canonical shards: AAPL:2023, MSFT:2022" in captured.out
    assert built_pairs == [("AAPL", 2023), ("MSFT", 2022)]
