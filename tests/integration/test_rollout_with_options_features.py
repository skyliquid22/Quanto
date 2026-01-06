from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from research.datasets.canonical_options_loader import CanonicalOptionData
from scripts import run_sma_finrl_rollout as rollout


UTC = timezone.utc


def test_rollout_determinism_with_options_features(tmp_path, monkeypatch, capsys):
    args = [
        "--symbol",
        "AAPL",
        "--start-date",
        "2023-01-02",
        "--end-date",
        "2023-01-04",
        "--data-root",
        str(tmp_path),
        "--fast",
        "2",
        "--slow",
        "3",
        "--interval",
        "daily",
    ]

    manifest = rollout.ManifestMetadata(
        path=tmp_path / "canonical" / "manifests" / "equity_ohlcv" / "mock.json",
        run_id="canon",
        start_date=date(2023, 1, 2),
        end_date=date(2023, 1, 4),
        created_at=datetime(2023, 1, 5, tzinfo=UTC),
        domain="equity_ohlcv",
        vendor="polygon",
    )
    manifest.path.parent.mkdir(parents=True, exist_ok=True)
    manifest.path.write_text("{}", encoding="utf-8")

    def fake_load_equity(symbols, start, end, data_root=None, interval="daily"):
        slice_data = SimpleNamespace(
            rows=[{"timestamp": datetime(2023, 1, 2, tzinfo=UTC)}],
            timestamps=[
                datetime(2023, 1, 2, tzinfo=UTC),
                datetime(2023, 1, 3, tzinfo=UTC),
                datetime(2023, 1, 4, tzinfo=UTC),
            ],
            closes=[100.0, 101.0, 102.0],
        )
        return {symbols[0]: slice_data}, {"canonical/equity/AAPL.parquet": "sha256:equity"}

    def fake_sma(*_, **__):
        return SimpleNamespace(
            timestamps=[
                datetime(2023, 1, 2, tzinfo=UTC),
                datetime(2023, 1, 3, tzinfo=UTC),
                datetime(2023, 1, 4, tzinfo=UTC),
            ],
            closes=[100.0, 101.0, 102.0],
            fast_sma=[99.0, 100.0, 101.0],
            slow_sma=[98.5, 99.5, 100.5],
            signal=[1, 1, 0],
        )

    options_reference = pd.DataFrame(
        [
            {"option_symbol": "OPT1", "option_type": "call", "underlying_symbol": "AAPL", "strike": 150},
            {"option_symbol": "OPT2", "option_type": "put", "underlying_symbol": "AAPL", "strike": 145},
        ]
    )
    options_oi = pd.DataFrame(
        [
            {"timestamp": datetime(2023, 1, 2, tzinfo=UTC), "option_symbol": "OPT1", "open_interest": 100},
            {"timestamp": datetime(2023, 1, 2, tzinfo=UTC), "option_symbol": "OPT2", "open_interest": 200},
            {"timestamp": datetime(2023, 1, 3, tzinfo=UTC), "option_symbol": "OPT1", "open_interest": 120},
            {"timestamp": datetime(2023, 1, 3, tzinfo=UTC), "option_symbol": "OPT2", "open_interest": 210},
        ]
    )
    options_ohlcv = pd.DataFrame(
        [
            {"timestamp": datetime(2023, 1, 2, tzinfo=UTC), "option_symbol": "OPT1", "volume": 10},
            {"timestamp": datetime(2023, 1, 3, tzinfo=UTC), "option_symbol": "OPT2", "volume": 20},
        ]
    )

    def fake_load_options(symbol, start_date, end_date, data_root=None):
        data = CanonicalOptionData(
            reference=options_reference.copy(),
            ohlcv=options_ohlcv.copy(),
            open_interest=options_oi.copy(),
            file_paths=[],
        )
        return data, {"canonical/options/AAPL.parquet": "sha256:options"}

    monkeypatch.setattr(rollout, "load_canonical_equity", fake_load_equity)
    monkeypatch.setattr(rollout, "run_sma_crossover", fake_sma)
    monkeypatch.setattr(rollout, "maybe_run_live_bootstrap", lambda **kwargs: rollout.BootstrapMetadata(mode="none"))
    monkeypatch.setattr(rollout, "_canonical_files_exist", lambda *args, **kwargs: True)
    monkeypatch.setattr(rollout, "_locate_canonical_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr("research.features.feature_registry.load_canonical_options", fake_load_options)

    def run_once(extra_args):
        monkeypatch.setattr(sys, "argv", ["run"] + args + extra_args)
        assert rollout.main() == 0
        out = capsys.readouterr().out.strip()
        payload = json.loads(out)
        report_path = Path(payload["report"])
        report = json.loads(report_path.read_text())
        return report

    report1 = run_once(["--feature-set", "options_v1"])
    report2 = run_once(["--feature-set", "options_v1"])

    assert report1["hashes"]["report_json"] == report2["hashes"]["report_json"]
    assert report1["hashes"]["plot_png"] == report2["hashes"]["plot_png"]

    options_obs_len = len(report1["parameters"]["observation_columns"])
    report_sma = run_once(["--feature-set", "sma_v1"])
    sma_obs_len = len(report_sma["parameters"]["observation_columns"])

    assert options_obs_len > sma_obs_len
