"""Integration-style tests for live bootstrap orchestration decisions."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import List

import pytest

from scripts import run_sma_finrl_rollout as rollout


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _extract_run_id(cmd: List[str]) -> str:
    if "--run-id" not in cmd:
        raise AssertionError("missing --run-id in command")
    idx = cmd.index("--run-id")
    return cmd[idx + 1]


def _create_canonical_file(root: Path, symbol: str, day: str) -> None:
    year, month, day_of_month = day.split("-")
    path = root / "canonical" / "equity_ohlcv" / symbol / "daily" / year / month / f"{day_of_month}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[]", encoding="utf-8")


def test_live_bootstrap_runs_ingest_and_build_when_canonical_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    symbol = "AAPL"
    start = date(2023, 1, 2)
    end = date(2023, 1, 5)
    vendor = "polygon"
    domain = "equity_ohlcv"
    calls: List[List[str]] = []
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    def fake_check_call(cmd: List[str], cwd=None, env=None):
        calls.append(list(cmd))
        run_id = _extract_run_id(cmd)
        if "scripts.ingest_equity_ohlcv" in cmd:
            manifest_path = tmp_path / "raw" / vendor / domain / "manifests" / f"{run_id}.json"
            _write_json(
                manifest_path,
                {
                    "run_id": run_id,
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "created_at": "2023-01-06T00:00:00Z",
                },
            )
        elif "scripts.build_canonical_datasets" in cmd:
            manifest_path = tmp_path / "canonical" / "manifests" / domain / f"{run_id}.json"
            _write_json(
                manifest_path,
                {
                    "domain": domain,
                    "run_id": run_id,
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "creation_timestamp": "2023-01-07T00:00:00Z",
                },
            )
        return 0

    monkeypatch.setattr(rollout.subprocess, "check_call", fake_check_call)
    result = rollout.maybe_run_live_bootstrap(
        symbols=[symbol],
        start=start,
        end=end,
        data_root=tmp_path,
        domain=domain,
        vendor=vendor,
        ingest_mode="rest",
        force_ingest=False,
        force_canonical=False,
        run_id_seed=None,
    )

    assert [cmd[2] for cmd in calls] == ["scripts.ingest_equity_ohlcv", "scripts.build_canonical_datasets"]
    assert result.refreshed is True
    assert result.mode == "live"
    assert len(result.raw_manifests) == 1


def test_live_bootstrap_skips_when_canonicals_fresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    symbol = "MSFT"
    start = date(2023, 2, 1)
    end = date(2023, 2, 3)
    vendor = "polygon"
    domain = "equity_ohlcv"
    _create_canonical_file(tmp_path, symbol, "2023-02-01")
    _write_json(
        tmp_path / "canonical" / "manifests" / domain / "canon.json",
        {
            "domain": domain,
            "run_id": "canon",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "creation_timestamp": "2023-02-10T00:00:00Z",
        },
    )
    _write_json(
        tmp_path / "raw" / vendor / domain / "manifests" / "raw.json",
        {
            "run_id": "raw",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "created_at": "2023-02-09T00:00:00Z",
        },
    )
    calls: List[List[str]] = []
    monkeypatch.setattr(rollout.subprocess, "check_call", lambda *args, **kwargs: (calls.append(list(args[0])) or 0))

    result = rollout.maybe_run_live_bootstrap(
        symbols=[symbol],
        start=start,
        end=end,
        data_root=tmp_path,
        domain=domain,
        vendor=vendor,
        ingest_mode="rest",
        force_ingest=False,
        force_canonical=False,
        run_id_seed=None,
    )

    assert calls == []
    assert result.refreshed is False
    assert result.mode == "none"
    assert result.canonical_manifest and result.canonical_manifest.run_id == "canon"


def test_live_bootstrap_rebuilds_when_canonical_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    symbol = "TSLA"
    start = date(2023, 3, 1)
    end = date(2023, 3, 4)
    vendor = "polygon"
    domain = "equity_ohlcv"
    _create_canonical_file(tmp_path, symbol, "2023-03-01")
    _write_json(
        tmp_path / "canonical" / "manifests" / domain / "old.json",
        {
            "domain": domain,
            "run_id": "old",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "creation_timestamp": "2023-03-05T00:00:00Z",
        },
    )
    _write_json(
        tmp_path / "raw" / vendor / domain / "manifests" / "new_raw.json",
        {
            "run_id": "new_raw",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "created_at": "2023-03-06T00:00:00Z",
        },
    )
    created: List[List[str]] = []

    def fake_check_call(cmd, cwd=None, env=None):
        created.append(list(cmd))
        run_id = _extract_run_id(cmd)
        manifest_path = tmp_path / "canonical" / "manifests" / domain / f"{run_id}.json"
        _write_json(
            manifest_path,
            {
                "domain": domain,
                "run_id": run_id,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "creation_timestamp": "2023-03-07T00:00:00Z",
            },
        )
        return 0

    monkeypatch.setattr(rollout.subprocess, "check_call", fake_check_call)
    result = rollout.maybe_run_live_bootstrap(
        symbols=[symbol],
        start=start,
        end=end,
        data_root=tmp_path,
        domain=domain,
        vendor=vendor,
        ingest_mode="rest",
        force_ingest=False,
        force_canonical=False,
        run_id_seed=None,
    )

    assert len(created) == 1
    assert "scripts.build_canonical_datasets" in created[0]
    assert result.refreshed is True
    assert result.raw_manifests and result.raw_manifests[0].run_id == "new_raw"


def test_live_bootstrap_runs_ingest_when_coverage_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    symbol = "AMZN"
    start = date(2023, 4, 1)
    end = date(2023, 4, 5)
    vendor = "polygon"
    domain = "equity_ohlcv"
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    _create_canonical_file(tmp_path, symbol, "2023-04-01")
    _write_json(
        tmp_path / "canonical" / "manifests" / domain / "partial.json",
        {
            "domain": domain,
            "run_id": "partial",
            "start_date": start.isoformat(),
            "end_date": date(2023, 4, 2).isoformat(),
            "creation_timestamp": "2023-04-03T00:00:00Z",
        },
    )
    _write_json(
        tmp_path / "raw" / vendor / domain / "manifests" / "partial_raw.json",
        {
            "run_id": "partial_raw",
            "start_date": start.isoformat(),
            "end_date": date(2023, 4, 2).isoformat(),
            "created_at": "2023-04-03T00:00:00Z",
        },
    )

    recorded: List[List[str]] = []

    def fake_check_call(cmd, cwd=None, env=None):
        recorded.append(list(cmd))
        run_id = _extract_run_id(cmd)
        if "scripts.ingest_equity_ohlcv" in cmd:
            manifest_path = tmp_path / "raw" / vendor / domain / "manifests" / f"{run_id}.json"
            _write_json(
                manifest_path,
                {
                    "run_id": run_id,
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "created_at": "2023-04-06T00:00:00Z",
                },
            )
        else:
            manifest_path = tmp_path / "canonical" / "manifests" / domain / f"{run_id}.json"
            _write_json(
                manifest_path,
                {
                    "domain": domain,
                    "run_id": run_id,
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "creation_timestamp": "2023-04-07T00:00:00Z",
                },
            )
        return 0

    monkeypatch.setattr(rollout.subprocess, "check_call", fake_check_call)
    result = rollout.maybe_run_live_bootstrap(
        symbols=[symbol],
        start=start,
        end=end,
        data_root=tmp_path,
        domain=domain,
        vendor=vendor,
        ingest_mode="rest",
        force_ingest=False,
        force_canonical=False,
        run_id_seed=None,
    )

    assert [cmd[2] for cmd in recorded] == ["scripts.ingest_equity_ohlcv", "scripts.build_canonical_datasets"]
    assert result.refreshed is True
    assert result.raw_manifests and result.raw_manifests[0].run_id.startswith("live_ingest_")


def test_live_bootstrap_requires_credentials_when_needed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    invoked: List[List[str]] = []
    monkeypatch.setattr(rollout.subprocess, "check_call", lambda cmd, cwd=None, env=None: invoked.append(list(cmd)))

    with pytest.raises(SystemExit, match="POLYGON_API_KEY"):
        rollout.maybe_run_live_bootstrap(
            symbols=["NFLX"],
            start=date(2023, 5, 1),
            end=date(2023, 5, 2),
            data_root=tmp_path,
            domain="equity_ohlcv",
            vendor="polygon",
            ingest_mode="rest",
            force_ingest=False,
            force_canonical=False,
            run_id_seed=None,
        )
    assert invoked == []
