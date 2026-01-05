from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import guard
    sys.path.insert(0, str(PROJECT_ROOT))

from infra.ingestion.adapters import OptionReferenceIngestionRequest, OptionTimeseriesIngestionRequest, PolygonOptionsAdapter
from infra.ingestion.options_pipeline import OptionPartition, OptionsIngestionPipeline, OptionsIngestionPlan
from infra.storage.raw_writer import RawOptionsWriter
from infra.validation import ValidationError


class _DeterministicOptionsRestClient:
    def __init__(self, *, negative_oi: bool = False) -> None:
        self.negative_oi = negative_oi
        self.reference_calls = 0
        self.ohlcv_calls = 0
        self.oi_calls = 0

    async def fetch_option_contracts(self, underlying, as_of, *, page_url, page_size):
        self.reference_calls += 1
        return {
            "results": [
                {
                    "contract_symbol": "OPT1",
                    "underlying_symbol": underlying,
                    "expiration_date": "2024-01-19",
                    "strike": 150,
                    "option_type": "C",
                    "multiplier": 100,
                }
            ]
        }

    async def fetch_option_ohlcv(self, option_symbol, start, end, *, page_url, page_size):
        self.ohlcv_calls += 1
        return {
            "results": [
                {
                    "t": 1704067200000,
                    "open": 1.0,
                    "high": 1.5,
                    "low": 0.9,
                    "close": 1.2,
                    "volume": 10.0,
                    "option_symbol": option_symbol,
                }
            ]
        }

    async def fetch_option_open_interest(self, option_symbol, start, end, *, page_url, page_size):
        self.oi_calls += 1
        value = -10 if self.negative_oi else 250
        return {"results": [{"t": 1704067200000, "option_symbol": option_symbol, "open_interest": value}]}

    async def aclose(self):  # pragma: no cover - compatibility
        return None


def _build_plan() -> OptionsIngestionPlan:
    reference_partition = OptionPartition(
        partition_id="ref-1",
        request=OptionReferenceIngestionRequest(
            underlying_symbols=("AAPL",),
            as_of_date=date(2024, 1, 1),
        ),
        mode="rest",
    )
    ohlcv_partition = OptionPartition(
        partition_id="ohlcv-1",
        request=OptionTimeseriesIngestionRequest(
            domain="option_contract_ohlcv",
            option_symbols=("OPT1",),
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
        ),
        mode="rest",
    )
    oi_partition = OptionPartition(
        partition_id="oi-1",
        request=OptionTimeseriesIngestionRequest(
            domain="option_open_interest",
            option_symbols=("OPT1",),
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
        ),
        mode="rest",
    )
    return OptionsIngestionPlan(
        vendor="polygon",
        reference=(reference_partition,),
        ohlcv=(ohlcv_partition,),
        open_interest=(oi_partition,),
    )


def _build_pipeline(tmp_path: Path, client: _DeterministicOptionsRestClient) -> OptionsIngestionPipeline:
    raw_root = tmp_path / "raw"
    checkpoint_root = tmp_path / "checkpoints"
    raw_writer = RawOptionsWriter(base_path=raw_root)
    adapter = PolygonOptionsAdapter(rest_client=client, rest_config={"concurrency": 1})
    return OptionsIngestionPipeline(
        adapter=adapter,
        raw_writer=raw_writer,
        manifest_base_dir=raw_root,
        checkpoint_dir=checkpoint_root,
    )


def test_options_pipeline_end_to_end_with_checkpoint_resume(tmp_path, monkeypatch):
    monkeypatch.setattr("infra.storage.parquet._PARQUET_AVAILABLE", False)
    plan = _build_plan()
    client = _DeterministicOptionsRestClient()
    pipeline = _build_pipeline(tmp_path, client)

    manifest = pipeline.run(plan, run_id="options-run")
    assert manifest["status"] == "succeeded"
    assert set(manifest["domains"].keys()) == {
        "option_contract_reference",
        "option_contract_ohlcv",
        "option_open_interest",
    }

    raw_root = tmp_path / "raw"
    reference_file = raw_root / "polygon" / "option_contract_reference" / "AAPL" / "2024" / "01" / "01.parquet"
    assert reference_file.exists()
    ohlcv_file = raw_root / "polygon" / "options_ohlcv" / "OPT1" / "daily" / "2024" / "01" / "01.parquet"
    assert ohlcv_file.exists()
    oi_file = raw_root / "polygon" / "options_oi" / "OPT1" / "daily" / "2024" / "01" / "01.parquet"
    assert oi_file.exists()

    # Re-run with a new adapter; checkpoint should prevent additional REST calls
    resumed_client = _DeterministicOptionsRestClient()
    resumed_pipeline = _build_pipeline(tmp_path, resumed_client)
    manifest_again = resumed_pipeline.run(plan, run_id="options-run")
    assert resumed_client.reference_calls == 0
    assert resumed_client.ohlcv_calls == 0
    assert resumed_client.oi_calls == 0
    assert manifest_again == manifest

    # manifests exist per domain
    for domain in manifest["domains"].keys():
        manifest_path = raw_root / "polygon" / DOMAIN_STORAGE[domain] / "manifests" / "options-run.json"
        assert manifest_path.exists()


def test_options_pipeline_records_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("infra.storage.parquet._PARQUET_AVAILABLE", False)
    plan = _build_plan()
    client = _DeterministicOptionsRestClient(negative_oi=True)
    pipeline = _build_pipeline(tmp_path, client)

    with pytest.raises(ValidationError):
        pipeline.run(plan, run_id="options-bad")

    manifest_path = (
        tmp_path
        / "raw"
        / "polygon"
        / "options_oi"
        / "manifests"
        / "options-bad.json"
    )
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "failed"
    assert manifest["failures"]
    assert manifest["partitions"][-1]["status"] == "failed"


DOMAIN_STORAGE = {
    "option_contract_reference": "option_contract_reference",
    "option_contract_ohlcv": "options_ohlcv",
    "option_open_interest": "options_oi",
}


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
