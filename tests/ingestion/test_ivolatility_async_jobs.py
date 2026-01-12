from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import guard
    sys.path.insert(0, str(PROJECT_ROOT))

from infra.ingestion import ivolatility_client as ivc
from infra.ingestion.ivolatility_client import (
    IvolatilityClient,
    IvolatilityClientError,
    TransportResponse,
)


class _MockTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, params, headers, timeout):
        if not self.responses:
            raise AssertionError("transport called more times than expected")
        payload = self.responses.pop(0)
        self.calls.append({"method": method, "url": url, "params": dict(params or {})})
        status = payload.get("status", 200)
        headers = payload.get("headers", {"Content-Type": "application/json"})
        body = payload.get("body", {})
        if isinstance(body, (dict, list)):
            text = json.dumps(body)
        else:
            text = str(body)
        return TransportResponse(status_code=status, headers=headers, text=text)


def test_async_dataset_polls_until_completion(tmp_path):
    detail_url = "https://restapi.ivolatility.com/data/info/job-123"
    responses = [
        {"body": {"status": {"code": "PENDING", "urlForDetails": detail_url}}},
        {"body": {"status": {"code": "PROCESSING", "urlForDetails": detail_url}}},
        {
            "body": {
                "status": {"code": "COMPLETED"},
                "data": [
                    {"symbol": "MSFT", "date": "2024-01-02"},
                    {"symbol": "AAPL", "date": "2024-01-01"},
                ],
            }
        },
    ]
    transport = _MockTransport(responses)
    client = IvolatilityClient(api_key="key", api_secret=None, transport=transport, cache_dir=tmp_path / "cache")

    params = {"stockGroup": "ALL_USA"}
    dataset = client.fetch_async_dataset(
        "equities/stock-market-data",
        params,
        poll_interval_s=0.0,
    )

    assert [row["symbol"] for row in dataset] == ["AAPL", "MSFT"]

    cached_dataset = client.fetch_async_dataset(
        "equities/stock-market-data",
        params,
        poll_interval_s=0.0,
    )
    assert cached_dataset == dataset
    assert len(transport.calls) == 3


def test_async_dataset_handles_immediate_completion(tmp_path):
    responses = [
        {
            "body": {
                "status": {"code": "COMPLETED"},
                "data": [
                    {"symbol": "AAPL", "date": "2024-01-01"},
                    {"symbol": "MSFT", "date": "2024-01-01"},
                ],
            }
        }
    ]
    transport = _MockTransport(responses)
    client = IvolatilityClient(api_key="key", api_secret=None, transport=transport, cache_dir=tmp_path / "cache")

    dataset = client.fetch_async_dataset("equities/stock-market-data", {"stockGroup": "TECH"})

    assert len(dataset) == 2
    assert dataset[0]["symbol"] == "AAPL"
    assert len(transport.calls) == 1


def test_async_dataset_times_out(monkeypatch):
    class _FakeTime:
        def __init__(self):
            self.current = 0.0

        def time(self):
            return self.current

        def sleep(self, seconds):
            self.current += max(0.0, seconds)

    fake_time = _FakeTime()
    monkeypatch.setattr(ivc, "time", fake_time)

    detail_url = "https://restapi.ivolatility.com/data/info/job-timeout"
    responses = [
        {"body": {"status": {"code": "PENDING", "urlForDetails": detail_url}}},
        {"body": {"status": {"code": "PENDING", "urlForDetails": detail_url}}},
        {"body": {"status": {"code": "PENDING", "urlForDetails": detail_url}}},
    ]
    transport = _MockTransport(responses)
    client = IvolatilityClient(api_key="key", api_secret=None, transport=transport, cache_dir=None)

    with pytest.raises(IvolatilityClientError) as excinfo:
        client.fetch_async_dataset(
            "equities/stock-market-data",
            {"stockGroup": "ALL_USA"},
            poll_timeout_s=3,
            poll_interval_s=1.0,
        )

    assert "timed out" in str(excinfo.value)


def test_async_fingerprint_invariance(tmp_path):
    responses = [
        {
            "body": {
                "status": {"code": "COMPLETED"},
                "data": [{"symbol": "AAPL", "date": "2024-01-01"}],
            }
        }
    ]
    transport = _MockTransport(responses)
    client = IvolatilityClient(api_key="key", api_secret=None, transport=transport, cache_dir=tmp_path / "cache")

    params_a = {"from": "2024-01-01", "to": "2024-01-02", "stockGroup": "TECH"}
    params_b = {"stockGroup": "TECH", "to": "2024-01-02", "from": "2024-01-01"}

    first = client.fetch_async_dataset("equities/stock-market-data", params_a)
    second = client.fetch_async_dataset("equities/stock-market-data", params_b)

    assert first == second
    assert len(transport.calls) == 1


def test_async_dataset_downloads_csv(tmp_path):
    detail_url = "https://restapi.ivolatility.com/data/info/job-file"
    download_url = "https://restapi.ivolatility.com/data/download/job-file"
    responses = [
        {"body": {"status": {"code": "PENDING", "urlForDetails": detail_url}}},
        {
            "body": {
                "status": {"code": "COMPLETED", "urlForDownload": download_url},
            }
        },
        {
            "headers": {"Content-Type": "text/csv"},
            "body": "symbol,date,value\nMSFT,2024-01-02,2\nAAPL,2024-01-01,1\n",
        },
    ]
    transport = _MockTransport(responses)
    client = IvolatilityClient(api_key="key", api_secret=None, transport=transport, cache_dir=tmp_path / "cache")

    payload = client.fetch_async_dataset(
        "equities/stock-market-data",
        {"stockGroup": "ALL_USA"},
        poll_interval_s=0.0,
    )

    assert isinstance(payload, bytes)
    decoded = payload.decode("utf-8").splitlines()
    assert decoded[0] == "symbol,date,value"
    assert decoded[1].startswith("AAPL")

    replay = client.fetch_async_dataset(
        "equities/stock-market-data",
        {"stockGroup": "ALL_USA"},
        poll_interval_s=0.0,
    )
    assert replay == payload
    assert len(transport.calls) == 3
