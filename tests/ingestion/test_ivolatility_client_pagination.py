from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import guard
    sys.path.insert(0, str(PROJECT_ROOT))

from infra.ingestion.ivolatility_client import (
    IvolatilityClient,
    IvolatilityClientError,
    TransportResponse,
)


class _FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, params, headers, timeout):
        call = {"method": method, "url": url, "params": dict(params or {})}
        self.calls.append(call)
        if not self.responses:
            raise AssertionError("transport called more times than expected")
        payload = self.responses.pop(0)
        status = payload.get("status", 200)
        body = json.dumps(payload.get("body", {}))
        return TransportResponse(
            status_code=status,
            headers=payload.get("headers", {}),
            text=body,
            body=body.encode("utf-8"),
        )


def test_ivolatility_client_paginates_and_sorts_records():
    first_page = {
        "body": {
            "results": [
                {"id": 2, "symbol": "MSFT", "date": "2024-01-02"},
                {"id": 1, "symbol": "AAPL", "date": "2024-01-01"},
            ],
            "next_page_token": "cursor-1",
        }
    }
    second_page = {
        "body": {
            "results": [
                {"id": 3, "symbol": "AAPL", "date": "2024-01-03"},
            ],
        }
    }
    transport = _FakeTransport([first_page, second_page])
    client = IvolatilityClient(
        api_key="key",
        api_secret="secret",
        transport=transport,
        cache_dir=None,
        max_retries=0,
    )

    records = client.fetch("options/contracts", {"underlying": "AAPL"})

    assert [record["id"] for record in records] == [1, 2, 3]
    assert transport.calls[0]["params"].get("page_token") is None
    assert transport.calls[1]["params"].get("page_token") == "cursor-1"


def test_fetch_one_requires_exactly_one_record():
    transport = _FakeTransport(
        [
            {
                "body": {
                    "results": [
                        {"id": 1, "symbol": "AAPL"},
                        {"id": 2, "symbol": "AAPL"},
                    ]
                }
            }
        ]
    )
    client = IvolatilityClient(
        api_key="key",
        api_secret="secret",
        transport=transport,
        cache_dir=None,
        max_retries=0,
    )

    with pytest.raises(IvolatilityClientError):
        client.fetch_one("options/contracts", {"underlying": "AAPL"})


def test_ivolatility_client_allows_missing_secret():
    transport = _FakeTransport(
        [
            {
                "body": {
                    "results": [
                        {"id": 1, "symbol": "AAPL"},
                    ]
                }
            }
        ]
    )
    client = IvolatilityClient(
        api_key="key-only",
        api_secret=None,
        transport=transport,
        cache_dir=None,
        max_retries=0,
    )

    client.fetch("options/contracts", {"underlying": "AAPL"})

    params = transport.calls[0]["params"]
    assert params.get("apiKey") == "key-only"
    assert "apiSecret" not in params
