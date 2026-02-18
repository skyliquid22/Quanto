from __future__ import annotations

from pathlib import Path

import pytest

from infra.storage.raw_writer import RawFinancialDatasetsWriter


def test_raw_financialdatasets_company_facts_layout(tmp_path: Path):
    policy_map = {
        "company_facts": {
            "policy": "snapshot_single_csv",
            "dedup_keys": ["ticker"],
        },
        "insider_trades": {
            "policy": "timeseries_csv_yearly",
            "date_priority": ["filing_date"],
            "date_kind": "date",
            "dedup_keys": ["ticker", "filing_date", "name", "transaction_date", "transaction_value", "transaction_shares", "security_title"],
        },
    }
    writer = RawFinancialDatasetsWriter(base_path=tmp_path, policy_map=policy_map)
    record = {
        "symbol": "AAPL",
        "as_of_date": "2024-01-05",
        "source_vendor": "financialdatasets",
    }
    result = writer.write_company_facts("financialdatasets", [record])

    path = Path(result["files"][0]["path"])
    assert path == tmp_path / "financialdatasets" / "company_facts" / "Facts.csv"


def test_raw_financialdatasets_insider_trades_layout(tmp_path: Path):
    policy_map = {
        "company_facts": {"policy": "snapshot_single_csv", "dedup_keys": ["ticker"]},
        "insider_trades": {
            "policy": "timeseries_csv_yearly",
            "date_priority": ["filing_date"],
            "date_kind": "date",
            "dedup_keys": ["ticker", "filing_date", "name", "transaction_date", "transaction_value", "transaction_shares", "security_title"],
        },
    }
    writer = RawFinancialDatasetsWriter(base_path=tmp_path, policy_map=policy_map)
    record = {
        "symbol": "AAPL",
        "filing_date": "2024-01-03",
        "source_vendor": "financialdatasets",
    }
    result = writer.write_insider_trades("financialdatasets", [record])

    path = Path(result["files"][0]["path"])
    assert path == tmp_path / "financialdatasets" / "insider_trades" / "AAPL" / "2024.csv"


def test_raw_financialdatasets_company_facts_upsert(tmp_path: Path):
    policy_map = {
        "company_facts": {"policy": "snapshot_single_csv", "dedup_keys": ["ticker"]},
    }
    writer = RawFinancialDatasetsWriter(base_path=tmp_path, policy_map=policy_map)
    first = {"symbol": "AAPL", "as_of_date": "2024-01-05", "name": "Apple"}
    second = {"symbol": "AAPL", "as_of_date": "2024-02-01", "name": "Apple Inc"}

    writer.write_company_facts("financialdatasets", [first])
    writer.write_company_facts("financialdatasets", [second])

    import pandas as pd
    path = tmp_path / "financialdatasets" / "company_facts" / "Facts.csv"
    frame = pd.read_csv(path)
    assert len(frame) == 1
    assert frame.loc[0, "name"] == "Apple Inc"


def test_raw_financialdatasets_insider_dedup_keys(tmp_path: Path):
    policy_map = {
        "company_facts": {"policy": "snapshot_single_csv", "dedup_keys": ["ticker"]},
        "insider_trades": {
            "policy": "timeseries_csv_yearly",
            "date_priority": ["filing_date"],
            "date_kind": "date",
            "dedup_keys": [
                "ticker",
                "filing_date",
                "name",
                "transaction_date",
                "transaction_value",
                "transaction_shares",
                "security_title",
            ],
        },
    }
    writer = RawFinancialDatasetsWriter(base_path=tmp_path, policy_map=policy_map)
    first = {
        "symbol": "AAPL",
        "filing_date": "2024-01-03",
        "name": "Jane Doe",
        "transaction_date": "2024-01-02",
        "transaction_value": 1000,
        "transaction_shares": 10,
        "security_title": "Common Stock",
    }
    second = {
        "symbol": "AAPL",
        "filing_date": "2024-01-03",
        "name": "Jane Doe",
        "transaction_date": "2024-01-04",
        "transaction_value": 1000,
        "transaction_shares": 10,
        "security_title": "Common Stock",
    }
    writer.write_insider_trades("financialdatasets", [first, second])

    import pandas as pd

    path = tmp_path / "financialdatasets" / "insider_trades" / "AAPL" / "2024.csv"
    frame = pd.read_csv(path)
    assert len(frame) == 2
