from __future__ import annotations

from pathlib import Path

import pytest

from infra.storage.parquet import _PARQUET_AVAILABLE
from infra.storage.raw_writer import RawFinancialDatasetsWriter

pytestmark = pytest.mark.skipif(not _PARQUET_AVAILABLE, reason="pyarrow is required for storage tests")


def test_raw_financialdatasets_company_facts_layout(tmp_path: Path):
    writer = RawFinancialDatasetsWriter(base_path=tmp_path)
    record = {
        "symbol": "AAPL",
        "as_of_date": "2024-01-05",
        "source_vendor": "financialdatasets",
    }
    result = writer.write_company_facts("financialdatasets", [record])

    path = Path(result["files"][0]["path"])
    assert path == tmp_path / "financialdatasets" / "company_facts" / "AAPL" / "2024" / "01" / "05.parquet"


def test_raw_financialdatasets_insider_trades_layout(tmp_path: Path):
    writer = RawFinancialDatasetsWriter(base_path=tmp_path)
    record = {
        "symbol": "AAPL",
        "transaction_date": "2024-01-03",
        "source_vendor": "financialdatasets",
    }
    result = writer.write_insider_trades("financialdatasets", [record])

    path = Path(result["files"][0]["path"])
    assert path == tmp_path / "financialdatasets" / "insider_trades" / "AAPL" / "2024" / "01" / "03.parquet"
