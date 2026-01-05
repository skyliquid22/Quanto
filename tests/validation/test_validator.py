from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import guard
    sys.path.insert(0, str(PROJECT_ROOT))

from infra.validation import ValidationError, validate_records


def _base_config(tmp_path: Path):
    return {
        "manifest_base_path": tmp_path / "manifests",
        "creation_timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "input_file_hashes": ["sha256:dummy"],
    }


def test_validate_equity_records_success(tmp_path):
    config = _base_config(tmp_path)
    records = [
        {
            "timestamp": datetime(2024, 1, 1, 14, 30, tzinfo=timezone.utc),
            "symbol": "AAPL",
            "open": 100.0,
            "high": 101.0,
            "low": 99.5,
            "close": 100.5,
            "volume": 1000,
            "source_vendor": "polygon",
        },
        {
            "timestamp": datetime(2024, 1, 1, 9, 30, tzinfo=timezone(timedelta(hours=-5))),
            "symbol": "MSFT",
            "open": 250.0,
            "high": 251.0,
            "low": 249.5,
            "close": 250.5,
            "volume": 2000,
            "source_vendor": "polygon",
        },
    ]

    validated, manifest = validate_records(
        "equity_ohlcv", records, source_vendor="polygon", run_id="equity-success", config=config
    )

    assert len(validated) == 2
    assert all(item["timestamp"].tzinfo == timezone.utc for item in validated)
    assert manifest["validation_status"] == "passed"
    assert manifest["valid_records"] == 2
    assert Path(manifest["manifest_path"]).exists()


def test_validate_equity_duplicate_records_fail(tmp_path):
    config = _base_config(tmp_path)
    timestamp = datetime(2024, 1, 1, 14, 30, tzinfo=timezone.utc)
    records = [
        {
            "timestamp": timestamp,
            "symbol": "AAPL",
            "open": 100.0,
            "high": 101.0,
            "low": 99.5,
            "close": 100.5,
            "volume": 1000,
            "source_vendor": "polygon",
        },
        {
            "timestamp": timestamp,
            "symbol": "AAPL",
            "open": 100.0,
            "high": 101.0,
            "low": 99.5,
            "close": 100.5,
            "volume": 1000,
            "source_vendor": "polygon",
        },
    ]

    with pytest.raises(ValidationError) as excinfo:
        validate_records(
            "equity_ohlcv",
            records,
            source_vendor="polygon",
            run_id="equity-dupe",
            config=config,
        )

    err = excinfo.value
    assert err.manifest["validation_status"] == "failed"
    assert err.manifest["invalid_records"] == 1
    assert Path(err.manifest["manifest_path"]).exists()


def test_option_contract_ohlcv_rejects_naive_timestamp_by_default(tmp_path):
    config = _base_config(tmp_path)
    config.pop("creation_timestamp")  # ensure default path exercised
    records = [
        {
            "timestamp": datetime(2024, 1, 1, 9, 30),  # Naive
            "option_symbol": "AAPL240119C00150000",
            "open": 1.0,
            "high": 1.5,
            "low": 0.9,
            "close": 1.2,
            "volume": 10,
            "source_vendor": "polygon",
        }
    ]

    with pytest.raises(ValidationError):
        validate_records(
            "option_contract_ohlcv",
            records,
            source_vendor="polygon",
            run_id="option-naive",
            config=config,
        )


def test_option_contract_ohlcv_naive_allowed_via_config(tmp_path):
    config = _base_config(tmp_path)
    config["allow_naive_timestamps"] = True
    records = [
        {
            "timestamp": datetime(2024, 1, 1, 9, 30),
            "option_symbol": "AAPL240119C00150000",
            "open": 1.0,
            "high": 1.5,
            "low": 0.9,
            "close": 1.2,
            "volume": 10,
            "source_vendor": "polygon",
        }
    ]

    validated, manifest = validate_records(
        "option_contract_ohlcv",
        records,
        source_vendor="polygon",
        run_id="option-naive-allowed",
        config=config,
    )

    assert validated[0]["timestamp"].tzinfo == timezone.utc
    assert manifest["validation_status"] == "passed"


def test_equity_negative_price_rejected(tmp_path):
    config = _base_config(tmp_path)
    records = [
        {
            "timestamp": datetime(2024, 1, 1, 14, 30, tzinfo=timezone.utc),
            "symbol": "AAPL",
            "open": -1.0,
            "high": 1.0,
            "low": 0.5,
            "close": 0.8,
            "volume": 10,
            "source_vendor": "polygon",
        }
    ]

    with pytest.raises(ValidationError) as excinfo:
        validate_records(
            "equity_ohlcv",
            records,
            source_vendor="polygon",
            run_id="equity-negative",
            config=config,
        )

    assert "positive" in str(excinfo.value.errors[0]["error"]).lower()


def test_fundamentals_require_finite_values(tmp_path):
    config = _base_config(tmp_path)
    records = [
        {
            "symbol": "AAPL",
            "report_date": "2023-12-31",
            "fiscal_period": "FY23",
            "revenue": float("nan"),
            "net_income": 1.0,
            "eps": 0.5,
            "total_assets": 10.0,
            "total_liabilities": 5.0,
            "shareholder_equity": 5.0,
            "operating_income": 2.0,
            "free_cash_flow": 1.0,
            "shares_outstanding": 100.0,
            "source_vendor": "polygon",
        }
    ]

    with pytest.raises(ValidationError):
        validate_records(
            "fundamentals",
            records,
            source_vendor="polygon",
            run_id="fundamentals-nan",
            config=config,
        )


def test_validator_is_deterministic_for_same_input(tmp_path):
    config = _base_config(tmp_path)
    config["allow_naive_timestamps"] = True
    run_id = "deterministic"
    records = [
        {
            "timestamp": datetime(2024, 1, 1, 9, 30),
            "symbol": "AAPL",
            "open": 100.0,
            "high": 101.0,
            "low": 99.5,
            "close": 100.5,
            "volume": 1000,
            "source_vendor": "polygon",
        }
    ]

    manifest_bytes = []
    for _ in range(2):
        validated, manifest = validate_records(
            "equity_ohlcv", records, source_vendor="polygon", run_id=run_id, config=config
        )
        assert len(validated) == 1
        manifest_bytes.append(Path(manifest["manifest_path"]).read_bytes())

    assert hashlib.sha256(manifest_bytes[0]).hexdigest() == hashlib.sha256(manifest_bytes[1]).hexdigest()


def test_extra_fields_are_rejected(tmp_path):
    config = _base_config(tmp_path)
    records = [
        {
            "timestamp": datetime(2024, 1, 1, 14, 30, tzinfo=timezone.utc),
            "symbol": "AAPL",
            "open": 100.0,
            "high": 101.0,
            "low": 99.5,
            "close": 100.5,
            "volume": 1000,
            "source_vendor": "polygon",
            "unexpected": 1,
        }
    ]

    with pytest.raises(ValidationError):
        validate_records(
            "equity_ohlcv",
            records,
            source_vendor="polygon",
            run_id="equity-extra",
            config=config,
        )
