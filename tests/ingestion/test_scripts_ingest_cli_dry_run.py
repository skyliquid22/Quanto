import json

from scripts import ingest


def test_cli_dry_run_polygon_equity(tmp_path, capsys):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "symbols": ["AAPL"],
                "start_date": "2023-01-01",
                "end_date": "2023-01-02",
                "vendor": "polygon",
            }
        )
    )
    exit_code = ingest.main(
        [
            "--config",
            str(config_path),
            "--domain",
            "equity_ohlcv",
            "--run-id",
            "dry-run-test",
            "--dry-run",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["status"] == "dry_run"
    assert summary["adapter"] == "PolygonEquityAdapter"
    assert summary["vendor"] == "polygon"


def test_cli_ivolatility_config_never_builds_polygon_client(tmp_path, capsys, monkeypatch):
    config_path = tmp_path / "ivol.json"
    config_path.write_text(
        json.dumps(
            {
                "symbols": ["AAPL"],
                "start_date": "2023-01-01",
                "end_date": "2023-01-02",
                "vendor": "ivolatility",
            }
        )
    )

    def _fail(*args, **kwargs):
        raise AssertionError("PolygonRESTClient should not be instantiated for iVolatility runs")

    monkeypatch.setattr(ingest, "PolygonRESTClient", _fail)

    exit_code = ingest.main(
        [
            "--config",
            str(config_path),
            "--domain",
            "equity_ohlcv",
            "--run-id",
            "ivol-dry-run",
            "--dry-run",
        ]
    )
    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["adapter"] == "IvolatilityEquityAdapter"
    assert summary["vendor"] == "ivolatility"
