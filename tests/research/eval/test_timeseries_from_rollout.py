from __future__ import annotations

from research.eval.timeseries import extract_timeseries_from_rollout


def _rollout_fixture():
    return {
        "metadata": {
            "symbols": ["AAA", "BBB"],
            "rollout": {"symbols": ["AAA", "BBB"]},
        },
        "series": {
            "timestamps": [
                "2023-01-02T00:00:00+00:00",
                "2023-01-03T00:00:00+00:00",
                "2023-01-04T00:00:00+00:00",
            ],
            "returns": [0.01, -0.005],
            "weights": {"AAA": [0.0, 0.6, 0.4], "BBB": [0.0, 0.4, 0.6]},
            "regime": {
                "feature_names": ["market_vol_20d", "market_trend_20d"],
                "values": [[0.2, 0.1], [0.3, 0.05]],
            },
        },
    }


def test_extract_timeseries_computes_exposures_and_turnover():
    payload = _rollout_fixture()
    timeseries = extract_timeseries_from_rollout(payload, float_precision=6)
    assert timeseries["timestamps"] == [
        "2023-01-03T00:00:00+00:00",
        "2023-01-04T00:00:00+00:00",
    ]
    assert timeseries["returns"] == [0.01, -0.005]
    assert timeseries["exposures"] == [1.0, 1.0]
    assert timeseries["turnover_by_step"] == [1.0, 0.4]
    regime = timeseries["regime"]
    assert regime["feature_names"] == ["market_vol_20d", "market_trend_20d"]
    assert regime["series"] == [[0.2, 0.1], [0.3, 0.05]]
