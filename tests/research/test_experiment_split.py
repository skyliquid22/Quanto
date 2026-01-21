from __future__ import annotations

from datetime import date

import pytest

from research.experiments import runner as experiment_runner


def _sessions_by_month(year: int, months: int, sessions_per_month: int) -> list[date]:
    sessions: list[date] = []
    for month in range(1, months + 1):
        for day in range(1, sessions_per_month + 1):
            sessions.append(date(year, month, day))
    return sessions


def test_split_selects_closest_allowed_window():
    sessions = _sessions_by_month(2023, months=10, sessions_per_month=20)
    split = experiment_runner._resolve_split_from_sessions(
        sessions,
        train_ratio=0.625,
        test_ratio=0.375,
        test_window_months=None,
    )
    assert split.test_window_months == 4


def test_split_enforces_minimum_one_month():
    sessions = _sessions_by_month(2023, months=2, sessions_per_month=20)
    split = experiment_runner._resolve_split_from_sessions(
        sessions,
        train_ratio=0.8,
        test_ratio=0.2,
        test_window_months=None,
    )
    assert split.test_window_months == 1
    assert split.test_start == date(2023, 2, 1)


def test_split_short_range_falls_back_to_half():
    sessions = [date(2023, 1, day) for day in range(1, 16)]
    split = experiment_runner._resolve_split_from_sessions(
        sessions,
        train_ratio=0.8,
        test_ratio=0.2,
        test_window_months=None,
    )
    assert split.test_window_months is None
    assert split.test_start == sessions[5]
    assert split.test_ratio == pytest.approx(10 / 15)
