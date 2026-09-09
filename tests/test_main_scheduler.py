"""Tests for EOD scheduler duplicate prevention + missed-window recovery."""

import json

import pytest

import main as m
from datetime import date


@pytest.fixture
def eod_state(tmp_path, monkeypatch):
    state_file = tmp_path / "eod_state.json"
    monkeypatch.setattr(m, "_EOD_STATE_FILE", str(state_file))
    yield state_file
    if state_file.exists():
        state_file.unlink()


@pytest.fixture
def scheduler(eod_state):
    return m.EODScheduler(conn_factory=lambda: None, enabled=True,
                          hour=23, email_on_complete=True)


def test_fresh_day_not_done(scheduler):
    assert not scheduler._day_done(date(2026, 9, 1), True)


def test_duplicate_prevention_single_run(eod_state, scheduler, monkeypatch):
    calls = []

    def real_gen(conn, hint, email_on_complete=False, day=None):
        calls.append(day)
        # mimic production: report exists + email delivered
        m._mark_report_done(day.isoformat(), True, True)
        return "x.xlsx"

    monkeypatch.setattr(m, "generate_and_maybe_email", real_gen)
    scheduler._run_eod(date(2026, 9, 1))
    scheduler._run_eod(date(2026, 9, 1))   # same day must be skipped
    assert len(calls) == 1


def test_email_pending_retries_next_cycle(eod_state, scheduler, monkeypatch):
    state = {"2026-09-01": {"report_generated": True, "email_sent": False}}
    eod_state.write_text(json.dumps(state), encoding="utf-8")
    calls = []
    monkeypatch.setattr(m, "generate_and_maybe_email",
                        lambda conn, hint, email_on_complete=False, day=None: (calls.append(day) or "x.xlsx"))
    scheduler._run_eod(date(2026, 9, 1))
    assert calls == [date(2026, 9, 1)]   # retried exactly once


def test_email_sent_day_skipped(eod_state, scheduler, monkeypatch):
    state = {"2026-09-01": {"report_generated": True, "email_sent": True}}
    eod_state.write_text(json.dumps(state), encoding="utf-8")
    calls = []
    monkeypatch.setattr(m, "generate_and_maybe_email",
                        lambda conn, hint, email_on_complete=False, day=None: (calls.append(day) or "x.xlsx"))
    scheduler._run_eod(date(2026, 9, 1))
    assert calls == []


def test_mark_report_done_records_state(eod_state, scheduler):
    m._mark_report_done("2026-09-01", True, True)
    state = json.loads(eod_state.read_text(encoding="utf-8"))
    assert state["2026-09-01"] == {"report_generated": True, "email_sent": True}
    assert scheduler._day_done(date(2026, 9, 1), True)


def test_missed_window_catches_up(eod_state, scheduler, monkeypatch):
    """Day before today, never marked done -> scheduler recovers it."""
    calls = []
    monkeypatch.setattr(m, "generate_and_maybe_email",
                        lambda conn, hint, email_on_complete=False, day=None: (calls.append(day) or "x.xlsx"))
    scheduler._run_eod(date(2026, 8, 31))
    assert calls == [date(2026, 8, 31)]