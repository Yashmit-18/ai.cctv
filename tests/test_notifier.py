"""Tests for bounded email retry and visible delivery status."""

import smtplib

import pytest

import src.notifier as N


@pytest.fixture(autouse=True)
def reset_status():
    N._set_status(ok=None, report=None, attempts=0, message="", at=None)


def _make_report(tmp_path):
    p = tmp_path / "daily_report_2026-09-01.xlsx"
    p.write_bytes(b"PK\x03\x04 dummy xlsx")
    return str(p)


def test_multiple_attempts_then_success(monkeypatch, tmp_path):
    report = _make_report(tmp_path)
    calls = {"n": 0}

    def flaky_send_once(rp, recipients, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            raise smtplib.SMTPRecipientsRefused({})
        return True

    monkeypatch.setattr(N, "_send_once", flaky_send_once)
    monkeypatch.setattr(N, "time", __import__("time"))
    ok = N.send_daily_report(report, recipient_emails=["a@example.com"],
                             max_retries=4, retry_delay=0)
    assert ok is True
    assert calls["n"] == 3            # two failures then success
    st = N.last_email_status()
    assert st["ok"] is True and st["attempts"] == 3


def test_exhaustion_marks_failed(monkeypatch, tmp_path):
    report = _make_report(tmp_path)
    calls = {"n": 0}

    def always_fail(rp, recipients, timeout):
        calls["n"] += 1
        raise smtplib.SMTPServerDisconnected("gone")

    monkeypatch.setattr(N, "_send_once", always_fail)
    ok = N.send_daily_report(report, recipient_emails=["a@example.com"],
                             max_retries=2, retry_delay=0)
    assert ok is False
    assert calls["n"] == 2            # exactly max retries, not infinite
    st = N.last_email_status()
    assert st["ok"] is False and st["attempts"] == 2


def test_non_retryable_bails_immediately(monkeypatch, tmp_path):
    report = _make_report(tmp_path)
    calls = {"n": 0}

    def local_error(rp, recipients, timeout):
        calls["n"] += 1
        raise ValueError("bad encoding")   # not SMTP/OSError -> non-retryable

    monkeypatch.setattr(N, "_send_once", local_error)
    ok = N.send_daily_report(report, recipient_emails=["a@example.com"],
                             max_retries=5, retry_delay=0)
    assert ok is False
    assert calls["n"] == 1            # non-retryable short-circuits


def test_missing_sender_short_circuits(monkeypatch):
    monkeypatch.setattr(N, "SENDER_EMAIL", "")
    ok = N.send_daily_report("x.xlsx", recipient_emails=[], max_retries=3)
    assert ok is False
    assert N.last_email_status()["ok"] is False


def test_missing_report_file(monkeypatch, tmp_path):
    ok = N.send_daily_report(str(tmp_path / "nope.xlsx"),
                             recipient_emails=["a@example.com"],
                             max_retries=3)
    assert ok is False
    assert N.last_email_status()["ok"] is False
    assert "not found" in N.last_email_status()["message"]