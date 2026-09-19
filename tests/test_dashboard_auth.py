"""Dashboard auth regression tests (2026-09-10 security audit).

Locks in three confirmed dashboard-auth defects:

  H1  : fail-closed dead code.  ``do_auth()`` returned ``"admin"`` (open
        dashboard) even when ``CCTV_DASH_FAIL_CLOSED=1`` and no password was
        configured, because ``_auth_enabled()`` short-circuited *before*
        ``_auth_fail_closed()``.  Fail-closed must gate first.
  H3  : empty admin password.  With only ``CCTV_DASH_VIEWER_PASS`` set, the
        old admin branch accepted ``user="admin", pw=""`` as admin because it
        matched ``pw == DASH_ADMIN_PASS`` with ``DASH_ADMIN_PASS=""``.
  H2  : ``_audit_report_export`` called ``logging.getLogger`` without
        ``import logging`` -- a ``NameError`` inside the very exception
        handler meant to survive audit-write failures.
"""

import logging
from unittest.mock import MagicMock

import pytest

import app
from src import database as db


# ----------------------------------------------------------------------
# H1 - fail-closed must block even when no password is configured
# ----------------------------------------------------------------------

def test_fail_closed_blocks_before_admin_shortcut(monkeypatch):
    monkeypatch.setattr(app, "DASH_AUTH_ENABLED", True)
    monkeypatch.setattr(app, "DASH_FAIL_CLOSED", True)
    monkeypatch.setattr(app, "DASH_ADMIN_PASS", "")
    monkeypatch.setattr(app, "DASH_VIEWER_PASS", "")
    # Regression: this used to return "admin" (unreachable fail-closed block).
    assert app.do_auth() == ""


def test_auth_fail_closed_requires_all_conditions(monkeypatch):
    monkeypatch.setattr(app, "DASH_AUTH_ENABLED", True)
    monkeypatch.setattr(app, "DASH_FAIL_CLOSED", True)
    monkeypatch.setattr(app, "DASH_ADMIN_PASS", "")
    monkeypatch.setattr(app, "DASH_VIEWER_PASS", "")
    assert app._auth_fail_closed() is True

    monkeypatch.setattr(app, "DASH_VIEWER_PASS", "view1")
    assert app._auth_fail_closed() is False   # a password exists -> login flow

    monkeypatch.setattr(app, "DASH_ADMIN_PASS", "adm1")
    monkeypatch.setattr(app, "DASH_VIEWER_PASS", "")
    assert app._auth_fail_closed() is False

    monkeypatch.setattr(app, "DASH_FAIL_CLOSED", False)   # documented dev mode
    monkeypatch.setattr(app, "DASH_VIEWER_PASS", "")
    assert app._auth_fail_closed() is False


def test_auth_disabled_dev_mode_still_returns_admin(monkeypatch):
    # Documented local-dev behaviour (auth intended but not enforced) is kept.
    monkeypatch.setattr(app, "DASH_AUTH_ENABLED", True)
    monkeypatch.setattr(app, "DASH_FAIL_CLOSED", False)
    monkeypatch.setattr(app, "DASH_ADMIN_PASS", "")
    monkeypatch.setattr(app, "DASH_VIEWER_PASS", "")
    assert app.do_auth() == "admin"


# ----------------------------------------------------------------------
# H3 - empty admin password never escalates to admin
# ----------------------------------------------------------------------

def test_authenticate_empty_admin_password_never_admin(monkeypatch):
    monkeypatch.setattr(app, "DASH_ADMIN_PASS", "")
    monkeypatch.setattr(app, "DASH_VIEWER_PASS", "view1")
    monkeypatch.setattr(app, "DASH_USERNAME", "admin")
    # Regression: blank user/pw used to match the empty admin password.
    assert app._authenticate("admin", "") == ""
    assert app._authenticate("admin", "view1") == "viewer"


def test_authenticate_requires_username_for_admin(monkeypatch):
    monkeypatch.setattr(app, "DASH_ADMIN_PASS", "adm1")
    monkeypatch.setattr(app, "DASH_VIEWER_PASS", "view1")
    monkeypatch.setattr(app, "DASH_USERNAME", "admin")
    assert app._authenticate("admin", "adm1") == "admin"
    assert app._authenticate("other", "adm1") == ""     # right pw, wrong user
    assert app._authenticate("admin", "") == ""


def test_authenticate_viewer_password_is_usernameless(monkeypatch):
    monkeypatch.setattr(app, "DASH_ADMIN_PASS", "adm1")
    monkeypatch.setattr(app, "DASH_VIEWER_PASS", "view1")
    monkeypatch.setattr(app, "DASH_USERNAME", "admin")
    assert app._authenticate("viewer", "view1") == "viewer"


# ----------------------------------------------------------------------
# H2 - report-export audit failure must never break the download
# ----------------------------------------------------------------------

def test_audit_report_export_survives_audit_write_failure(monkeypatch, caplog):
    conn = MagicMock()
    monkeypatch.setattr(app, "_open_write_conn", lambda: conn)

    def boom(*_args, **_kwargs):
        raise RuntimeError("audit db unavailable")

    monkeypatch.setattr(db, "audit", boom)
    with caplog.at_level(logging.WARNING, logger="cctv.dashboard"):
        app._audit_report_export("daily.xlsx", "viewer")
    assert "audit failed" in caplog.text
    conn.close.assert_called_once()


def test_audit_report_export_success_path_audits(monkeypatch):
    conn = MagicMock()
    monkeypatch.setattr(app, "_open_write_conn", lambda: conn)
    writes = []
    monkeypatch.setattr(db, "audit", lambda c, *a, **k: writes.append((a, k)))
    app._audit_report_export("daily.xlsx", "viewer")
    assert writes, "audit row expected on success"
    conn.close.assert_called_once()