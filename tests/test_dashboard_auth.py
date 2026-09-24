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

import inspect
import logging
from unittest.mock import MagicMock

import pytest

import app
from src import database as db


# ----------------------------------------------------------------------
# Phase 64B -- cloud hardening regression (SECURITY BLOCKER)
# ----------------------------------------------------------------------
# Before this phase `CCTV_DASH_FAIL_CLOSED` defaulted to "0", so a bare cloud
# deployment with no dashboard passwords silently granted the `admin` role to
# unauthenticated visitors.  The default is now fail-closed: missing
# credentials never mean "admin".

def test_config_fail_closed_default_is_secure():
    """config default (no env override) must be fail-closed ON."""
    import importlib

    import config as _cfg

    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("CCTV_DASH_FAIL_CLOSED", raising=False)
        mp.delenv("CCTV_DASH_AUTH", raising=False)
        importlib.reload(_cfg)
        assert _cfg.DASH_AUTH_ENABLED is True
        assert _cfg.DASH_FAIL_CLOSED is True
    importlib.reload(_cfg)  # restore session environment state


def test_missing_credentials_default_fail_closed_blocks(monkeypatch):
    """64B: no admin/viewer credential + fail-closed => NOT admin."""
    monkeypatch.setattr(app, "DASH_AUTH_ENABLED", True)   # production default
    monkeypatch.setattr(app, "DASH_FAIL_CLOSED", True)    # hardened default
    monkeypatch.setattr(app, "DASH_ADMIN_PASS", "")
    monkeypatch.setattr(app, "DASH_VIEWER_PASS", "")
    monkeypatch.setattr(app, "DASH_USERNAME", "admin")
    assert app.do_auth() != "admin"
    assert app.do_auth() == ""
    assert app._authenticate("admin", "") == ""


def test_correct_admin_credentials_grant_admin(monkeypatch):
    """64B: only the configured admin username+password grants admin."""
    monkeypatch.setattr(app, "DASH_ADMIN_PASS", "64b-admin-dummy")
    monkeypatch.setattr(app, "DASH_VIEWER_PASS", "64b-viewer-dummy")
    monkeypatch.setattr(app, "DASH_USERNAME", "admin")
    assert app._authenticate("admin", "64b-admin-dummy") == "admin"


def test_correct_viewer_credentials_grant_viewer(monkeypatch):
    """64B: the shared viewer password grants read-only viewer."""
    monkeypatch.setattr(app, "DASH_ADMIN_PASS", "64b-admin-dummy")
    monkeypatch.setattr(app, "DASH_VIEWER_PASS", "64b-viewer-dummy")
    assert app._authenticate("anyone", "64b-viewer-dummy") == "viewer"


def test_incorrect_credentials_rejected(monkeypatch):
    """64B: wrong user/password combinations never grant a role."""
    monkeypatch.setattr(app, "DASH_ADMIN_PASS", "64b-admin-dummy")
    monkeypatch.setattr(app, "DASH_VIEWER_PASS", "64b-viewer-dummy")
    monkeypatch.setattr(app, "DASH_USERNAME", "admin")
    assert app._authenticate("admin", "wrong") == ""
    assert app._authenticate("nobody", "64b-admin-dummy") == ""
    assert app._authenticate("", "") == ""


def test_viewer_cannot_obtain_admin(monkeypatch):
    """64B: a viewer password must never double as an admin password."""
    monkeypatch.setattr(app, "DASH_ADMIN_PASS", "64b-admin-dummy")
    monkeypatch.setattr(app, "DASH_VIEWER_PASS", "64b-viewer-dummy")
    monkeypatch.setattr(app, "DASH_USERNAME", "admin")
    assert app._authenticate("admin", "64b-viewer-dummy") != "admin"


# ----------------------------------------------------------------------
# Phase 64C -- admin username is configurable (email-style) + credential
# edge cases.  NEVER use real deployment credentials in tests; the real
# admin password lives only in Streamlit Secrets.
# ----------------------------------------------------------------------

def test_email_style_admin_username_grants_admin(monkeypatch):
    """64C: the admin username is a config value; email-style works."""
    monkeypatch.setattr(app, "DASH_ADMIN_PASS", "64c-admin-dummy")
    monkeypatch.setattr(app, "DASH_VIEWER_PASS", "")
    monkeypatch.setattr(app, "DASH_USERNAME", "admin@example.local")
    assert app._authenticate("admin@example.local", "64c-admin-dummy") == "admin"
    assert app._authenticate("admin@example.local", "wrong-pw") == ""
    assert app._authenticate("someone@else.local", "64c-admin-dummy") == ""


def test_empty_username_rejected_with_valid_password(monkeypatch):
    """64C: blank username (admin path) is never admin."""
    monkeypatch.setattr(app, "DASH_ADMIN_PASS", "64c-admin-dummy")
    monkeypatch.setattr(app, "DASH_VIEWER_PASS", "64c-viewer-dummy")
    monkeypatch.setattr(app, "DASH_USERNAME", "admin@example.local")
    assert app._authenticate("", "64c-admin-dummy") == ""


def test_empty_password_rejected_with_valid_username(monkeypatch):
    """64C: blank password (admin path) is never admin."""
    monkeypatch.setattr(app, "DASH_ADMIN_PASS", "64c-admin-dummy")
    monkeypatch.setattr(app, "DASH_VIEWER_PASS", "64c-viewer-dummy")
    monkeypatch.setattr(app, "DASH_USERNAME", "admin@example.local")
    assert app._authenticate("admin@example.local", "") == ""


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


# ======================================================================
# Phase 66B -- auth RENDERING gate regression (mutually exclusive states)
# ----------------------------------------------------------------------
# Root cause fixed: ``st.fragment(run_every=...)`` auto-refresh timers kept
# firing in the deployed (Cloud) session AFTER logout and re-painted the
# authenticated page bodies (Live Overview KPIs, camera cards, security
# ledger) NEXT TO the freshly rendered login form, using the stale ``role``
# captured in the fragment closure.  The unauthenticated page can now never
# render authenticated content because:
#   * logout/expiry clears EVERY session marker via ``_clear_auth_session``;
#   * the fragment-wrapped tab bodies re-check ``_current_role()`` on every
#     execution and render nothing when no valid session exists.
# ======================================================================

_APP = __import__("pathlib").Path(__file__).resolve().parent.parent / "app.py"
_AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def _auth_env(monkeypatch, admin="ph66-admin", viewer="ph66-viewer",
              minutes: float = 15.0):
    """Enable auth for the AppTest-executed app.py copy AND the test module.

    ``AppTest.from_file`` re-executes ``app.py`` (a fresh ``from config
    import``), so the config module attributes must be patched for the app
    script; the ``app`` module attributes are patched too for unit helpers.
    """
    import config
    monkeypatch.setattr(config, "DASH_AUTH_ENABLED", True)
    monkeypatch.setattr(config, "DASH_ADMIN_PASS", admin)
    monkeypatch.setattr(config, "DASH_VIEWER_PASS", viewer)
    monkeypatch.setattr(config, "DASH_USERNAME", "admin")
    monkeypatch.setattr(config, "DASH_FAIL_CLOSED", True)
    monkeypatch.setattr(config, "DASH_SESSION_MINUTES", minutes)
    monkeypatch.setattr(app, "DASH_AUTH_ENABLED", True)
    monkeypatch.setattr(app, "DASH_ADMIN_PASS", admin)
    monkeypatch.setattr(app, "DASH_VIEWER_PASS", viewer)
    monkeypatch.setattr(app, "DASH_USERNAME", "admin")
    monkeypatch.setattr(app, "DASH_FAIL_CLOSED", True)
    monkeypatch.setattr(app, "DASH_SESSION_MINUTES", minutes)


def _html_body(at) -> str:
    """Concatenated html-block markup WITHOUT the always-shipped <style> block.

    Every ``st.html`` output embeds ``_COMPONENT_CSS`` (a stylesheet whose
    selectors legitimately mention p42-topbar / p42-session-card / etc.), so
    content assertions must only look at the markup after ``</style>``.
    """
    parts = []
    for _el in at.get("html"):
        try:
            raw = str(_el.proto.body)
            if "</style>" in raw:
                raw = raw.split("</style>")[-1]
            parts.append(raw)
        except AttributeError:  # pragma: no cover - defensive
            continue
    return "\n".join(parts)


def _ss_get(_at, key: str, default=None):
    """SafeSessionState has no .get(); read via in / [] instead."""
    return _at.session_state[key] if key in _at.session_state else default


def _login(_at, username: str, password: str):
    fields = {str(w.label): w for w in _at.text_input}
    if "Username" in fields:
        fields["Username"].set_value(username)
    fields["Password"].set_value(password)
    [b for b in _at.button if str(b.label) == "Sign In \u2192"][0].click()
    _at.run()
    _at.run()
    return _at


def _logout(_at):
    [b for b in _at.sidebar.button if "Log out" in str(b.label)][0].click()
    _at.run()
    return _at


def test_unauth_renders_login_ui_only_no_admin_shell(monkeypatch, tmp_path):
    """STATE A: fresh unauthenticated visit shows ONLY the login screen."""
    _auth_env(monkeypatch)
    monkeypatch.setenv("CCTV_LIVE_STATE_FILE", str(tmp_path / "missing.json"))
    at = _AppTest.from_file(str(_APP), default_timeout=90).run()
    assert not at.exception, at.exception

    # Login UI rendered exactly once.
    assert len({str(b.label) for b in at.button if "Sign In" in str(b.label)}) == 1
    assert "Welcome Back" in _html_body(at)

    # No authenticated application shell.
    assert not at.sidebar.radio                    # no navigation
    assert not at.metric                           # no KPIs
    body = _html_body(at)
    assert "Admin Control Center" not in body      # no admin nav / page
    assert "p42-topbar" not in body                # no topbar
    assert "p42-session-card" not in body          # no Role/System card
    assert "Live Overview" not in body             # no dashboard content
    assert "Role</span><span>Admin" not in body    # no stale admin identity


def test_admin_login_transitions_to_authenticated_state(monkeypatch, tmp_path):
    """STATE B (admin): credentials take a fresh session to the app."""
    _auth_env(monkeypatch)
    monkeypatch.setenv("CCTV_LIVE_STATE_FILE", str(tmp_path / "missing.json"))
    at = _AppTest.from_file(str(_APP), default_timeout=90).run()
    _login(at, "admin", "ph66-admin")
    assert not at.exception, at.exception
    assert _ss_get(at, "cctv_role") == "admin"
    assert len(at.sidebar.radio) == 1               # navigation returned
    assert "Live Overview" in " ".join(str(h.value) for h in at.header)
    body = _html_body(at)
    assert "p42-session-card" in body
    assert "Role</span><span>Admin" in body


def test_viewer_login_transitions_to_viewer_role(monkeypatch, tmp_path):
    """STATE B (viewer): viewer login is authenticated as read-only viewer."""
    _auth_env(monkeypatch)
    monkeypatch.setenv("CCTV_LIVE_STATE_FILE", str(tmp_path / "missing.json"))
    at = _AppTest.from_file(str(_APP), default_timeout=90).run()
    _login(at, "", "ph66-viewer")
    assert not at.exception, at.exception
    assert _ss_get(at, "cctv_role") == "viewer"
    assert len(at.sidebar.radio) == 1


def test_admin_logout_clears_all_markers_and_content(monkeypatch, tmp_path):
    """Logout: no session markers remain; login renders again; no admin UI."""
    _auth_env(monkeypatch)
    monkeypatch.setenv("CCTV_LIVE_STATE_FILE", str(tmp_path / "missing.json"))
    at = _AppTest.from_file(str(_APP), default_timeout=90).run()
    _login(at, "admin", "ph66-admin")
    _logout(at)
    assert not at.exception, at.exception

    for key in ("cctv_role", "cctv_login_ts"):
        assert key not in at.session_state
    assert _ss_get(at, "_p66_login_viewer") is None

    assert not at.sidebar.radio
    assert not at.metric
    body = _html_body(at)
    assert "Admin Control Center" not in body
    assert "p42-topbar" not in body
    assert "p42-session-card" not in body
    assert "Role</span><span>Admin" not in body
    assert "Live Overview" not in body
    assert any("Welcome Back" in t for t in body.split()) or \
        any("Sign In \u2192" in str(b.label) for b in at.button)


def test_viewer_logout_clears_all_markers_and_content(monkeypatch, tmp_path):
    """Viewer logout is symmetrical: clean unauthenticated login only."""
    _auth_env(monkeypatch)
    monkeypatch.setenv("CCTV_LIVE_STATE_FILE", str(tmp_path / "missing.json"))
    at = _AppTest.from_file(str(_APP), default_timeout=90).run()
    _login(at, "", "ph66-viewer")
    _logout(at)
    assert not at.exception, at.exception
    assert "cctv_role" not in at.session_state
    assert "cctv_login_ts" not in at.session_state
    assert not at.sidebar.radio
    assert not at.metric


def test_session_expiry_returns_to_clean_login_state(monkeypatch, tmp_path):
    """Expired session -> markers purged, login shown, only one login UI."""
    _auth_env(monkeypatch, minutes=15.0)
    monkeypatch.setenv("CCTV_LIVE_STATE_FILE", str(tmp_path / "missing.json"))
    at = _AppTest.from_file(str(_APP), default_timeout=90).run()
    _login(at, "admin", "ph66-admin")
    assert _ss_get(at, "cctv_role") == "admin"

    # Backdate the login timestamp past the session window, then rerun: the
    # session MUST return to the clean unauthenticated state.
    at.session_state["cctv_login_ts"] = __import__("time").time() - 3600
    at.run()
    assert not at.exception, at.exception
    assert "cctv_role" not in at.session_state
    assert "cctv_login_ts" not in at.session_state
    assert not at.sidebar.radio
    assert any("Session expired" in str(w.value) for w in at.warning)
    sign_ins = [b for b in at.button if "Sign In" in str(b.label)]
    assert len(sign_ins) == 1                       # login form only once


def test_fragment_wrapped_tab_bodies_are_auth_gated():
    """Auto-refresh fragments must no-op when unauthenticated.

    This is the Cloud-vector regression: a ``st.fragment(run_every=...)``
    timer survives logout and re-executes these bodies; without the gate it
    re-paints authenticated content next to the login form.
    """
    for fn in (app.tab_live_overview, app.page_live_monitoring,
               app.tab_security):
        src = inspect.getsource(fn)
        assert "if not _current_role():\n        return" in src, fn.__name__


def test_current_role_and_clear_auth_session_semantics(monkeypatch):
    """Session helpers: auth-disabled -> admin; empty -> ''; clears all."""
    # Suite default runs with CCTV_DASH_AUTH=0 -> open dev mode grants admin.
    assert app._auth_enabled() is False
    assert app._current_role() == "admin"
    assert app._clear_auth_session() is None        # idempotent

    # Auth enabled but no session -> genuinely unauthenticated ('').
    _auth_env(monkeypatch)
    assert app._current_role() == ""

    for key in ("cctv_role", "cctv_login_ts", "_p66_login_viewer"):
        assert key in app._AUTH_SESSION_KEYS