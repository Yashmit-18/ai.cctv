"""Phase 66 -- dashboard empty/data-state + premium landing/login UX.

Locks in the SOC data-state contract from the Phase 66 spec:

  * absent live data reads as a *system state*, never as fabricated/inflated
    numbers and never as a security incident;
  * "NO DATA != ZERO": KPIs show "—" (no data) instead of misleading zeros;
  * the daemon-offline dashboard copy is explicit: Dashboard ONLINE /
    CCTV Daemon OFFLINE / Live Data WAITING, plus the local-vs-cloud command
    block;
  * the productivity empty state shows the 3-step progression;
  * the security / analytics / reports empty states are premium and honest;
  * the landing hero (``?view=landing``) and the split-screen login keep the
    existing auth gate exactly (fail-closed and the admin/viewer matrix are
    untouched by the redesign).
"""

from __future__ import annotations

import datetime as _dt
import inspect

import pandas as pd
import pytest

import app

st = pytest.importorskip("streamlit.testing.v1").AppTest

_APP = __import__("pathlib").Path(__file__).resolve().parent.parent / "app.py"


# ----------------------------------------------------------------------
# Pure HTML builders (no Streamlit runtime required)
# ----------------------------------------------------------------------

def test_daemon_offline_html_status_trio_and_cloud_copy():
    h = app._daemon_offline_panel_html()
    assert "Live intelligence is currently offline" in h
    assert "STATUS: DAEMON OFFLINE" in h
    # status trio: Dashboard ONLINE / CCTV Daemon OFFLINE / Live Data WAITING
    assert ">ONLINE<" in h
    assert h.count(">OFFLINE<") >= 1
    assert ">WAITING<" in h
    assert "p66-dot-green" in h
    assert "p66-dot-amber" in h
    # a data-state card is muted/amber, never an alarm -- the status region
    # (after the CSS block) never uses a red dot or alarm copy
    body = h.split("</style>")[-1]
    assert "p66-dot-red" not in body
    assert "incident" not in body.lower()
    # Part 4: local-vs-cloud explanation
    assert "Streamlit Cloud provides the dashboard" in h
    assert "connected to the machine running the CCTV daemon" in h


def test_live_offline_html_has_four_statuses():
    h = app._live_offline_panel_html()
    assert "Live monitoring is unavailable" in h
    assert "INFERENCE: OFFLINE" in h
    assert ">ONLINE<" in h          # Dashboard ONLINE
    assert ">NOT CONNECTED<" in h   # Camera Feed NOT CONNECTED
    assert ">NONE<" in h            # Last Snapshot NONE
    assert "p66-dot-green" in h
    assert "p66-dot-red" not in h.split("</style>")[-1]  # muted, never red


def test_productivity_empty_html_has_3_step_progression():
    h = app._productivity_empty_html(pd.DataFrame())
    assert "NO PRODUCTIVITY DATA" in h
    assert "AWAITING DATA" in h
    for step in ("Add employees", "Start the CCTV daemon",
                 "Activity data appears"):
        assert step in h
    assert "<span class='p66-step-num'>01</span>" in h
    assert "<span class='p66-step-num'>02</span>" in h
    assert "<span class='p66-step-num'>03</span>" in h
    assert "No employees are configured" in h
    h2 = app._productivity_empty_html(
        pd.DataFrame([{"employee_id": "EMP001"}]))
    assert "Employees are registered" in h2
    assert "no observed activity has been logged yet" in h2


def test_analytics_period_empty_html_includes_range():
    h = app._analytics_period_empty_html(
        _dt.date(2026, 9, 20), _dt.date(2026, 9, 22))
    assert "No analytics data available for this period" in h
    assert "2026-09-20 → 2026-09-22" in h
    assert "NO DATA RECORDED" in h


def test_reports_empty_html_copy():
    h = app._reports_empty_html()
    assert "No report data available for the selected period" in h
    assert "NO REPORTS GENERATED" in h


# ----------------------------------------------------------------------
# NO DATA != ZERO -- observation vs roster semantics
# ----------------------------------------------------------------------

def test_has_observed_activity_never_confuses_roster_with_data():
    assert app._has_observed_activity(None) is False
    assert app._has_observed_activity(pd.DataFrame()) is False
    roster_only = pd.DataFrame([
        {"employee_id": "EMP001", "status": "NOT OBSERVED",
         "productive_pct": None, "active_hours": 0.0, "phone_mins": 0.0},
    ])
    assert app._has_observed_activity(roster_only) is False
    observed = pd.DataFrame([
        {"employee_id": "EMP001", "status": "OBSERVED",
         "productive_pct": 87.5, "active_hours": 1.2, "phone_mins": 3.0},
    ])
    assert app._has_observed_activity(observed) is True
    active_only = pd.DataFrame([
        {"employee_id": "EMP001", "status": "NOT OBSERVED",
         "productive_pct": None, "active_hours": 0.5, "phone_mins": 0.0},
    ])
    assert app._has_observed_activity(active_only) is True


# ----------------------------------------------------------------------
# Dashboard with the daemon offline (AppTest via the live-state seam)
# ----------------------------------------------------------------------

def _app():
    return st.from_file(str(_APP), default_timeout=60)


def test_apptest_daemon_offline_dashboard_shows_no_data_states(monkeypatch, tmp_path):
    monkeypatch.setenv("CCTV_LIVE_STATE_FILE", str(tmp_path / "missing-live.json"))
    at = _app()
    at.run()
    assert not at.exception, at.exception

    values = {str(el.label): str(el.value) for el in at.metric}
    # live-derived KPIs are "—" (no data), never misleading 0s
    assert values.get("People Active") == "—"
    assert values.get("Away") == "—"
    assert values.get("On Phone") == "—"
    assert values.get("Open Incidents") == "—"
    assert values.get("Cameras Online") == "—"
    assert all("n/a" not in str(el.value) for el in at.metric)

    # runnable daemon command is offered as copyable st.code
    assert any("python main.py --source webcam --headless" in str(el.value)
               for el in at.code)
    # legacy misleading empty copy is gone from the dashboard
    assert not any("No data available yet today" in str(el.value)
                   for el in at.info)
    assert len(at.get("html")) >= 3


def test_base_app_defaults_to_dashboard_not_landing(monkeypatch, tmp_path):
    monkeypatch.setenv("CCTV_LIVE_STATE_FILE", str(tmp_path / "missing-live.json"))
    at = _app()
    at.run()
    assert not at.exception, at.exception
    assert len(list(at.sidebar.radio[0].options)) == 10
    headers = " ".join(str(el.value) for el in at.header)
    assert "Live Overview" in headers
    assert not any(str(b.label) == "Open Dashboard" for b in at.button)


# ----------------------------------------------------------------------
# Landing hero (opt-in via ?view=landing)
# ----------------------------------------------------------------------

def test_landing_hero_renders_and_explore_toggles_features():
    at = _app()
    at.query_params["view"] = "landing"
    at.run()
    assert not at.exception, at.exception
    assert not at.sidebar.radio  # landing is pre-nav by design

    labels = {str(b.label) for b in at.button}
    assert any("Open Dashboard" in label for label in labels)
    assert any("Explore Capabilities" in label for label in labels)

    src = inspect.getsource(app._render_landing)
    assert "Smarter Surveillance." in src
    assert "Safer Workplaces." in src
    before = len(at.get("html"))
    buttons = [b for b in at.button if "Explore Capabilities" in str(b.label)]
    assert buttons
    buttons[0].click()
    at.run()
    assert not at.exception, at.exception
    assert len(at.get("html")) > before  # feature grid appeared


# ----------------------------------------------------------------------
# Split-screen login keeps the auth gate intact
# ----------------------------------------------------------------------

def test_login_split_screen_keeps_auth_gate_and_flow(monkeypatch):
    import config
    monkeypatch.setattr(config, "DASH_AUTH_ENABLED", True)
    monkeypatch.setattr(config, "DASH_ADMIN_PASS", "dummy-admin")
    monkeypatch.setattr(config, "DASH_VIEWER_PASS", "dummy-viewer")
    monkeypatch.setattr(config, "DASH_USERNAME", "admin")
    monkeypatch.setattr(config, "DASH_FAIL_CLOSED", False)
    monkeypatch.setattr(config, "DASH_SESSION_MINUTES", 30)

    at = _app()
    at.run()
    assert not at.exception, at.exception
    assert not at.sidebar.radio          # gated: no nav before login

    fields = {str(w.label): w for w in at.text_input}
    assert "Username" in fields
    assert "Password" in fields
    sign_in = [b for b in at.button if str(b.label) == "Sign In →"]
    assert sign_in
    assert any(str(b.label) == "Viewer Access" for b in at.button)

    # premium copy present in the login flow
    src = inspect.getsource(app.do_auth)
    assert "Welcome Back" in src
    assert "Smarter Surveillance." in src
    assert "Safer Workplaces." in src
    assert "Sign in to access your Security" in src
    assert "Operations Center" in src
    assert "Admin" in src and "Full system control" in src
    assert "Viewer" in src and "Monitoring and reports access" in src

    # wrong credentials stay gated
    fields["Username"].set_value("admin")
    fields["Password"].set_value("wrong")
    sign_in[0].click()
    at.run()
    assert not at.exception, at.exception
    assert any("Invalid credentials" in str(e.value) for e in at.error)
    assert not at.sidebar.radio

    # correct credentials land on the SOC dashboard
    fields = {str(w.label): w for w in at.text_input}
    fields["Username"].set_value("admin")
    fields["Password"].set_value("dummy-admin")
    [b for b in at.button if str(b.label) == "Sign In →"][0].click()
    at.run()
    at.run()
    assert not at.exception, at.exception
    assert at.sidebar.radio
    headers = " ".join(str(el.value) for el in at.header)
    assert "Live Overview" in headers


def test_login_viewer_access_routes_to_viewer_role(monkeypatch):
    import config
    monkeypatch.setattr(config, "DASH_AUTH_ENABLED", True)
    monkeypatch.setattr(config, "DASH_ADMIN_PASS", "dummy-admin")
    monkeypatch.setattr(config, "DASH_VIEWER_PASS", "dummy-viewer")
    monkeypatch.setattr(config, "DASH_USERNAME", "admin")
    monkeypatch.setattr(config, "DASH_FAIL_CLOSED", False)
    monkeypatch.setattr(config, "DASH_SESSION_MINUTES", 30)

    at = _app()
    at.run()
    assert not at.exception, at.exception
    assert not at.sidebar.radio

    # "Viewer Access" hints at the viewer path, still routes through the SAME gate
    viewer_btn = [b for b in at.button if str(b.label) == "Viewer Access"]
    assert viewer_btn
    viewer_btn[0].click()
    at.run()
    assert not at.exception, at.exception
    assert any("Viewer sign-in" in str(c.value) for c in at.caption)

    fields = {str(w.label): w for w in at.text_input}
    fields["Password"].set_value("dummy-viewer")
    [b for b in at.button if str(b.label) == "Sign In →"][0].click()
    at.run()
    at.run()
    assert not at.exception, at.exception
    assert at.sidebar.radio
    headers = " ".join(str(el.value) for el in at.header)
    assert "Live Overview" in headers


def test_auth_helpers_unchanged_by_redesign():
    # Real (CCTV_DASH_AUTH=0) environment: open to admin, no fail-closed.
    assert app._auth_enabled() is False
    assert app._auth_fail_closed() is False
    # Without a configured password, blank credentials never escalate.
    assert app._authenticate("", "") == ""
    assert app._authenticate("admin", "") == ""


# ----------------------------------------------------------------------
# Page empty states + architecture copy
# ----------------------------------------------------------------------

def test_cameras_offline_chips_copy_present():
    src = inspect.getsource(app._render_camera_cards)
    assert "CAMERA CONFIGURATION READY" in src
    assert "CCTV INFERENCE OFFLINE" in src
    assert "No cameras configured" in src
    assert "Camera Management" in src
    assert "office CCTV/IP camera" in src


def test_security_empty_and_offline_copy_present():
    src = inspect.getsource(app.tab_security)
    assert "No security events detected" in src
    assert "MONITORING STATUS: OFFLINE" in src
    assert "NO INCIDENT DATA" in src


def test_deployment_labels_dashboard_cloud_vs_inference_edge():
    src = inspect.getsource(app.tab_deployment)
    assert "CLOUD" in src
    assert "OFFICE / EDGE MACHINE" in src
    assert "Streamlit Cloud provides the dashboard" in src or \
        "runs in Streamlit Cloud" in src


def test_landing_and_login_hero_make_no_unsupported_claims():
    src = inspect.getsource(app._render_landing) + inspect.getsource(app.do_auth)
    assert "Smarter Surveillance." in src
    assert "Safer Workplaces." in src
    # landing features (4)
    for feat in ("AI-Powered Detection", "Real-Time Monitoring",
                 "Employee Intelligence", "Enterprise Security"):
        assert feat in src
    # login features (3)
    for feat in ("Secure Access", "Real-Time Intelligence",
                 "Enterprise Security"):
        assert feat in src
    for banned in ("uptime", "99.9", "reduces", "guaranteed"):
        assert banned not in src.lower()


# ----------------------------------------------------------------------
# Part 18 -- performance & cloud-safety boundaries stay intact
# ----------------------------------------------------------------------

def test_performance_boundaries_intact():
    src = inspect.getsource(app)
    assert "@st.cache_data(ttl=DASH_DAY_METRICS_TTL_SEC" in src
    assert "@st.cache_data(ttl=DASH_RANGE_METRICS_TTL_SEC" in src
    assert "def get_readonly_connection" in src
    assert "load_live_state" in src            # live snapshot stays uncached
    # No MODULE-LEVEL OpenCV import (startup cost).  A lazy, in-function
    # ``import cv2`` is allowed: _validate_enrollment_bytes() decodes bytes at
    # upload time, and the face model via src.face_registry is also deferred.
    top_level = (l for l in src.splitlines() if l and not l[0].isspace())
    assert not any("import cv2" in l for l in top_level)


# ----------------------------------------------------------------------
# Phase 66 premium SOC dashboard additions (full spec, Part 1-30)
# ----------------------------------------------------------------------

def test_system_status_html_separates_component_states():
    # daemon offline + DB fine + nothing configured
    h = app._system_status_html({}, True, 0, None)
    assert "System Status" in h
    assert "Dashboard" in h and "ONLINE" in h
    assert "Database" in h and "CONNECTED" in h
    assert "CCTV Daemon" in h and "OFFLINE" in h
    assert "Live Data" in h and "WAITING" in h
    assert "Camera Runtime" in h and "NOT CONFIGURED" in h
    assert ">NO SNAPSHOT<" in h          # chip
    # a data-state card stays muted: no red anything in the status rows
    assert "p66-dot-red" not in h.split("</style>")[-1]

    # daemon live + db fine + runtime ready
    live = {"camera_health": {"cam1": {"health": "ONLINE"}}}
    h2 = app._system_status_html(live, True, 1, None)
    assert ">LIVE<" in h2
    assert "ONLINE" in h2                 # daemon row
    assert "AVAILABLE" in h2              # live data row
    assert "READY" in h2                  # camera runtime row

    # db problem + stale snapshot => DEGRADED, honest per component
    h3 = app._system_status_html(None, False, 1, 123.0)
    assert ">STALE<" in h3
    assert "FAILED" in h3
    assert "UNAVAILABLE" in h3
    assert "NOT CONFIGURED" not in h3


def test_system_status_renderer_used_on_dashboard(monkeypatch, tmp_path):
    src = inspect.getsource(app.tab_live_overview)
    assert "_render_system_status(live)" in src
    assert "_p66_page_head_html(" in src
    monkeypatch.setenv("CCTV_LIVE_STATE_FILE", str(tmp_path / "missing-live.json"))
    at = _app()
    at.run()
    assert not at.exception, at.exception
    html_parts = []
    for _el in at.get("html"):
        try:
            html_parts.append(str(_el.proto.body))
        except AttributeError:  # some html blocks surface without a readable body
            continue
    html_src = " ".join(html_parts)
    assert "Real-time overview of your security and workplace intelligence" in html_src
    assert "System Status" in html_src
    assert "DAEMON OFFLINE" in html_src     # topbar system chip (real state)


def test_empty_state_component_renders_kinds_and_escapes():
    h = app._p66_empty_state_html(
        "No cameras configured", "Add your camera.",
        kind="NOT_CONFIGURED", action="Open Camera Management")
    assert "p66-empty" in h
    assert "p66-empty-ic" in h
    assert "No cameras configured" in h
    assert "Open Camera Management" in h
    assert "<div class='p66-empty'>" in h        # normal card, no warn class
    hw = app._p66_empty_state_html("Engine error", "bad input", kind="ERROR")
    assert "<div class='p66-empty p66-empty-warn'>" in hw
    assert "error_outline" in hw
    hx = app._p66_empty_state_html("<script>", "& < ok")
    assert "&lt;script&gt;" in hx
    assert "<script>" not in hx


def test_error_card_has_support_copy_and_diagnostic():
    h = app._error_card_html("Something went wrong", "boom")
    assert "Something went wrong" in h
    assert "Diagnostic ID:" in h
    assert "DIAG-" in h
    assert "Please check System Check or contact an administrator." in h
    assert "boom" in h


def test_topbar_and_sidebar_tokens_present():
    src = inspect.getsource(app._render_topbar)
    for token in ("SYSTEM ONLINE", "DAEMON OFFLINE", "SYSTEM DEGRADED",
                  "p66-avatar", "Notifications"):
        assert token in src
    src2 = inspect.getsource(app.main)
    assert "p42-sb-groups" in src2
    assert "Monitoring" in src2 and "People" in src2 \
        and "Security" in src2 and "Infrastructure" in src2


def test_camera_cards_feed_offline_placeholder_and_location():
    src = inspect.getsource(app._render_camera_cards)
    assert "CAMERA FEED OFFLINE" in src
    assert "Waiting for the CCTV inference daemon." in src
    assert "Location" in src


def test_live_employees_table_has_premium_columns():
    src = inspect.getsource(app.tab_live_employees)
    for col in ('"Desk"', '"Chair"', '"Confidence"', '"Productivity"',
                '"Employee"', '"Status"', '"Last Seen"'):
        assert col in src


def test_security_page_premium_kpis_and_empties():
    src = inspect.getsource(app.tab_security)
    assert 'metric("Events Today"' in src
    assert 'metric("Open Incidents"' in src
    assert 'metric("High Severity"' in src
    assert 'metric("Critical"' in src
    assert "_p66_empty_state_html(" in src
    assert "_severity_color" in src
    assert "MONITORING STATUS: OFFLINE" in src
    assert "NO INCIDENT DATA" in src


def test_reports_page_range_and_summary_kpis():
    src = inspect.getsource(app.tab_reports)
    for kpi in ("Attendance", "Productivity", "Phone Usage",
                "Security Events", "Incidents"):
        assert f'"{kpi}"' in src
    assert '"rep_from"' in src
    assert '"rep_to"' in src
    assert "selected range" in src


def test_analytics_page_polish_tokens():
    src = inspect.getsource(app.tab_historical)
    for token in ("_style_fig", "Phone Usage (minutes)",
                  "Employee Activity", "Security Events in Range",
                  "Attendance", "Productivity", "Tracked Days"):
        assert token in src
    # empty-data path stays honest via the premium panel + component
    h = app._analytics_period_empty_html(_dt.date(2026, 9, 20), _dt.date(2026, 9, 22))
    assert "NO DATA RECORDED" in h


def test_settings_sections_and_landing_illustration():
    src = inspect.getsource(app.tab_settings)
    for sec in ("Dashboard", "Authentication", "Cameras", "Metrics & Detection",
                "Office Schedule", "Security & Privacy", "Alerts",
                "System & Storage"):
        assert sec in src
    landing = inspect.getsource(app._render_landing)
    assert "p66-illus" in landing
    assert "ILLUSTRATION" in landing
    assert "Employee Intelligence" in landing
    assert "Enterprise Security" in landing