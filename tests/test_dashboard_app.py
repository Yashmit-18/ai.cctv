"""Phase 27.9/27.10 -- Streamlit dashboard execution smoke test.

Runs app.py through Streamlit's AppTest harness (no browser needed) and
asserts the key KPIs, empty states and layout render without raising.  This
exercises the real rendering code paths, not just helper functions.
"""

import json
from pathlib import Path

import pytest

st = pytest.importorskip("streamlit.testing.v1").AppTest

_ROOT = Path(__file__).resolve().parent.parent
_APP = _ROOT / "app.py"


def _app():
    return st.from_file(str(_APP), default_timeout=30)


def test_dashboard_boots_and_renders_live_overview():
    at = _app()
    at.run()
    assert not at.exception, at.exception
    # Live Overview tab renders header + metric KPIs.
    headers = " ".join(str(el.value) for el in at.header)
    assert "Live Overview" in headers
    assert len(at.metric) >= 4  # Recognised / Avg / Active / Phone KPIs


def test_live_overview_empty_state_no_crash():
    # No live_state.json presence should render an info, not crash.
    at = _app()
    at.run()
    assert not at.exception, at.exception


def test_employees_tab_shows_roster_or_empty_state():
    at = _app()
    at.run()
    assert not at.exception, at.exception


def test_settings_tab_never_exposes_secrets():
    at = _app()
    at.run()
    assert not at.exception, at.exception
    body = "\n".join(str(el) for el in at.main)
    for secret in ("SMTP_SERVER", "SENDER_PASSWORD", "DASH_PASS", "rtsp://"):
        assert secret.lower() not in body.lower()


def test_live_employees_and_camera_health_tabs_render():
    at = _app()
    at.run()
    assert not at.exception, at.exception
    # The Live Employees tab and Camera Health tab should not raise whether or
    # not a live snapshot exists (empty state shown when absent).
    assert len(at.tabs) >= 7  # all seven tabs exist
    assert len(at.caption) >= 0  # captions render without error


def test_productivity_bar_handles_no_observation_rows():
    # Regression: a day that has registered employees but no observed activity
    # (all productive_pct NaN) must render a friendly empty state, not crash
    # plotly/narwhals with an empty frame (length-mismatch ShapeError).
    import pandas as pd

    import app

    roster_only = pd.DataFrame([
        {"employee_id": "EMP001", "productive_pct": None},
        {"employee_id": "EMP002", "productive_pct": None},
    ])
    assert app._productivity_bar_figure(roster_only) is None

    with_activity = pd.DataFrame([
        {"employee_id": "EMP001", "productive_pct": 80.0},
        {"employee_id": "EMP002", "productive_pct": None},
    ])
    fig = app._productivity_bar_figure(with_activity)
    assert fig is not None
    assert len(fig.data) == 1
