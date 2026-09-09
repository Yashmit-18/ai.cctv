"""SQLite database access layer.

Schema
------
* ``employees``     -- enrolled employees (metadata, active flag, schedule).
* ``activity_logs`` -- per-interval state rows with a normalised ``date``
  column and an optional ``source`` (camera id).

Thread safety
-------------
The daemon owns a single write connection on the main thread; reader threads
(Streamlit dashboard) use read-only URI connections.  WAL mode allows one
writer + concurrent readers without blocking.  ``init_db`` performs a safe
schema migration so existing databases gain the new columns/indexes.
"""

import logging
import os
import sqlite3

from config import DB_PATH

logger = logging.getLogger("cctv.database")


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    db_dir = os.path.dirname(os.path.abspath(db_path))
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    logger.debug("Opened database: %s", db_path)
    return conn


EMPLOYEES_SCHEMA = """
    CREATE TABLE IF NOT EXISTS employees (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id TEXT NOT NULL UNIQUE,
        name        TEXT NOT NULL DEFAULT '',
        department  TEXT NOT NULL DEFAULT '',
        designation TEXT NOT NULL DEFAULT '',
        active      INTEGER NOT NULL DEFAULT 1,
        enrolled    INTEGER NOT NULL DEFAULT 0,
        created_at  TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
        updated_at  TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
    )
"""

ACTIVITY_SCHEMA = """
    CREATE TABLE IF NOT EXISTS activity_logs (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp         TEXT NOT NULL,
        date              TEXT,
        employee_id       TEXT NOT NULL,
        state             TEXT NOT NULL,
        duration_seconds  REAL,
        source            TEXT
    )
"""

INDEXES = """
    CREATE INDEX IF NOT EXISTS idx_activity_emp ON activity_logs(employee_id);
    CREATE INDEX IF NOT EXISTS idx_activity_date ON activity_logs(date);
    CREATE INDEX IF NOT EXISTS idx_activity_state ON activity_logs(state);
    CREATE INDEX IF NOT EXISTS idx_activity_emp_date ON activity_logs(employee_id, date);
    CREATE INDEX IF NOT EXISTS idx_employees_emp ON employees(employee_id);
"""

# ======================================================================
# Phase 31 -- security schema  (additive, canonical)
# ======================================================================

SECURITY_EVENTS_SCHEMA = """
    CREATE TABLE IF NOT EXISTS security_events (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        incident_id       TEXT,
        event_type        TEXT NOT NULL,
        severity          TEXT NOT NULL,
        timestamp         TEXT NOT NULL,
        date              TEXT,
        camera            TEXT,
        zone              TEXT,
        employee_id       TEXT,
        person_count      INTEGER,
        confidence        REAL,
        duration_seconds  REAL,
        status            TEXT,
        evidence_ref      TEXT,
        details           TEXT,
        created_at        TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
    )
"""

INCIDENTS_SCHEMA = """
    CREATE TABLE IF NOT EXISTS incidents (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        incident_id   TEXT UNIQUE NOT NULL,
        event_type    TEXT NOT NULL,
        severity      TEXT NOT NULL,
        status        TEXT NOT NULL DEFAULT 'OPEN',
        first_seen    TEXT NOT NULL,
        last_seen     TEXT,
        count         INTEGER NOT NULL DEFAULT 1,
        camera        TEXT,
        zone          TEXT,
        employee_id   TEXT,
        confidence    REAL,
        evidence_ref  TEXT,
        notes         TEXT NOT NULL DEFAULT '',
        resolved_at   TEXT,
        created_at    TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
    )
"""

ALERT_RULES_SCHEMA = """
    CREATE TABLE IF NOT EXISTS alert_rules (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        rule_name     TEXT UNIQUE NOT NULL,
        event_type    TEXT DEFAULT '',
        min_severity  TEXT NOT NULL DEFAULT 'MEDIUM',
        channels      TEXT NOT NULL DEFAULT 'dashboard,log',
        cooldown_sec  INTEGER NOT NULL DEFAULT 300,
        enabled       INTEGER NOT NULL DEFAULT 1,
        schedule_start TEXT,
        schedule_end  TEXT,
        recipients    TEXT DEFAULT ''
    )
"""

ALERTS_SCHEMA = """
    CREATE TABLE IF NOT EXISTS alerts (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at   TEXT NOT NULL,
        rule_name    TEXT,
        event_type   TEXT,
        severity     TEXT,
        camera       TEXT,
        zone         TEXT,
        employee_id  TEXT,
        incident_id  TEXT,
        channel      TEXT,
        message      TEXT,
        delivered    INTEGER DEFAULT 0
    )
"""

AUDIT_LOG_SCHEMA = """
    CREATE TABLE IF NOT EXISTS audit_log (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp  TEXT NOT NULL,
        actor      TEXT NOT NULL DEFAULT 'system',
        action     TEXT NOT NULL,
        resource   TEXT,
        detail     TEXT
    )
"""

SECURITY_ZONES_SCHEMA = """
    CREATE TABLE IF NOT EXISTS security_zones (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        zone_name        TEXT UNIQUE NOT NULL,
        camera           TEXT NOT NULL,
        enabled          INTEGER NOT NULL DEFAULT 1,
        polygon          TEXT NOT NULL,
        allowed_employees TEXT NOT NULL DEFAULT '',
        schedule         TEXT,
        alert_policy     TEXT NOT NULL DEFAULT 'INTRUSION',
        created_at       TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
    )
"""

EVIDENCE_FILES_SCHEMA = """
    CREATE TABLE IF NOT EXISTS evidence_files (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        incident_id  TEXT NOT NULL,
        path         TEXT NOT NULL,
        kind         TEXT,
        captured_at  TEXT,
        size_bytes   INTEGER
    )
"""

SECURITY_INDEXES = """
    CREATE INDEX IF NOT EXISTS idx_sec_type ON security_events(event_type);
    CREATE INDEX IF NOT EXISTS idx_sec_date ON security_events(date);
    CREATE INDEX IF NOT EXISTS idx_sec_incident ON security_events(incident_id);
    CREATE INDEX IF NOT EXISTS idx_sec_camera ON security_events(camera);
    CREATE INDEX IF NOT EXISTS idx_sec_zone ON security_events(zone);
    CREATE INDEX IF NOT EXISTS idx_inc_status ON incidents(status);
    CREATE INDEX IF NOT EXISTS idx_inc_type ON incidents(event_type);
    CREATE INDEX IF NOT EXISTS idx_inc_first_seen ON incidents(first_seen);
    CREATE INDEX IF NOT EXISTS idx_alert_created ON alerts(created_at);
    CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(timestamp);
    CREATE INDEX IF NOT EXISTS idx_zones_camera ON security_zones(camera);
    CREATE INDEX IF NOT EXISTS idx_evidence_incident ON evidence_files(incident_id);
"""

# ======================================================================
# Phase 32 -- investigation / intelligence schema (additive, Phase 31 re-use)
# ======================================================================

INCIDENT_EVENTS_SCHEMA = """
    CREATE TABLE IF NOT EXISTS incident_events (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        incident_id  TEXT NOT NULL,
        event_id     INTEGER NOT NULL,
        event_type   TEXT,
        camera       TEXT,
        zone         TEXT,
        correlation_reason TEXT,
        linked_at    TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
    )
"""

INCIDENT_NOTES_SCHEMA = """
    CREATE TABLE IF NOT EXISTS incident_notes (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        incident_id  TEXT NOT NULL,
        author       TEXT NOT NULL DEFAULT 'system',
        note         TEXT NOT NULL,
        created_at   TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
    )
"""

CAMERA_CONFIG_SCHEMA = """
    CREATE TABLE IF NOT EXISTS camera_config (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        camera_id    TEXT UNIQUE NOT NULL,
        name         TEXT NOT NULL DEFAULT '',
        location     TEXT NOT NULL DEFAULT '',
        zone         TEXT NOT NULL DEFAULT '',
        source       TEXT NOT NULL DEFAULT '',
        enabled      INTEGER NOT NULL DEFAULT 1,
        fps_target   INTEGER,
        schedule     TEXT,
        notes        TEXT NOT NULL DEFAULT '',
        created_at   TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
    )
"""

CAMERA_HEALTH_SCHEMA = """
    CREATE TABLE IF NOT EXISTS camera_health (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        camera_id    TEXT NOT NULL,
        health       TEXT NOT NULL,
        fps          REAL,
        frames_read  INTEGER,
        reconnects   INTEGER,
        last_frame   REAL,
        downtime_sec REAL,
        tampered     INTEGER NOT NULL DEFAULT 0,
        recorded_at  TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
    )
"""

DETECTOR_REGISTRY_SCHEMA = """
    CREATE TABLE IF NOT EXISTS detector_registry (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        detector_id    TEXT UNIQUE NOT NULL,
        name           TEXT NOT NULL DEFAULT '',
        version        TEXT NOT NULL DEFAULT '',
        enabled        INTEGER NOT NULL DEFAULT 1,
        model_path     TEXT NOT NULL DEFAULT '',
        event_types    TEXT NOT NULL DEFAULT '',
        conf_threshold REAL,
        persist_sec    REAL,
        cooldown_sec   REAL,
        camera_scope   TEXT NOT NULL DEFAULT '',
        zone_scope     TEXT NOT NULL DEFAULT '',
        status         TEXT NOT NULL DEFAULT 'UNAVAILABLE',
        status_detail  TEXT NOT NULL DEFAULT '',
        updated_at     TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
    )
"""

PHASE32_INDEXES = """
    CREATE INDEX IF NOT EXISTS idx_incev_incident ON incident_events(incident_id);
    CREATE INDEX IF NOT EXISTS idx_incev_event ON incident_events(event_id);
    CREATE INDEX IF NOT EXISTS idx_incnotes_incident ON incident_notes(incident_id);
    CREATE INDEX IF NOT EXISTS idx_camhealth_camera ON camera_health(camera_id);
    CREATE INDEX IF NOT EXISTS idx_camhealth_time ON camera_health(recorded_at);
"""

# ======================================================================
# Phase 33 -- AI intelligence / pilot-hardening schema (additive)
# ======================================================================

# A3 -- camera topology graph (admin-configured, NOT auto-inferred).
CAMERA_TOPOLOGY_SCHEMA = """
    CREATE TABLE IF NOT EXISTS camera_topology (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        from_camera  TEXT NOT NULL,
        to_camera    TEXT NOT NULL,
        relation     TEXT NOT NULL DEFAULT 'NEXT',
        note         TEXT NOT NULL DEFAULT '',
        enabled      INTEGER NOT NULL DEFAULT 1,
        created_at   TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
    )
"""

# A6 -- evidence access audit trail (who viewed / exported / verified what).
EVIDENCE_ACCESS_SCHEMA = """
    CREATE TABLE IF NOT EXISTS evidence_access (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        evidence_id  INTEGER NOT NULL,
        incident_id  TEXT,
        action       TEXT NOT NULL,
        actor        TEXT NOT NULL DEFAULT 'system',
        detail       TEXT,
        accessed_at  TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
    )
"""

# B1 -- explainable anomaly events (advisory only; never accusatory).
ANOMALY_EVENTS_SCHEMA = """
    CREATE TABLE IF NOT EXISTS anomaly_events (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        anomaly_type TEXT NOT NULL,
        scope        TEXT NOT NULL,          -- zone / camera / campus
        scope_value  TEXT NOT NULL,
        severity     TEXT NOT NULL,
        reason       TEXT NOT NULL,
        data         TEXT,
        observed_at  TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
    )
"""

# B2 -- behavioral baselines (INSUFFICIENT_DATA until enough samples).
BEHAVIORAL_BASELINES_SCHEMA = """
    CREATE TABLE IF NOT EXISTS behavioral_baselines (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        scope        TEXT NOT NULL,
        scope_value  TEXT NOT NULL,
        metric       TEXT NOT NULL,
        mean         REAL,
        std          REAL,
        sample_count INTEGER NOT NULL DEFAULT 0,
        status       TEXT NOT NULL DEFAULT 'INSUFFICIENT_DATA',
        updated_at   TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
        UNIQUE(scope, scope_value, metric)
    )
"""

# B4 -- transparent per-incident security-risk score (incidents, never people).
INCIDENT_RISK_SCHEMA = """
    CREATE TABLE IF NOT EXISTS incident_risk (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        incident_id  TEXT NOT NULL,
        score        REAL NOT NULL,
        components   TEXT,
        rationale    TEXT,
        scored_at    TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
    )
"""

PHASE33_INDEXES = """
    CREATE INDEX IF NOT EXISTS idx_topology_from ON camera_topology(from_camera);
    CREATE INDEX IF NOT EXISTS idx_topology_to ON camera_topology(to_camera);
    CREATE INDEX IF NOT EXISTS idx_evaccess_evidence ON evidence_access(evidence_id);
    CREATE INDEX IF NOT EXISTS idx_anomaly_scope ON anomaly_events(scope, scope_value);
    CREATE INDEX IF NOT EXISTS idx_baseline_scope ON behavioral_baselines(scope, scope_value);
    CREATE INDEX IF NOT EXISTS idx_incrisk_incident ON incident_risk(incident_id);
"""

# ======================================================================
# Phase A/B schema (core vision + entry/exit + motion)  (additive)
# ======================================================================
# Motion events are advisory (frame-level activity); they are correlated into
# incidents only when genuinely related to a threatened context.  Stored
# separately from security_events so raw motion is never misread as a
# security alarm.
MOTION_EVENTS_SCHEMA = """
    CREATE TABLE IF NOT EXISTS motion_events (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        camera     TEXT NOT NULL,
        timestamp  TEXT NOT NULL,
        date       TEXT,
        score      REAL NOT NULL,
        bbox_x1    REAL,
        bbox_y1    REAL,
        bbox_x2    REAL,
        bbox_y2    REAL,
        duration_seconds REAL NOT NULL DEFAULT 0,
        event_id   TEXT
    )
"""

# Entry/exit line-crossing events (PHASE A A10 + B6).  ``track_id`` ties the
# crossing to the spatial person track (identity is separate and smoothed by
# the tracker -- never derived from a single frame).
ENTRY_EXIT_EVENTS_SCHEMA = """
    CREATE TABLE IF NOT EXISTS entry_exit_events (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        camera       TEXT NOT NULL,
        timestamp    TEXT NOT NULL,
        date         TEXT,
        event_type   TEXT NOT NULL,       -- ENTRY_CROSSING / EXIT_CROSSING
        line         TEXT NOT NULL DEFAULT '',
        direction    TEXT NOT NULL DEFAULT '',
        track_id     TEXT NOT NULL,
        employee_id  TEXT,
        confidence   REAL,
        security_event_id INTEGER
    )
"""

PHASE_AB_INDEXES = """
    CREATE INDEX IF NOT EXISTS idx_motion_cam_time ON motion_events(camera, timestamp);
    CREATE INDEX IF NOT EXISTS idx_ee_cam_time ON entry_exit_events(camera, timestamp);
    CREATE INDEX IF NOT EXISTS idx_ee_track ON entry_exit_events(track_id);
"""


def _migrate_security_phase32(conn: sqlite3.Connection):
    """Add Phase 32 columns to existing Phase 31 tables when missing."""
    # incidents: correlation + investigation metadata
    inc_cols = _table_columns(conn, "incidents")
    _add_col(conn, "incidents", inc_cols, "correlation_ids", "TEXT DEFAULT ''")
    _add_col(conn, "incidents", inc_cols, "review_state", "TEXT DEFAULT 'DETECTED'")
    _add_col(conn, "incidents", inc_cols, "acknowledged_at", "TEXT")
    _add_col(conn, "incidents", inc_cols, "resolved_ts", "TEXT")
    _add_col(conn, "incidents", inc_cols, "updated_at", "TEXT")

    # evidence_files: integrity + availability metadata
    ev_cols = _table_columns(conn, "evidence_files")
    _add_col(conn, "evidence_files", ev_cols, "sha256", "TEXT")
    _add_col(conn, "evidence_files", ev_cols, "mime", "TEXT")
    _add_col(conn, "evidence_files", ev_cols, "capture_type", "TEXT")
    _add_col(conn, "evidence_files", ev_cols, "camera", "TEXT")
    _add_col(conn, "evidence_files", ev_cols, "available", "INTEGER DEFAULT 1")
    _add_col(conn, "evidence_files", ev_cols, "retention_deadline", "TEXT")

    # alerts: lifecycle + delivery status
    al_cols = _table_columns(conn, "alerts")
    _add_col(conn, "alerts", al_cols, "status", "TEXT DEFAULT 'SENT'")
    _add_col(conn, "alerts", al_cols, "attempts", "INTEGER DEFAULT 1")
    _add_col(conn, "alerts", al_cols, "acknowledged_at", "TEXT")
    _add_col(conn, "alerts", al_cols, "resolved_at", "TEXT")
    _add_col(conn, "alerts", al_cols, "escalated", "INTEGER DEFAULT 0")
    _add_col(conn, "alerts", al_cols, "error", "TEXT DEFAULT ''")
    _add_col(conn, "alerts", al_cols, "last_attempt_epoch", "REAL DEFAULT 0")

    conn.commit()


def _migrate_security_phase33(conn: sqlite3.Connection):
    """Add Phase 33 columns to existing tables when missing."""
    # incidents: temporal intelligence state (A1).
    inc_cols = _table_columns(conn, "incidents")
    _add_col(conn, "incidents", inc_cols, "temporal_state", "TEXT DEFAULT 'FIRST_SEEN'")
    _add_col(conn, "incidents", inc_cols, "occurrences", "INTEGER DEFAULT 1")

    # detector_registry: model/health versioning (B9).
    det_cols = _table_columns(conn, "detector_registry")
    _add_col(conn, "detector_registry", det_cols, "last_inference_at", "TEXT")
    _add_col(conn, "detector_registry", det_cols, "last_run_at", "TEXT")
    _add_col(conn, "detector_registry", det_cols, "last_run_ok", "INTEGER DEFAULT 1")
    _add_col(conn, "detector_registry", det_cols, "last_error", "TEXT DEFAULT ''")
    _add_col(conn, "detector_registry", det_cols, "inference_count", "INTEGER DEFAULT 0")

    # camera_config: reliability scoring (B7).
    cam_cols = _table_columns(conn, "camera_config")
    _add_col(conn, "camera_config", cam_cols, "coverage_note", "TEXT DEFAULT ''")
    _add_col(conn, "camera_config", cam_cols, "location_type", "TEXT DEFAULT ''")

    conn.commit()


def _migrate_phase_ab(conn: sqlite3.Connection):
    """Phase A/B schema additions: security_events.track_id, security_zones.capacity."""
    ev_cols = _table_columns(conn, "security_events")
    _add_col(conn, "security_events", ev_cols, "track_id", "TEXT")

    zone_cols = _table_columns(conn, "security_zones")
    # Optional per-zone capacity override for occupancy (falls back to the
    # global CROWD_THRESHOLD when NULL/<=0).
    _add_col(conn, "security_zones", zone_cols, "capacity", "INTEGER")

    conn.commit()


def _add_col(conn: sqlite3.Connection, table: str, cols: set[str],
             name: str, ddl: str) -> None:
    if name not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
        logger.info("Migrating %s: adding '%s' column.", table, name)


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {r[1] for r in rows}
    except sqlite3.Error:
        return set()


def _migrate_activity_logs(conn: sqlite3.Connection):
    """Add ``date``/``source`` columns + indexes to an existing DB."""
    cols = _table_columns(conn, "activity_logs")
    if cols and "date" not in cols:
        logger.info("Migrating activity_logs: adding 'date' column.")
        conn.execute(
            "ALTER TABLE activity_logs ADD COLUMN date TEXT"
        )
        conn.execute(
            "UPDATE activity_logs SET date = substr(timestamp, 1, 10)"
        )
    if cols and "source" not in cols:
        logger.info("Migrating activity_logs: adding 'source' column.")
        conn.execute("ALTER TABLE activity_logs ADD COLUMN source TEXT DEFAULT ''")


def _migrate_employees(conn: sqlite3.Connection):
    """Rebuild the legacy Phase-1 ``employees`` table (desk ROI columns) with
    the current metadata schema (department/designation/active/enrolled).

    The original prototype stored per-desk ROIs (``desk_x/y/w/h``) instead of
    employee metadata.  ``CREATE TABLE IF NOT EXISTS`` cannot alter an existing
    table, so ``upsert_employee`` would otherwise fail with "no such column"
    on any database created by the Phase-1 prototype.  The rebuild preserves
    ``id``, ``employee_id`` and ``name``.
    """
    cols = _table_columns(conn, "employees")
    if not cols or "department" in cols:
        return  # already current schema (or table absent, to be created below)
    logger.warning(
        "Migrating legacy employees table (desk schema) -> metadata schema."
    )
    conn.execute("ALTER TABLE employees RENAME TO employees_legacy")
    conn.executescript(EMPLOYEES_SCHEMA)
    conn.execute(
        """
        INSERT INTO employees (id, employee_id, name)
        SELECT id, employee_id, name FROM employees_legacy
        """
    )
    conn.execute("DROP TABLE employees_legacy")
    conn.commit()


def init_db(conn: sqlite3.Connection):
    conn.executescript(EMPLOYEES_SCHEMA)
    conn.executescript(ACTIVITY_SCHEMA)
    _migrate_employees(conn)      # rebuild legacy desk-schema employees table
    _migrate_activity_logs(conn)  # adds date/source columns if missing
    conn.executescript(INDEXES)     # safe once all columns exist
    # Phase 31 security schema (additive; never removes existing tables).
    conn.executescript(SECURITY_EVENTS_SCHEMA)
    conn.executescript(INCIDENTS_SCHEMA)
    conn.executescript(ALERT_RULES_SCHEMA)
    conn.executescript(ALERTS_SCHEMA)
    conn.executescript(AUDIT_LOG_SCHEMA)
    conn.executescript(SECURITY_ZONES_SCHEMA)
    conn.executescript(EVIDENCE_FILES_SCHEMA)
    conn.executescript(SECURITY_INDEXES)
    # Phase 32 intelligence schema (additive; never removes existing tables).
    conn.executescript(INCIDENT_EVENTS_SCHEMA)
    conn.executescript(INCIDENT_NOTES_SCHEMA)
    conn.executescript(CAMERA_CONFIG_SCHEMA)
    conn.executescript(CAMERA_HEALTH_SCHEMA)
    conn.executescript(DETECTOR_REGISTRY_SCHEMA)
    conn.executescript(PHASE32_INDEXES)
    _migrate_security_phase32(conn)
# Phase 33 intelligence/pilot-hardening schema (additive; never removes).
    conn.executescript(CAMERA_TOPOLOGY_SCHEMA)
    conn.executescript(EVIDENCE_ACCESS_SCHEMA)
    conn.executescript(ANOMALY_EVENTS_SCHEMA)
    conn.executescript(BEHAVIORAL_BASELINES_SCHEMA)
    conn.executescript(INCIDENT_RISK_SCHEMA)
    conn.executescript(PHASE33_INDEXES)
    _migrate_security_phase33(conn)
    # Phase A/B schema (core vision + motion + entry/exit) (additive).
    conn.executescript(MOTION_EVENTS_SCHEMA)
    conn.executescript(ENTRY_EXIT_EVENTS_SCHEMA)
    conn.executescript(PHASE_AB_INDEXES)
    _migrate_phase_ab(conn)
    conn.commit()
    logger.info("Database schema verified/migrated.")


# ======================================================================
# Activity logging
# ======================================================================

def log_interval(
    conn: sqlite3.Connection,
    timestamp: str,
    employee_id: str,
    state: str,
    duration_seconds: float,
    source: str = "",
):
    """Insert an interval row.  ``date`` is derived from ``timestamp``."""
    conn.execute(
        """
        INSERT INTO activity_logs
            (timestamp, date, employee_id, state, duration_seconds, source)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (timestamp, timestamp[:10], employee_id, state, duration_seconds, source or None),
    )
    conn.commit()


def log_interval_batch(conn: sqlite3.Connection, rows: list[tuple]):
    """Insert many rows in one transaction (faster, atomic)."""
    if not rows:
        return
    conn.executemany(
        """
        INSERT INTO activity_logs
            (timestamp, date, employee_id, state, duration_seconds, source)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [(ts, ts[:10], emp, state, dur, src or None)
         for ts, emp, state, dur, src in rows],
    )
    conn.commit()


# ======================================================================
# Employee metadata (Phase 4/5 -- Option A: properly use the table)
# ======================================================================

def upsert_employee(
    conn: sqlite3.Connection,
    employee_id: str,
    name: str | None = None,
    department: str | None = None,
    designation: str | None = None,
    active: bool = True,
    enrolled: bool | None = None,
) -> bool:
    """Insert or update an employee.  Returns True on insert, False on update.

    ``None`` for a metadata field means *leave unchanged* on update (a bare
    ``ensure_known`` call must never wipe an existing name).  Empty string
    explicitly clears the field.
    """
    if not employee_id:
        return False
    existing = conn.execute(
        "SELECT employee_id FROM employees WHERE employee_id = ?", (employee_id,)
    ).fetchone()
    if existing:
        sets = ["active = ?", "updated_at = datetime('now', 'localtime')"]
        params: list = [1 if active else 0]
        for col, val in (("name", name), ("department", department),
                         ("designation", designation)):
            if val is not None:
                sets.append(f"{col} = ?")
                params.append(val)
        params.append(employee_id)
        conn.execute(
            f"UPDATE employees SET {', '.join(sets)} WHERE employee_id = ?",
            params,
        )
        if enrolled is not None:
            conn.execute(
                "UPDATE employees SET enrolled = ? WHERE employee_id = ?",
                (1 if enrolled else 0, employee_id),
            )
        conn.commit()
        return False
    conn.execute(
        """
        INSERT INTO employees
            (employee_id, name, department, designation, active, enrolled)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (employee_id, name or "", department or "", designation or "",
         1 if active else 0, 1 if (enrolled if enrolled is not None else False) else 0),
    )
    conn.commit()
    return True


def set_employee_enrolled(conn: sqlite3.Connection, employee_id: str,
                          enrolled: bool):
    conn.execute(
        "UPDATE employees SET enrolled = ?, updated_at = datetime('now', 'localtime') WHERE employee_id = ?",
        (1 if enrolled else 0, employee_id),
    )
    conn.commit()


def set_employee_active(conn: sqlite3.Connection, employee_id: str, active: bool):
    conn.execute(
        "UPDATE employees SET active = ?, updated_at = datetime('now', 'localtime') WHERE employee_id = ?",
        (1 if active else 0, employee_id),
    )
    conn.commit()


def list_employees(conn: sqlite3.Connection, include_inactive: bool = True) -> list[dict]:
    sql = "SELECT * FROM employees"
    if not include_inactive:
        sql += " WHERE active = 1"
    sql += " ORDER BY employee_id"
    rows = conn.execute(sql).fetchall()
    cols = [d[1] for d in conn.execute("PRAGMA table_info(employees)").fetchall()]
    return [dict(zip(cols, r)) for r in rows]


def get_employee(conn: sqlite3.Connection, employee_id: str) -> dict | None:
    rows = list_employees(conn)
    for r in rows:
        if r["employee_id"] == employee_id:
            return r
    return None


def delete_employee(conn: sqlite3.Connection, employee_id: str):
    conn.execute("DELETE FROM employees WHERE employee_id = ?", (employee_id,))
    conn.commit()


# ======================================================================
# Analytics queries (read-only, date-safe via the ``date`` column)
# ======================================================================

def query_day_intervals(conn: sqlite3.Connection, day: str) -> list[tuple]:
    """Return (timestamp, employee_id, state, duration_seconds, source) for a day.

    A row is included when its interval *overlaps* the target calendar day
    (``start <= day <= end``), so intervals that cross midnight still have
    their spillover segment attributed to the following day instead of being
    silently lost by the single-day ``date`` column.
    """
    return conn.execute(
        """
        SELECT timestamp, employee_id, state, duration_seconds, source
        FROM activity_logs
        WHERE substr(timestamp, 1, 10) <= ?
          AND substr(datetime(timestamp, '+' || duration_seconds || ' seconds'), 1, 10) >= ?
        ORDER BY timestamp
        """,
        (day, day),
    ).fetchall()


def query_range_intervals(conn: sqlite3.Connection,
                          start_day: str, end_day: str) -> list[tuple]:
    return conn.execute(
        """
        SELECT timestamp, employee_id, state, duration_seconds, source
        FROM activity_logs
        WHERE date >= ? AND date <= ?
        ORDER BY timestamp
        """,
        (start_day, end_day),
    ).fetchall()


def retention_cleanup(conn: sqlite3.Connection, days: int) -> int:
    """Delete activity rows older than ``days``.  Returns rows removed.

    Explicitly opt-in; never called by the daemon automatically unless
    ``ENABLE_RETENTION`` is set.
    """
    if days < 0:
        return 0
    from datetime import date, timedelta
    cutoff = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    cur = conn.execute("DELETE FROM activity_logs WHERE date < ?", (cutoff,))
    conn.commit()
    return cur.rowcount


# ======================================================================
# Phase 31 -- security persistence helpers
# ======================================================================

def insert_security_event(conn: sqlite3.Connection, ev: dict) -> int:
    """Persist one security event; returns the new row id.

    ``ev`` mirrors the :class:`~src.security_events.SecurityEvent` fields
    (canonical event model).  ``date`` is derived from ``timestamp``.
"""
    ts = ev.get("timestamp", "")
    cur = conn.execute(
        """
        INSERT INTO security_events
            (incident_id, event_type, severity, timestamp, date, camera,
             zone, employee_id, person_count, confidence, duration_seconds,
             status, evidence_ref, details, track_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ev.get("incident_id"),
            ev.get("event_type"),
            ev.get("severity"),
            ts,
            ts[:10] if ts else None,
            ev.get("camera"),
            ev.get("zone"),
            ev.get("employee_id"),
            ev.get("person_count"),
            ev.get("confidence"),
            ev.get("duration_seconds"),
            ev.get("status"),
            ev.get("evidence_ref"),
            ev.get("details"),
            ev.get("track_id"),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def set_event_incident(conn: sqlite3.Connection, event_id: int,
                       incident_id: str) -> None:
    """Back-link an already-inserted security event to its incident."""
    conn.execute(
        "UPDATE security_events SET incident_id = ? WHERE id = ?",
        (incident_id, event_id),
    )
    conn.commit()


def query_security_events(conn: sqlite3.Connection, *, day: str | None = None,
                          event_type: str | None = None,
                          camera: str | None = None, zone: str | None = None,
                          employee_id: str | None = None,
                          incident_id: str | None = None,
                          severity: str | None = None,
                          status: str | None = None,
                          limit: int = 500) -> list[dict]:
    """Read-only event query with optional filters (all safe raw SQL)."""
    clauses: list[str] = []
    params: list = []
    if day:
        clauses.append("date = ?")
        params.append(day)
    if event_type:
        clauses.append("event_type = ?")
        params.append(event_type)
    if camera:
        clauses.append("camera = ?")
        params.append(camera)
    if zone:
        clauses.append("zone = ?")
        params.append(zone)
    if employee_id:
        clauses.append("employee_id = ?")
        params.append(employee_id)
    if incident_id:
        clauses.append("incident_id = ?")
        params.append(incident_id)
    if severity:
        clauses.append("severity = ?")
        params.append(severity)
    if status:
        clauses.append("status = ?")
        params.append(status)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = (
        "SELECT * FROM security_events " + where +
        " ORDER BY timestamp DESC LIMIT ?"
    )
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    cols = [d[1] for d in conn.execute("PRAGMA table_info(security_events)").fetchall()]
    return [dict(zip(cols, r)) for r in rows]


# ----------------------------------------------------------------------
# Incidents
# ----------------------------------------------------------------------

def insert_incident(conn: sqlite3.Connection, inc: dict) -> str:
    """Create an incident row; returns the stable ``incident_id``."""
    conn.execute(
        """
        INSERT INTO incidents
            (incident_id, event_type, severity, status, first_seen,
             last_seen, count, camera, zone, employee_id, confidence,
             evidence_ref, notes, resolved_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            inc.get("incident_id"), inc.get("event_type"),
            inc.get("severity") or "INFO", inc.get("status", "OPEN"),
            inc.get("first_seen", inc.get("last_seen", "")),
            inc.get("last_seen"), inc.get("count", 1),
            inc.get("camera"), inc.get("zone"), inc.get("employee_id"),
            inc.get("confidence"), inc.get("evidence_ref"),
            inc.get("notes", ""), inc.get("resolved_at"),
        ),
    )
    conn.commit()
    return inc["incident_id"]


def get_incident(conn: sqlite3.Connection, incident_id: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM incidents WHERE incident_id = ?", (incident_id,)
    ).fetchone()
    if not row:
        return None
    cols = [d[1] for d in conn.execute("PRAGMA table_info(incidents)").fetchall()]
    return dict(zip(cols, row))


def update_incident(conn: sqlite3.Connection, incident_id: str, **fields) -> None:
    """Update non-null fields on an incident (status, notes, ...)."""
    sets, params = [], []
    allowed = {"status", "last_seen", "count", "confidence", "evidence_ref",
               "notes", "resolved_at", "correlation_ids", "review_state",
               "acknowledged_at", "resolved_ts", "temporal_state", "occurrences"}
    for key, val in fields.items():
        if key in allowed and val is not None:
            sets.append(f"{key} = ?")
            params.append(val)
    if not sets:
        return
    params.append(incident_id)
    conn.execute(
        f"UPDATE incidents SET {', '.join(sets)} WHERE incident_id = ?",
        params,
    )
    conn.commit()


def list_incidents(conn: sqlite3.Connection, *, status: str | None = None,
                   event_type: str | None = None, camera: str | None = None,
                   zone: str | None = None, employee_id: str | None = None,
                   day: str | None = None, limit: int = 500) -> list[dict]:
    clauses: list[str] = []
    params: list = []
    if status:
        clauses.append("status = ?")
        params.append(status)
    if event_type:
        clauses.append("event_type = ?")
        params.append(event_type)
    if camera:
        clauses.append("camera = ?")
        params.append(camera)
    if zone:
        clauses.append("zone = ?")
        params.append(zone)
    if employee_id:
        clauses.append("employee_id = ?")
        params.append(employee_id)
    if day:
        clauses.append("substr(first_seen, 1, 10) = ?")
        params.append(day)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = (
        "SELECT * FROM incidents " + where +
        " ORDER BY first_seen DESC LIMIT ?"
    )
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    cols = [d[1] for d in conn.execute("PRAGMA table_info(incidents)").fetchall()]
    return [dict(zip(cols, r)) for r in rows]


def find_open_incident(conn: sqlite3.Connection, event_type: str,
                       camera: str | None, zone: str | None) -> dict | None:
    """Return the most recent OPEN/ACKNOWLEDGED incident matching the dedup
    key.  ``None`` or ``""`` for ``camera``/``zone`` acts as a wildcard."""
    rows = conn.execute(
        """
        SELECT * FROM incidents
        WHERE status IN ('OPEN', 'ACKNOWLEDGED')
          AND event_type = ?
          AND (? IS NULL OR ? = '' OR camera = ?)
          AND (? IS NULL OR ? = '' OR zone = ?)
        ORDER BY first_seen DESC LIMIT 1
        """,
        (event_type, camera, camera, camera,
         zone, zone, zone),
    ).fetchall()
    if not rows:
        return None
    cols = [d[1] for d in conn.execute("PRAGMA table_info(incidents)").fetchall()]
    return dict(zip(cols, rows[0]))


# ----------------------------------------------------------------------
# Audit log (append-only)
# ----------------------------------------------------------------------

def audit(conn: sqlite3.Connection, action: str, *, actor: str = "system",
          resource: str | None = None, detail: str | None = None) -> None:
    """Record one append-only audit row.  Never records secrets."""
    conn.execute(
        "INSERT INTO audit_log (timestamp, actor, action, resource, detail) "
        "VALUES (datetime('now', 'localtime'), ?, ?, ?, ?)",
        (actor, action, resource, detail),
    )
    conn.commit()


def list_audit_events(conn: sqlite3.Connection, limit: int = 200) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    cols = [d[1] for d in conn.execute("PRAGMA table_info(audit_log)").fetchall()]
    return [dict(zip(cols, r)) for r in rows]


# ----------------------------------------------------------------------
# Alert rules + alert log
# ----------------------------------------------------------------------

def upsert_alert_rule(conn: sqlite3.Connection, rule: dict) -> None:
    conn.execute(
        """
        INSERT INTO alert_rules
            (rule_name, event_type, min_severity, channels, cooldown_sec,
             enabled, schedule_start, schedule_end, recipients)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(rule_name) DO UPDATE SET
            event_type=excluded.event_type,
            min_severity=excluded.min_severity,
            channels=excluded.channels,
            cooldown_sec=excluded.cooldown_sec,
            enabled=excluded.enabled,
            schedule_start=excluded.schedule_start,
            schedule_end=excluded.schedule_end,
            recipients=excluded.recipients
        """,
        (
            rule.get("rule_name"), rule.get("event_type", ""),
            rule.get("min_severity", "MEDIUM"), rule.get("channels", "dashboard,log"),
            rule.get("cooldown_sec", 300), int(bool(rule.get("enabled", True))),
            rule.get("schedule_start"), rule.get("schedule_end"),
            rule.get("recipients", ""),
        ),
    )
    conn.commit()


def list_alert_rules(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM alert_rules ORDER BY rule_name").fetchall()
    cols = [d[1] for d in conn.execute("PRAGMA table_info(alert_rules)").fetchall()]
    return [dict(zip(cols, r)) for r in rows]


def insert_alert(conn: sqlite3.Connection, alert: dict) -> int:
    cur = conn.execute(
        """
        INSERT INTO alerts
            (created_at, rule_name, event_type, severity, camera, zone,
             employee_id, incident_id, channel, message, delivered, status,
             attempts)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            alert.get("created_at", ""), alert.get("rule_name"),
            alert.get("event_type"), alert.get("severity"),
            alert.get("camera"), alert.get("zone"), alert.get("employee_id"),
            alert.get("incident_id"), alert.get("channel"),
            alert.get("message", ""), int(bool(alert.get("delivered", False))),
            alert.get("status", "SENT"), alert.get("attempts", 1),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def query_alerts(conn: sqlite3.Connection, limit: int = 200) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    cols = [d[1] for d in conn.execute("PRAGMA table_info(alerts)").fetchall()]
    return [dict(zip(cols, r)) for r in rows]


def last_alert_time(conn: sqlite3.Connection) -> str | None:
    row = conn.execute("SELECT MAX(created_at) FROM alerts").fetchone()
    return str(row[0]) if row and row[0] else None


# ----------------------------------------------------------------------
# Security zones
# ----------------------------------------------------------------------

def upsert_zone(conn: sqlite3.Connection, zone: dict) -> None:
    conn.execute(
        """
INSERT INTO security_zones
            (zone_name, camera, enabled, polygon, allowed_employees,
             schedule, alert_policy, capacity)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(zone_name) DO UPDATE SET
            camera=excluded.camera,
            enabled=excluded.enabled,
            polygon=excluded.polygon,
            allowed_employees=excluded.allowed_employees,
            schedule=excluded.schedule,
            alert_policy=excluded.alert_policy,
            capacity=excluded.capacity
        """,
        (
            zone.get("zone_name"), zone.get("camera"),
            int(bool(zone.get("enabled", True))), zone.get("polygon"),
            zone.get("allowed_employees", ""), zone.get("schedule"),
            zone.get("alert_policy", "INTRUSION"),
            zone.get("capacity"),
        ),
    )
    conn.commit()


def list_zones(conn: sqlite3.Connection, *, enabled_only: bool = False) -> list[dict]:
    sql = "SELECT * FROM security_zones"
    if enabled_only:
        sql += " WHERE enabled = 1"
    sql += " ORDER BY zone_name"
    rows = conn.execute(sql).fetchall()
    cols = [d[1] for d in conn.execute("PRAGMA table_info(security_zones)").fetchall()]
    return [dict(zip(cols, r)) for r in rows]


def delete_zone(conn: sqlite3.Connection, zone_name: str) -> None:
    conn.execute("DELETE FROM security_zones WHERE zone_name = ?", (zone_name,))
    conn.commit()


# ----------------------------------------------------------------------
# Evidence files
# ----------------------------------------------------------------------

def insert_evidence_file(conn: sqlite3.Connection, incident_id: str, path: str,
                         kind: str, captured_at: str, size_bytes: int,
                         camera: str | None = None,
                         capture_type: str | None = None) -> None:
    conn.execute(
        "INSERT INTO evidence_files "
        "(incident_id, path, kind, captured_at, size_bytes, camera, capture_type) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (incident_id, path, kind, captured_at, size_bytes, camera, capture_type),
    )
    conn.commit()


def list_evidence_files(conn: sqlite3.Connection,
                        incident_id: str | None = None) -> list[dict]:
    if incident_id:
        rows = conn.execute(
            "SELECT * FROM evidence_files WHERE incident_id = ? "
            "ORDER BY captured_at", (incident_id,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM evidence_files ORDER BY captured_at").fetchall()
    cols = [d[1] for d in conn.execute("PRAGMA table_info(evidence_files)").fetchall()]
    return [dict(zip(cols, r)) for r in rows]


def delete_evidence_file(conn: sqlite3.Connection, path: str) -> None:
    conn.execute("DELETE FROM evidence_files WHERE path = ?", (path,))
    conn.commit()


def delete_evidence_file_id(conn: sqlite3.Connection, evidence_id: int) -> None:
    """Remove an evidence row by its id (used by RBAC-guarded deletion)."""
    conn.execute("DELETE FROM evidence_files WHERE id = ?", (evidence_id,))
    conn.commit()


# ----------------------------------------------------------------------
# Security retention (explicit, opt-in)
# ----------------------------------------------------------------------

def security_retention_cleanup(conn: sqlite3.Connection, *, events_days: int = -1,
                               alerts_days: int = -1,
                               audit_days: int = -1) -> dict[str, int]:
    """Purge old security_events / alerts / audit_log rows.

    Incidents are NEVER purged automatically (they are the investigation
    record).  Evidence-file pruning is file-based, not here.  Negative days
    => that domain is left untouched.
    """
    removed: dict[str, int] = {"events": 0, "alerts": 0, "audit": 0}
    from datetime import date, timedelta

    def _cutoff(days: int) -> str:
        return (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")

    if events_days >= 0:
        cur = conn.execute(
            "DELETE FROM security_events WHERE date < ?", (_cutoff(events_days),)
        )
        removed["events"] = cur.rowcount
    if alerts_days >= 0:
        cur = conn.execute(
            "DELETE FROM alerts WHERE date(created_at) < ?", (_cutoff(alerts_days),)
        )
        removed["alerts"] = cur.rowcount
    if audit_days >= 0:
        cur = conn.execute(
            "DELETE FROM audit_log WHERE date(timestamp) < ?", (_cutoff(audit_days),)
        )
        removed["audit"] = cur.rowcount
    conn.commit()
    return removed


# ======================================================================
# Phase 32 -- investigation / intelligence persistence helpers
# ======================================================================

def link_incident_event(conn, incident_id: str, event_id: int,
                        event_type: str | None, camera: str | None,
                        zone: str | None, correlation_reason: str | None = None) -> None:
    """Record that ``event_id`` belongs to ``incident_id`` (with reason).

    Idempotent (Phase 35): re-linking the same incident/event pair is a no-op
    so a daemon restart or reprocess does not create duplicate link rows.
    """
    exists = conn.execute(
        "SELECT 1 FROM incident_events WHERE incident_id = ? AND event_id = ?",
        (incident_id, event_id)).fetchone()
    if exists:
        return
    conn.execute(
        "INSERT INTO incident_events "
        "(incident_id, event_id, event_type, camera, zone, correlation_reason) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (incident_id, event_id, event_type, camera, zone, correlation_reason),
    )
    conn.commit()


def list_incident_events(conn, incident_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM incident_events WHERE incident_id = ? "
        "ORDER BY linked_at", (incident_id,)
    ).fetchall()
    cols = [d[1] for d in conn.execute("PRAGMA table_info(incident_events)").fetchall()]
    return [dict(zip(cols, r)) for r in rows]


def add_incident_note(conn, incident_id: str, author: str, note: str) -> None:
    conn.execute(
        "INSERT INTO incident_notes (incident_id, author, note) VALUES (?, ?, ?)",
        (incident_id, author, note),
    )
    conn.commit()


def list_incident_notes(conn, incident_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM incident_notes WHERE incident_id = ? ORDER BY created_at",
        (incident_id,),
    ).fetchall()
    cols = [d[1] for d in conn.execute("PRAGMA table_info(incident_notes)").fetchall()]
    return [dict(zip(cols, r)) for r in rows]


# ----------------------------------------------------------------------
# Camera configuration (Phase 32)
# ----------------------------------------------------------------------

def upsert_camera_config(conn, cfg: dict) -> None:
    conn.execute(
        """
        INSERT INTO camera_config
            (camera_id, name, location, zone, source, enabled, fps_target,
             schedule, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(camera_id) DO UPDATE SET
            name=excluded.name, location=excluded.location,
            zone=excluded.zone, source=excluded.source,
            enabled=excluded.enabled, fps_target=excluded.fps_target,
            schedule=excluded.schedule, notes=excluded.notes
        """,
        (
            cfg.get("camera_id"), cfg.get("name", ""), cfg.get("location", ""),
            cfg.get("zone", ""), cfg.get("source", ""),
            int(bool(cfg.get("enabled", True))), cfg.get("fps_target"),
            cfg.get("schedule"), cfg.get("notes", ""),
        ),
    )
    conn.commit()


def list_camera_configs(conn, *, enabled_only: bool = False) -> list[dict]:
    sql = "SELECT * FROM camera_config"
    if enabled_only:
        sql += " WHERE enabled = 1"
    sql += " ORDER BY camera_id"
    rows = conn.execute(sql).fetchall()
    cols = [d[1] for d in conn.execute("PRAGMA table_info(camera_config)").fetchall()]
    return [dict(zip(cols, r)) for r in rows]


def get_camera_config(conn, camera_id: str) -> dict | None:
    rows = conn.execute(
        "SELECT * FROM camera_config WHERE camera_id = ?", (camera_id,)
    ).fetchall()
    if not rows:
        return None
    cols = [d[1] for d in conn.execute("PRAGMA table_info(camera_config)").fetchall()]
    return dict(zip(cols, rows[0]))


def set_camera_enabled(conn, camera_id: str, enabled: bool) -> None:
    conn.execute(
        "UPDATE camera_config SET enabled = ? WHERE camera_id = ?",
        (int(bool(enabled)), camera_id),
    )
    conn.commit()


# ----------------------------------------------------------------------
# Persistent camera health history (Phase 32)
# ----------------------------------------------------------------------

def record_camera_health(conn, camera_id: str, health: dict) -> None:
    conn.execute(
        "INSERT INTO camera_health "
        "(camera_id, health, fps, frames_read, reconnects, last_frame, "
        " downtime_sec, tampered) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            camera_id, health.get("health") or "UNKNOWN",
            health.get("fps"), health.get("frames_read"),
            health.get("reconnects"), health.get("last_frame"),
            health.get("downtime_sec"), int(bool(health.get("tampered", 0))),
        ),
    )
    conn.commit()


def camera_health_history(conn, camera_id: str, limit: int = 200) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM camera_health WHERE camera_id = ? "
        "ORDER BY recorded_at DESC LIMIT ?", (camera_id, limit)
    ).fetchall()
    cols = [d[1] for d in conn.execute("PRAGMA table_info(camera_health)").fetchall()]
    return [dict(zip(cols, r)) for r in rows]


def camera_downtime_total(conn, camera_id: str | None = None) -> float:
    sql = ("SELECT SUM(downtime_sec) FROM camera_health "
           "WHERE downtime_sec > 0")
    params: list = []
    if camera_id:
        sql += " AND camera_id = ?"
        params.append(camera_id)
    row = conn.execute(sql, params).fetchone()
    return float(row[0] or 0.0)


# ----------------------------------------------------------------------
# Detector registry (Phase 32)
# ----------------------------------------------------------------------

def upsert_detector(conn, det: dict) -> None:
    conn.execute(
        """
        INSERT INTO detector_registry
            (detector_id, name, version, enabled, model_path, event_types,
             conf_threshold, persist_sec, cooldown_sec, camera_scope,
             zone_scope, status, status_detail)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(detector_id) DO UPDATE SET
            name=excluded.name, version=excluded.version,
            enabled=excluded.enabled, model_path=excluded.model_path,
            event_types=excluded.event_types,
            conf_threshold=excluded.conf_threshold,
            persist_sec=excluded.persist_sec,
            cooldown_sec=excluded.cooldown_sec,
            camera_scope=excluded.camera_scope,
            zone_scope=excluded.zone_scope,
            status=excluded.status,
            status_detail=excluded.status_detail,
            updated_at=datetime('now', 'localtime')
        """,
        (
            det.get("detector_id"), det.get("name", ""), det.get("version", ""),
            int(bool(det.get("enabled", True))), det.get("model_path", ""),
            det.get("event_types", ""), det.get("conf_threshold"),
            det.get("persist_sec"), det.get("cooldown_sec"),
            det.get("camera_scope", ""), det.get("zone_scope", ""),
            det.get("status", "UNAVAILABLE"), det.get("status_detail", ""),
        ),
    )
    conn.commit()


def list_detectors(conn, *, enabled_only: bool = False) -> list[dict]:
    sql = "SELECT * FROM detector_registry"
    if enabled_only:
        sql += " WHERE enabled = 1"
    sql += " ORDER BY detector_id"
    rows = conn.execute(sql).fetchall()
    cols = [d[1] for d in conn.execute("PRAGMA table_info(detector_registry)").fetchall()]
    return [dict(zip(cols, r)) for r in rows]


def get_detector(conn, detector_id: str) -> dict | None:
    rows = conn.execute(
        "SELECT * FROM detector_registry WHERE detector_id = ?", (detector_id,)
    ).fetchall()
    if not rows:
        return None
    cols = [d[1] for d in conn.execute("PRAGMA table_info(detector_registry)").fetchall()]
    return dict(zip(cols, rows[0]))


def set_detector_status(conn, detector_id: str, status: str,
                        detail: str = "") -> None:
    conn.execute(
        "UPDATE detector_registry SET status = ?, status_detail = ?, "
        "updated_at = datetime('now', 'localtime') WHERE detector_id = ?",
        (status, detail, detector_id),
    )
    conn.commit()


# ----------------------------------------------------------------------
# Evidence integrity / availability (Phase 32)
# ----------------------------------------------------------------------

def set_evidence_metadata(conn, path: str, *, sha256: str | None = None,
                          mime: str | None = None, capture_type: str | None = None,
                          camera: str | None = None,
                          retention_deadline: str | None = None) -> None:
    sets, params = [], []
    if sha256 is not None:
        sets.append("sha256 = ?"); params.append(sha256)
    if mime is not None:
        sets.append("mime = ?"); params.append(mime)
    if capture_type is not None:
        sets.append("capture_type = ?"); params.append(capture_type)
    if camera is not None:
        sets.append("camera = ?"); params.append(camera)
    if retention_deadline is not None:
        sets.append("retention_deadline = ?"); params.append(retention_deadline)
    if not sets:
        return
    params.append(path)
    conn.execute(
        f"UPDATE evidence_files SET {', '.join(sets)} WHERE path = ?", params,
    )
    conn.commit()


def set_evidence_available(conn, path: str, available: bool) -> None:
    conn.execute(
        "UPDATE evidence_files SET available = ? WHERE path = ?",
        (int(bool(available)), path),
    )
    conn.commit()


# ----------------------------------------------------------------------
# Alert lifecycle / delivery (Phase 32)
# ----------------------------------------------------------------------

def set_alert_status(conn, alert_id: int, status: str, *, error: str = "",
                     attempts: int | None = None,
                     last_attempt_epoch: float | None = None) -> None:
    sets = ["status = ?", "error = ?"]
    params: list = [status, error]
    if attempts is not None:
        sets.append("attempts = ?")
        params.append(attempts)
    if last_attempt_epoch is not None:
        sets.append("last_attempt_epoch = ?")
        params.append(last_attempt_epoch)
    params.append(alert_id)
    conn.execute(
        f"UPDATE alerts SET {', '.join(sets)} WHERE id = ?", params,
    )
    conn.commit()


def acknowledge_alert(conn, alert_id: int) -> None:
    conn.execute(
        "UPDATE alerts SET status = 'ACKNOWLEDGED', "
        "acknowledged_at = datetime('now', 'localtime') WHERE id = ?", (alert_id,)
    )
    conn.commit()


def resolve_alert(conn, alert_id: int) -> None:
    conn.execute(
        "UPDATE alerts SET status = 'RESOLVED', "
        "resolved_at = datetime('now', 'localtime') WHERE id = ?", (alert_id,)
    )
    conn.commit()


def escalate_alert(conn, alert_id: int) -> None:
    conn.execute(
        "UPDATE alerts SET status = 'ESCALATED', escalated = 1 WHERE id = ?",
        (alert_id,),
    )
    conn.commit()


def query_alerts_filtered(conn, *, status: str | None = None,
                          severity: str | None = None, incident_id: str | None = None,
                          limit: int = 200) -> list[dict]:
    clauses: list[str] = []
    params: list = []
    if status:
        clauses.append("status = ?"); params.append(status)
    if severity:
        clauses.append("severity = ?"); params.append(severity)
    if incident_id:
        clauses.append("incident_id = ?"); params.append(incident_id)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = "SELECT * FROM alerts " + where + " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    cols = [d[1] for d in conn.execute("PRAGMA table_info(alerts)").fetchall()]
    return [dict(zip(cols, r)) for r in rows]


# ----------------------------------------------------------------------
# Correlation / incident query (Phase 32)
# ----------------------------------------------------------------------

def set_incident_review_state(conn, incident_id: str, review_state: str,
                              actor: str = "system") -> None:
    conn.execute(
        "UPDATE incidents SET review_state = ?, updated_at = datetime('now', 'localtime') "
        "WHERE incident_id = ?", (review_state, incident_id),
    )
    conn.commit()
    audit(conn, "incident.review_state", actor=actor,
          resource=incident_id, detail=review_state)


def set_incident_correlation(conn, incident_id: str,
                             correlation_reason: str | None) -> None:
    """Record the correlation reason for an incident.

    The explainable reason is persisted per linked event on ``incident_events``
    (see :func:`link_incident_event`) and surfaced through the investigation
    view.  This helper only bumps the incident's update marker so the timeline
    reflects the correlation.
    """
    if not correlation_reason:
        return
    conn.execute(
        "UPDATE incidents SET updated_at = datetime('now', 'localtime') WHERE incident_id = ?",
        (incident_id,),
    )
    conn.commit()


def set_incident_ack_resolve(conn, incident_id: str, *, acknowledged_at: str = None,
                             resolved_ts: str = None) -> None:
    sets, params = [], []
    if acknowledged_at is not None:
        sets.append("acknowledged_at = ?"); params.append(acknowledged_at)
    if resolved_ts is not None:
        sets.append("resolved_ts = ?"); params.append(resolved_ts)
    if not sets:
        return
    params.append(incident_id)
    conn.execute(
        f"UPDATE incidents SET {', '.join(sets)}, updated_at = datetime('now', 'localtime') "
        "WHERE incident_id = ?", params,
    )
    conn.commit()


# ======================================================================
# Phase 33 -- intelligence / pilot-hardening helpers
# ======================================================================

# --- Camera topology (A3) ----------------------------------------------

def upsert_camera_topology(conn, from_camera: str, to_camera: str,
                           relation: str = "NEXT", note: str = "") -> None:
    conn.execute(
        "INSERT INTO camera_topology (from_camera, to_camera, relation, note) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT DO NOTHING",
        (from_camera, to_camera, relation, note),
    )
    conn.commit()


def list_camera_topology(conn, *, enabled_only: bool = True) -> list[dict]:
    sql = "SELECT * FROM camera_topology"
    if enabled_only:
        sql += " WHERE enabled = 1"
    rows = conn.execute(sql + " ORDER BY from_camera, id").fetchall()
    cols = [d[1] for d in conn.execute("PRAGMA table_info(camera_topology)").fetchall()]
    return [dict(zip(cols, r)) for r in rows]


def set_camera_topology_enabled(conn, edge_id: int, enabled: bool) -> None:
    conn.execute("UPDATE camera_topology SET enabled = ? WHERE id = ?",
                 (int(bool(enabled)), edge_id))
    conn.commit()


def delete_camera_topology(conn, edge_id: int) -> None:
    conn.execute("DELETE FROM camera_topology WHERE id = ?", (edge_id,))
    conn.commit()


def incident_adjacent_cameras(conn, camera_id: str) -> list[str]:
    """Direct neighbours of ``camera_id`` in the admin-config graph."""
    rows = conn.execute(
        "SELECT to_camera AS c FROM camera_topology WHERE from_camera = ? AND enabled = 1 "
        "UNION SELECT from_camera AS c FROM camera_topology WHERE to_camera = ? AND enabled = 1",
        (camera_id, camera_id),
    ).fetchall()
    return [r[0] for r in rows]


# --- Evidence access audit (A6) ----------------------------------------

def log_evidence_access(conn, evidence_id: int, action: str, *,
                        actor: str = "system", incident_id: str | None = None,
                        detail: str | None = None) -> None:
    conn.execute(
        "INSERT INTO evidence_access (evidence_id, incident_id, action, actor, detail) "
        "VALUES (?, ?, ?, ?, ?)",
        (evidence_id, incident_id, action, actor, detail),
    )
    conn.commit()


def list_evidence_access(conn, *, evidence_id: int | None = None,
                         limit: int = 500) -> list[dict]:
    if evidence_id is not None:
        rows = conn.execute(
            "SELECT * FROM evidence_access WHERE evidence_id = ? "
            "ORDER BY accessed_at DESC LIMIT ?", (evidence_id, limit)).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM evidence_access ORDER BY accessed_at DESC LIMIT ?",
            (limit,)).fetchall()
    cols = [d[1] for d in conn.execute("PRAGMA table_info(evidence_access)").fetchall()]
    return [dict(zip(cols, r)) for r in rows]


# --- Anomaly events (B1) ------------------------------------------------

def insert_anomaly(conn, *, anomaly_type: str, scope: str, scope_value: str,
                   severity: str, reason: str, data: str | None = None) -> None:
    conn.execute(
        "INSERT INTO anomaly_events (anomaly_type, scope, scope_value, severity, reason, data) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (anomaly_type, scope, scope_value, severity, reason, data),
    )
    conn.commit()


def list_anomalies(conn, limit: int = 200) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM anomaly_events ORDER BY observed_at DESC LIMIT ?", (limit,)
    ).fetchall()
    cols = [d[1] for d in conn.execute("PRAGMA table_info(anomaly_events)").fetchall()]
    return [dict(zip(cols, r)) for r in rows]


# --- Behavioral baselines (B2) ------------------------------------------

def upsert_baseline(conn, *, scope: str, scope_value: str, metric: str,
                    mean: float, std: float, sample_count: int, status: str) -> None:
    conn.execute(
        "INSERT INTO behavioral_baselines (scope, scope_value, metric, mean, std, sample_count, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(scope, scope_value, metric) DO UPDATE SET "
        "mean = excluded.mean, std = excluded.std, "
        "sample_count = excluded.sample_count, status = excluded.status, "
        "updated_at = datetime('now', 'localtime')",
        (scope, scope_value, metric, mean, std, sample_count, status),
    )
    conn.commit()


def get_baseline(conn, *, scope: str, scope_value: str, metric: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM behavioral_baselines WHERE scope = ? AND scope_value = ? AND metric = ?",
        (scope, scope_value, metric),
    ).fetchone()
    if not row:
        return None
    cols = [d[1] for d in conn.execute("PRAGMA table_info(behavioral_baselines)").fetchall()]
    return dict(zip(cols, row))


def list_baselines(conn, limit: int = 500) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM behavioral_baselines ORDER BY updated_at DESC LIMIT ?", (limit,)
    ).fetchall()
    cols = [d[1] for d in conn.execute("PRAGMA table_info(behavioral_baselines)").fetchall()]
    return [dict(zip(cols, r)) for r in rows]


# --- Incident risk score (B4) -------------------------------------------

def upsert_incident_risk(conn, incident_id: str, score: float, *,
                         components: str | None, rationale: str | None) -> None:
    """Store/replace the risk score for an incident.

    Idempotent (Phase 35): keeps exactly one latest row per incident instead of
    appending a new row on every rescore (the previous implementation was a
    plain INSERT despite the ``upsert_`` name, so repeated rescoring grew the
    ``incident_risk`` table unboundedly).
    """
    conn.execute(
        "DELETE FROM incident_risk WHERE incident_id = ?", (incident_id,)
    )
    conn.execute(
        "INSERT INTO incident_risk (incident_id, score, components, rationale) "
        "VALUES (?, ?, ?, ?)",
        (incident_id, score, components, rationale),
    )
    conn.commit()


def get_incident_risk(conn, incident_id: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM incident_risk WHERE incident_id = ? "
        "ORDER BY scored_at DESC LIMIT 1", (incident_id,),
    ).fetchone()
    if not row:
        return None
    cols = [d[1] for d in conn.execute("PRAGMA table_info(incident_risk)").fetchall()]
    return dict(zip(cols, row))


def list_incident_risk(conn, limit: int = 200) -> list[dict]:
    """Return risk rows joined with their incident context.

    ``incident_risk`` alone only stores ``incident_id/score/components/rationale``.
    For the dashboard risk panel we need the incident's ``severity/status/
    review_state/event_type/zone`` so operators can filter to *open* incidents
    and see what the score refers to.  The join is LEFT so a risk row is still
    returned even if its incident was deleted (advisory meta-data is preserved).
    The score column is aliased ``risk_score`` for reader clarity.
    """
    rows = conn.execute(
        """
        SELECT r.incident_id AS incident_id,
               r.score AS risk_score,
               r.components AS factors,
               r.scored_at AS scored_at,
               i.event_type AS event_type,
               i.severity AS severity,
               i.status AS status,
               i.review_state AS review_state,
               i.zone AS zone,
               i.camera AS camera
        FROM incident_risk r
        LEFT JOIN incidents i ON i.incident_id = r.incident_id
        ORDER BY r.score DESC LIMIT ?
        """, (limit,)
    ).fetchall()
    cols = ["incident_id", "risk_score", "factors", "scored_at", "event_type",
            "severity", "status", "review_state", "zone", "camera"]
    return [dict(zip(cols, r)) for r in rows]


# --- Detector health (B9) -----------------------------------------------

def record_detector_run(conn, detector_id: str, *, ok: bool,
                        error: str = "", last_inference: str | None = None) -> None:
    conn.execute(
        "UPDATE detector_registry SET "
        "last_run_at = datetime('now', 'localtime'), last_run_ok = ?, last_error = ?, "
        "inference_count = inference_count + 1 "
        + (", last_inference_at = ? " if last_inference else " ")
        + "WHERE detector_id = ?",
        (int(bool(ok)), error, last_inference, detector_id)
        if last_inference else (int(bool(ok)), error, detector_id),
    )
    conn.commit()


# ======================================================================
# Phase A/B -- motion + entry/exit persistence
# ======================================================================

def insert_motion_event(conn, camera: str, timestamp: str, score: float,
                        *, duration_seconds: float = 0.0,
                        bbox: tuple | None = None,
                        event_id: str | None = None) -> int:
    """Persist one motion event; returns the row id."""
    bx1 = by1 = bx2 = by2 = None
    if bbox and len(bbox) >= 4:
        bx1, by1, bx2, by2 = (None if v is None else round(float(v), 2)
                              for v in bbox[:4])
    cur = conn.execute(
        """
        INSERT INTO motion_events
            (camera, timestamp, date, score, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
             duration_seconds, event_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (camera, timestamp, timestamp[:10], float(score),
         bx1, by1, bx2, by2, float(duration_seconds), event_id),
    )
    conn.commit()
    return int(cur.lastrowid)


def query_motion_events(conn, *, camera: str | None = None,
                        day: str | None = None,
                        event_id: str | None = None,
                        limit: int = 500) -> list[dict]:
    clauses: list[str] = []
    params: list = []
    if camera:
        clauses.append("camera = ?")
        params.append(camera)
    if day:
        clauses.append("date = ?")
        params.append(day)
    if event_id:
        clauses.append("event_id = ?")
        params.append(event_id)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = (
        "SELECT * FROM motion_events " + where +
        " ORDER BY timestamp DESC LIMIT ?"
    )
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    cols = [d[1] for d in conn.execute("PRAGMA table_info(motion_events)").fetchall()]
    return [dict(zip(cols, r)) for r in rows]


def insert_entry_exit_event(conn, *, camera: str, timestamp: str,
                            event_type: str, line: str, direction: str,
                            track_id: str, employee_id: str | None = None,
                            confidence: float = 0.0,
                            security_event_id: int | None = None) -> int:
    """Persist one entry/exit crossing; returns the row id."""
    cur = conn.execute(
        """
        INSERT INTO entry_exit_events
            (camera, timestamp, date, event_type, line, direction, track_id,
             employee_id, confidence, security_event_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (camera, timestamp, timestamp[:10], event_type, line, direction,
         track_id, employee_id, float(confidence), security_event_id),
    )
    conn.commit()
    return int(cur.lastrowid)


def query_entry_exit_events(conn, *, camera: str | None = None,
                            day: str | None = None, event_type: str | None = None,
                            line: str | None = None, track_id: str | None = None,
                            limit: int = 500) -> list[dict]:
    clauses: list[str] = []
    params: list = []
    if camera:
        clauses.append("camera = ?")
        params.append(camera)
    if day:
        clauses.append("date = ?")
        params.append(day)
    if event_type:
        clauses.append("event_type = ?")
        params.append(event_type)
    if line:
        clauses.append("line = ?")
        params.append(line)
    if track_id:
        clauses.append("track_id = ?")
        params.append(track_id)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = (
        "SELECT * FROM entry_exit_events " + where +
        " ORDER BY timestamp DESC LIMIT ?"
    )
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    cols = [d[1] for d in conn.execute("PRAGMA table_info(entry_exit_events)").fetchall()]
    return [dict(zip(cols, r)) for r in rows]
