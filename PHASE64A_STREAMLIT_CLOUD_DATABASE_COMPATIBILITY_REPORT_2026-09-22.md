# Phase 64A — Streamlit Cloud Database Compatibility Report

**Date:** 2026-09-22 (re-verified 2026-09-23)  
**Project:** CCTV Employee Productivity Tracker  
**Scope:** Streamlit dashboard database initialization only

## Status

- **CODE VERIFIED:** The current local code resolves the database from `config.PROJECT_ROOT`; no hardcoded Windows path is used by the application database configuration.
- **LOCAL VERIFIED:** SQLite bootstrap, dashboard smoke tests, and the full test suite pass (both literal `python -m pytest -q` and `python -m pytest -q -W error`).
- **CLOUD VERIFIED:** **NOT VALIDATED.** The deployed Streamlit Cloud Employees page was not accessible from this execution environment.
- **SECURITY BLOCKER:** The default configuration grants the `admin` role when no dashboard passwords are configured and `CCTV_DASH_FAIL_CLOSED=0`.
- **ENVIRONMENT LIMITATION:** Streamlit Cloud's local filesystem/database is runtime storage, not permanent production storage.

## Original Error

The deployed Employees page displayed:

> Database unavailable.

The deployed branch was inspected directly. `origin/main` at commit `541beb4` still contained:

```python
if not os.path.exists(db_path):
    return None
```

Therefore, when the cloud checkout had no `data/database/sessions.db`, the dashboard returned `None` before attempting directory creation or schema initialization. No SQLite exception was exposed by that code path; the failure was an early existence check.

## Database Path

The configured path is defined in `config.py` as:

```python
DB_PATH = str(PROJECT_ROOT / "data" / "database" / "sessions.db")
```

`PROJECT_ROOT` is:

```python
Path(__file__).resolve().parent
```

This is Linux-compatible and does not contain a hardcoded `F:\` or `C:\` path. On Streamlit Cloud, the resolved path is expected to be the checked-out application directory followed by:

```text
<cloud checkout>/data/database/sessions.db
```

The exact absolute cloud path is **NOT VALIDATED** because the deployed runtime was not available for inspection.

Local diagnostic result:

```text
cwd=F:\synilogic\cctv monitoring
project_root=F:\synilogic\cctv monitoring
db_path=F:\synilogic\cctv monitoring\data\database\sessions.db
parent_exists_before=True
db_exists_before=True
connect=True
select=1
tables=29
required_tables=True
```

## Existing Database Layer

`src/database.py` remains the authoritative implementation. It was not replaced or redesigned.

`get_connection()`:

1. Creates the parent directory with `os.makedirs(..., exist_ok=True)`.
2. Opens SQLite with `sqlite3.connect()`.
3. Enables WAL mode, busy timeout, and normal synchronous mode.

`init_db()` is idempotent and successfully initializes an empty database. The local direct diagnostic confirmed that the existing initializer creates the expected `employees` and `activity_logs` tables, along with the rest of the application schema.

No SQLite URI, `mode=ro`, or `immutable=1` connection is used by the current dashboard reader. The query-only connection is opened only after normal initialization/probing.

## Exact Fix

`app.py:get_readonly_connection()` now:

1. Resolves the configured path with `Path`.
2. Detects a missing database or an existing empty database.
3. Opens the database through `src.database.get_connection()`.
4. Runs the existing `src.database.init_db()` initializer.
5. Closes the bootstrap writer connection.
6. Opens the normal SQLite dashboard connection.
7. Enables `PRAGMA query_only=ON` for dashboard reads.
8. Records safe diagnostics containing only stage, exception type/message, path, current directory, parent existence, and file existence when a failure occurs.

First-run bootstrap is serialized with a module-level `threading.Lock` (`_DB_BOOTSTRAP_LOCK`) because Streamlit Cloud runs concurrent sessions in separate threads; without it, two simultaneous first loads could initialize the same fresh database at once and transiently fail on a busy/locked SQLite handle. With the lock, the schema needs are re-probed inside the critical section so only one initializer ever runs.

The Employees page includes that safe diagnostic detail when its connection fails. No passwords, API keys, RTSP credentials, embeddings, or biometric data are included.

The database schema file `src/database.py` was not modified.

## Focused Tests

Added to `tests/test_dashboard_helpers.py`:

- Missing database file with missing parent directory.
- First-run schema initialization.
- Query-only connection after initialization.
- Required tables exist.
- Existing zero-byte SQLite file initialization.
- Concurrent first-run (4 threads race on a brand-new database; every caller gets a schema-bootstrapped read-only connection).

Focused result:

```text
12 passed in 3.34s
```

## Local Verification

Streamlit was started successfully with:

```text
python -m streamlit run app.py --server.headless true --server.port 8505
```

Health endpoint result:

```text
ok
```

Local fresh-deployment smoke (simulates the cloud first-run against a
brand-new database path with no parent directory):

```text
db_path= ...\fresh-cloud-dir\database\sessions.db
parent_exists=True
file_exists=True
conn=True
employees=True activity_logs=True
write_blocked=OperationalError        # PRAGMA query_only=ON enforced
db_error=''
```

Corrupt-file diagnostic (surfaces the real exception instead of silently
returning None):

```text
stage=probe; exception=DatabaseError: file is not a database;
database_path=...; cwd=...; parent_exists=True; file_exists=True
```

The existing Streamlit AppTest suite exercises the dashboard shell and switches
through all ten navigation pages (Dashboard, Live Monitoring, Employees,
Security, Reports, Analytics, Cameras, Admin Control Center, Settings,
Deploy / System Check) without exceptions and without any `Database unavailable.`
error. This covers Employees, Cameras, Admin Control Center, Settings, and
Reports rendering paths without requiring a live camera.

Full test results (literal commands requested by the phase):

```text
python -m pytest -q            1285 passed (pre-hardening); 1286 passed (post-hardening)
python -m pytest -q -W error   1286 passed
```

Both literal commands complete successfully; nothing is skipped by using
`-W error`. (The top-level `test_droidcam*.py` scripts are not collected
because `pytest.ini` sets `testpaths = tests`.)

## Git / Deployment State

Current branch:

```text
main...origin/main
```

Current remote commit:

```text
541beb4c10b5b22cf9edfdd9200407989bf2a2f5
fix: prepare Streamlit Cloud deployment
```

The database fix and focused tests are currently local uncommitted changes. `origin/main` was verified to contain the old early-return implementation and does not contain this fix. No commit or push was performed, per instruction.

Unrelated pre-existing deleted phase report files remain in the worktree and were not changed by this phase.

## Authentication / RBAC

The current defaults are:

```text
DASH_AUTH_ENABLED=True
DASH_ADMIN_PASS=""
DASH_VIEWER_PASS=""
DASH_FAIL_CLOSED=False
```

With those defaults, `_auth_enabled()` is false and `do_auth()` returns `admin` without a login. The deployed UI showing `Role: admin` therefore cannot be treated as authenticated admin access unless Streamlit Cloud secrets/environment variables configure a password or enable fail-closed mode.

**SECURITY BLOCKER:** Configure `CCTV_DASH_PASS` and preferably `CCTV_DASH_FAIL_CLOSED=1` in the deployment environment before exposing the dashboard. Authentication was not redesigned in this phase.

## Cloud Verification

**NOT VALIDATED.** The actual deployed Streamlit Cloud Employees page was not tested after this local change. The required result remains pending:

- Employees page must load without `Database unavailable.`.
- An empty database may show an empty employee state.
- The fix must first be committed and pushed, then redeployed by Streamlit Cloud.

## Remaining Limitations

- SQLite under Streamlit Cloud's application filesystem is suitable for a temporary/demo deployment, not durable production storage. A restart, rebuild, or redeploy may lose local database changes. A managed database (e.g. Postgres on the cloud) is the intended production path but was explicitly out of scope for this phase.
- The CCTV daemon and camera hardware remain intentionally absent from the cloud deployment. This does not prevent database-backed dashboard pages from initializing.
- A corrupt/non-SQLite `sessions.db` still surfaces `Database unavailable.` — but now with the exact exception, path and filesystem state shown in the message rather than silently returned as `None`.
