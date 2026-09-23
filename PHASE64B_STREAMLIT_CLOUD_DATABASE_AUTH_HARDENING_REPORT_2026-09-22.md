# Phase 64B — Streamlit Cloud Database + Authentication Hardening Report

**Date:** 2026-09-23  
**Project:** CCTV Employee Productivity Tracker  
**Repo:** Yashmit-18/ai.cctv  
**Scope:** Cloud database bootstrap + dashboard authentication fail-closed (Phase 64B)

## Status Summary

- **CODE VERIFIED:** Database bootstrap and authentication fail-closed fix implemented and unit-tested.
- **LOCAL VERIFIED:** Full suite green under both literal commands; Streamlit AppTest scenarios pass (fail-closed / admin / viewer / rejected).
- **CLOUD VERIFIED:** **NOT VALIDATED.** No access to the deployed Streamlit Cloud runtime from this environment.
- **SECURITY BLOCKER:** RESOLVED (in code). With the new default `CCTV_DASH_FAIL_CLOSED=1`, a deployment with no dashboard credentials is blocked instead of opening `admin`. The live cloud deployment still needs a redeploy plus Secrets configuration to confirm.
- **ENVIRONMENT LIMITATION:** Streamlit Cloud's application filesystem is runtime storage, not durable production storage; the CCTV daemon/camera is intentionally absent from the cloud (`NO SNAPSHOT` state is expected and out of scope here).

---

## 1. Database Root Cause

The deployed `origin/main` (commit `541beb4`) contained the old
`get_readonly_connection()`:

```python
if not os.path.exists(db_path):
    return None
```

On a fresh Streamlit Cloud checkout there is no
`data/database/sessions.db`, so the dashboard returned `None` and rendered
`Database unavailable.` on the Employees (and other DB-backed) pages. No SQLite
exception occurred — it was a silent early existence check.

The configured path is Linux-compatible with no hardcoded `C:\`/`F:\` windows
path:

```python
DB_PATH = str(PROJECT_ROOT / "data" / "database" / "sessions.db")   # config.py
```

Cloud path: `<cloud checkout>/data/database/sessions.db` (exact absolute path
**NOT VALIDATED**).

## 2. Database Fix

`app.py:get_readonly_connection()` now (Phase 64A code, preserved in 64B):

1. Probes for the canonical schema (`employees` + `activity_logs`).
2. When the file is missing, empty, or schema-less: creates the parent
   directory, initializes the schema through the existing
   `src.database.get_connection()` + `src.database.init_db()` (idempotent,
   existing layer — `database.py` was not redesigned).
3. Reopens the file with a normal `sqlite3.connect` +
   `PRAGMA query_only=ON` (read-only semantics preserved, WAL rows visible).
4. First-run bootstrap is serialized by a `threading.Lock` because Streamlit
   Cloud runs concurrent sessions in threads; a race here could transiently
   surface `Database unavailable.`.
5. On failure, safe diagnostics (exception type/message, db path, cwd, parent
   existence, file existence — no secrets) are appended to the message instead
   of silently returning `None`.

Missing file → parent dir created → schema initialized → query-only connection
→ Employees page works. Covered by
`tests/test_dashboard_helpers.py` (12 tests incl. concurrent first-run).

## 3. Authentication Root Cause (SECURITY BLOCKER)

`config.py` previously defaulted:

```python
DASH_FAIL_CLOSED = os.getenv("CCTV_DASH_FAIL_CLOSED", "0") == "1"
```

With the combined defaults `CCTV_DASH_AUTH=1`, `CCTV_DASH_PASS=""`,
`CCTV_DASH_VIEWER_PASS=""`, `CCTV_DASH_FAIL_CLOSED=0`, `app.do_auth()` returned
`admin` with no login — the public cloud UI showed `Role: admin` /
`Full access` to an unauthenticated visitor.

## 4. Authentication Fix

`config.py`:

```python
DASH_FAIL_CLOSED = os.getenv("CCTV_DASH_FAIL_CLOSED", "1") == "1"
```

Fail-closed is now the secure default:

- NO VALID ADMIN/VIEWER CREDENTIAL + fail-closed default → access **denied**,
  never `admin`.
- Correct admin credentials (`CCTV_DASH_USER` + `CCTV_DASH_PASS`) → `admin`.
- Correct viewer password (`CCTV_DASH_VIEWER_PASS`, usernameless) → `viewer`.
- Wrong credentials → rejected (no role).
- Explicit `CCTV_DASH_AUTH=0` restores the documented open dev mode on a
  trusted network (existing behavior preserved; no passwords in source).

The existing RBAC architecture (`AccessGuard`, roles, session expiry) is
unchanged. `database.py` unchanged.

## 5. Exact Existing Config / Secrets Variable Names

Use exactly these (no invented names). Streamlit Cloud → Manage app → Settings →
Secrets, add these keys (each becomes an environment variable):

```toml
# .streamlit/secrets.toml on Streamlit Cloud
CCTV_DASH_AUTH = "1"                 # 1 enables the login gate
CCTV_DASH_FAIL_CLOSED = "1"          # secure default; 1 = deny without creds
CCTV_DASH_USER = "admin"             # admin username
CCTV_DASH_PASS = "replace-with-a-strong-password"
CCTV_DASH_VIEWER_PASS = "replace-with-viewer-password"
CCTV_DASH_SESSION_MINUTES = "30"     # session expiry
```

Other consts read from env at `config.py`:

| Variable | Default | Meaning |
|---|---|---|
| `CCTV_DASH_AUTH` | `1` | 1 = login gate, 0 = open dev mode |
| `CCTV_DASH_FAIL_CLOSED` | `1` (was `0`) | block when no credentials |
| `CCTV_DASH_USER` | `admin` | admin username |
| `CCTV_DASH_PASS` | `` | admin password (empty = no admin) |
| `CCTV_DASH_VIEWER_PASS` | `` | shared viewer password |
| `CCTV_DASH_SESSION_MINUTES` | `15` | session expiry minutes |

`validate_config()` preflight now flags `CCTV_DASH_FAIL_CLOSED=1` without a
password as a blocking problem (existing Phase 39 B4 check, now reachable by
default).

## 6. Fail-Closed Behavior

- Bare deployment (no secrets) → `do_auth()` returns `""`, dashboard shows the
  blocking error, no navigation rendered (`nav_options = 0`), no `admin` role.
- Configured admin credentials → dashboard renders all 10 nav pages.
- Configured viewer credentials → dashboard renders all 10 nav pages (read-only
  role).
- Incorrect credentials → `Invalid credentials.` error, no access.

## 7. Local Tests

```text
python -m pytest -q             1292 passed
python -m pytest -q -W error    1292 passed
```

Focused:
```text
tests/test_dashboard_auth.py    18 passed (H1/H2/H3 + Phase 64B matrix)
tests/test_dashboard_helpers.py 12 passed (DB bootstrap incl. concurrency)
tests/test_dashboard_app.py      7 passed (all 10 pages render, no exceptions)
tests/test_config.py            12 passed
tests/test_phase39.py           full suite green (fail-closed preflight intact)
```

Note: `pytest.ini` uses `testpaths = tests`; the top-level `test_droidcam*.py`
scripts are not collected and no unrelated CCTV code was modified to hide
anything. Both literal requested commands complete successfully.

## 8. Streamlit Local Verification

`python -m streamlit run app.py --server.headless true` → health `ok`.

AppTest scenario matrix (fresh processes, controlled env):

| Scenario | Result |
|---|---|
| Default (no creds, fail-closed=1) | BLOCKED, `nav_options=0`, no admin role — PASS |
| `admin` + correct `CCTV_DASH_PASS` | 10 nav pages, no errors — PASS |
| `viewer` + correct `CCTV_DASH_VIEWER_PASS` | 10 nav pages, no errors — PASS |
| `admin` + wrong password | `Invalid credentials.` rejected — PASS |
| Admin login → Employees page | No `Database unavailable.` error — PASS |

The open-mode AppTest sweep (Phase 42 shell) renders all ten pages (Employees,
Cameras, Admin Control Center, Settings, Reports, ...) without exceptions and
without any `Database unavailable.` state.

## 9. Cloud Deployment Status

**NOT VALIDATED.** The fix is not yet committed/pushed, so Streamlit Cloud still
runs the old code. Required steps after this phase:

1. Push this commit to `main` (authorized).
2. Add `CCTV_DASH_PASS` (and optionally `CCTV_DASH_VIEWER_PASS`) in
   Streamlit Cloud → Settings → Secrets (see §5). `CCTV_DASH_FAIL_CLOSED`
   already defaults to `1`.
3. Streamlit Cloud redeploys; verify live:
   - no credentials → access blocked (not `Role: admin`);
   - with credentials → Employees page loads, empty DB shows empty roster;
   - no `Database unavailable.`.

## 10. Remaining Limitations

- SQLite on Streamlit Cloud's application filesystem is runtime/ephemeral
  storage — a restart/redeploy may lose DB changes. A managed database is the
  intended production path (out of scope this phase; no second DB
  implementation was introduced).
- Cloud runtime and authentication have not been exercised against the live
  deployment (requires push + Secrets + manual verification).
- `CCTV_DASH_VIEWER_PASS` is a shared viewer password (existing design).
- The `NO SNAPSHOT` message on Live Monitoring is expected without the daemon;
  it is deliberately untouched.
- The 11 pre-existing deleted phase report files in the worktree are unrelated
  and left untouched.

---

## Labels

- Database code: **CODE VERIFIED**
- Authentication code: **CODE VERIFIED**
- Local tests: **LOCAL VERIFIED** (1292 passed, both literal commands)
- Local Streamlit: **LOCAL VERIFIED** (health ok, AppTest matrix PASS)
- Live cloud deployment: **NOT VALIDATED**
- Security blocker (unauthenticated admin): **RESOLVED in code; cloud confirm pending**
- Cloud filesystem persistence: **ENVIRONMENT LIMITATION**