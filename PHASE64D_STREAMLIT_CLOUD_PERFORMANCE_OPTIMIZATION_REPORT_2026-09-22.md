# Phase 64D — Streamlit Cloud Dashboard Performance Optimization Report

**Date:** 2026-09-23
**Project:** CCTV Employee Productivity Tracker
**Repo:** Yashmit-18/ai.cctv
**Scope:** Reduce dashboard navigation/render latency on Streamlit Cloud without
changing any CCTV/AI/PB semantics.

## Status Summary

- **CODE VERIFIED:** Optimization implemented (session-shared read-only SQLite
  handle + configurable read-query cache TTLs). No YOLO/phone/Face/ReID/tracker/
  productivity/security/RTSP/camera/enrollment logic touched.
- **LOCAL VERIFIED:** Full suite green (both literal commands), focused bundle
  green, headless boot `health=ok`, authenticated 10-page nav sweep renders with
  zero `Database unavailable.` errors, and the session now holds exactly **one**
  SQLite handle across all page navigations + refresh clicks.
- **CLOUD VERIFIED:** **NOT VERIFIED.** A live cloud perf delta cannot be
  measured from this Windows environment. Do not treat these numbers as cloud
  timings.
- **NO SNAPSHOT behavior:** unchanged. The cloud dashboard still reads the
  missing/stale live snapshot and shows the daemon-offline message; it never
  attempts to open a physical camera.

---

## 1. Current Rerun Architecture (audit)

`app.py` (3602 lines) is a single-file Streamlit app:

- **Shell (Phase 42):** `main()` (`app.py:3525`) → `do_auth()` → CSS injection →
  sidebar (`Navigation` radio `p42_nav`, `Refresh now`, `Log out`) → dispatch to
  exactly ONE page function per run by `_NAV_KEYS[choice]`. Pages:
  Dashboard, Live Monitoring, Employees, Security, Reports, Analytics, Cameras,
  Admin Control Center, Settings, Deploy / System Check (10 pages).
- **Widget→rerun:** every widget interaction (radio change, button click, form
  submit, checkbox) triggers a **full top-to-bottom script rerun** in Streamlit.
  There is no partial nav; `main()` runs `do_auth()`, re-renders the sidebar
  (including `_snapshot_status(load_live_state())`) and then the selected page.
- **Auto-refresh:** already isolated with **`st.fragment(run_every=X)`**
  (`_auto_fragment`, `app.py:3592`) for the three live-capable tabs
  (Dashboard `tab_live_overview`, Live Monitoring `page_live_monitoring`,
  Security `tab_security`). Static/admin pages (Employees, Cameras, Admin,
  Settings, Reports, Analytics, Deploy) **do not auto-refresh** — they rerun only
  on genuine interaction. This was already correct; no global 5s rerun exists.
- **`time.sleep` / loops:** none in `app.py`. The daemon (`main.py`) has its own
  sleep pacing and is untouched.
- **Repeated configuration loading:** `config.*` constants are read once at
  module import; `import config` inside page functions is a `sys.modules` no-op.
- **Model initialization:** none at import or nav (see §7).
- **Camera initialization:** none from the dashboard (see §8).

Per-render work that runs on every interaction:

| Operation | Where | Cost class |
|---|---|---|
| `do_auth()` + session expiry | `main()` every run | cheap (dict lookups) |
| Sidebar snapshot caption file read | `main()` every run | cheap (~0.03 ms absent) |
| `load_live_state()`/`live_state_age()` | live + security pages | cheap file read, **uncached** |
| SQLite **open + schema probe + close** | 13 read sites across pages | the primary target — was per-render |
| `cached_day_metrics` (ttl was 5 s) | Dashboard/Live every auto-tick | recomputed every 5 s tick |
| `cached_range_metrics` (ttl was 10 s) | Analytics/Reports | recompute on interaction |
| Employees faces `glob` | Employees page only | small |
| Plotly figure builds / dataframes | Dashboard/Analytics | small (~10-30 ms) |
| Report dir listing | Reports page only | small |
| Preflight checks | Deploy page only | small (read-only module checks) |

## 2. Identified Bottlenecks / 3. Root Cause of Navigation Delay

Measured on this machine (SQLite local cheap; Python 3.12.6, `.venv`):

```
cold_import_app                       2.195 s   (per browser session, one-time)
_probe_schema                         3.256 ms/op
get_readonly_connection open+close    3.377 ms/op
_read_live_file (absent)              0.026 ms
employee_day_metrics (cached)         0.974 ms
employee_range_metrics (30d, cached) 22.337 ms
EmployeeStore.list                    6.308 ms
faces_dir glob                        0.230 ms
DB rows: 4 employees / 168 activity_logs
```

**Root cause of the cloud navigation delay (triple, ordered by importance for
cloud):**

1. **Per-render connection churn.** Each of the 13 read sites re-opened a fresh
   SQLite connection AND re-ran the schema probe (`_probe_schema` → another
   open + `sqlite_master` query) then closed it, on every page render. On a slow
   filesystem (Streamlit Cloud GFFS / ephemeral disk) each open+probe is
   proportionally far more expensive than the ~6.6 ms seen locally, and an
   active session pays it once per page turn.
2. **Cache TTL == auto-refresh period.** `cached_day_metrics` used `ttl=5` while
   the live fragment reruns every 5 s → the entry was stale on every tick, so
   the live page re-opened a connection and re-ran the query every 5 seconds
   forever. `cached_range_metrics` (`ttl=10`) churned similarly during
   Analytics/Reports interaction.
3. **Full-app rerun + cold session import.** Every nav is a full rerun (Streamlit
   architecture — retained), and the first run of a session pays ~2.2 s locally
   (pandas/plotly + module graph; higher on cloud). This is the cost the user
   perceives while the spinner is up on the very first navigation of a session.

Local per-page wall-times (AppTest, this machine) were ~0.46-0.64 s before AND
after — Streamlit's in-process execution overhead dominates at local sqlite
speeds, so wall-clock deltas here are within noise. The saving is structural
(one handle per session instead of one per render) and matters where open cost
is non-trivial (cloud FS).

## 4. Auto-Refresh

- Live tabs auto-refresh via `st.fragment(run_every=DASH_REFRESH_SEC)`
  (default 5 s). Static/admin pages do not auto-refresh. **Unchanged.**
- The live snapshot file read (`load_live_state`) is **never cached** — live
  KPIs stay 5 s fresh.
- The metrics caches now expire on **longer, configurable** TTLs
  (`CCTV_DASH_DAY_METRICS_TTL` default 15 s, `CCTV_DASH_RANGE_METRICS_TTL`
  default 60 s) so an auto-tick no longer forces a reconnect+requery every
  round. Both bounds are visible on the Settings page (read-only) so the trade
  is transparent.
- Refresh now button and logout are unchanged (`st.rerun()`).

## 5. Database Behavior

- `get_readonly_connection()` is now **session-scoped shared**:
  - opened once per Streamlit **session** (`st.session_state["_cctv_readonly_conn_<path>"]`),
  - reused across all pages and navigations (verified: same object after
    Employees → Settings → … 10 pages + 2 refreshes, still queryable),
  - `close()` is a no-op on the shared handle (`_ReadonlyConnection` subclass),
    so the existing `finally: conn.close()` guards can stay in place,
  - thread-safe by construction: a Streamlit session runs script turns
    sequentially on one thread (SQLite `check_same_thread=False` retained for
    WAL correctness as before),
  - a thread-safe first-run bootstrap (`_DB_BOOTSTRAP_LOCK`, schema init) is
    preserved exactly for the initial open.
- New `_open_readonly_connection()` keeps the old per-call open+bootstrap+probe
  semantics for short-lived callers (the cached metric helpers — run in the
  Streamlit cache thread where `session_state` is unsupported — and any caller
  outside a running runtime, e.g. unit tests). Unshared handles close normally
  (new test `test_open_readonly_connection_unshared_closes_normally`).
- Write path unchanged: `_open_write_conn()` still opens a fresh write
  connection per mutation; read/write layering and WAL visibility are
  unchanged (write-side `st.cache_data.clear()`/`st.rerun()` invalidation paths
  preserved).
- SQL semantics unchanged. No schema change. No second database layer.

## 6. Cache Changes

| Cached value | Before | After | Safe? |
|---|---|---|---|
| `cached_day_metrics` | `ttl=5` | `CCTV_DASH_DAY_METRICS_TTL` (default 15 s) | read-only today-aggregate; admin mutations already clear metrics caches |
| `cached_range_metrics` | `ttl=10` | `CCTV_DASH_RANGE_METRICS_TTL` (default 60 s) | read-only history aggregate; worst staleness 60 s |
| live snapshot (`load_live_state`) | uncached | uncached (kept) | live state must never be cached |
| auth/session/role | uncached | uncached (kept) | never cache auth state |
| camera config / mutations | via DB queries per render | shared read handle | same freshness as before |

TTL knobs are additive env vars; defaults preserve honest real-data behavior
("refreshed with the rest of this page" stays truthful within ≤15/≤60 s).

## 7. Model Initialization

Verified by AST scan of `app.py` module-level imports: **NONE** of
torch/ultralytics/insightface/onnx/onnxruntime/cv2/face are imported at module
level or during navigation. Dashboard imports are streamlit/pandas/plotly plus
light `src.*` store modules. `src.face_registry` (the heavy face stack) is
imported only inside the "Validate enrollment" button handler — a deliberate,
already-lazy path. Opening any admin page triggers no AI model load.

## 8. Camera Initialization

The dashboard never opens a camera device and never constructs the camera
manager. It only reads `data/live_state.json` (uncached) and renders whatever
the snapshot says. On the cloud, with no snapshot, it continues to show the
required **NO SNAPSHOT / daemon offline** state without touching a camera.
Local Windows daemon behavior untouched.

## 9. Files Changed

- `app.py` — connection refactor (§5), TTL wiring + Settings display (§6).
- `config.py` — `DASH_DAY_METRICS_TTL_SEC`, `DASH_RANGE_METRICS_TTL_SEC`.
- `.env.example`, `README.md` — new env knobs documented.
- `tests/test_config.py` — default + configurable TTL tests.
- `tests/test_dashboard_helpers.py` — unshared-close test.
- `tests/test_dashboard_app.py` — session-shared-handle reuse test.
  (`tests/test_dashboard_auth.py` carries the unrelated Phase 64C additions
  already present in the worktree; not touched in 64D.)

## 10. Tests

```
python -m pytest -q           1301 passed
python -m pytest -q -W error  1301 passed
(1297 at 64C → 1301: +4 Phase 64D tests)

Focused: test_dashboard_helpers + test_config + test_dashboard_app +
         test_dashboard_auth + test_phase39  → 68 passed
```

The 10-page navigation AppTest (`test_sidebar_navigation_switches_every_page_...`)
and the new shared-handle test run under the AppTest harness. Existing
DroidCam/headless-OpenCV limitation is pre-existing and unrelated; no camera
code modified.

## 11. Before / After (LOCAL measurements only)

Sub-op measurements (this machine):
- Cold `import app` ≈ 2.2 s (session-level, unchanged).
- Per-render SQLite handles: before = every read site opened+probed+closed;
  after = **1 per session** (verified object identity across 10 pages + 2
  refreshes in one AppTest session).
- Cache recompute cadence: day metrics 5 s → 15 s; range 10 s → 60 s.
- Page nav wall-times (AppTest): ~0.46-0.64 s before / ~0.47-0.64 s after —
  equivalent at local sqlite speeds; the benefit targets slow cloud filesystems
  and is stated as structural, not measured-in-ms locally.

## 12. Local Verification

- `python -m streamlit run app.py --server.headless true` → `health=ok`.
- Authenticated AppTest (admin@gmail.com / dummy password): login → all 10 nav
  pages render, `db_unavailable_errors= none`.
- Session-shared handle reuse + usability confirmed.
- Logout/login, viewer role, fail-closed, session expiry behaviors untouched
  (64B/64C suites still pass).

## 13. Cloud Verification Status

**NOT VERIFIED.** Requires the deployed Streamlit Cloud instance. Perf deltas on
cloud cannot be inferred as measured from this local Windows environment.

## 14. Remaining Limitations

- Every nav interaction still triggers a full Streamlit rerun (platform
  architecture; not removed, not fake-partitioned). Auto-refresh is already
  fragment-scoped.
- Cold session import (~2.2 s local) is unchanged; it is one-time per session
  but appears on the first render after app wake/sleep.
- Long-lived read handles hold the DB file open for the session lifetime; a
  mid-session manual DB delete would raise operational errors until the session
  reconnects (same surface as before, rare, and diagnosable via the existing
  `_format_db_error` path). Schema-bootstrap/probe still runs on the initial
  open.
- Metric caches are up to 15/60 s stale (configurable); live snapshots remain
  5 s fresh.
- Cloud wall-clock gains are expected on connection/open costs and not locally
  measurable — do not treat LOCALLY captured timings as cloud performance.

---

## Labels

- Optimization code: **CODE VERIFIED**
- Structural reuse (one handle/session): **LOCAL VERIFIED**
- Local tests: **LOCAL VERIFIED** (1301 passed, both literal commands)
- Local Streamlit (headless + authed nav sweep): **LOCAL VERIFIED**
- Live cloud performance delta: **NOT VERIFIED / ENVIRONMENT LIMITATION**
- NO SNAPSHOT / daemon-offline cloud state: **UNCHANGED (expected)**
- Unauthenticated-auth, fail-closed, session handling: **UNCHANGED (64B/64C suites green)**