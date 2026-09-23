# PHASE 65 — Streamlit Cloud Stability + Premium SOC UI Redesign (Report)

**Date:** 2026-09-23
**Branch:** `main` (HEAD `5af0e94` — Phase 64C/64D, committed & pushed earlier)
**Status labels used below:** `CODE VERIFIED` / `LOCAL VERIFIED` / `CLOUD NOT VERIFIED` /
`ENVIRONMENT LIMITATION` / `NOT VALIDATED`

---

## 1. Scope

Make the dashboard reliable on Streamlit Cloud while redesigning it as a
"premium AI CCTV Security Operations Center" UI. Concretely:

- **Camera import failure (crash fix):** the Cameras page and Admin Control
  Center crashed on Streamlit Cloud because `src/camera_store.py` imported
  `src.camera` at module import time, and `src/camera.py` runs `import cv2`.
  On Cloud, `insightface` pulls the GUI OpenCV build, whose `cv2` needs
  `libGL.so.1`, so `import cv2` raised an ImportError => `cameras.py` and the
  Admin Control Center pages rendered a raw crash screen.
- **Premium SOC UI redesign** of the shell and pages (nav groups, KPI grid,
  day-one panels) without touching any semantic CCTV behaviour.
- **Never fabricate live data:** no snapshot / no daemon => honest "DAEMON
  OFFLINE" empty states, never invented cameras, employees or incidents.
- **Do not break the local daemon:** `main.py` and `src/camera_manager.py`
  keep importing `src.camera`/OpenCV and must keep working on Windows.

## 2. Root cause — `ImportError` on Streamlit Cloud  `CODE VERIFIED`

### 2.1 Immediate import chain

| Step | Location | What it imports | Verdict |
|---|---|---|---|
| `app.py: tab_admin_cameras` (~:2428) | `from src.camera_store import ...` | camera_store | 𝒪k after fix |
| `app.py` Admin Control Center (~:3009) | `from src.camera_store import CameraStore` | camera_store | 𝒪k after fix |
| `src/camera_store.py` (was line 49) | `from src.camera import VideoCapture` at module top | camera | **crash** (removed) |
| `src/camera.py:17` | `import cv2` at module top | OpenCV | **crash on Cloud** |

This is now **async-guarded**: `src/camera_store.py` no longer imports
`src.camera` at import time; the capture runtime is loaded lazily only by
"Test Connection".

### 2.2 Why Cloud resolves to a broken OpenCV

- `requirements.txt` has declared `opencv-python-headless>=4.9` since commit
  `541beb4`.
- BUT `insightface>=0.7.3` resolves to `insightface==1.0.1`, whose METADATA
  declares `Requires-Dist: opencv-python` (the GUI build). On Cloud, pip
  installs **both** OpenCV variants; the GUI build's files land last in the
  single `cv2` package directory, so `import cv2` imports GUI-cv2, which
  requires `libGL.so.1` (absent on Cloud images) => ImportError.
- Error text on the deployed app is **redacted**; the local machine cannot
  reproduce it (see §3).

### 2.3 Fix plan

1. **Lazy boundary (runtime fix):** `src/camera_store.py` now loads
   `VideoCapture` on demand via `load_camera_runtime()`; when the import
   fails it reports `RUNTIME_UNAVAILABLE` (clean dict, sanitized URL,
   no password) instead of crashing. `CODE VERIFIED`
2. **Dependency ordering (installer fix):** reorder `requirements.txt` so
   `opencv-python-headless>=4.9` is declared **last** (with an explanatory
   comment). pip processes a requirements file top-to-bottom; installing
   headless last makes headless overwrite the GUI `cv2` directory on Cloud,
   so `import cv2` resolves to the headless build. `CODE VERIFIED`
   `LOCAL VERIFIED` (requirement file parsing / install order); the actual
   Cloud resolve-check is `CLOUD NOT VERIFIED` until the app is re-deployed.

## 3. Local reproduction boundary  `ENVIRONMENT LIMITATION`

- Local Windows venv has BOTH `opencv-python==5.0.0.93` and
  `opencv-python-headless==5.0.0.93` installed (venv artifact); local
  `import cv2` succeeds (Windows OpenCV does not need libGL), so the exact
  Cloud exception cannot be triggered natively.
- Local simulations therefore **inject** the Cloud condition in tests:
  block `import cv2` at the interpreter level (ImportError, like a missing
  libGL) and evict already-imported camera-runtime modules so the lazy
  import genuinely re-runs (`tests/test_phase65.py`).
- The exact behaviour "on the deployed Cloud app" independently is NOT
  validated from this machine => `CLOUD NOT VERIFIED`.

## 4. Camera runtime boundary  `CODE VERIFIED` / `LOCAL VERIFIED`

| Concern | Implementation |
|---|---|
| Capture runtime import | `src/camera_store.load_camera_runtime()` tries `from src.camera import VideoCapture`; on ImportError returns `(None, error)` and logs |
| Test Connection in cloud | Returns `ok=False, connected=False, health="RUNTIME_UNAVAILABLE"`, `error="camera runtime unavailable in this deployment environment: ..."`, URL redacted, password never present |
| Test Connection in a real backend | Uses the real `VideoCapture` path; start failures are caught, capture always `stop()`ed in `finally`, failed probes return clean `ok=False` dicts |
| Module import safety | `import src.camera_store` no longer needs OpenCV at all; all pure configuration ops (CRUD, validation, apply handshake, resolution) are runtime-free |

`tests/test_phase61.py` (real local probe path, incl. start-failure/provide-dict
test at line 492) and `tests/test_phase65.py` (cloud simulation) both cover this.

## 5. Dependency hardening in `requirements.txt`  `CODE VERIFIED`

- Moved `opencv-python-headless>=4.9` to the very end of the file, AFTER
  `insightface` (and after torch), with a comment explaining the
  insightface GUI-OpenCV dependency and why installing headless last wins
  the `cv2` slot.
- No `libGL`, `libGLU1`, `xvfb`, `GTK`, or `X11` OS packages are added; the
  dashboard only uses headless-compatible camera APIs.
- Local machines that installed BOTH cv2 wheels can clean up with
  `pip uninstall opencv-python` (advice only; the venv is left untouched).
- Streamlit Cloud Python version is environment-determined and unknown from
  here; the fix is version-agnostic. `ENVIRONMENT LIMITATION`

## 6. Error UX — raw tracebacks are never shown  `CODE VERIFIED`

- `main()` page dispatch is wrapped in `try/except`.
- On error the user sees a clean error card (title + message + diagnostic
  id) via `_render_error_card(...)`; the technical detail is logged
  internally (`logging.getLogger("cctv.dashboard").error(...)`).
- Streamlit's own red screen (which shows the raw traceback and the
  `st.exception` flag in AppTest) is avoided, so a cloud-only camera import
  failure can never again produce a raw traceback UI.
- All pages still render without exception (`AppTest at.exception` None).

## 7. Premium SOC UI redesign  `CODE VERIFIED` / `LOCAL VERIFIED`

### 7.1 Shell / global theme
- Extended `_COMPONENT_CSS` and `_inject_shell_css`: deeper navy/charcoal
  palette, top brand accent bar, metric accent top-border, and `p65-*`
  utility classes (page header, panels, error card, session card, nav legend).
- Both old and new component classes shipped in the same `_COMPONENT_CSS`
  string so prior element classes keep working.

### 7.2 Topbar (`_render_topbar`)
- Adds a real **SYSTEM ONLINE / SYSTEM DEGRADED / DAEMON OFFLINE** chip,
  keeps the snapshot chip, "Last updated", the role badge and the clock.
- Preserves the pre-existing chip semantics and the html-element count bars
  that the Phase 42 regression tests lock.

### 7.3 Sidebar navigation (`main()` block)
- Session card on top (Role / System / Snapshot).
- Group legend line: MONITORING (Dashboard · Live), PEOPLE (Employees),
  SECURITY (Security · Reports · Analytics), INFRASTRUCTURE (Cameras · Admin ·
  Settings · Deploy).
- **One radio** (`key="p42_nav"`) with the exact 10 options (required by
  `test_phase38` / `test_dashboard_phase42`); legend is only a caption.
- Refresh now / Log out buttons keep `p42_refresh` / `p42_logout` keys;
  banner and auto-refresh caption retained.

### 7.4 Dashboard page (`tab_live_overview`)
- Purpose: `st.header("Live Overview")` + `p42-page-sub` premium subtitle
  (keeps the Phase-42 header assertion).
- **9-card KPI grid** (`_render_dashboard_metrics`) replacing the old
  5+4 rows. Labels preserved exactly:
  Cameras Online / People Active / Away / On Phone / Open Incidents /
  Recognised Today / Avg Productivity / Active Hours Today / Phone Today
  (the 4 labels asserted by `test_dashboard_phase42` keep their values).
- DAEMON OFFLINE premium panel + stale-data warn panel when there is no
  (or a stale) snapshot. Nothing is fabricated; KPI helpers unchanged.
- Productivity chart + optional detailed table unchanged.

### 7.5 Live Monitoring page (`page_live_monitoring`)
- Page-head + premium DAEMON OFFLINE panel when no snapshot.
- People cards only rendered for a live snapshot; otherwise the honest
  "PEOPLE PRESENCE UNAVAILABLE" panel (no simulation).
- html-element count still >=5 with the fixture snapshot (Phase 42 test).

### 7.6 Employees page (`tab_employees` / `tab_live_employees`)
- `p42-page-head` header + "NO EMPLOYEES CONFIGURED" premium empty panel
  when the roster is empty.

### 7.7 Cameras page (`tab_admin_cameras`)
- Overview KPI row: **Total Cameras / Online / Offline / Disabled** (real
  data; no live snapshot => Offline shows "—", never inventing numbers).
- Richer "Camera list" table: Name / ID / Type / Location / Status /
  Resolution / FPS / Reconnect Policy / **Credentials** (CONFIGURED/NONE).
- Credentials are never rendered (no password, no credential-bearing URL);
  the redacted example line previously shown in the caption is dropped.
- Test Connection now reports **"TEST CONNECTION UNAVAILABLE IN CLOUD"**
  when the runtime is absent (given health == RUNTIME_UNAVAILABLE),
  distinct from "Connection unavailable" for real backend failures.
- Add / Edit / Enable-Disable / Test / Apply / Delete semantics unchanged.

### 7.8 Other pages
- Security, Reports, Analytics, Admin Control Center, Settings, Deployment
  keep their content/behaviour (only shell/global styles changed). Note:
  the Admin Control Center also imports `src.camera_store` which is now
  cv2-safe (`CODE VERIFIED` via cloud-sim AppTest of all 10 pages).

## 8. Preserved semantics (explicit)  `CODE VERIFIED`

- **No CCTV semantic changes:** ACTIVE / ON_PHONE / AWAY / Unknown handling,
  state thresholds, schedule, productivity metrics, incidents, evidence and
  retention rules are untouched in `src/tracker.py`, `src/analytics.py`,
  `src/incidents.py`, `src/security_*`.
- **64D performance work preserved:** session-shared read-only DB handle
  (`_cctv_readonly_conn_<path>` in `app.py`), day/range analytics cache TTLs
  (15 s / 60 s), lazy heavy-AI imports, uncached live snapshot reads.
- **Auth flow untouched:** `do_auth`, fail-closed `CCTV_DASH_AUTH`, viewer
  restrictions, no-credential path — unchanged (`test_dashboard_auth.py` green).
- **Never display/store secrets:** runtime boundary, `safe_dict`,
  `_render_topbar`, tables and probes all avoid secrets; `test_phase38`
  credential-leak tests green.

## 9. Tests  `CODE VERIFIED` / `LOCAL VERIFIED`

| Command | Result |
|---|---|
| `python -m pytest -q` (full) | **1308 passed** in 174.50 s (was 1301 + 7 new) |
| `python -m pytest -q -W error` (full) | **1308 passed** in 99.30 s |
| `tests/test_phase65.py` (new, cloud simulation) | 7 passed |
| `tests/test_phase61.py` (camera store, real probe) | 33 passed |
| `tests/test_phase62.py test_phase63.py` | 54 passed |
| `tests/test_phase38.py test_dashboard_phase42.py test_dashboard_app.py test_dashboard_auth.py test_config.py test_dashboard_helpers.py` | 83 passed |

New test coverage (`tests/test_phase65.py`):
1. store pure ops exist with cv2 blocked;
2. CameraStore CRUD works with cv2 blocked (password never leaks in any shape);
3. `test_camera_connection` returns `RUNTIME_UNAVAILABLE` and no credential leak;
4. real capture path used when the runtime is available (ok=True) with redaction;
5. failed real probe never leaks credentials;
6. Cameras page renders in cloud mode, shows the 4 KPIs, and the
   Test-Connection button yields "TEST CONNECTION UNAVAILABLE IN CLOUD";
7. all 10 navigation pages render without exception in cloud mode.

## 10. Local Streamlit (headless)  `LOCAL VERIFIED`

- `python -m streamlit run app.py --server.headless true --server.port 8513`
  starts cleanly.
- `/_stcore/health` => `ok`; base path => HTTP 200.
- No traceback in the captured stderr during boot/render.
- All 10 pages additionally rendered without exception under the cloud
  simulation via AppTest (see §9).

## 11. Streamlit Cloud verification  `CLOUD NOT VERIFIED`

Not verified from this machine: the app has not been re-deployed to
Streamlit Cloud, so the actual `import cv2` behaviour on the Cloud image is
not independently confirmed. The runtime boundary + headless-last dependency
ordering make the crash impossible, but that is only proven by simulation
here, not on the deployed app. A follow-up deploy + smoke test is required.

## 12. Files changed (working tree, NOT committed)

```
 M app.py                (shell/CSS/sidebar/topbar redesign + error UX +
                          Cameras page + DAEMON OFFLINE panels +
                          9-card KPI grid + phase-65 helpers)
 M requirements.txt     (headless moved last, after insightface + comment)
 M src/camera_store.py  (lazy load_camera_runtime boundary + start-failure
                          hardening in probe)
?? tests/test_phase65.py (new cloud-simulation suite)
git diff --stat: 3 files changed, 383 insertions(+), 113 deletions(-)
```

## 13. Commit / push  `NOT PERFORMED`

No commit and no push were performed in Phase 65 (per the phase rule). The
working tree contains the changes listed in §12 on top of committed HEAD
`5af0e94`.

## 14. Conclusion

- The Cloud crash root cause is understood and fixed at two levels: a lazy
  capture-runtime boundary in `src/camera_store.py` and an opencv-headless
  install-order fix in `requirements.txt`, with the UI never exposing raw
  tracebacks.
- The dashboard is redesigned as a premium SOC UI while preserving every
  semantic guarantee (no fabricated live data, unchanged CCTV semantics,
  64D performance, auth/redaction rules).
- Full suite green: **1308 passed** (plain and `-W error`), including 7 new
  cloud-simulation tests and the hardened probe test.
- Remaining: deploy to Streamlit Cloud and run a smoke test to flip the
  `CLOUD NOT VERIFIED` label to `CLOUD VERIFIED`.