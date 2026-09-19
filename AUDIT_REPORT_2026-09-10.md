# AI CCTV Intelligence & Security Platform — Full-System Audit Report

**Date:** 2026-09-10
**Auditor:** opencode (senior architect / AI-CV / security / QA / production-readiness sweep)
**Method:** evidence-based audit — source (`file:line`), config, tests, and runtime correctness. NO commit/push performed. No working architecture refactored.

---

## 1. Repository Summary

| Item | Value |
|---|---|
| Root | `F:\synilogic\cctv monitoring` |
| Daemon | `main.py` (orchestrator) + `src/` (45 modules) |
| Dashboard | `app.py` (Streamlit, 1701+ lines, 10 tabs) |
| Tests | `tests/` — **745 tests at baseline**, **753 after this audit** |
| Models | `models/yolov8n.pt` (COCO) only; InsightFace `buffalo_l` auto-downloaded to user cache |
| Docs | `README.md`, `PRODUCTION_READINESS.md`, `PROJECT_FULL_SYSTEM_AUDIT.md`, `VALIDATION_REPORT.md`, `FIX_REPORT_2026-09-09.md`, `GAP_ANALYSIS_PHASE_AB.md` |

## 2. Git Baseline

- Branch `main`; HEAD `c95f391` "Fix security events during full camera outage".
- Prior commits: `1188a93` ("Ignore local backup files"), `882b696` ("Initial AI CCTV intelligence platform").
- Ignored: `.env`, `.venv/`, `data/`, `models/`.
- Pre-existing uncommitted change before audit: `src/alerts.py` — **trailing-whitespace only** (line 178); left untouched.
- Pre-existing untracked artifact: a file literally named `s -ExecutionPolicy RemoteSigned) ; (& ￢f￥synilogiccctv monitoring.venvScriptsActivate.ps1￢)` — accidental PowerShell activation output; **candidate for manual deletion**, not project data.

## 3. Environment Baseline

| Item | Value |
|---|---|
| OS | Windows (win32) |
| Global Python | `C:\Python314\python.exe` 3.14.6 — **incomplete deps** (no pytest/streamlit/insightface) |
| Project env | `.venv` Python 3.12.6, `pytest 9.1.1`, `streamlit 1.62.0`, `torch 2.13.0+cpu`, `ultralytics 8.4.135`, cv2 5.0.0, pandas 3.0.5, openpyxl 3.1.5 |
| OS constraint | Windows Application Control blocks `scipy/_distance_wrap` DLL inside the venv → `sklearn` unusable in the venv. Affects nothing that pytest exercises (no test imports sklearn; insightface used in-process avoids it). |

**Interpretation of 745 tests collected/passing** — 55 test files; regex `^def test_*` gave 489; pytest collection is authoritative (**745**), the extra coming from parametrized/class tests. README/report files used counts 161/564/585/595/628/642/700/744 — **all stale/contradictory; corrected in README to 753.**

## 4. Initial Test Status (baseline)

```
745 passed in 80.13s   (pytest, .venv, default config)
```
No failures, no errors, no skips reported (real-feed camera tests ran — webcam index 0 reachable).

## 5. Final Test Status

```
753 passed in 78.20s   (pytest -W error, .venv)
```
745 baseline + **8 new dashboard-auth regression tests** (`tests/test_dashboard_auth.py`). Clean under `-W error` (the repo's documented production standard).

## 6. Architecture Overview (verified against source)

- **Capture layer** (`src/camera.py`, `src/camera_manager.py`): one reader thread per camera; reconnect backoff 2 s→30 s (×2.5); read timeout 4 s; stale frame 10 s; FROZEN streak 30 frames / ≥6 s; LOW_FPS threshold 5.0. `latest_frames()` excludes stale; `_usable_camera_frames()` gates on `{ONLINE, LOW_FPS, FROZEN}` + brightness — dark/blank feed ⇒ `camera_online=False`.
- **Detection** (`src/detector.py`): YOLO classes **person(0) + cell phone(67) only** (`_DETECT_CLASSES=[0,67]`); InsightFace detect + 512-D embed; device auto cuda/mps/cpu; `PHONE_PROXIMITY_RATIO=0.4`, `FACE_EXPAND_RATIO=1.5`. No fire/smoke/weapon/fall/PPE/violence — reserved `*SUSPECTED_*` types are honest forward keys, never claimed active.
- **Face registry** (`src/face_registry.py`): `data/faces/` scan, cache `data/embeddings.pkl` (CACHE_VERSION 3), `MIN_FACE_AREA` 120×120, duplicate rejection `COS ≥ 0.985`, 512-D vectors never logged.
- **Tracker** (`src/tracker.py`): `EmployeeTracker` FSM **ACTIVE / ON_PHONE / AWAY**; first observation adopted immediately (never AWAY-first); **camera-offline freeze** — no state change, no time accrual (`_since` advanced ⇒ offline gaps excluded from intervals); presence-patience holds candidate while any person visible (face-occluded), buffer widened when departing with person present; `MultiTracker` dedups multi-camera and logs transitions to SQLite; `SpatialTracker` IoU tracks with adopt-after-3 / never-demote / switch-after-3 identity voting.
- **Productivity** (`src/productivity.py` + `src/analytics.py`): single canonical formula `score = productive/(productive+phone+away)`, unobserved time excluded from denominator, lunch/grace/break knobs honored, Unknown filtered.
- **Security** (`src/security_engine.py`, `incidents.py`, `alerts.py`, `evidence.py`, `correlation.py`): orchestrates unknown-presence / after-hours / zone intrusion / loitering / occupancy / camera-offline-recovered / tamper; `INC-YYYYMMDD-NNNN` incidents; alert lifecycle `PENDING→DELIVERED→ACKNOWLEDGED→RESOLVED/ESCALATED` with retry=3, retry 60 s, escalate 900 s; evidence SHA-256 + OFF/HIGH_SEVERITY_ONLY/EVENT_ONLY capture gate + per-record retention. **Offline camera never emits employee AWAY/ACTIVE events.**
- **Database** (`src/database.py`): SQLite WAL, `busy_timeout 5000`, `synchronous NORMAL`, localtime timestamps; backup/restore + integrity drill.
- **Dashboard** (`app.py`): RBAC admin/viewer with audited denials, fail-closed auth (fixed this audit), session expiry (15 m default), auto-refresh 5 s fragments, honest "AI Capabilities" vocabulary (AVAILABLE/DISABLED/DEGRADED/NOT_CONFIGURED/FUTURE_MODEL_REQUIRED), deployment checklist + `src.preflight` integration.
- **Simulation** (`src/simulation/`): opt-in only, **never imported by production entry points**; `FaultInjector`, `failing_detector`, `AlertDelivererStub`, deterministic virtual camera; isolated from production data.

## 7. Invariant Conformance (test-confirmed)

| Invariant | Result |
|---|---|
| `UNKNOWN != EMPLOYEE` | ✅ Unknown tracked separately, excluded from roster/productivity |
| `CAMERA OFFLINE != EMPLOYEE AWAY` | ✅ FSM freeze; regression tests `test_tracker.py`, `test_tracker_regression.py`, `test_multiperson_pipeline.py` |
| Camera failure never fabricates AWAY / productivity loss | ✅ `test_offline_gap_excluded_from_committed_interval`, `test_camera_disconnect_never_manufactures_away` |
| Security/camera-health work with zero usable frames | ✅ `test_camera_failure.py`, offline failsafe in `security_engine` |
| N-persons → N detections, N-faces → N identities | ✅ `test_multiperson_pipeline.py` |

## 8. External Dependencies & Models

`requirements.txt` + `.env.example` are consistent with code. The only trained model present is `yolov8n.pt` (COCO, person+phone). InsightFace `buffalo_l` is runtime-downloaded; enrollment images live under `data/faces/`. RTSP/GPU/SMTP/Docker are config- and code-wired but **not exercisable in this environment** (honestly reported by `preflight` as NOT EXECUTED / environment limitation).

## 9. Security Review

- Dashboard auth fail-closed is now **correct** (see Findings H1–H3). Admin escalation with empty admin password is closed.
- No credentials/embeddings/face images in logs or README; `validate_config` and `preflight` redact secrets; existing secret-leak tests green.
- Session expiry, audited RBAC denials, evidence access audit all present.
- Residual: dashboard should be deployed behind TLS/reverse-proxy/VPN; `.env` must set real `CCTV_DASH_PASS` (default is development-only).

## 10. Issue Classification (severity legend)

| Severity | Meaning |
|---|---|
| **CRITICAL** | Data loss / security bypass / production-break |
| **HIGH** | Confirmed defect with security/reliability impact |
| **MEDIUM** | Confirmed defect, moderate impact |
| **LOW** | Cosmetic / doc / maintainability |

## 11. Issues Found — Fixed

| ID | Severity | Module | Finding (evidence) | Fix |
|---|---|---|---|---|
| **H1** | **HIGH** | `app.py:294-309` | Auth **fail-closed dead code**: `do_auth()` returns `"admin"` when `CCTV_DASH_FAIL_CLOSED=1` and no password configured, because `_auth_enabled()` short-circuited before `_auth_fail_closed()`. The dashboard (a separate process from the daemon) silently opened as admin despite the operator's fail-closed intent. | Reorder: `_auth_fail_closed()` now gates **before** the `_auth_enabled()` shortcut in `do_auth()`. |
| **H2** | **HIGH** | `app.py:794-796` | `_audit_report_export()` calls `logging.getLogger("cctv.dashboard")` but `app.py` never imported `logging` → **`NameError` inside the very exception handler meant to survive audit-write failures**, breaking report downloads exactly when the DB write fails. | Added `import logging` (top of file). |
| **H3** | **HIGH** | `app.py:329` | Empty-admin-password escalation: with only `CCTV_DASH_VIEWER_PASS` set, login `user="admin", pw=""` matched `pw == DASH_ADMIN_PASS` (`""`) and **granted admin**. | Refactored credential decision into pure `_authenticate(user, pw)`; admin only granted when `DASH_ADMIN_PASS` is non-empty. |
| **L1** | **LOW** | `app.py:1581` | Deployment checklist hardcoded stale claim `passes (642 tests)` (now 753). | Reworded to `passes (full suite green)`. |
| **L2** | **LOW** | `README.md` (4 spots) | Contradictory/stale test counts (161/564/744) in the primary doc. | Updated to verified **753**; added `-W error` run line + honest validation-status matrix (labels below). |

## 12. Issues Found — Not Fixed (accepted / environment-limited / design choice)

| ID | Severity | Area | Finding | Rationale / recommendation |
|---|---|---|---|---|
| N1 | MEDIUM (config) | Auth defaults | With no `.env` passwords and `FAIL_CLOSED=0` (default), dashboard open-as-admin is the documented dev mode. | Keep for dev; require `CCTV_DASH_PASS` in production (`preflight` enforces). Consider flipping default to fail-closed in release. |
| N2 | MEDIUM (UX/permission) | Tabs | Security tab is not hidden for `viewer` (actions inside are RBAC-gated). | Observed design choice (viewer = monitor). If stricter separation needed, add tab-level gating. |
| N3 | MEDIUM (AI honesty) | Tracker | When the entire office empties (`camera_online=True`, zero detections), `process_batch` freezes instead of promoting to AWAY — deliberate, heavily tested ("never fabricate AWAY"). Consequence: AWAY unreachable in single-person empty-scene via the batch path. | Keep (matches invariant philosophy). Documented; do not "fix" without product confirmation. |
| N4 | LOW (cleanup) | Repo | Stray untracked PowerShell-activation artifact file. | Delete manually (not project data). |
| N5 | LOW (redundancy) | Docs | Overlapping secret/redaction tests across several files; historical phase reports carry stale counts. | Optional dedupe; dated reports intentionally left. |

## 13. Files Modified (all uncommitted, per policy)

| File | Change |
|---|---|
| `app.py` | H1 reorder + H3 `_authenticate` helper + H2 `import logging` + L1 reword (48 lines touched) |
| `tests/test_dashboard_auth.py` | **NEW** — 8 regression tests (H1/H2/H3) |
| `README.md` | Test-count corrections (4 places), `-W error` line, validation-status matrix |

`src/alerts.py` — pre-existing whitespace change, **not touched** by this audit.

## 14. Verification After Fixes

- `pytest -q -W error` → **753 passed** (no failures, no warnings).
- Behavior locks: fail-closed blocks to `""`, not `"admin"`; blank admin password yields `""` (viewer remains possible); audit-write failure inside `_audit_report_export` is swallowed + logged, download survives.

## 15. Real vs. Code vs. Simulated vs. Not Validated vs. Not Implemented

**REAL VERIFIED** (this environment, real hardware/feed):
- Webcam capture (device 0), CPU YOLO inference, InsightFace recognition (EMP001 + Unknown simultaneously ACTIVE, documented live session).
- FROZEN_FRAME classification on **real webcam pixels** (5 tests; no sklearn/scipy needed; ran, not skipped, during this audit).

**CODE VERIFIED** (pytest): tracker FSM/invariants, productivity/analytics, security engine, incidents/alerts, evidence, RBAC + auth (incl. 8 new), backup/restore, config/preflight, detector registry honesty, simulation isolation.

**SIMULATED**: deterministic virtual-camera/fault scenarios; bounded fast-time soak.

**NOT VALIDATED**: real RTSP/NVR pool, GPU/CUDA, SMTP delivery, Docker compose, multi-hour wall-clock soak, two-or-more simultaneous roster-face recognition (only single-operator webcam + known/unknown live).

**NOT IMPLEMENTED**: fire/smoke/weapon/fall/PPE/violence classes (reserved keys for future models only).

## 16. Production Readiness Verdict

### PILOT READY

- **Not PRODUCTION READY / not CONDITIONALLY PRODUCTION READY** because: no real RTSP/NVR validation, no GPU inference validation, no SMTP delivery validation, no Docker run, no wall-clock soak, and models are mocked in automated tests (real-inference evidence is manual webcam smoke only).
- **Ahead of NOT READY / DEVELOPMENT READY** because: 753/753 automated tests green under `-W error`; the critical security defects found this audit are fixed with regression coverage; invariant behavior (no fabricated AWAY/absence) is strong and test-locked; the product ships an honest pre-flight gate, RTSP harness, pilot mode, and opt-in security/RBAC/auth fail-closed specifically to de-risk a controlled client pilot.

**Conditions to enter pilot:** set real `.env` (passwords, cameras, SMTP), run `python -m src.preflight` at the client site, validate RTSP endpoints with `python -m src.rtsp_harness`, enroll roster faces, and run with `CCTV_PILOT_MODE=1` + daily audit review.

## 17. Remaining Risks (ranked)

1. **AI-accuracy in the field** — recognition thresholds only tuned on a single live webcam; multi-camera office layouts unvalidated (risk to productivity/security quality).
2. **Email channel is the primary real-time alert path** — SMTP delivery itself never exercised; a silent SMTP failure reduces alerts to dashboard-only.
3. **Camera-offline != AWAY philosophy** can *under*-report away time (empty-scene freeze) — acceptable per design, but operators must expect lower "away" numbers than reality in some layouts.
4. **InsightFace model download at first run** — needs network; failure degrades to person-only (honest via detector registry).
5. **scipy/WDAC block in this venv** — a client-machine environment risk, not a product defect (no test path needs sklearn).

## 18. Recommended Next Steps (outside this audit, no commit performed)

1. Client-site gate: preflight → RTSP harness → real SMTP test (`main.py --test-email`) → Docker compose run → 48h wall-clock soak.
2. Enroll ≥2 roster faces and validate simultaneous multi-identity on the target camera layout; tune `FACE_SIMILARITY_THRESHOLD` from results.
3. Release hardening: default `CCTV_DASH_FAIL_CLOSED=1`; document reverse-proxy/TLS; consider tab-level viewer gating for the Security tab.
4. Sweep: dedupe overlapping secret tests; remove the stray PowerShell-artifact file.
5. Re-run this entire audit after any future change; rebuild README counts from `pytest` rather than hand-editing.

---

*Prepared per the 25-phase audit workflow. All findings stem from direct source/test/DB evidence. No code was committed or pushed. Model claims never exceed what was actually executed.*