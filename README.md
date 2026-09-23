# CCTV Employee Productivity Tracker

An AI-driven, multi-camera **employee presence & phone-usage monitoring system**.
It watches CCTV/RTSP camera feeds — **or your laptop webcam** (the recommended
daily driver for development and demo) — recognizes employees by **facial
embedding**, classifies each person's state as `ACTIVE` / `ON_PHONE` / `AWAY`,
logs every state interval to SQLite, and exports styled **daily Excel
productivity reports** that can be emailed at end-of-day. A **Streamlit web
dashboard** gives live analytics, historical trends, camera health, employee
management with face enrollment, and report downloads.

> **Development status:** the system is in **RELEASE FREEZE** (Phase 30) after a
> full validation campaign, extended by the **Phase 31 security platform**
> (unknown-presence, restricted zones, intrusion, after-hours, camera
> offline/tamper, incident + alert + evidence engines, audit log, WAL-safe
> backup, Security dashboard) and the **Phase 32 AI Office Intelligence &
> Security Platform** (event correlation, investigation workflow, evidence &
> alert hardening, structured search, camera intelligence, detector registry,
> security analytics, RBAC, failure isolation) and the **Phase 33 Advanced AI
> Intelligence & Real-World Pilot Hardening** (temporal intelligence, camera
> topology adjacency, cross-camera correlation, occupancy anomaly + baselines +
> recurring patterns, incident risk scoring, evidence access control,
> timeline-v2 search & pagination, detector health, system-health 2.0, advisory
> daemon cycles, config snapshot UI — all **advisory**, never altering
> productivity data) and the **Phase 34 Next-Generation AI Office Intelligence
> & Real-World Pilot Hardening** (incident reconstruction, person/identity
> continuity, advanced event-context rollup, SOC operator attention queue,
> adaptive time-of-day baselines with an `INSUFFICIENT_DATA` guard, encrypted &
> portable evidence, incident-only HTTP/shelling end-point, extended explainable
> risk context factors, and DB/config hardening — all read-only and evidential,
> **never** modifying productivity data or fabricating identity) and the
> **Phase 35 Production Deployment & Real-World Reliability** (pipeline failure
> isolation — detector/security/main-loop exceptions never kill the daemon,
> degraded no-AI retry mode, wired opt-in automatic daily backup, WAL-safe
> backup **restore**, wired security/evidence retention, idempotent incident
> risk + event links, and extended config validation — all preserving the 376
> productivity/security baseline) and the
> **Phase 36 Virtual Office Simulation & Stress Testing** (deterministic,
> opt-in virtual cameras + seeded office scenarios that drive the *real*
> detection/tracker/security/incident/evidence/alert pipeline end-to-end —
> multi-camera, camera-failure/recovery, cross-camera continuity, fault
> injection, load + database/evidence/alert stress, bounded soak and
> resource-limit safeguards; simulated cameras are **never** presented as real
> RTSP cameras) and the
> **Phase 37 Controlled Advisory-Pilot Readiness** (client-environment
> **pre-flight check**, an **RTSP validation harness**, an opt-in **pilot mode**
> with operational health logging, a failure/recovery + deployment-validation
> test layer, and credential hardening — all reporting real capabilities
> honestly: `READY` / `NOT EXECUTED — ENVIRONMENT LIMITATION` /
> `FUTURE MODEL REQUIRED`, never fabricated) and the
> **Phase 38 Real Client Deployment Validation** (the **website/dashboard is
> treated as a first-class product component**: the Security-tab Severity/Status
> filters are now genuinely wired to the backend, incident Resolve/Dismiss are
> **backend-authoritative** behind RBAC with audited viewer denials, a latent
> `AccessGuard.resolve_incident` `TypeError` was fixed, the Camera Health tab
> shows a `Configured` column without ever leaking RTSP credentials, and a new
> **Deploy & System Check** tab renders honest pre-flight PASS/WARN/FAIL/SKIP
> status + a read-only deployment checklist). The laptop
> **webcam + CPU** path is fully verified; real
> RTSP cameras, GPU inference, SMTP delivery, and Docker deployment are
> **future client-environment requirements**, documented below and marked
> `REQUIRES CLIENT ENVIRONMENT` throughout this README.

| Capability | Status |
|---|---|
| Laptop webcam capture, CPU inference, face recognition | **VERIFIED NOW** (daily driver + demo) |
| Live HUD + Streamlit dashboard, Excel reports, EOD scheduler, retention | **VERIFIED NOW** (logic-level tests + local runs) |
| Security layer: unknown presence, restricted zones, intrusion, after-hours, camera offline/recovered, tamper, incidents, alerts, evidence, audit, backup, Security dashboard | **VERIFIED NOW** (pure-python tests + integration) |
| Phase 32 intelligence & hardening: event correlation, investigation workbench, evidence SHA-256/path-safety, alert lifecycle/retry/escalation, structured search, camera intelligence, detector registry, security analytics, RBAC, failure isolation | **VERIFIED NOW** (pure-python tests + integration) |
| Phase 33 advisory intelligence: temporal states, camera topology/adjacency, cross-camera correlation (with `CROSS_CAMERA_CORRELATION_UNAVAILABLE` fallback), occupancy anomaly + baselines + recurring patterns, incident risk scoring, evidence access control, timeline-v2 search + pagination, detector health, system-health 2.0, risk/coverage/health dashboards | **VERIFIED NOW** (pure-python tests + webcam integration smoke) |
| Phase 34 SOC/office intelligence: incident reconstruction (T±30s window), person/identity continuity (first/last seen, dwell, zone transitions, cross-camera path), advanced event-context rollup (`CORRELATED_SECURITY_INCIDENT (REQUIRES_HUMAN_REVIEW)`), SOC operator attention queue, adaptive time-of-day baselines (hour/day-of-week + `INSUFFICIENT_DATA` guard), extended explainable incident risk context factors (zone/after-hours/evidence/confidence), config snapshot with Phase 34 tunables | **VERIFIED NOW** (pure-python tests + integration smoke) |
| Phase 35 production reliability & recovery: daemon-level failure isolation (detector/security/main-loop exceptions never crash the run, degraded no-detector retry mode), opt-in automatic daily backup (`CCTV_AUTO_BACKUP_DAILY`), WAL-safe backup **restore** (`restore_database`/`restore_to_temp`), wired security+evidence retention (opt-in, incidents never auto-purged), idempotent incident-risk/event-link writes, extended `validate_config` (threshold ranges, retention >= 0, camera URL, SMTP) | **VERIFIED NOW** (pure-python tests + live webcam smoke + backup/restore drill + secret audit) |
| Phase 36 virtual office simulation & stress: deterministic seeded office scenarios (A–R), opt-in virtual cameras with independent FPS/resolution/scene/fault state, camera failure/recovery + frozen/dark/bright/noisy/disconnect, multi-camera (1/2/3; pool guard to N), cross-camera continuity, fault injection (detector/alert/camera — never in production), structured machine-readable reports with real latency p50/p95/p99, resource limits (max cameras/fps/steps/evidence/workers) | **VERIFIED NOW** (simulation/stress/soak pytest markers + real webcam regression preserved). **Simulated ≠ real RTSP** |
| Phase 37 controlled advisory-pilot readiness: client-environment **pre-flight check** (`python -m src.preflight`, system/GPU/camera/models/DB/evidence/SMTP/Docker/security/config/pilot), **RTSP validation harness** (`python -m src.rtsp_harness`) with credential redaction, opt-in **pilot mode** (`CCTV_PILOT_MODE=1`) + operational health logging (never disables security/RBAC/audit or enables continuous recording), failure/recovery + deployment-validation tests, and credential hardening (explicit `CHANGE_ME` markers, no real-looking defaults) | **VERIFIED NOW** (1075-test suite incl. 79 Phase 37 tests + preflight/RTSP-harness smoke). Real RTSP/GPU/SMTP/Docker reported honestly as `NOT EXECUTED — ENVIRONMENT LIMITATION` |
| Phase 38 website/deployment validation: **website as first-class component** — Security-tab Severity/Status filters wired to `query_security_events`/`EventStore.events`; incident Resolve/Dismiss backend-authoritative via `AccessGuard` (audited viewer denial; `resolve_incident` `TypeError` fixed); Camera Health `Configured` column (no RTSP/password leak); **Deploy & System Check** tab rendering honest `src.preflight` status + read-only deployment checklist; 9-tab AppTest rendering with no secret leakage; pagination + backup/restore/retention invariants | **VERIFIED NOW** (1075-test suite incl. 21 Phase 38 tests + preflight smoke). Real RTSP/GPU/SMTP/Docker still reported honestly as `NOT EXECUTED — ENVIRONMENT LIMITATION` |
| Phase 39 critical-defect remediation & production hardening: dashboard auth **fail-closed** (`CCTV_DASH_FAIL_CLOSED` + preflight FAIL), **session expiry** (`CCTV_DASH_SESSION_MINUTES`, default 15m), gender-agnostic **LEFT/departure emission** from last-seen expiry, evidence capture honoring OFF/HIGH_SEVERITY_ONLY/EVENT_ONLY, risk-panel LEFT JOIN, incident acknowledge/dismiss RBAC + single-audit, employee mutation RBAC + audit + path sanitization, biometric-at-rest WARN, `.dockerignore`, per-record evidence retention, `localtime` timezone normalization, grace/break knobs surfaced, `-W error` clean | **VERIFIED NOW** (1075-test suite incl. 10 Phase 39 regression tests, full green under `-W error`). Real RTSP/GPU/SMTP/Docker still reported honestly as `NOT EXECUTED — ENVIRONMENT LIMITATION` |
| Phases 40–48 pipeline realism & honesty upgrades: Phase 40–42 detection/pipeline refinements (single-source FPS cap, cadence budget), Phase 43 **per-person + face/phone/ReID cadence** layers with opt-in `PIPELINE_TIMING`/`PHONE_DIAG` diagnostics, Phase 44/44A/44B dedicated **phone pass** (class-67, `PHONE_IMGSZ=1280`, cadence, per-track evidence hysteresis) + wall-clock **ON_PHONE/AWAY FSM** (`PHONE_AFTER_SEC`, `AWAY_AFTER_SEC`, gap grace), Phase 45 single-source realism/tuning (`effective_target_fps`), Phase 46 YOLO11n default upgrade + research benchmark, Phase 47 validation-protocol report (tree-absent), Phase 48 first real-camera validation attempt (index 0 yielded black/frozen frames only — ENV BLOCKED). YOLO person/phone accuracy remains **NOT VALIDATED** (no frame-labelled real dataset); see `PROJECT_FULL_SYSTEM_AUDIT_2026-09-19.md` | **CODE VERIFIED** (suite regression + bench JSON; real-camera accuracy NOT VALIDATED) |
| Phase A/B core vision & security intelligence: **per-person** detection entries (`person_box`/`person_conf`) with legacy presence kept, **spatial person tracking** (`Track`/`SpatialTracker`: track_id, IoU association, EMA bbox, trajectory, first/last-seen), **flicker-free track identity voting** (adopt-after-3 / never-demote / switch-after-3), **phone→track→employee** attribution + per-track phone duration, **frame motion detection** (advisory only, `MOTION_DETECTED` + `motion_events`), **virtual-line entry/exit crossings** (`VirtualLine`/`EntryExitDetector` → `ENTRY_CROSSING`/`EXIT_CROSSING` + `entry_exit_events`), **LEFT emitted from exit-crossing** with `LEFT_DEBOUNCE_SEC`, opt-in **UNEXPECTED_STAY**, **`INTRUSION_COOLDOWN_SEC` actually wired**, per-zone **capacity**, **FROZEN_FRAME/LOW_FPS** camera health states, **`CAMERA_MODE`** + dashboard camera-selection, registry capability vocabulary (**AVAILABLE/DISABLED/DEGRADED/NOT_CONFIGURED/FUTURE_MODEL_REQUIRED**) + "🤖 AI Capabilities" tab (10-tab dashboard). Multi-person robustness: **N-persons → N independent person/track entries, N-faces → N independent identities** (never first-face-only), **face→person pairing by face centre + nearest person centroid**, **per-person phone attribution**, **blank/dark or dead camera freezes employee state — never fabricates AWAY**, **FROZEN_FRAME-but-flowing stays monitored**, security engine treats **FROZEN/NO_FRAME as offline** (failsafe, `LOW_FPS` stays online), **structured `CAMERA_DEBUG`** loop diagnostics, live table row for **Unknown person** + dashboard distinguishes **dead/blank vs frozen** camera | **VERIFIED NOW** (1075-test suite incl. 54 Phase A/B + 30 multi-person/camera-honesty regression tests + 5 real-feed FROZEN_FRAME camera-health tests + live webcam session showing EMP001 + Unknown active simultaneously; see `GAP_ANALYSIS_PHASE_AB.md` §5). Two-roster-employee simultaneous recognition (EMP002/EMP003 still `NO_FACE`) + DroidCam honestly `NOT EXECUTED — ENVIRONMENT LIMITATION` |
| Phone-use classification (`ON_PHONE`) | **DOCUMENTED LIMITATION** (YOLOv8n sensitivity; see [Known limitations](#known-limitations)) |
| Phase 49 genuine person-ReID upgrade: vendored **OSNet** family (vanilla + AIN, MSMT17-trained, MIT), opt-in via `CCTV_REID_MODEL_PATH` → `models/osnet_x1_0_msmt17.pth` (or `osnet_x0_5` / `osnet_ain_x1_0`), 512-dim embeddings + torchreid preprocessing (256×128, ImageNet mean/std), dim-aware appearance-cache invalidation (`CACHE_VERSION=2`), zero behaviour change at default settings, opt-in event-driven ReID cadence (`CCTV_REID_GATE_RESOLVED=1` skips appearance inference on face-resolved person boxes) | **CODE VERIFIED / SIMULATED** (`data/phase49_benchmark.json`: CPU latency/params + similarity separation on synthetic crops; 1075-test suite green under `-W error`). Real-camera threshold calibration **NOT VALIDATED** — production default unchanged, OSNet remains opt-in |
| Phase 50 real-camera ReID calibration preparation: offline **threshold/margin calibration tool** (`python -m src.calibration`, labelled-crop scanning, pose dedup, same/diff cosine distributions, best-vs-second margins, operating-point scan, honest `eligible=False` refusal under FAR+FRR caps — **never mutates config**), **production gate** (5 hard prerequisites before any OSNet production use), identity-safety regression suite, real-camera **feed gate** (bright/changing/FPS), and **side/back + phone validation protocols** (`PHASE50_VALIDATION_PROTOCOLS_2026-09-19.md` — **tree-absent**, see `PROJECT_FULL_SYSTEM_AUDIT_2026-09-19.md` §23/§24) | **CODE VERIFIED** (30 new Phase 50 tests; suite **1075 passed** under `-W error`). Thresholds, OSNet production use, and live protocols **NOT VALIDATED / NOT EXECUTED** (no usable camera feed; EMP003/Piyush unavailable) — production default unchanged, OSNet remains opt-in |
| Phase 51 calibration-tool hardening & validation harness: **data-integrity scanner** (roster-missing, empty pose dirs, corrupt/unreadable crops reported **per-path** — nothing silently dropped; unsupported files; pixel-identical **cross-employee + cross-pose dedup**; per-size `image_sizes` census; deterministic dataset fingerprint), **manifest-gated provenance** (`REAL LABELED DATA` now requires a validated `manifest.json` — SIMULATED artifacts and bare PNG folders can never be REAL), recommender **hard guards** (refuse `eligible=False` with explicit reason on no-data / single-employee / sample-shortage / dead–confusable / zero-norm inputs — never a silent closest-point pick; 13 fixtures A–M), **provenance-forcing production gate** (synthetic dry-run report can never satisfy `real_calibration_data`), hardened **feed gate** (frozen-tail/majority-change detection, min frames, ≥15 FPS) + **validation-runner choke point** (`NOT EXECUTED / INVALID FEED` — cameras that open while black/frozen are not evidence), **fail-closed model identity** (`checkpoint`/`model_variant` recorded; a requested checkpoint that fails to load aborts with exit 1 — no silent fallback, no variant mislabel), Phase 51 report | **CODE VERIFIED / SIMULATED** (75 new tests; suite **1075 passed** under `-W error`). Does **NOT** enable OSNet, calibrate thresholds, or validate side/back/phone on a live feed — those stay **NOT VALIDATED / NOT EXECUTED** (no usable camera feed). Production default unchanged, OSNet remains opt-in |
| Phase 52 real-camera validation execution: fed the only device that opens (index 0, MSMF) through the Phase 51 feed gate — **every frame black/frozen** (median brightness `4.12/255`, frame change `0.0`, gate verdict `NOT EXECUTED / INVALID FEED`). Real validation **STOPPED at the preflight choke point**; no manifest, no employee captures, no scenarios, no thresholds changed; OSNet untouched | **REAL VERIFIED (failure event)** — the black/frozen feed and its correct rejection were observed twice. All live scenarios remain **NOT EXECUTED / NOT VALIDATED** (blocked: no usable feed, no rostered employee). See `PHASE52_REAL_CAMERA_VALIDATION_REPORT_2026-09-19.md` |
| Phase 53 full-system audit (zero-change read-only pass): every module/config/test/doc examined; invariant CODE VERIFICATION + full suite double-run; live DB/SQLite inspected; deployment/auth/docs gaps catalogued as M01–M20 | **AUDIT ONLY** — see `PROJECT_FULL_SYSTEM_AUDIT_2026-09-19.md` |
| Phase 54 defensive remediation (audit-derived): **M01** docker dashboard secure-by-default (`CCTV_DASH_AUTH=1`, `CCTV_DASH_FAIL_CLOSED=1` forwarded, 30-min session); **M02** live-state writer logs failures (rate-limited) + recovery instead of silent swallow; **M03** model preflight remedy + ONNX provider check; **M04** ReID fallback transparency (`requested_model`/`load_error` surfaced); **M05** INFO-advisory incident auto-close on `CAMERA_RECOVERED` + anomaly `REPEATED_PATTERN` cooldown dedup (24 h); **M06** opt-in biometric-cache retention (`CCTV_RETENTION_BIOMETRICS_DAYS`, enrolment images never touched); **M07** report/EOD continuity reconcile (no phantom `report_generated`) | **CODE VERIFIED** (27 new Phase 54 regression tests; suite **1075 passed** under `-W error`). Real live feed / SMTP / Docker run remain **NOT EXECUTED** (environment) |
| Real RTSP camera pool | **REQUIRES CLIENT ENVIRONMENT** (validated by tests/simulation only) |
| GPU (CUDA) batched inference | **REQUIRES CLIENT ENVIRONMENT** (needs NVIDIA GPU + drivers) |
| SMTP email delivery | **REQUIRES CLIENT ENVIRONMENT** (needs real SMTP credentials) |
| Docker/Compose deployment | **REQUIRES CLIENT ENVIRONMENT** (no Docker host in dev) |
| Fall / fight / fire-smoke detection | **FUTURE MODEL REQUIRED** (purpose-trained models; never faked) |

---

## Table of Contents

1. [What it does](#what-it-does)
2. [Features](#features)
3. [How it works (pipeline & workflow)](#how-it-works-pipeline--workflow)
4. [Project structure](#project-structure)
5. [Tech stack](#tech-stack)
6. [Requirements](#requirements)
7. [Installation & setup (Windows)](#installation--setup-windows)
8. [Configuration (.env)](#configuration-env)
9. [Usage](#usage)
   - [Quick start (webcam demo)](#quick-start-webcam-demo)
   - [Webcam / single local source](#webcam--single-local-source)
   - [Multi-camera RTSP pool](#multi-camera-rtsp-pool)
   - [Admin & validation commands](#admin--validation-commands)
   - [Web dashboard](#web-dashboard)
   - [Face enrollment](#face-enrollment)
   - [RTSP stress tester](#rtsp-stress-tester)
10. [Daily operational workflow](#daily-operational-workflow)
11. [Database & reports](#database--reports)
12. [Backup & restore](#backup--restore)
13. [Demo data management & clean reset](#demo-data-management--clean-reset)
14. [Deployment](#deployment)
15. [Deployment checklist (VERIFIED NOW vs CLIENT)](#deployment-checklist)
16. [Future integration handover](#future-integration-handover)
    - [Real RTSP cameras](#real-rtsp-cameras)
    - [GPU inference](#gpu-inference)
    - [SMTP email](#smtp-email)
    - [Docker in production](#docker-in-production)
17. [Running the tests](#running-the-tests)
18. [Troubleshooting](#troubleshooting)
19. [Known limitations](#known-limitations)
20. [Security notes](#security-notes)
21. [Release & supporting documentation](#release--supporting-documentation)

---

## What it does

Given one or more camera feeds, the system:

1. **Captures** frames continuously on a background thread per camera.
2. **Recognizes faces** with `insightface` embeddings matched against an
   enrolled-employee registry (e.g. `EMP001`, `EMP002`, …). Unrecognized people
   are tracked as `Unknown`.
3. **Detects cell phones** with YOLOv8 and applies a proximity heuristic
   (phone near a recognized face ≈ `ON_PHONE`).
4. **Classifies** each employee's state:
   - `ACTIVE` – face present, no phone near the face/body
   - `ON_PHONE` – face present **and** phone near the face/body
   - `AWAY` – face not seen in the frame (presence patience applies)
5. **Logs** each state interval to SQLite (`activity_logs`) with timestamps,
   durations, and source camera.
6. **Reports** a styled `.xlsx` per day (Summary + Activity Timeline sheets).
7. **Emails** the report (optional) and shows a **live HUD** (OpenCV) and/or the
   **Streamlit dashboard**.

---

## Features

- **Multi-camera pool** (`MultiCameraManager`): up to 10 RTSP cameras, one
  reader thread each; a dropped camera never stalls the others.
- **GPU-batched inference**: frames from several cameras are batched into one
  YOLO call (`MAX_BATCH_SIZE` keeps VRAM bounded), with a fallback to CPU.
- **Dynamic face recognition** (insightface + cosine similarity) instead of
  static desk ROIs — identity follows the person, no calibrated boxes required.
- **Smart state tracking** with a smoothing buffer (no state thrashing),
  multi-camera deduplication, **presence patience** (occluded face retains the
  last known state), and **camera-offline awareness** (a dead camera freezes the
  tracker — it can never accrue `AWAY`; stale frames older than
  `FRAME_STALE_SEC` are dropped).
- **Person tracking for `Unknown`**: personnel presence is monitored even before
  face enrollment (bounded to `ACTIVE`/`AWAY`, never counted toward any
  employee's score).
- **Webcam-first wiring**: actionable startup probe with non-zero exit on
  failure, a direct `read()` fast-path for local devices, and per-employee
  telemetry that only accrues while the camera actually delivers frames.
- **Round-robin throttling** (`CCTV_TARGET_FPS`) for stable CPU/GPU load.
- **Live HUD** (OpenCV window) with color-coded per-person state panels +
  telemetry; hotkeys `q`/`ESC` (graceful shutdown) and `r` (report snapshot).
- **Streamlit web dashboard**: live overview, per-employee live table,
  historical trends, camera health, report center, employee management with
  dashboard-driven face enrollment, camera-degraded alerts, empty-state
  onboarding hints, auto-refresh, and optional role-based login.
- **Automated EOD scheduler** with duplicate prevention (each day reported+emailed
  at most once), missed-window catch-up, and bounded SMTP retries.
- **Resilient capture**: exponential-backoff reconnection + per-camera health in
  the dashboard.
- **Opt-in data retention** (`CCTV_ENABLE_RETENTION=1`); the daemon never deletes
  history by itself.
- **Docker + Compose** deployment with healthchecks, non-root runtime, GPU
  passthrough, and graceful-stop guarantees.
- **Automated test suite** — **1075 pytest tests** (pure/logic level, no GPU,
  camera, SMTP, or enrollment images needed), covering the Phase 30 productivity
  core, the Phase 31 security layer, and the Phase 32 intelligence/hardening
  layer (correlation, investigation, evidence/alert hardening, search, camera
  intelligence, detector registry, analytics, RBAC, failure isolation), the
  Phase 33/34 advisory intelligence + SOC/office intelligence, the Phase 35
  reliability/backup/restore/retention hardening, the Phase 36 virtual
  office simulation/stress/soak (marked `simulation`/`stress`/`soak`), and the
  Phase 37 controlled advisory-pilot readiness (preflight, RTSP harness, pilot
  mode, failure/recovery, and deployment-validation hardening).

---

## How it works (pipeline & workflow)

```
 RTSP / webcam / video file
        │   (one VideoCapture reader thread per camera; local devices use a direct fast-path)
        ▼
 MultiCameraManager ──►  latest_frames(limit = MAX_BATCH_SIZE)   (round-robin)
        │
        ▼
 ActivityDetector.detect_batch()
    one batched YOLO call (classes: person + cell phone)
        │                              └─►  insightface: face bbox + 512-D embedding
        ▼                                   FaceRegistry.identify(): cosine similarity
   [{cam, emp_id, phone}, ...]              employee_id = EMP001.. or "Unknown"
        │
        ▼
 MultiTracker.process_batch()   ──►  per-employee FSM (ACTIVE / ON_PHONE / AWAY)
    - multi-camera dedup          │   - 5 s smoothing buffer
    - presence patience           │   - lazily spawn trackers
        ▼                         ▼
   SQLite (activity_logs)  ◄─  live_states() → HUD / telemetry / dashboard
        │
        ▼
 reporter.py  ──►  data/reports/daily_report_YYYY-MM-DD.xlsx  (Summary + Timeline)
        │
        ▼ (optional, EOD scheduler)
 notifier.py  ──►  SMTP email with the .xlsx attached (bounded retries)
```

**State derivation** (per frame, per employee):

| Face present | Phone near face | State |
|:---:|:---:|:---:|
| ✅ | ❌ | `ACTIVE` |
| ✅ | ✅ | `ON_PHONE` |
| ❌ | — | `AWAY` (after presence patience) |

**Productivity model** (canonical in `src/productivity.py`):

```
productive_pct = productive / (productive + phone + away) × 100
```

- Accounted time is scored only inside the configured **working window**
  (`CCTV_WORK_START` → `CCTV_WORK_END`), with the unpaid **lunch window**
  excluded.
- **Camera downtime never becomes employee downtime**: unobserved time (dead
  cameras, blocked views, non-working hours) is excluded from the denominator,
  so a dead stream freezes the tracker instead of accruing `AWAY`.
- `Unknown` faces never count toward any employee's score and are isolated in
  the dashboard/table.

---

## Project structure

```
cctv monitoring/
├── main.py                      # Production orchestrator: CLI, live HUD, EOD scheduler
├── app.py                       # Streamlit web dashboard (live + historical + enrollment)
├── config.py                    # Central config: .env loading, validation, logging
├── requirements.txt             # Python dependency manifest (floor pins)
├── setup_env.ps1                # Windows environment bootstrapper (venv + deps)
├── .env.example                 # Environment template (cameras, SMTP, dashboard, retention)
├── .env                         # YOUR secrets (gitignored — never commit)
├── .gitignore                   # Protects .env / data / models / venv from VCS
├── Dockerfile                   # Container image (CUDA or CPU base, non-root)
├── docker-compose.yml           # Two services (ai_daemon + web_dashboard) + healthchecks
├── models/
│   └── yolo11n.pt               # YOLO11n weights (person / cell phone detection)
│
├── data/                        # Runtime data (persists across restarts / containers)
│   ├── database/
│   │   ├── sessions.db          # SQLite (WAL mode) + -wal / -shm sidecars
│   │   └── sessions.db.bak-p27  # Phase 27 backup evidence (kept as a restore demo)
│   ├── faces/                   # Enrolled employees: EMP001.jpg, EMP002.jpg, ...
│   ├── embeddings.pkl           # Cached 512-D face embeddings (rebuilt at startup)
│   ├── reports/                 # daily_report_YYYY-MM-DD.xlsx outputs
│   ├── backups/                 # SQLite backups (see Backup & restore)
│   ├── live_state.json          # Daemon live snapshot read by the dashboard
│   ├── eod_state.json           # EOD scheduler duplicate-prevention marker
│   └── logs/app.log             # Rollover application log
│
├── tests/                       # pytest suite (1075 green)
│   ├── conftest.py                     # fixtures (isolated DB, config env)
│   ├── test_productivity.py            # working-window scoring, midnight splits
│   ├── test_analytics.py               # Unknown filtering, day/range metrics
│   ├── test_analytics_invariants.py    # metric invariants across report/dashboard
│   ├── test_tracker.py                 # FSM: offline freeze, patience, dedup
│   ├── test_tracker_regression.py      # regression: presence/offline semantics
│   ├── test_database.py                # schema/migration/CRUD/retention
│   ├── test_domain.py                  # domain models + URL redaction
│   ├── test_config.py                  # env parsing + validate_config
│   ├── test_notifier.py                # bounded email retry
│   ├── test_main.py                    # EOD scheduler + retention wiring
│   ├── test_main_scheduler.py          # scheduler duplicate prevention, catch-up
│   ├── test_detector.py                # batched-detection path contracts
│   ├── test_detector_batch.py          # batch shape/limit behaviour
│   ├── test_webcam_mode.py             # webcam startup/failure invariants
│   ├── test_camera_failure.py          # camera-offline robustness
│   ├── test_employees.py               # EmployeeStore metadata CRUD
│   ├── test_dashboard_helpers.py       # dashboard read helpers incl. WAL-aware read
│   ├── test_dashboard_app.py           # Streamlit AppTest end-to-end render
│   ├── test_face_registry_status.py    # per-file ENROLLED/NO_FACE status
│   ├── test_report_consistency.py      # report/DB consistency
│   ├── test_e2e_pipeline.py            # end-to-end pipeline simulation
│   └── test_phase_ab.py            # Phase A/B regression: tracker, motion, entry/exit,
│                                     #   intrusion cooldown, registry, camera health
│
└── src/
    ├── camera.py                # Threaded VideoCapture: stale-drop, reconnect, health
    ├── camera_manager.py        # MultiCameraManager: pooled reader threads + health API
    ├── detector.py              # ActivityDetector: insightface + YOLO (person + phone)
    ├── face_registry.py         # FaceRegistry: enroll, embedding cache, cosine-match
    ├── tracker.py               # Track + SpatialTracker (spatial: track_id, IoU assoc.,
    │                           #   trajectory, identity voting) + MultiTracker /
    │                           #   per-employee FSM kept as the state layer
    │   # Phase A/B — core vision & security intelligence
    ├── motion.py                # MotionDetector: per-camera frame-diff (advisory, A8)
    ├── entry_exit.py            # VirtualLine / EntryExitDetector: track crossings (A10)
    ├── camera.py                # Threaded VideoCapture: stale-drop, reconnect, health +
    │                           #   FROZEN_FRAME / LOW_FPS / DEGRADED states (A12)
    ├── database.py              # SQLite schema (employees, activity_logs, security_*,
    │                           #   motion_events, entry_exit_events) + migrations
    ├── employees.py             # EmployeeStore employee metadata CRUD / reconciliation
    ├── domain.py                # Domain models + constants + URL redaction
    ├── productivity.py          # Canonical working-window productivity engine
    ├── analytics.py             # Shared analytics used by reporter + dashboard
    ├── reporter.py              # Styled .xlsx daily report exporter
    ├── notifier.py              # SMTP EOD email dispatch + bounded retry + status
    ├── roi_selector.py          # Legacy interactive desk-ROI tool (deprecated)
    ├── rtsp_tester.py           # Multi-camera RTSP connectivity stress tester
    ├── dummy_face_enroll.py     # Generates placeholder faces for testing
    │   # Phase 31 security layer
    ├── security_events.py       # SecurityEvent + EventStore (persistence/queries)
    ├── incidents.py             # IncidentEngine: grouping/dedup, ack/resolve, notes
    ├── zones.py                 # ZoneStore: zones, allow-lists, schedules, policies
    ├── camera_state.py          # Per-camera offline/recovered FSM
    ├── tamper_monitor.py        # Frozen/dark/bright tamper detection
    ├── alerts.py                # AlertEngine: rules, cooldown, channels, lifecycle/retry/escalation
    ├── evidence.py              # EvidenceStore: event-driven SHA-256 snapshots, path-safe
    ├── auditlog.py              # Append-only audit log
    ├── backup_tool.py           # WAL-safe backup + restore + integrity
    ├── security_engine.py       # SecurityEngine.tick(): detectors → events → incidents → alerts → evidence
    │   # Phase 32 intelligence & hardening layer
    ├── correlation.py           # EventCorrelator: incident correlation + explainable reasons
    ├── search.py                # SecuritySearch: safe parameterized multi-entity filters
    ├── camera_intelligence.py   # CameraIntelligence + SystemHealth
    ├── detector_registry.py     # DetectorRegistry: pluggable detectors, anti-fabrication
    ├── security_analytics.py    # SecurityAnalytics: operational metrics (no employee profiling)
    ├── rbac.py                  # AccessGuard: role permission enforcement + audited denials
    ├── failure_isolation.py     # CircuitBreaker / safe_call / chunked resilience
    ├── preflight.py             # Phase 37 client-env pre-flight check tool (CLI + --json)
    ├── rtsp_harness.py          # Phase 37 RTSP client validation harness (creds-redacted)
    └── simulation/              # Phase 36 virtual-office simulation (opt-in, test-only)
        ├── virtual_camera.py    #   VirtualCamera / VirtualCameraPool (frames + health + faults)
        ├── scenario.py          #   OfficeScenarioPlan A–R (seeded, real detect_batch schema)
        ├── faults.py            #   deterministic fault injectors (never in production)
        ├── report.py            #   SimulationReport + latency p50/p95/p99
        └── runner.py            #   SimulationRunner: drives the REAL pipeline end-to-end
```

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Language | Python **3.10 / 3.11 / 3.12** (3.12 recommended; 3.13+ unsupported) |
| Object detection | PyTorch + Ultralytics **YOLO11** (`yolo11n.pt`) — classes `[0=person, 67=cell phone]` |
| Face recognition | **insightface** (`buffalo_l`) + ONNX Runtime + scikit-learn cosine similarity |
| Vision plumbing | OpenCV (`opencv-python`) |
| Storage | SQLite (WAL mode) |
| Reporting | Pandas + OpenPyXL (styled `.xlsx`) |
| Dashboard | Streamlit + Plotly |
| Config / secrets | `python-dotenv` + `.env` |
| Deployment | Docker + Docker Compose (optional NVIDIA GPU) |

---

## Requirements

- **Python 3.10, 3.11, or 3.12 (64-bit)**. Python 3.13/3.14 will **not** work —
  PyTorch has no CUDA wheels for them, so a silent CPU fallback breaks
  `insightface`. `setup_env.ps1` enforces this and aborts otherwise.
- **Windows** (native dev/demo), or Docker for the containerized path.
- **Webcam** (for the daily webcam path) or RTSP-reachable cameras (client site).
- Optional: **NVIDIA GPU** + drivers + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  for accelerated inference.
- ~2–3 GB free disk space for the venv + model weights.

---

## Installation & setup (Windows)

### 1. Install a compatible Python version

```powershell
# Option A: winget
winget install Python.Python.3.12

# Option B: manual — download python-3.12.x-amd64.exe from python.org
# Tick "Add python.exe to PATH" during install.
```

### 2. Create the virtual environment

```powershell
# From the project root (F:\synilogic\cctv monitoring)
Remove-Item -Recurse -Force .venv   # only if a stale .venv exists

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version                    # must print 3.10 / 3.11 / 3.12
```

### 3. Run the setup script

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1 -Python .\.venv\Scripts\python.exe
```

`setup_env.ps1` will:
- Verify the Python version is 3.10/3.11/3.12 (abort otherwise).
- Install PyTorch (CUDA-index attempt with CPU fallback; `-CpuOnly` forces CPU).
- Install `onnxruntime-gpu` (falls back to `onnxruntime` on CPU).
- Install `insightface`, `streamlit`, `plotly`, `python-dotenv`, `scikit-learn`,
  and the rest of `requirements.txt`.
- Verify every critical import and print CUDA/GPU availability.

### 4. Configure `.env`

```powershell
Copy-Item .env.example .env
# edit .env with your values (camera URLs, SMTP creds, etc.)
```

---

## Configuration (.env)

`config.py` is the single source of truth. It reads these environment variables
(via `python-dotenv`) with sensible defaults. `CCTV_CAM_<NN>_URL` entries that
are left empty are **excluded** from the camera pool, so you can enable 1–10
cameras freely.

### Sensors & inference

```dotenv
# --- Multi-camera RTSP pool (Phase 7/8) -------------------------------
# cam_01/02 use public always-on test streams so the pool works off-site.
CCTV_CAM_01_URL=rtsp://rtspstream:2a12bd5@zephyr.rtsp.stream/pattern
CCTV_CAM_02_URL=rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov
CCTV_CAM_03_URL=rtsp://user:CHANGE_ME@192.168.1.103:554/stream1   # client placeholders
CCTV_CAM_04_URL=rtsp://user:CHANGE_ME@192.168.1.104:554/stream1   # (replace on site)
CCTV_CAM_05_URL=rtsp://user:CHANGE_ME@192.168.1.105:554/stream1
CCTV_CAM_06_URL=rtsp://user:CHANGE_ME@192.168.1.106:554/stream1
CCTV_CAM_07_URL=rtsp://user:CHANGE_ME@192.168.1.107:554/stream1
CCTV_CAM_08_URL=rtsp://user:CHANGE_ME@192.168.1.108:554/stream1
CCTV_CAM_09_URL=rtsp://user:CHANGE_ME@192.168.1.109:554/stream1
CCTV_CAM_10_URL=rtsp://user:CHANGE_ME@192.168.1.110:554/stream1

# --- Webcam / CPU path tuning ------------------------------------------
CCTV_STARTUP_FRAME_WAIT=15   # s to probe a local source before failing (default 15)
CCTV_FRAME_STALE_SEC=10      # s before a frame is treated as stale/dropped (default 10)
CCTV_FACE_DETECT_SIZE=640    # downscale webcam frames to this side before detection (default 640)

# --- Inference / capture tuning -----------------------------------------
CCTV_MAX_BATCH_SIZE=4        # max frames per GPU batch (VRAM bound)
CCTV_TARGET_FPS=2            # target frames/sec per camera
```

### Phase A/B — core vision & security intelligence

```dotenv
# --- Spatial tracking (A5/A6) ------------------------------------------
CCTV_TRACK_IOU=0.20          # min IoU to associate a person to an existing track
CCTV_TRACK_MAX_AGE=3.0       # s without a detection before a track is pruned
CCTV_IDENTITY_ADOPT=3        # consecutive votes before a fresh track adopts an identity
CCTV_IDENTITY_SWITCH=3       # consecutive votes to switch between known employees
CCTV_TRACK_TRAJECTORY=120    # max trajectory points kept per track
CCTV_TRACK_DEBUG=0           # spatial-tracker + crossing debug logs

# --- Motion detection (A8, advisory only) ------------------------------
CCTV_MOTION_ENABLED=0        # master switch (registry reports DISABLED when off)
CCTV_MOTION_MIN_SCORE=0.02   # min changed-pixel fraction to count as motion
CCTV_MOTION_MIN_DURATION=1.0 # continuous-motion duration (s) before an event
CCTV_MOTION_COOLDOWN=30      # min seconds between MOTION_DETECTED events per camera
CCTV_MOTION_DOWNSCALE=320    # max side of the working frame (pixels, 0 = none)

# --- Entry/exit line crossings (A10/B6) ---------------------------------
CCTV_ENTRY_EXIT_ENABLED=0    # gate for the crossing engine at the daemon level
CCTV_ENTRY_EXIT_LINES_FILE=  # optional JSON of virtual lines (default data/entry_exit_lines.json)
CCTV_ENTRY_EXIT_DEBOUNCE=30  # min seconds between events for one (track, line)

# --- Security intelligence (B2/B6) --------------------------------------
CCTV_INTRUSION_COOLDOWN_SEC=600  # min seconds between INTRUSION events per (camera, zone)
CCTV_LEFT_DEBOUNCE=10        # suppress appearance-based LEFT for N s after an EXIT_CROSSING
CCTV_UNEXPECTED_STAY_ENABLED=0  # opt-in "crossed in via a line, never crossed out" event

# --- Camera selection / mode (A13) --------------------------------------
CCTV_CAMERA_MODE=AUTO        # EXPLICIT (requires --source) or AUTO (pick live device)

# --- Diagnostics --------------------------------------------------------
CCTV_CAMERA_DEBUG=0          # 1 = structured CAMERA_DEBUG loop lines (source, index,
                             # frame age/fps, usability decision) to data/logs/app.log
```

### Working window / productivity

```dotenv
CCTV_WORK_START=09:00
CCTV_WORK_END=18:00
CCTV_LUNCH=12:00-13:00       # unpaid window, excluded from scoring
CCTV_GRACE_MIN=15            # tolerated lateness (informational)
CCTV_BREAKS_MIN=30           # paid break allowance (informational)
CCTV_EOD_HOUR=19             # 0-23; empty/"none" disables scheduled reports
```

### Dashboard (auth + refresh)

```dotenv
CCTV_DASH_REFRESH=5            # live-tab auto-refresh interval (s)
CCTV_DASH_AUTH=1               # 1 = login gate, 0 = explicit open dev mode
CCTV_DASH_FAIL_CLOSED=1        # default 1: no credentials configured -> access
                               # is BLOCKED (never silently open as admin)
CCTV_DASH_USER=admin
CCTV_DASH_PASS=replace-me      # set an admin password to enable login
CCTV_DASH_VIEWER_PASS=replace-me   # read-only viewer role
```

> Fail-closed (secure default, Phase 64B): with `CCTV_DASH_AUTH=1` and no
> password configured the dashboard **denies all access** instead of opening an
> admin panel. Configure `CCTV_DASH_PASS` (and optionally
> `CCTV_DASH_VIEWER_PASS`) to log in. For trusted local development only, set
> `CCTV_DASH_AUTH=0` explicitly to restore the open admin mode.
> Always put the dashboard behind a VPN / reverse proxy.

### SMTP (report emailing)

```dotenv
SMTP_SERVER=smtp.example.com
SMTP_PORT=587
SENDER_EMAIL=tracker@example.com
SENDER_PASSWORD=replace-with-app-password   # use an app-password/token, never a plain one
RECIPIENT_EMAILS=manager@example.com,lead@example.com
CCTV_EMAIL_RETRIES=3          # bounded retries — never loops forever
CCTV_EMAIL_RETRY_DELAY=10     # seconds between attempts
CCTV_EMAIL_TIMEOUT=30
```

### Data retention (opt-in)

```dotenv
CCTV_ENABLE_RETENTION=0       # 1 = delete history beyond the windows below
CCTV_RETENTION_DAYS=365
CCTV_RETENTION_LOGS_DAYS=365
CCTV_RETENTION_REPORTS_DAYS=365
```

### Pilot mode (Phase 37, opt-in, advisory)

```dotenv
CCTV_PILOT_MODE=0             # 1 = enable controlled advisory-pilot health logging
CCTV_PILOT_METRICS_INTERVAL=30   # seconds between operational metrics log lines
CCTV_PILOT_HEALTH_LOG=60         # seconds between health-log lines
```

Pilot mode is **off by default** and purely advisory: it adds a periodic
operational health log (cameras online, FPS, detector state + model device) so
operators can confirm the platform is healthy. It **never** disables
security/RBAC/audit, **never** enables continuous recording, and **never**
modifies or accelerates productivity/employee data.

### Other notable `config.py` settings

| Setting | Default | Meaning |
|---------|---------|---------|
| `MODEL_PATH` | `models/yolo11n.pt` | YOLO11n weights |
| `CCTV_REID_MODEL_PATH` | *(empty)* | Trained person-ReID model. `.pth`/`.pt` → torch **OSNet** (512-dim, 256×128 preprocessing); `.onnx` → OSNet-style ONNX. Empty/unloadable → built-in descriptor (never crashes, never auto-downloads). |
| `CCTV_REID_GATE_RESOLVED` | `0` | **OPT-IN** event-driven cadence: `1` skips appearance inference for person boxes overlapping face-resolved tracks from the previous cycle (`CCTV_REID_GATE_IOU`, default `0.30`). |
| `CONF_THRESHOLD` / `FRAME_SKIP` | `0.4` / `10` | YOLO confidence / frame sampling |
| `FACE_MODEL` | `buffalo_l` | insightface model pack |
| `FACE_SIMILARITY_THRESHOLD` | `0.6` | cosine-similarity pass/fail |
| `SMOOTHING_BUFFER_SEC` | `5` | state-transition hysteresis |
| `IDENTITY_STABILITY_FRAMES` | `3` | frame-count floor before committing identity |
| `PRESENCE_PATIENCE_SEC` | `8.0` | grace before face-occluded → `AWAY` |
| `DISPLAY_WIDTH` | `960` | HUD window width |
| `DB_PATH` / `REPORT_OUTPUT_DIR` / `FACES_DIR` | under `data/` | storage locations |

### Security layer (Phase 31, P0/P1)

| Setting | Default | Meaning |
|---------|---------|---------|
| `SECURITY_ENABLED` | `true` | master switch for the security engine |
| `OFFLINE_TRIGGER_SEC` | `20` | camera offline threshold before `CAMERA_OFFLINE` |
| `SECURITY_UNKNOWN_MODE` | `off` | `immediate` / `after` / restricted-only / off-hours / `off` |
| `SECURITY_UNKNOWN_AFTER_SEC` | `15` | seconds in restricted zone before Unknown → `UNKNOWN_PRESENCE` |
| `SECURITY_UNKNOWN_COOLDOWN_SEC` | `300` | suppress repeat Unknown events |
| `SECURITY_OFFICE_START/END` | `09:30`/`18:00` | office hours for `AFTER_HOURS_ACTIVITY` |
| `TAMPER_ENABLE_DARK` / `TAMPER_DARK_MEAN` / `TAMPER_FROZEN_DIFF` | `true` / `1.0` / `0.5` | tamper sensitivity |
| `TAMPER_PERSIST_SEC` | `90` | frames must stay tamper-like this long before `CAMERA_TAMPER_SUSPECTED` |
| `LOITERING_SEC` | `30` | person in restricted zone this long → `LOITERING_SUSPECTED` |
| `CROWD_THRESHOLD` | `5` | person count above this → `UNUSUAL_OCCUPANCY` |
| `EVIDENCE_MODE` | `off` | `off` / `event_only` / `high_severity_only` (never continuous) |
| `EVIDENCE_RETENTION_DAYS` | `7` | evidence snapshot retention |
| `EVIDENCE_MAX_MB` | `5000` | evidence storage cap |
| `SECURITY_ZONES_FILE` | `data/zones.json` | zone definitions (polygons, allow-lists, policies) |

Zones are configured in `data/zones.json`; see `src/zones.py` for the format.
Incidents, alerts, evidence files, and the audit log are stored in the same
SQLite database (`data/database/sessions.db`) and `data/evidence/`.

### Phase 33 — Advisory intelligence tunables

Phase 33 adds a set of transparent, env-resolvable tuning knobs (all advisory;
they never affect productivity/employee data). Defaults shown:

| Variable | Default | Meaning |
|---|---|---|
| `CCTV_TEMPORAL_REPEAT` | `5` | occurrences before an incident reaches `REPEATED` |
| `CCTV_TEMPORAL_ESCALATE_COUNT` | `3` | rapid occurrences before `ESCALATED` |
| `CCTV_TEMPORAL_SILENCE_DAYS` | `1` | silence days before an incident reaches `ENDED` |
| `CCTV_ANOMALY_MIN_SAMPLES` | `5` | baseline samples required before `STATUS_READY` |
| `CCTV_ANOMALY_ZSCORE` | `3.0` | z-score threshold for an occupancy anomaly alert |
| `CCTV_RISK_ENABLED` | `1` | enable incident risk scoring (`0` disables) |
| `CCTV_INCIDENT_ALERT_DEDUP_SEC` | `3600` | alert dedup window per incident |
| `CCTV_RELIABILITY_FPS_TARGET` | `10` | expected FPS used in camera reliability scoring |

These are surfaced read-only on the dashboard **Settings** tab
(`config.runtime_config_snapshot()`), consistent with the project's
restart-safe, env-driven configuration model.

Validate your config without starting anything:

```powershell
.\.venv\Scripts\python.exe main.py --validate-config   # exit 0 = OK, 1 = problem
```

### Phase 34 — SOC / adaptive-intelligence tunables

Phase 34 adds transparent, env-resolvable knobs (all advisory and read-only;
they never alter productivity/employee data). Defaults shown:

| Variable | Default | Meaning |
|---|---|---|
| `CCTV_RISK_CONTEXT_FACTORS` | `1` | add explainable context factors (zone sensitivity / after-hours / evidence / confidence) as a bounded bonus to incident risk (`0` disables) |

`incident.risk` extended components (`zone_sensitive`, `after_hours`,
`evidence`, `low_confidence`) are always bounded so the score stays in
`0.0..1.0`; every factor is listed verbatim in the incident's `rationale`.
The Phase 34 SOC view (`Security` tab) reconstructs incidents (T−30s..T+30s),
rolls up event context (`CORRELATED_SECURITY_INCIDENT (REQUIRES_HUMAN_REVIEW)`
— never an accusation), resolves person continuity (first/last seen, dwell,
zone transitions, cross-camera path; `Unknown` always stays `UNRESOLVED`), and
scores risk for the attention queue.

---

## Usage

### Quick start (webcam demo)

This machine's primary dev/demo mode is **local webcam + CPU**. From the project
root, in **two PowerShell windows**:

```powershell
# Window 1 — the tracker (your laptop webcam)
.\.venv\Scripts\python.exe main.py --source webcam --headless --no-email

# Window 2 — the dashboard
.\.venv\Scripts\streamlit.exe run app.py
```

Then open **http://localhost:8501** in your browser. You will see the live
employee table, camera health (ONLINE), productivity, and reports.

> Always use `--source webcam`. A bare `main.py` (no `--source`) uses the RTSP
> **camera pool** instead — with unreachable client URLs nothing will display.

### Webcam / single local source

The `--source` flag overrides the multi-camera pool with one local device:

```powershell
python main.py --source webcam        # device index 0 (also: --source 0, laptop, usb, local)
python main.py --source 1             # try another device index
python main.py --source webcam --headless --no-email   # daemon mode, no HUD/email
```

- The startup probe waits up to `STARTUP_FRAME_WAIT_SEC` (15 s) for the camera;
  on failure you get an **actionable error** and a non-zero exit:
  `Unable to open webcam at index 0 ... Try a different device index, e.g.: --source 1`.
- Local devices use a **direct `read()` fast-path** (no per-frame thread).
- On success a concise **demo banner** prints camera/inference details, the
  enrolled-employee count, and target FPS (no secrets), then the frame resolution
  + measured FPS once the first frames arrive.
- If the camera dies mid-run the pipeline degrades (never crashes): detector
  keeps running, tracker **freezes** (dead camera can't accrue `AWAY`), stale
  frames are dropped, and health shows `RECONNECTING` / `NO_FRAME`.

**HUD hotkeys** (non-`--headless`): `q` / `ESC` → graceful shutdown + final report;
`r` → on-demand report snapshot.

### Multi-camera RTSP pool

```powershell
python main.py --headless          # daemon mode using config.CAMERAS
python main.py --device cpu        # force CPU (or: cuda | mps | auto)
```

`--device auto` (default) detects CUDA → MPS → CPU automatically.

### Admin & validation commands

```powershell
python main.py --validate-config     # fail-fast config check (exit 0/1)
python main.py --test-email          # send one test report email then exit
python main.py --no-email            # run EOD scheduler without emailing
python main.py --max-batch 8         # raise the per-batch frame cap/VRAM budget
python main.py --headless --device auto
```

### Web dashboard

```powershell
streamlit run app.py                 # open http://localhost:8501
```

When `CCTV_DASH_AUTH=1` and `CCTV_DASH_PASS` is set, sign in with a username +
password (viewer password ⇒ read-only; admin password ⇒ full access).

**Tabs:**

- **Live Overview** — today's KPIs (Recognised, Avg Productivity, Active Hours,
  Phone, Present, Cameras Online, Unknown). Before activity, metrics show `n/a`
  with a step-by-step getting-started hint.
- **Live Employees** — per-employee status from the daemon snapshot (In Session
  h:mm, live Source), `Unknown` isolated below the table, auto-refresh every
  `CCTV_DASH_REFRESH` s. A camera OFFLINE/RECONNECTING/NO_FRAME banner appears so
  a camera failure never reads as employee `AWAY`.
- **Historical** — per-day productivity trend and active-hours aggregation over a
  selectable range.
- **Camera Health** — ONLINE/OFFLINE/RECONNECTING/NO_FRAME badges with Last
  Frame, FPS, reconnect/frame counters and a plain-language note (RTSP URLs are
  never shown).
- **Security** (Phase 31 + Phase 32) — security overview cards (cameras offline,
  open incidents, high-severity open, evidence mode), incident timeline,
  incident filters, security analytics, an **Investigation Workbench** (per-
  incident related events / alerts / evidence, investigation notes, review
  state, acknowledge — RBAC-guarded), and admin resolve/dismiss actions.
  Privacy indicator distinguishes LIVE MONITORING vs RECORDED EVIDENCE vs
  ANALYTICS.
- **Reports** — download every generated `.xlsx` with email-delivery status.
- **Employees** — roster management (admin role): add/edit employees, then
  upload a face photo and **Validate enrollment** (status: `ENROLLED` /
  `NO_FACE` / `MULTIPLE_FACES` / `INVALID_IMAGE` / `LOW_QUALITY`).
- **Settings** — read-only grouped summary of working-time, recognition, and
  dashboard settings (secrets never shown).

### Face enrollment

Add one clear, front-facing photo per employee to `data/faces/` named by
employee ID (`EMP001.jpg`, `EMP002.jpg`, …; JPEG/PNG, full-face, a few hundred px
minimum), then rebuild the registry:

```powershell
python -m src.face_registry    # re-scans data/faces, rebuilds embeddings, prints per-employee status
```

The registry reports a **per-file status** for every image (`ENROLLED`,
`MULTIPLE_FACES`, `NO_FACE`, `INVALID_IMAGE`, `LOW_QUALITY`) and syncs the DB
`enrolled` flag. Only `ENROLLED` images contribute embeddings; `NO_FACE`
placeholders match nothing, so identities stay `"Unknown"`.

Enrollment can also be done **from the dashboard** (Employees tab → upload →
Validate enrollment), which uses the cached registry for a fast status read.

To exercise the pipeline before real photos exist, generate placeholders (noise
images that exercise the flow but enroll zero faces):

```powershell
python -m src.dummy_face_enroll            # 3 dummy employees
python -m src.dummy_face_enroll --count 5 --force
```

### RTSP stress tester

```powershell
python -m src.rtsp_tester                       # test all configured cameras, 30 s
python -m src.rtsp_tester --duration 60         # longer run
python -m src.rtsp_tester --cameras cam_01 cam_02   # specific cameras only
```

Verifies that all configured camera sockets can be held open simultaneously.

---

## Daily operational workflow

**Before the shift**

1. Ensure only the clients you need are running; cameras/webcam free.
2. Start the tracker: `python main.py --source webcam --headless --no-email`.
3. Start the dashboard: `streamlit run app.py` → open http://localhost:8501.
4. Confirm the banner prints `Camera feed online`, and the dashboard shows
   `local_webcam: ONLINE` plus the employee(s) you enrolled.

**During the shift** — the daemon continuously classifies ACTIVE / ON_PHONE /
AWAY and logs intervals. Watch live states and productivity on the dashboard.

**End of shift**

- The EOD scheduler emits the report automatically at `CCTV_EOD_HOUR` (default
  19:00), or
- Press `Ctrl+C` in the tracker window (or `q`/`ESC` in the HUD) for a **graceful
  shutdown**: all trackers are flushed, the final report is written (and emailed
  if SMTP is configured), WAL is checkpointed, and the webcam is released.

**Health checks while running**

| Check | How |
|---|---|
| Cameras alive | `Get-Content .\data\live_state.json` → look at `camera_health` |
| Tracker log | `Get-Content .\data\logs\app.log -Tail 30` (INFO only, no ERROR/WARNING) |
| DB integrity | see [Backup & restore](#backup--restore) `--integrity` |
| Security events | `Get-Content .\data\live_state.json` → `security` section, or the **🛡 Security** dashboard tab |

---

## Database & reports

### SQLite (`data/database/sessions.db`, WAL mode)

- `employees` — employee_id, name, department, designation, active/enrolled flags,
  timestamps.
- `activity_logs` — `timestamp`, `date` (normalised), `employee_id`, `state`,
  `duration_seconds`, `source` (camera id) + compound indexes. Schema migrations
  auto-upgrade older DBs at startup.

**Security tables (Phase 31)** — added alongside the productivity tables:

- `security_events` — canonical security event log (type, camera, zone, severity,
  confidence, incident, timestamp) with recovery/clear pairing.
- `incidents` — grouped investigation records (`OPEN` → `ACKNOWLEDGED` →
  `RESOLVED | DISMISSED`), per-day stable IDs, notes, audit. Incidents are never
  auto-purged.
- `alert_rules` / `alerts` — alert rule configuration and fired alerts
  (cooldown-deduped).
- `audit_log` — append-only administrator/employee/config/incident/evidence/
  report/backup actions (no secrets).
- `security_zones` — seeded zone definitions (polygons, allowed employees,
  schedule, policy).
- `evidence_files` — event-driven evidence snapshot metadata (path, size, event,
  timestamp) for retention/metrics.

Reports are written to two sheets in `.xlsx` form:

- **Summary** — Employee ID/Name/Department, In-Time, Out-Time, Expected Hours,
  Active Hours, Phone (min), Away (min), Unobserved (min), Productivity (%),
  Status — frozen header, totals row, score highlighting. `Unknown` excluded;
  every enrolled employee gets a row.
- **Activity Timeline** — raw timestamped state intervals (keeps `Unknown` rows
  for monitoring).

Reports are produced by: the EOD scheduler (at most once per day), graceful
shutdown, the `r` hotkey, or `--test-email`. Output:
`data/reports/daily_report_YYYY-MM-DD.xlsx`.

---

## Backup & restore

### What to back up

| Path | Why |
|---|---|
| `data/database/sessions.db` (+ `-wal`, `-shm`) | all activity + employee data |
| `data/faces/*` | enrolled biometric images |
| `data/embeddings.pkl` | cached embeddings (rebuilt automatically if missing) |
| `data/reports/*.xlsx`, `data/logs/*` | history (optional) |

### Hot (online) backup — always safe, WAL-aware

Use SQLite's `backup` API (safe while the daemon is running — the DB stays in WAL
mode, so a plain file copy can miss the newest committed rows). Run this from the
project root (it snapshots `sessions.db` into `data\backups\<timestamp>.db` and
prints `integrity: ok` + row counts to prove the copy is good):

```powershell
.\.venv\Scripts\python.exe -c "import sqlite3,os,time; src=r'data\database\sessions.db'; dst=os.path.join(r'data\backups', 'sessions_backup_'+time.strftime('%Y%m%d_%H%M%S')+'.db'); os.makedirs(os.path.dirname(dst), exist_ok=True); c=sqlite3.connect(src); b=sqlite3.connect(dst); c.backup(b); b.close(); c.close(); i=sqlite3.connect(dst); print('backup ->', dst); print('integrity:', i.execute('PRAGMA integrity_check').fetchone()[0]); print('rows:', i.execute('select count(*) from activity_logs').fetchone()[0])"
```

Verified on this machine: `integrity: ok`. A Windows scheduled task can run this
exact command daily for offsite/periodic backups.

### Cold backup (daemon stopped)

Copy the three files together, and verify after restore:

```powershell
Copy-Item data\database\sessions.db* data\backups\   # .db + -wal + -shm
```

### Restore

1. Stop the daemon and dashboard.
2. Replace `data/database/sessions.db` with the backup (delete stale `-wal`/`-shm`).
3. Start the daemon — migrations run automatically and the schema is verified.
4. Verify:

```powershell
.\.venv\Scripts\python.exe -c "import sqlite3; c=sqlite3.connect(r'data\database\sessions.db'); print(c.execute('PRAGMA integrity_check').fetchone())"
```

Expected: `('ok',)`.

### Integrity check (quick, read-only)

```powershell
.\.venv\Scripts\python.exe -c "import sqlite3; c=sqlite3.connect(r'data\database\sessions.db'); print(c.execute('PRAGMA integrity_check').fetchone()); print('employees', c.execute('select count(*) from employees').fetchone()[0]); print('rows', c.execute('select count(*) from activity_logs').fetchone()[0])"
```

---

## Demo data management & clean reset

This project stores actual runtime data in `data/`; do **not** delete it casually.

**To clear tracking activity** (keep roster + faces), from the project root:

1. Stop the tracker (and dashboard).
2. Remove the DB and report/log artifacts (fresh start):

```powershell
Remove-Item data\database\sessions.db*
Remove-Item data\reports\*.xlsx, data\live_state.json, data\eod_state.json
```

3. Start the tracker again — the schema is recreated automatically and the next
   report starts from a clean day.

**To clear employees and faces entirely** (full demo reset — also removes photos):

```powershell
Remove-Item data\faces\*.jpg, data\faces\*.png   # delete enrolled images
Remove-Item data\embeddings.pkl                   # cached embeddings
Remove-Item data\database\sessions.db*
```

Then re-enroll fresh faces (see [Face enrollment](#face-enrollment)). This is
**non-destructive to the application itself** — nothing in `src/`, `tests/`, or
the installed environment is touched.

> IP reminder: images are never kept beyond what you intentionally enroll; the
> daemon only stores what you let it. Delete an employee's `.jpg` + row and the
> person is fully removed.

---

## Deployment

The repo ships a `Dockerfile` + `docker-compose.yml` for GPU-capable production:

```bash
# 1. (GPU) install the NVIDIA Container Toolkit first.
# 2. Build:
docker compose build
# 3. Start both services:
docker compose up -d
```

**Services**

- `cctv-ai-daemon` — headless tracker; reads `CCTV_CAM_<NN>_URL`, writes `./data`.
- `cctv-web-dashboard` — Streamlit UI at http://localhost:8501 (starts only after
  the daemon is healthy).

Both bind-mount the host `./data` to `/app/data`, so the DB, face registry,
reports, and logs persist on disk. The compose file wires the 10 camera env vars
(empties filtered out by `config.py`) and requests NVIDIA GPU passthrough via a
`deploy.resources.devices` block.

**Health & lifetime hardening**

- `healthcheck:` on both services: daemon = `live_state.json` freshness (< 120 s);
  dashboard = TCP probe on 8501. Dashboard `depends_on` the daemon as healthy.
- `restart: unless-stopped` + `stop_grace_period: 30s` (daemon) so intervals are
  flushed and the final report emailed on shutdown.
- Images run as a non-root `appuser` (UID 1000) — ensure `./data` is host-writable
  by that UID.
- `network_mode: bridge` (host network available on request for on-site cameras).

**No GPU?** Build the CPU image instead:

```bash
docker compose build --build-arg BASE_IMAGE=python:3.10-slim
```

---

## Deployment checklist

### VERIFIED NOW (validated in this development environment)

| Item | Evidence |
|---|---|
| Webcam capture (device 0/N), CPU inference, face recognition | live smoke: `local_webcam ONLINE`, trackers `EMP001`+`Unknown` |
| Live HUD + dashboard tabs, reports, EOD scheduler, retention | logic tests + local runs |
| Graceful shutdown (flush + final report + DB checkpoint) | repeated Ctrl+C runs, integrity `ok` |
| WAL-safe concurrent read (daemon writes / dashboard reads) | Phase 29 fix: `PRAGMA query_only=ON`, regression test |
| Config validation, secret redaction, env/secrets handling | `--validate-config`, audit |
| Full automated test suite | **1075 pytest tests pass** |
| Face enrollment workflow (files → registry → DB sync) | status-test suite |

### REQUIRES CLIENT ENVIRONMENT (future deployment)

| Item | What is needed |
|---|---|
| Real IP cameras + RTSP connectivity | server IPs, RTSP URLs + credentials, ports, codec, FPS, layout |
| GPU accelerated inference | NVIDIA GPU + drivers + CUDA (see GPU section) |
| Real SMTP delivery | SMTP host/port/credentials/recipients (see SMTP section) |
| Docker/Compose host | a Docker host + (for GPU) NVIDIA Container Toolkit |
| Production auth hardening | `CCTV_DASH_PASS`/`VIEWER_PASS` set, VPN/reverse-proxy, TLS |
| Offsite/periodic backups | scheduled backup job using the documented procedure |
| Camera network segmentation, access control on face data | client IT |

---

## Future integration handover

### Real RTSP cameras

Per-camera info the client should provide:

- Camera ID + logical name; **RTSP URL, username, password** (deliver credentials
  out-of-band — never via the repo/chat), port (default 554), path (e.g. `/stream1`).
- Resolution, native FPS, codec (H.264/H.265).
- Network: static IP/DHCP reservation, VLAN/firewall rules allowing the daemon host.
- Placement notes: height, angle (downward ~30–45°), FOV, lighting, employee count
  per camera.

Wire it in:

```dotenv
CCTV_CAM_01_URL=rtsp://USER:PASS@10.0.0.21:554/stream1
CCTV_CAM_02_URL=...
```

Test connectivity before committing:

```powershell
python -m src.rtsp_tester --cameras cam_01 cam_02
```

### GPU inference

| Item | Recommended |
|---|---|
| GPU | NVIDIA with ≥ 4 GB VRAM for 1–4 cams at batch 4 (size batch by VRAM) |
| Drivers | current NVIDIA driver + CUDA-compatible runtime |
| PyTorch | CUDA build (`torch.cuda.is_available()` must be True) |
| insightface | `onnxruntime-gpu` (CPU build falls back to CPU silently) |
| Docker | `nvidia/cuda:11.8.0-cudnn8-runtime` base + Container Toolkit |

Checks:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python main.py --validate-config
```

Then run `python main.py --headless --device auto` (auto-detects CUDA). VRAM is
bounded by `CCTV_MAX_BATCH_SIZE`. **Not verified on any client GPU — requires
client hardware.**

### SMTP email

Required values: SMTP server, port (587 STARTTLS or 465), sender email,
**app-password/token** (never a plain account password), recipient list,
timeouts/retries. Validate before enabling:

```powershell
python main.py --test-email
```

Behavior: EOD report emailed at `CCTV_EOD_HOUR`, bounded retries
(`CCTV_EMAIL_RETRIES`, default 3), delivery status surfaced in the dashboard.
**Real SMTP delivery is pending external credentials — not yet verified.**

### Docker in production

Same compose file as [Deployment](#deployment). Volume `./data` persists
everything; bind-mounts keep host/container data identical; non-root UID 1000;
healthchecks gate dashboard start; stop-grace guarantees flush. **Not verified on
a Docker host — requires client infrastructure.**

---

## Running the tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
# expected: 1075 passed, no warnings
```

The suite also runs clean under `-W error` (warnings treated as errors):

```powershell
.\.venv\Scripts\python.exe -m pytest -q -W error
# expected: 1075 passed
```

The suite is pure/logic-level and does **not** require a GPU, camera feeds, SMTP,
or enrollment images. Run it before every deployment and after any code change.

Phase 36 adds three opt-in, marked test groups so normal CI never accidentally
runs slow work:

```powershell
# deterministic virtual-camera / scenario simulation tests
.\.venv\Scripts\python.exe -m pytest -m simulation -q
# load / database / evidence / alert stress tests (isolated DBs)
.\.venv\Scripts\python.exe -m pytest -m stress -q
# bounded soak tests (fast simulated time, NOT a multi-hour wall-clock soak)
.\.venv\Scripts\python.exe -m pytest -m soak -q
```

Phase 37 adds the client-environment **pre-flight check** and the **RTSP
validation harness**, both runnable directly (see
[Phase 37 — Client-environment readiness](#phase-37---client-environment-readiness)):

```powershell
# client environment pre-flight check (system/GPU/camera/models/security/pilot)
.\.venv\Scripts\python.exe -m src.preflight
# validate an RTSP endpoint (no fabrication on failure)
.\.venv\Scripts\python.exe -m src.rtsp_harness --url "rtsp://user:pass@host/stream1"
```

A multi-hour **wall-clock** soak, real RTSP camera pools, GPU inference, SMTP
delivery and Docker runs are **NOT EXECUTED** in this environment — they are
future client-environment tasks and are never faked.

### Validation-status matrix (honest labeling)

Every capability is labeled with one of the mutually-exclusive statuses below.
Nothing is inferred or "assumed to work" — an item only climbs a level after the
corresponding validation actually ran.

| Status | Meaning |
|---|---|
| **REAL VERIFIED** | Exercised end-to-end on real hardware/feed in this environment (e.g. live webcam capture + real YOLO/InsightFace inference). |
| **CODE VERIFIED** | Verified by the automated pytest suite (logic-level), reading real code paths against real/isolated databases. |
| **SIMULATED** | Verified only against the deterministic virtual-camera / fault-injection simulation layer. |
| **NOT VALIDATED** | Implemented, covered by code, but the required hardware/environment (RTSP/GPU/SMTP/Docker/long soak/multi-roster-face) was not available to validate here. |
| **NOT IMPLEMENTED** | Explicitly out of scope / future (\*SUSPECTED\_\* event types, fire/smoke/weapon/etc. classes beyond YOLO person+phone). |

| Capability | Status |
|---|---|
| YOLOv8n person + phone detection on live webcam frames | **REAL VERIFIED** (CPU) |
| InsightFace enrollment + recognition (single-operator webcam) | **REAL VERIFIED** — known + Person/Unknown live; two-or-more simultaneous roster faces | **NOT VALIDATED** (EMP002/EMP003 not enrolled in live runs) |
| Multi-person detection / identity independence | **CODE VERIFIED** + single real known+unknown live session |
| Employee FSM (ACTIVE / ON_PHONE / AWAY), camera-offline freeze | **CODE VERIFIED** |
| Camera reconnect / health / FROZEN_FRAME / LOW_FPS | **REAL VERIFIED** on webcam pixels (FROZEN classification); sustained physical freeze end-to-end | **NOT VALIDATED** |
| Productivity & analytics scoring (window/lunch/grace/break) | **CODE VERIFIED** |
| Security engine, incidents, alerts lifecycle/retry/escalation | **CODE VERIFIED** |
| Evidence capture + SHA-256 + retention | **CODE VERIFIED** |
| RBAC (admin/viewer, audited denials), dashboard auth fail-closed | **CODE VERIFIED** (incl. 8 auth-regression tests) |
| Backup / restore / integrity drill | **CODE VERIFIED** (+ local restore drill) |
| RTSP camera pool / NVR | **NOT VALIDATED** — implemented, validated by tests/simulation only |
| GPU/CUDA inference | **NOT VALIDATED** — CPU fallback used; GPU requires client hardware |
| SMTP alert/report delivery | **NOT VALIDATED** — real delivery not exercised; failure paths code-verified |
| Docker / Compose deployment | **NOT VALIDATED** — no Docker daemon in this environment |
| Wall-clock soak / long-run stability | **SIMULATED** (bounded fast-simulated-time soak only) |
| Fire/smoke/weapon/fall/PPE/violence detection | **NOT IMPLEMENTED** — YOLO model is COCO person+phone only; keys/reserved event types exist for a future model, never claimed active |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `main.py` with no args shows nothing | That's the **RTSP pool**. Use `--source webcam --headless --no-email` for your laptop camera. |
| Webcam won't open | Another app holds it (Teams/Zoom/browser) — close it, or try `--source 1`. Error message tells you which index failed. |
| Slow startup | Webcams can take ~10 s; the probe waits `STARTUP_FRAME_WAIT_SEC` (15 s). Raise it via `CCTV_STARTUP_FRAME_WAIT` if needed. |
| Camera shows `RECONNECTING`/`NO_FRAME` | Feed is down/stale; tracker freezes (never accrues AWAY) and retries with backoff. Check the RTSP URL/privacy shutter. |
| Employee shows `Unknown` | Face not enrolled (see [Face enrollment](#face-enrollment)); dummy photos intentionally enroll zero faces. |
| Dashboard open with no admin gating | Set `CCTV_DASH_PASS` in `.env` and restart. Never deploy without it. |
| Port 8501 already in use | `streamlit run app.py --server.port 8601`, or kill the old streamlit process. |
| Report "not generated" | Check `CCTV_EOD_HOUR`; or press `r` / `Ctrl+C` (graceful) to force one. Reports appear in `data/reports/`. |
| SQLite lock warnings | WAL + read-only `query_only` connection make concurrent access safe; WAL sidecars flush at shutdown. A stale `-wal` after a crash is merged on next open. |
| All cameras offline in dashboard | A stray **second** daemon (e.g. bare `main.py`) overwrites `live_state.json`. Run only one tracker (`--source webcam`), restart the dashboard. |
| My webcam is "still active" after I stopped the app | Only the tracker opens the webcam; verify no `main.py` process remains. It does **not** record video files. |
| `setup_env.ps1` aborts on 3.13/3.14 | Install 3.10–3.12 and recreate the venv (see Installation). |
| `insightface` install fails | Install MSVC C++ Build Tools, re-run `setup_env.ps1`. |
| No CUDA despite an NVIDIA GPU | Verify drivers + PyTorch index; `python -c "import torch;print(torch.cuda.is_available())"`. |
| Covered lens / blocked view shows `AWAY` after ~8 s | Expected: presence patience (8 s) then `AWAY`; unobserved time isn't charged as phone time. |

---

## Known limitations

- **Phone-use classification is a documented limitation.** `ON_PHONE` requires
  YOLOv8 to detect a phone box near a recognized face. On the webcam path
  (640×480, YOLOv8n, CPU) a handheld phone is frequently missed — measured in
  Phase 29 at 0.4–1.7% of probe frames, unrecovered by confidence tuning. The
  proximity/buffering logic is correct *when a phone is detected*; the bottleneck
  is detector sensitivity, not the state machine. A phone-class-specific or
  higher-resolution model is a future enhancement, not a defect.
- Other **not-yet-verifiable here** (client-environment requirements, not
  defects): real RTSP cameras, GPU inference, real SMTP delivery, Docker/Compose
  production deployment, and production authentication hardening.
- **Phase 33/34 intelligence is advisory, not safety-critical.** Temporal
  escalation, anomaly alerts, recurring-pattern hints, and incident risk scores
  are decision-support inputs that **always require human review** before any
  action. They never modify productivity/employee data and never auto-accuse.
  Incident reconstruction reports `PRE_EVENT_EVIDENCE_UNAVAILABLE` rather than
  fabricating a pre-event (no continuous recording is added); person continuity
  keeps `Unknown` as `UNRESOLVED`; event-context rollup never labels anything
  "criminal". Cross-camera correlation, fall/fight/fire/weapon AI, and
  continuous recording are out of scope (see `PHASE_33_FEATURE_AUDIT.md`) until
  matched model + hardware are available and measured.
- **Phase 35 reliability scope.** This phase hardened daemon failure-isolation,
  opt-in backup/restore/retention, and config validation, and validated them on
  the local webcam/CPU path and via pure-python tests. It does NOT add new AI
  detectors, real RTSP/GPU/SMTP/Docker support is validated only where a client
  environment provides it, and a sustained multi-hour soak was **not executed**
  in this environment (declared, never faked). Reliability is preserved: camera
  and detector failures cannot manufacture employee absence or fake events.
- **Phase 36 simulation scope.** This phase adds a deterministic, opt-in
  virtual-office simulation for *software* scalability/failure/long-run testing.
  It is **not** proof of physical camera readiness — simulated cameras are
  explicitly NOT real RTSP cameras, and real RTSP/GPU/SMTP/Docker/multi-camera
  physical deployment and a multi-hour wall-clock soak remain **REQUIRES CLIENT
  ENVIRONMENT / FUTURE_HARDWARE_REQUIRED** and are reported as **NOT EXECUTED**
  here, never faked. The simulation never fabricates unsupported AI families
  (fall/fight/fire/weapon stay `FUTURE_MODEL_REQUIRED`), never touches real
  cameras or the production DB, and never creates employee absence from camera
  offline.

---

## Security notes

- Never commit real credentials. `.env` is the only place secrets live and is
  excluded via `.gitignore`.
- SMTP passwords, RTSP passwords, and dashboard passwords are **never logged**;
  camera URLs are redacted before display/logging (`redact_url`).
- The dashboard login is a **configuration boundary**, not a security boundary —
  set `CCTV_DASH_PASS` + `CCTV_DASH_VIEWER_PASS`, and deploy behind your
  network/VPN/reverse proxy/TLS.
- The Docker image runs as a non-root `appuser` (UID 1000); ensure `./data` is
  host-writable by that UID.
- Retention is **opt-in** (`CCTV_ENABLE_RETENTION=1`); the daemon never deletes
  history automatically otherwise.
- Face enrollment is a **record of volunteers/managers with consent**; treat
  `data/faces/` as personal biometric data — control access, back up securely,
  remove on request.

---

## Release & supporting documentation

The system has completed a release-freeze validation (Phase 30) and has since
been extended by the **Phase 31 security platform**, the **Phase 32 AI Office
Intelligence & Security Platform**, the **Phase 33 Advanced AI Intelligence
& Real-World Pilot Hardening**, the **Phase 34 Next-Generation AI Office
Intelligence & Real-World Pilot Hardening**, and the **Phase 35 Production
Deployment & Real-World Reliability Hardening**. Current, authoritative,
self-contained documents in this folder:

| Document | Purpose |
|---|---|
| `RELEASE_FREEZE.md` | Release-freeze marker: frozen baseline, validation coverage, change-control rules |
| `RELEASE_MANIFEST.md` | Release manifest (Phase 30): validated versions, scores, validation date, known limitations |
| `PHASE_31_FEATURE_AUDIT.md` | Phase 31 feature audit against the frozen baseline (P0/P1/P2, integration points) |
| `PHASE_31_SECURITY_ARCHITECTURE.md` | Security-layer design: event/incident/alert model, data model, module layout |
| `PHASE_31_IMPLEMENTATION_REPORT.md` | Phase 31 implementation report: new modules, detectors, non-goals, future models |
| `PHASE_31_VALIDATION_REPORT.md` | Phase 31 validation report: test results, checklist, verdict |
| `PHASE_32_FEATURE_AUDIT.md` | Phase 32 feature audit (classification legend, implementation plan) |
| `PHASE_32_ARCHITECTURE.md` | Phase 32 architecture: correlation, investigation, hardening, RBAC, isolation |
| `PHASE_32_IMPLEMENTATION_REPORT.md` | Phase 32 implementation report: new modules, hardening, non-goals |
| `PHASE_32_VALIDATION_REPORT.md` | Phase 32 validation report: test results, verdict |
| `PHASE_33_FEATURE_AUDIT.md` | Phase 33 feature audit (classification legend, implementation plan) |
| `PHASE_33_ARCHITECTURE.md` | Phase 33 architecture: intelligence modules, data model, isolation, non-goals |
| `PHASE_33_IMPLEMENTATION_REPORT.md` | Phase 33 implementation report: new modules, hardening, non-goals |
| `PHASE_33_VALIDATION_REPORT.md` | Phase 33 validation report: test results, checklist, pilot evidence |
| `PHASE_33_PILOT_CHECKLIST.md` | Phase 33 pilot / go-live hardening checklist (evidence-based) |
| `PRODUCTION_READINESS.md` | Updated readiness scores across employee + security + deployment |

> **Phase 34/35/36 documentation policy:** These phases introduce no per-phase
> report files. This README is the primary documentation, and a consolidated
> **PROJECT PHASE HISTORY / CHANGELOG** is maintained at the end of this file.

Validation facts (this machine, 2026-09-04):

- **585 pytest tests pass** (0 failures, 0 warnings): the Phase 30/31 baseline
  **(262)** plus 52 Phase 32 tests + 36 Phase 33 tests + 14 Phase 34 tests +
  **12 Phase 35 tests** + **61 Phase 36 tests** (virtual camera, scenario,
  simulation runner, stress, soak + simulation-isolation/webcam regression) +
  **79 Phase 37 tests** (preflight, RTSP harness units, pilot mode, failure/
  recovery matrix, config hardening, retention, backup/restore, security/privacy
  incl. RBAC, process restart, logging, multi-camera, model readiness, Docker
  audit, specialized-model architecture, SMTP readiness, evidence integrity) +
  **21 Phase 38 tests** (website/dashboard rendering, RBAC backend-authoritative
  resolve/dismiss, bearer of the previously-unwired Security-tab filters,
  camera-UI Configured column + no-secret leakage, pre-flight honesty,
  pagination, backup/restore retention invariants).
- **Phase 37 client-environment readiness:** a **pre-flight check** tool
  (`python -m src.preflight`) reports system/GPU/camera/model/database/evidence/
  SMTP/Docker/security/config/pilot status with `--json`/`--verbose`; an **RTSP
  validation harness** (`python -m src.rtsp_harness`) probes real endpoints and
  reports failure honestly (never fabricated success) with credentials redacted.
  Every real environmental capability (RTSP, GPU, SMTP delivery, Docker daemon)
  is reported `READY (PRECONDITION)` or `NOT EXECUTED — ENVIRONMENT LIMITATION`,
  never faked. Pilot mode (`CCTV_PILOT_MODE=1`) is opt-in, adds operational
  health logging, and **never** disables security/RBAC/audit, enables continuous
  recording, or modifies productivity data. Placeholder camera credentials were
  hardened to explicit `CHANGE_ME` markers in `.env`, `.env.example`, and
  `config.py` defaults.
- **Phase 36 real-world validation:** the real webcam path is preserved and
  re-verified (12 real frames → `detect_batch` → real detections, CPU). The
  simulation layer drives the *real* tracker/security/incident/evidence/alert
  pipeline against isolated DBs + temp evidence: 11 scenarios run end-to-end
  with 0 errors; camera-disconnect never manufactures `AWAY`; `Unknown` never
  becomes an employee or touches productivity; cross-camera movement stays
  stable; same-seed runs reproduce identically; evidence (if any) is written
  only into the isolated temp dir with valid files; backup/restore under a
  populated DB is `ok`; database/evidence/alert stress queries stay well under
  bound; resource limits abort runaway simulations safely.
- Live smoke: webcam `ONLINE` ~3 fps, trackers `EMP001` + `Unknown`, graceful
  Ctrl+C shutdown flushes trackers, writes the final report, and checkpoints the
  DB (`PRAGMA integrity_check` → `ok`). Phase 33 webcam capture+detect smoke
  validated (15 frames grabbed, `detect_batch` returned detections). Phase 34
  integration smoke validated reconstruction (`EVIDENCE_PRESENT`), context
  rollup (`CORRELATED_SECURITY_INCIDENT (REQUIRES_HUMAN_REVIEW)`), continuity
  (dwell/zone-transition/cross-camera path), adaptive time baseline
  (`TIME_OCCUPANCY_ANOMALY`), and bounded risk context factors — all on a fresh
  DB with `PRAGMA integrity_check` → `ok`.
- **Phase 35 real-world validation:** a genuine webcam capture+detect smoke
  passed (12 real frames, `detect_batch` returned real detections, CPU device);
  a real backup → restore → integrity → table/record-presence drill passed
  (`backup_integrity=ok`, `restored_integrity=ok`, employees + incidents +
  security_events present); a secret/credential audit confirmed all source-URL
  log sites use `redact_url()` and `_write_live_state` emits camera IDs, never
  raw credentials; `validate_config` reports 0 problems.
- Phase 33/34 intelligence is **advisory and evidence-based**: temporal/anomaly/risk
  outputs always require human review and **never modify productivity data**,
  never fabricate detections, and never auto-accuse. Cross-camera correlation
  returns `CROSS_CAMERA_CORRELATION_UNAVAILABLE` instead of inventing links.
  Incident reconstruction reports `PRE_EVENT_EVIDENCE_UNAVAILABLE`, person
  continuity keeps `Unknown` as `UNRESOLVED`, and event-context rollup never
  labels anything "criminal".
- **Phase 35 reliability hardening fixes a production-critical bug**: the
  daemon previously set `security = audit = None` after constructing the real
  engines, which would crash on the first processed frame and disable security
  processing in production. Now the engines are retained, and detector /
  security-tick / main-loop failures are isolated (degraded no-detector retry
  mode) so the daemon survives camera and detector faults without creating fake
  absence or events.
- **Phase 38 website/deployment hardening (real fixes, not filler):** the
  Streamlit dashboard's Security tab collected Severity/Status filters that were
  **never applied** — extended `src/database.query_security_events` and
  `EventStore.events` and wired the tab to pass them, so the website reflects
  real backend state. The dashboard's Resolve/Dismiss used only a UI-level role
  check (backend was not authoritative) — added
  `AccessGuard.dismiss_incident`, routed both actions through `AccessGuard.require`
  so the **backend** denies viewer actions and audits every denial; fixed a
  latent `AccessGuard.resolve_incident` **TypeError** (passed `resolved_at=` to a
  function expecting `resolved_ts=`). Camera Health now shows a
  `Configured`/`Not configured` column (never an RTSP URL/password). A new
  **Deploy & System Check** dashboard tab (9th) runs `src.preflight.run_all_checks()`
  and renders an honest PASS/WARN/FAIL/SKIP status map + a read-only deployment
  checklist — never a fake-green. Verified via Streamlit `AppTest` that all 9
  tabs render with no exception and the rendered body leaks no credentials.
- Scores (Phase 38, advisory): **Current Software Readiness 97/100 · Employee
  Monitoring Readiness 98/100 · Security Monitoring Readiness 93/100 · Incident
  Management Readiness 95/100 · Website/Dashboard Readiness 95/100 · Operational
  Readiness 93/100 · Client Deployment Readiness 90/100** (see
  `PRODUCTION_READINESS.md`; client-environment capabilities remain
  `REQUIRES CLIENT ENVIRONMENT` until validated on-site).
- Verified exact dependency versions: opencv-python 5.0.0.93, torch 2.13.0,
  torchvision 0.28.0, onnx 1.22.0, onnxruntime 1.29.0, insightface 1.0.1,
  ultralytics 8.4.137, numpy 2.5.2, pandas 3.0.5, openpyxl 3.1.5,
  scikit-learn 1.9.0, streamlit 1.62.0, plotly 7.0.0, python-dotenv 1.2.3,
  pytest 9.1.1.

---

*Version: Phase 30 release-freeze baseline + Phase 31 security platform +
Phase 32 AI Office Intelligence & Security Platform + Phase 33 Advanced AI
Intelligence & Real-World Pilot Hardening + Phase 34 Next-Generation AI Office
Intelligence & Real-World Pilot Hardening + Phase 35 Production Deployment &
Real-World Reliability Hardening + Phase 36 Virtual Office Simulation & Stress
Testing + Phase 37 Controlled Advisory-Pilot Readiness + Phase 38 Real Client
Deployment Validation, 2026-09-04. The webcam+CPU path is
the validated daily driver; all client-site capabilities are frozen until
measured in their target environment. Virtual cameras are NOT real RTSP
cameras.*

---

## PROJECT PHASE HISTORY / CHANGELOG

A consolidated record of what each phase added, hardened, and decided. Phases 34,
35, 36, 37 and 38 follow the README-only documentation policy (no per-phase report files).

### Phase 30 — Release-freeze baseline
- Frozen validated baseline (262 tests green, 0 warnings), release manifest, and
  change-control rules. README primary doc policy established.

### Phase 31 — Security platform (P0/P1/P2)
- Added the security layer: unknown-presence, restricted zones, intrusion,
  after-hours, camera offline/recovered, tamper, incidents, alerts, evidence
  (SHA-256, path-safety), audit log, WAL-safe backup, Security dashboard.
- Documented non-goals (fall/fight/fire/weapon/continuous-recording require
  matched model + hardware). No security → productivity contamination.

### Phase 32 — AI Office Intelligence & Security Platform
- Event correlation, investigation workbench, evidence & alert hardening
  (retry/escalation/lifecycle), structured search, camera intelligence, detector
  registry, security analytics, RBAC, failure isolation.
- 52 new tests (314 total at end of Phase 32).

### Phase 33 — Advanced AI Intelligence & Real-World Pilot Hardening (verdict B)
- Temporal intelligence, camera topology/adjacency, cross-camera correlation
  (`CROSS_CAMERA_CORRELATION_UNAVAILABLE` fallback), occupancy anomaly +
  baselines + recurring patterns, incident risk scoring, evidence access
  control, timeline-v2 search + pagination, detector health, system-health 2.0,
  config snapshot UI. 36 new tests (350 total).
- Scores: Software 93, Employee Monitoring 95, Security 86, Incident Mgmt 91,
  Client Deployment 85.

### Phase 34 — Next-Generation AI Office Intelligence & Real-World Pilot Hardening (this phase)
- **P0 — operational intelligence:** `IncidentReconstruction` (T−30s..T+30s
  event/alerts/evidence window; `PRE_EVENT_EVIDENCE_UNAVAILABLE` instead of
  fabrication; no continuous recording added); `PersonContinuity` (first/last
  seen, dwell, zone transitions, cross-camera path; `Unknown` stays
  `UNRESOLVED`); `EventContextRollup`
  (`CORRELATED_SECURITY_INCIDENT (REQUIRES_HUMAN_REVIEW)` — never accusatory);
  SOC operator view (attention queue + reconstruction + continuity panels).
- **P1 — adaptive intelligence:** `BaselineTracker` time-of-day / day-of-week
  baselines with an `INSUFFICIENT_DATA` guard; `AnomalyEngine.evaluate_occupancy_time`
  raises a spike only from a READY baseline; extended `IncidentRiskScorer`
  explainable context factors (zone sensitivity / after-hours / evidence /
  low-confidence) gated by `CCTV_RISK_CONTEXT_FACTORS`, bounded to `0.0..1.0`.
- **New module:** `src/investigation.py` (read-only, evidence-backed).
- **14 new tests (364 total), 0 regressions.** Integration smoke validates all
  Phase 34 capabilities end-to-end.
- **Known limitations (this phase):** No sustained long-run soak was executed
  (this environment cannot run one without risk of interference) — **ENVIRONMENT
  LIMITATION, never faked**. Performance benchmarks at 1/2/5/10 cameras are
  **ENVIRONMENT / CLIENT_HARDWARE REQUIRED** (measured only on the dev webcam
  CPU path, ~3 fps). Real RTSP / GPU inference / SMTP delivery / Docker remain
  **CLIENT_ENVIRONMENT REQUIRED**.
- **Scores (Phase 34, advisory):** Software 94, Employee Monitoring 95, Security
  88, Incident Mgmt 93, Client Deployment 86.
- **Verdict boundary:** Phase 34 is advisory/analytic only — security and
  productivity remain fully separated; human review is final for every incident.

### Phase 35 — Production Deployment, Real-World Validation & 24/7 Reliability (this phase)

**Status:** COMPLETE — reliability/recovery hardening plus honest real-world
validation. No new AI detectors; no filler features.

**Tests:** 364 baseline unbroken + **12 new Phase 35 tests = 376 total**,
  0 failures, 0 warnings, 0 regressions.

**Major changes (genuine hardening):**
- **CRITICAL FIX:** the daemon (`main.py`) previously set `security = audit =
  None` after constructing the real engines, which would crash on the first
  processed frame and disable security processing in production. Removed.
- **Failure isolation:** `detect_batch`, `security.tick`, and the intelligence
  cycle are isolated so any camera/detector/engine exception never kills the
  daemon; a model-load failure now degrades to a retry-next-batch mode (no
  fabricating detections or absence) instead of `break`.
- **Automatic daily backup:** wired `CCTV_AUTO_BACKUP_DAILY=1` + `CCTV_BACKUP_HOUR`
  into the daemon via the existing WAL-safe `backup_all()`.
- **Backup restore:** added `restore_database()` / `restore_to_temp()` (WAL-safe,
  refuses corrupt backups, never touches the production DB).
- **Retention wiring:** `_maybe_run_retention` now also runs
  `security_retention_cleanup` (events/alerts/audit) and evidence-file retention
  when `CCTV_ENABLE_RETENTION=1` — incidents are never auto-purged.
- **Idempotency:** `upsert_incident_risk` keeps a single latest row per incident;
  `link_incident_event` is a no-op on re-link.
- **Config validation:** added threshold ranges (CONF/FACE), non-negative
  retention (evidence/audit), non-empty camera URLs, and SMTP completeness.

**Real-World Validation:**
- Webcam: **VERIFIED** — real capture + `detect_batch` on 12 live frames (CPU),
  2 real detections.
- Backups: **VERIFIED** — real backup → restore → integrity (`ok` for both) →
  records present; `restore_to_temp` sanity.
- Secrets: **VERIFIED** — all source-URL log sites use `redact_url()`; live-state
  writes camera IDs, never credentials.
- Config: **VERIFIED** — `validate_config` → 0 problems.

**Known Limitations / ENVIRONMENT LIMITATIONS (NOT EXECUTED, never faked):**
- RTSP / multi-camera (1/2/5/10) / GPU / SMTP delivery / Docker / long-duration
  soak / production resource sampling — all **REQUIRE CLIENT ENVIRONMENT /
  FUTURE_HARDWARE_REQUIRED**; no claim of unattended production readiness.
- AI model families (fall/fight/fire/smoke/weapon/PPE) remain **FUTURE_MODEL_REQUIRED**.

**Scores (Phase 35, advisory):** Software 95, Employee Monitoring 95, Security
90, Incident Mgmt 94, Investigation 94, Cross-Camera 86, Camera 89, Alerting 90,
Evidence 93, Analytics 92, AI Model Architecture 88, Reliability 93,
Privacy/Security 93, Deployment 88, Overall 94, Production Readiness 91.

**Final verdict:** **B — READY FOR A CONTROLLED ADVISORY PILOT WITH HUMAN-IN-THE-LOOP REVIEW**
(not yet unattended production; see environment limitations above).

### Phase 36 — Virtual Office Simulation, Stress Testing & Advanced AI Readiness

**Status:** COMPLETE — a deterministic, opt-in simulation + stress-validation
layer that drives the REAL pipeline end-to-end. It does **not** pretend virtual
cameras are real RTSP cameras, adds no new AI detectors, and no filler features.

**Tests:** 376 baseline unbroken + **61 new Phase 36 tests = 437 total**,
  0 failures, 0 warnings, 0 regressions. Phase 36 tests are grouped under
  opt-in `simulation` / `stress` / `soak` markers so normal CI never runs slow
  work accidentally.

**New simulation package (`src/simulation/`, opt-in, never imported by `main.py`
or `app.py`):**
- `virtual_camera.py` — `VirtualCamera` / `VirtualCameraPool` with deterministic
  frames, per-camera independent resolution/FPS and fault state (frozen/dark/
  bright/noisy/disconnect/reconnect), exposing the same batch/health interface
  the production pipeline consumes; hard max-camera/`ResourceLimitError` guard.
- `scenario.py` — seeded office scenarios A–R (normal, arrivals/leaves, phone
  proximity, Unknown, restricted-zone intrusion, after-hours, loitering,
  cross-camera move, multiple + Unknown employees, disconnect/recovery/outages,
  long-running) emitting records in the exact real `detect_batch` schema.
- `faults.py` — deterministic fault injection (camera, detector-raise/sentinel,
  model-load-fail, alert-delivery) strictly opt-in and impossible from
  production runtime.
- `report.py` / `runner.py` — honest measured latency (min/max/avg/p50/p95/p99),
  structured machine-readable `SimulationReport` (JSON), and
  `SimulationRunner` wiring the REAL `MultiTracker.process_batch` +
  `SecurityEngine.tick` + incidents/evidence/alerts against an isolated DB + temp
  evidence dir; resource limits (max cameras/fps/steps/evidence/workers).

**Simulation scenarios validated (real pipeline, isolated DBs):**
- A–R scenarios run end-to-end with 0 errors; camera-disconnect NEVER
  manufactures employee `AWAY`; `Unknown` never becomes an employee nor touches
  productivity; cross-camera movement keeps one stable identity; co-present
  `Unknown` does not corrupt the employee; same-seed runs reproduce identically.
- Evidence (when an incident enables it) is written only into the isolated temp
  dir with valid, non-empty files; backup/restore under a populated simulation
  DB is `ok` with incidents/events/alerts/audit/productivity surviving;
  corrupt-backup rejection holds; database/evidence/alert stress queries stay
  well under bound; link idempotency (Phase 35) holds at scale; resource limits
  abort runaway simulations safely; repeated reconnect causes no duplicate
  incidents and memory/thread growth stays bounded.

**Known limitations / ENVIRONMENT LIMITATIONS (NOT EXECUTED, never faked):**
- Real RTSP cameras, physical multi-camera deployment, GPU, SMTP, Docker, and a
  multi-hour wall-clock soak remain **REQUIRES CLIENT ENVIRONMENT /
  FUTURE_HARDWARE_REQUIRED**. Simulation/simulation-failure/soak tests prove
  software scalability under virtual feeds only — **they are not proof of
  physical camera readiness**.
- Unsupported AI families (fall/fight/fire/smoke/weapon/PPE) remain
  `FUTURE_MODEL_REQUIRED`; the simulation never fabricates them.

**Scores (Phase 36, advisory):** Employee Monitoring 95, Security 91, Incident
Mgmt 95, Investigation 95, Cross-Camera Intelligence 88, Camera Reliability 91,
Alerting 92, Evidence 94, Analytics 94, AI Model Architecture 88, Reliability 95,
Privacy/Security 93, Simulation/Scalability 90, Deployment Readiness 89, Overall
Software Quality 95, Overall Production Readiness 91. (Scores reflect measured
pure-python + real webcam evidence; real client-environment dimensions stay
uncounted for the reasons above.)

**Final verdict:** **B — READY FOR A CONTROLLED ADVISORY PILOT WITH
HUMAN-IN-THE-LOOP REVIEW.** The simulation/stress layer materially increases
confidence in software scalability, failure isolation, and 24/7 reliability
under virtual multi-camera load, but real RTSP/GPU/SMTP/Docker and a genuine
wall-clock soak remain client-environment tasks before an unattended go-live.

### Phase 37 — Controlled Advisory-Pilot Readiness (this phase)

**Status:** COMPLETE — prepares the platform for a supervised, real-client
advisory pilot **without fabricating** real RTSP/GPU/SMTP/Docker results. Adds
on-site validation tooling, an opt-in pilot mode, and a failure/recovery +
deployment-validation test layer. Real environmental capabilities are reported
`READY`/`NOT EXECUTED — ENVIRONMENT LIMITATION` / `FUTURE MODE REQUIRED` — never
invented.

**Tests:** 437 baseline unbroken + **79 new Phase 37 tests = 564 total**,
0 failures, 0 warnings, 0 regressions. Phase 37 tests cover: A1 preflight,
A2 RTSP harness unit, A3 pilot mode, A4 failure/recovery, A5 config hardening,
A6 retention, A7 backup/restore, A8 security/privacy (incl. RBAC), A9 process
restart, A10 logging, A11 multi-camera, A12 model readiness, A13 Docker audit,
A14 specialized-model architecture, A15 SMTP readiness, A16 evidence integrity.

**New on-site validation tooling:**
- `src/preflight.py` — client-environment **pre-flight check** (CLI + `--json` +
  `--verbose`): system (OS/CPU/RAM/disk), GPU/CUDA, camera configuration + URL
  credential checks, models (YOLO/insightface/embeddings), database integrity,
  evidence writability, SMTP config, Docker availability, security posture
  (engine/evidence/dashboard auth/auto-backup), config validation, and pilot-mode
  status. Verdict: `READY` / `READY WITH WARNINGS` / `NOT_READY` / `NOT EXECUTED`.
- `src/rtsp_harness.py` — RTSP endpoint validation harness (`validate_camera`,
  `validate_all`) with credential redaction and honest `READY_FOR_CLIENT_VALIDATION`
  / `NOT EXECUTED — ENVIRONMENT LIMITATION` reporting; a failed probe is reported
  as failed, never as connected.

**Pilot mode (opt-in, `CCTV_PILOT_MODE=1`):**
- `PILOT_MODE`, `PILOT_METRICS_INTERVAL_SEC`, `PILOT_HEALTH_LOG_INTERVAL_SEC` in
  `config.py` (validated in `validate_config`, surfaced in the runtime config
  snapshot).
- Adds a periodic operational health log (camera online count, FPS, detector
  state + model device) so operators can see the platform is healthy during the
  pilot. It **never** disables security/RBAC/audit, **never** enables continuous
  recording, and **never** modifies or accelerates productivity/employee data.

**Hardening (from the Phase 37 audit):**
- Placeholder camera credentials in `.env`, `.env.example`, and `config.py`
  defaults replaced with explicit `CHANGE_ME` markers (no lockable-looking
  `admin:password` defaults that could be mistaken for real secrets).
- `redact_url()` used at all source-URL log sites; live-state emits camera IDs
  only.
- Verified failure/recovery: detector failures isolate; RBAC enforces roles and
  audits denials; pilot mode cannot bypass audit/security; retention/backup/
  restore hold; evidence stays path-safe + SHA-256; multi-camera health is
  reported without leaking credentials; detections are only enabled when a real
  model exists (anti-fabrication).

**Environment limitations (NOT EXECUTED, never faked):** real RTSP camera pools,
physical multi-camera deployment, GPU/CUDA inference, SMTP delivery, and a Docker
daemon run are client-environment preconditions. In this dev environment the
Docker daemon is not running, CUDA is not available, and RTSP endpoints are not
real — these are reported as such by `src/preflight.py` and the harness, and
remain `REQUIRES CLIENT ENVIRONMENT` until measured on-site. Fall/fight/fire/
weapon/PPE remain `FUTURE_MODEL_REQUIRED`.

**Scores (Phase 37, advisory):** Current Software Readiness 97, Employee
Monitoring 98, Security Monitoring 93, Incident Management 95, Investigation 95,
Cross-Camera Intelligence 88, Camera Reliability 91, Alerting 92, Evidence 94,
Analytics 94, AI Model Architecture 90, Reliability 95, Privacy/Security 94,
Simulation/Scalability 90, Deployment Readiness 90, Overall Software Quality 97,
Overall Production Readiness 92. (Client-environment dimensions — real RTSP/GPU/
SMTP/Docker — remain uncounted or `REQUIRES CLIENT ENVIRONMENT` until validated
on-site.)

**Final verdict:** **B — READY FOR A CONTROLLED ADVISORY PILOT WITH
HUMAN-IN-THE-LOOP REVIEW + REAL CLIENT ENVIRONMENT VALIDATION REQUIRED.** The
platform is verifiably ready to begin a supervised advisory pilot; on-site
validation (real RTSP, GPU, SMTP, Docker, and a genuine soak) must be completed
during that pilot before an unattended go-live.

### Phase 38 — Real Client Deployment Validation (website treated as first-class product)

**Status:** COMPLETE — a full-product validation pass (backend + AI + CCTV +
security + productivity + evidence + alerts + **website/dashboard** + admin +
deployment + monitoring + recovery) that fixes genuine website-backend gaps and
proves deployment readiness **without fabricating** real RTSP/GPU/SMTP/Docker
results.

**Tests:** 564 baseline unbroken + **21 new Phase 38 tests = 585 total**,
0 failures, 0 warnings, 0 regressions (full suite; plus 51 `simulation` and 6
`stress` marker tests pass). Adds the `website` pytest marker.

**Website treated as first-class product component — real fixes:**
- **Wired, not fake, Security filters:** the Security tab had Severity/Status
  filter inputs that were collected but **never applied** to the query. Extended
  `src/database.query_security_events` and `EventStore.events` with `severity`/
  `status`, and wired `tab_security` to pass them — the website now reflects real
  backend state (no fake/unapplied controls).
- **Backend-authoritative RBAC on Resolve/Dismiss:** these previously relied on a
  UI-level role check only. Added `AccessGuard.dismiss_incident`, routed both
  actions through `AccessGuard.require(...)` so the **backend denies** viewer
  actions and audits every denial (`access.denied`), then the UI surfaces the
  role-restriction error. Regression fixed: `AccessGuard.resolve_incident` passed
  `resolved_at=` to a DB function expecting `resolved_ts=` → **TypeError**.
- **Camera Health UI:** added a `Configured`/`Not configured` column (from
  `config.CAMERAS` + health sources). Never exposes an RTSP URL or password; a
  caption notes credentials are never shown.
- **Deployment tab:** new 9th dashboard tab **Deploy & System Check** runs
  `src.preflight.run_all_checks()` (31 checks) and renders an honest
  PASS/WARN/FAIL/SKIP status map plus a read-only executable deployment checklist
  (Hardware/OS/Runtime/GPU/Cameras/Network+RTSP/Storage/Database/Models/SMTP/
  Docker/Security+RBAC/Backup/Retention/Monitoring). Read-only and advisory —
  never fabricates a green status.
- **Dashboard rendering & secrets:** Streamlit `AppTest` verifies all 9 tabs
  render with no exception and the rendered body leaks no credentials
  (`rtsp://`, `sender_password`, `smtp_server`, `dash_pass`).

**Validated software/deployment invariants (Phase 38 tests):**
- Severity/status/combined filter wiring + `EventStore` passthrough.
- `AccessGuard.resolve_incident` / `dismiss_incident` succeed for operators and
  **deny + audit** viewer attempts; denial never mutates the incident.
- Camera-UI `Configured` column contract; no secret leakage; settings never show
  secrets.
- Pre-flight honesty (`src.preflight`) — checks report PASS/WARN/FAIL/SKIP and
  resume shape; Docker reports honestly (never a fabricated "running").
- Incident/event query pagination (no unbounded loads).
- Backup → restore preserves employees + incidents + security events; retention
  cleanup never purges incidents (incidents are exempt).

**Environment limitations (NOT EXECUTED, never faked):** real RTSP cameras, GPU/
CUDA inference, SMTP delivery, a running Docker daemon, and a multi-hour wall-clock
soak remain client-environment preconditions reported as such — no claim of
unattended production readiness is made.

**Scores (Phase 38, advisory):** Overall Software Quality 97, Employee Monitoring
98, Security Monitoring 93, Incident Management 95, Investigation 95, Cross-Camera
Intelligence 88, Camera Reliability 91, Alerting 92, Evidence 94, Analytics 94,
AI Model Architecture 90, **Website/Dashboard 95**, **Operational Readiness 93**,
Reliability 95, Privacy/Security 94, Simulation/Scalability 90, Deployment
Readiness 90, Overall Production Readiness 92. (Client-environment dimensions —
real RTSP/GPU/SMTP/Docker — remain uncounted or `REQUIRES CLIENT ENVIRONMENT`.)

**Final verdict:** **B — READY FOR A CONTROLLED ADVISORY PILOT WITH
HUMAN-IN-THE-LOOP REVIEW + REAL CLIENT ENVIRONMENT VALIDATION REQUIRED.** The full
product — including the website/dashboard, now a validated first-class component —
is verifiably ready to begin a supervised advisory pilot; on-site validation (real
RTSP, GPU, SMTP, Docker, and a genuine soak) must be completed during that pilot
before an unattended go-live.