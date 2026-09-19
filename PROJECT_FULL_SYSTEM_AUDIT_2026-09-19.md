# PROJECT FULL SYSTEM AUDIT — PHASE 53

Project: AI CCTV Office Intelligence & Security Platform
Date: 2026-09-19
Mode: AUDIT ONLY — zero code changes, zero config changes, zero tests changes, no git mutations.

Evidence classes: REAL VERIFIED / CODE VERIFIED / SIMULATED / PARTIALLY VERIFIED / NOT VALIDATED / NOT EXECUTED / BROKEN / INCOMPLETE / ENVIRONMENT BLOCKED / MODEL LIMITATION / UNKNOWN. None of these may be upgraded (CODE VERIFIED ≠ REAL VERIFIED, etc.).

Safe checks executed this session (all green unless noted):
- `pytest -q` → 1043 passed, 5 skipped, 116.71 s
- `pytest -q -W error` → 1043 passed, 5 skipped, 115.02 s
- AST/parse sweep of 130 `.py` files → clean (1 flagged file is a valid UTF-8-BOM source, parses under `utf-8-sig`)
- Streamlit health → `/_stcore/health` 200 "ok"; root page 200 (HTTP, no browser automation available)
- Camera discovery/preflight → index 0 opens (MSMF, 640×480, black/frozen), indices 1–5 closed, no RTSP/video
- SQLite read-only inspection of `data/database/sessions.db`

---

## 1. Executive Summary

The project is a single-host (Windows, CPU-only) CCTV office-intelligence
platform: YOLO person/phone detection, InsightFace face recognition (model
downloaded at runtime), genuine OSNet ReID (opt-in, disabled in production),
spatial + per-employee FSM tracking (ACTIVE/ON_PHONE/AWAY), productivity,
security/incident/evidence engine, SQLite persistence, and an 8-page Streamlit
dashboard with RBAC.

The software corpus is large, internally consistent, and heavily test-backed:
**1043 tests pass cleanly under both `pytest -q` and `pytest -q -W error`**
(this session, two fresh runs). Core safety invariants (camera-offline never
fabricates AWAY, trusted face identity cannot be demoted, credential
redaction, evidence SHA-256, in-app fail-closed auth) are CODE VERIFIED.

However, **real-world validation is essentially absent**. The only camera
available opens but yields black/frozen frames — a stable, reproducible,
correctly-classified failure (REAL VERIFIED as a failure event, and correctly
rejected by the Phase 51 feed gate). There is no real RTSP endpoint, no video
file, no real calibration crop set, no labelled phone dataset, no real SMTP
delivery, no GPU, no live multi-person session, and OSNet has never run in
production. The single CRITICAL engineering finding is a deployment-path
default: **`docker-compose.yml` defaults the dashboard to unauthenticated** on
`0.0.0.0:8501`, diverging from the code's `CCTV_DASH_AUTH=1` default. Multiple
documentation claims are stale or contradicted by current files (README test
counts "753"; Phase 52 report claims `osnet_x1_0_msmt17.pth` unavailable while
it exists and was benchmarked the same day; referenced Phase 47–51 reports and
`PHASE50_VALIDATION_PROTOCOLS` are absent from the tree).

Bottom line: extensive, well-tested code; near-zero real-world evidence;
deployment/auth and documentation gaps are the main actionable risks. The next
phase should be a focused remediation phase (P0 auth-default fix, model-
availability preflight hardening, live-state error visibility), then controlled
validation preparation once a usable feed + a rostered employee exist.

## 2. Audit Scope

- Read authoritative sources: `main.py` (1578 ln), `app.py` (2274 ln),
  `config.py` (952 ln), all 50 `src/*.py` modules, `src/simulation/` (6),
  `README.md`, `VALIDATION_REPORT.md`, `AUDIT_REPORT_2026-09-10.md`,
  `PHASE52_REAL_CAMERA_VALIDATION_REPORT_2026-09-19.md`, `.env.example`,
  `.streamlit/config.toml`, `requirements.txt`, `Dockerfile`,
  `docker-compose.yml`, `pytest.ini`, `setup_env.ps1`.
- Inspected live data read-only: SQLite DB (`data/database/sessions.db`),
  `data/live_state.json`, `data/eod_state.json`, `data/appearance.pkl`,
  `data/embeddings.pkl`, `models/` (5 files), `data/reports/*.xlsx` (8),
  `data/*.json` (phase46/49 benchmarks).
- Ran only safe commands (above). No source/config/test file was modified.
- No browser automation available; Streamlit verified via HTTP health only.

## 3. Repository Inventory

Full recursive inventory (excluding `.venv`, `.git`, `__pycache__`,
`.pytest_cache`):

| Path | Type | Purpose | Runtime-used | Tested | Status | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| `main.py` | daemon | pipeline entry | yes | yes | CODE VERIFIED | 1043 suite |
| `app.py` | Streamlit | dashboard (8 pages) | yes | partial (stub) | PARTIALLY VERIFIED | health 200 |
| `config.py` | config | all settings + validate | yes | yes | CODE VERIFIED | test_config |
| `src/camera*.py` (4) | capture/health | camera pool, health, state | yes (camera/manager/state): topology util | yes | CODE VERIFIED | camera tests |
| `src/detector.py` | detection | YOLO person/phone | yes | yes | CODE VERIFIED | test_detector* |
| `src/detector_registry.py` | registry | AI-family status | partial | yes | NOT VALIDATED | future models only |
| `src/face_registry.py` | face | InsightFace + enrollment | yes | yes | CODE VERIFIED | tests |
| `src/reid*.py` (2) | ReID | built-in + OSNet | built-in on; OSNet opt-in | yes | SIMULATED (real) | bench JSON |
| `src/tracker.py` | tracking | spatial + FSM + multi | yes | yes | CODE VERIFIED | robust tests |
| `src/phone_detector.py` | phone | contextual phone scan | yes | yes | CODE VERIFIED / NOT VALIDATED | tests only |
| `src/database.py` | storage | SQLite | yes | yes | CODE VERIFIED | 20+ tests |
| `src/productivity.py`/`analytics.py`/`reporter.py` | metrics | score + xlsx | yes | yes | CODE VERIFIED | tests |
| `src/security_*.py` (3) + `incidents.py`, `alerts.py`, `evidence*.py` (2), `correlation.py`, `risk_scoring.py`, `temporal_intelligence.py`, `anomaly_engine.py`, `tamper_monitor.py`, `zones.py`, `search.py`, `security_analytics.py`, `investigation.py` | SOC | security stack | wired (some lazy) | yes | PARTIALLY VERIFIED | DB shows real events |
| `src/rbac.py`, `domain.py`, `semantics.py`, `auditlog.py`, `evidence_access.py` | auth/vocab | RBAC, redaction | yes | yes | CODE VERIFIED | tests + source |
| `src/employees.py`, `motion.py`, `entry_exit.py` | misc | roster, motion, entry/exit | employees yes; motion/entry_exit partial | yes | PARTIALLY VERIFIED | empty tables |
| `src/failure_isolation.py`, `backup_tool.py`, `preflight.py`, `rtsp_harness.py`, `rtsp_tester.py`, `roi_selector.py`, `notifier.py`, `camera_intelligence.py`, `calibration.py` | ops | fail-isolate, backup, checks, RTSP, tools | calibration/preflight/notifier yes; roi/tester util | yes | CODE VERIFIED (util) | tests |
| `src/phase46_benchmark.py`, `phase46_research.py`, `phase49_benchmark.py` | research | offline benches | no | yes | SIMULATED | bench JSON |
| `src/dummy_face_enroll.py` | util | enrollment | util | no | SIMULATED | used pre-audit |
| `src/simulation/` (6) | sim | faults, virtual camera, scenarios | no (isolated) | yes | SIMULATED | isolation test |
| `tests/` (68 files) | tests | suite | n/a | n/a | 1043 pass | runs |
| `models/` (5) | models | yolo11n, yolov8n, osnet×3 | yolo yes; osnet opt-in | OSNet load tested | present | sizes verified |
| `data/faces/` (9) | enrollments | EMP001×6, 002, 003, 004 | face registry | no | real files (not live-validated) | face_registry_status |
| `data/embeddings.pkl`, `appearance.pkl` | caches | biometric/reid | yes | no | CODE VERIFIED (cache) | no retention (issue) |
| `data/database/sessions.db` | live DB | main state | yes | no | real data | SQLite inspect |
| `data/backups/…db` | backup | 09-02 snapshot | archive | yes | EXECUTED once | backup tool |
| `data/reports/*.xlsx` (8) | reports | daily 08-31…09-12 | archive | yes | real files | report tests |
| `data/logs/app.log` | log | audit/op | yes | no | real | tailed |
| `data/live_state.json`, `eod_state.json` | state | runtime + eod | yes | no | stale (frozen feed) | inspected |
| `data/phase46_*.json`, `phase49_benchmark.json` | bench | phase46/49 | no | yes | SIMULATED | contents read |
| `data/calibration/` | absent | — | — | — | NOT EXECUTED | does not exist |
| `samples/` | absent | — | — | — | NOT EXECUTED | does not exist |
| `.env`, `.env.example` | secrets/tpl | env config | yes | — | placeholders | key-level only |
| `.streamlit/config.toml` | theme | dark SOC | yes | — | OK | read |
| `requirements.txt`, `Dockerfile`, `docker-compose.yml`, `pytest.ini`, `.gitignore`, `.dockerignore`, `setup_env.ps1` | build | px | yes | partial | issues found | see register |
| `test_droidcam*.py`, `main_backup_before*.py` | root artifacts | legacy/debug | no | no | LEGACY | not imported |
| `models/…pth` untracked; 4 deleted docs (`D FIX_REPORT…`, `GAP_ANALYSIS…`, `PRODUCTION_READINESS.md`, `PROJECT_FULL_SYSTEM_AUDIT.md`) | docs | — | no | no | ORPHANED (deleted) | git status |
| stray `s -ExecutionPolicy…` file | artifact | accidental PS save | no | no | ORPHANED | prior audit flagged |
| `README.md`, `VALIDATION_REPORT.md`, `AUDIT_REPORT_2026-09-10.md`, `PHASE52…md`, this report | docs | — | — | — | stale claims | see §24 |

Duplicates/conflicts:
- Two YOLO weights (`yolov8n.pt` fallback + `yolo11n.pt` default) — intentional.
- `DASH_REFRESH_SEC` alias of `DASHBOARD_REFRESH_SEC` (`config.py:597-598`).
- `ROI_PRESET_FILE` legacy alias of `ROI_PRESETS_FILE` (dead).
- Motion detector exists (`motion.py`) but table empty and cadence tied to main loop — partially wired.
- `entry_exit.py` constructed unconditionally but `ENTRY_EXIT_ENABLED` unused.

## 4. Actual Architecture (traced, not copied)

Verified runtime pipeline with entry points (see §5 for the entry-point map):

```
CAMERA (CCTV_CAM_* pool or --source index)       config.py:52-107, main.py CLI
  ├─ webcam  index via MSMF/DSHOW; RTSP via URL (placeholders only)
  └─ camera.py + camera_manager.py: reader threads, reconnect 2s→30s,
     read timeout 4s, stale frame 10s (FRAME_STALE_SEC=10), health states
     ONLINE/LOW_FPS/FROZEN/FROZEN_FRAME, black-frame gate (BLACK_FRAME_MEAN)
     latest-frame policy (latest_frames(), excludes stale)
        ▼
MAIN LOOP (main.py): rounds frames → batch → detector
  ├─ person YOLO (detector.py, yolo11n.pt default / yolov8n.pt fallback,
  │   CONF threshold, person class COCO 0, per-camera cadence)
  ├─ phone detector (phone_detector.py, YOLO cls 67, PHONE_DETECT_CADENCE=3,
  │   IMGSZ 1280, per-person association later in tracker)
  ├─ face (face_registry.py, InsightFace buffalo_l, FACE_DETECT_CADENCE=5,
  │   similar/margin logic; threshold 0.60, candidate 0.50, margin 0.06)
  └─ ReID (reid.py AppearanceExtractor; built-in descriptor by default;
       OSNet only if CCTV_REID_MODEL_PATH set; REID_CADENCE=3)
        ▼
src/tracker.py
  ├─ SpatialTracker: IoU 0.20, max age 3s, child-of-3 adopt,
  │   never-demote-trusted, identity voting (tracker.py:701-733,
  │   1000-1035)
  ├─ MultiTracker dedup + per-camera feed
  └─ EmployeeTracker FSM ACTIVE/ON_PHONE/AWAY (PHONE_AFTER_SEC=5.0,
       AWAY_AFTER_SEC=3.0, PRESENCE_PATIENCE_SEC=8, GAP_GRACE=2.0,
       SMOOTHING_BUFFER_SEC=5; camera-offline freeze tracker.py:125-149)
        ▼
productivity.py → analytics.py (is_unknown filtered) → reporter.py (xlsx) → data/reports
        ▼
security_engine.py (unknown-presence, after-hours, intrusion, loitering,
  occupancy, offline/recovered, tamper; detector-blind handling) → incidents
  (INC-…, correlation ids, SOIX queue) → evidence.py (OFF default) →
  alerts.py (retry 3/60s/escalate 900s) → notifier (SMTP)
        ▼
database.py (SQLite WAL; transaction wraps)  ⇐ main loop + security + evidence
        ▼
app.py (Streamlit, read-only DB handle PRAGMA query_only) + data/live_state.json
  (atomic tmp→replace main.py:375-379)
```

Verified facts that differ from "README architecture":
- README implies a single webcam default; the real default config path is the
  `CAMERAS` pool from `CCTV_CAM_*` (config.py:52-77) unless `--source` is given.
- `SOURCE`, `VIDEO_FILE_PATH`, `RTSP_URL` config knobs are **not used** by the
  daemon (SOURCE dead; VIDEO_FILE/RTSP only in `roi_selector`).
- `ENTRY_EXIT_ENABLED`/`MOTION_ENABLED` are not honoured as gates at runtime.
- Simulation is fully isolated (never imported by production) — matches docs.
- OSNet load path exists; production default is built-in descriptor — matches.

## 5. Runtime Entry-Point Map

| Entry point | File | Starts | Notes |
| --- | --- | --- | --- |
| CCTV daemon | `main.py` (CLI `--source`, `--cam`, etc.) | detector+tracker+security+DB+live_state | `validate_config()` at main.py:923-928; scheduler |
| Dashboard | `app.py` (`streamlit run app.py`) | UI+RBAAC+queries | read-only DB |
| Calibration tool | `python -m src.calibration` | calibration CLI (Phase 50/51) | feed gate fail-closed |
| RTSP harness | `src.rtsp_harness`, `src.rtsp_tester` | RTSP checks | utilities |
| Preflight | `python -m src.preflight` | env/model checks | no repo adapter examined |
| Backup tool | `src.backup_tool` | WAL backup/restore | utility |
| Benchmarks | `src.phase46_*`, `src.phase49_benchmark` | offline | SIMULATED |
| Simulation | `src.simulation.*` | sims/tests | never in production |
| Streamlit startup | `streamlit run app.py` | verified 200 OK | — |
| DB init | inside `database.py` / `main.py` | schema create-if-missing | — |
| Camera startup | `camera_manager` (pool) or `--source` single | reader threads | — |

Modules/features NOT reachable at runtime (orphan/dead): `ROI_PRESET_FILE`,
`SOURCE`, several pilot knobs (`PILOT_METRICS_INTERVAL_SEC` etc.),
`NORMALIZE_SCHEMA`, `ALERT_COOLDOWN_SEC`, `DEFAULT_ALERT_CHANNELS`,
`INTRUSION_PERSIST_SEC`; legacy `test_droidcam`.py and `main_backup_*.py` at
root are not imported by any entry point.

## 6. Camera Audit

Current environment verification (re-run this session, matches Phase 52):

| Source | Backend | Opens | Res/FPS | Brightness | Change | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| index 0 | DSHOW | no (C++ exc) | — | — | — | ENVIRONMENT BLOCKED |
| index 0 | MSMF | yes | 640×480 / 30 nominal (~18.5-18.5 capture) | lum med 0.0 (12-frame) | diff 0.0 | ENVIRONMENT BLOCKED / INVALID FEED (black/frozen) |
| index 1–5 | any | no | — | — | — | ENVIRONMENT BLOCKED |
| RTSP | — | none configured | — | — | — | ENVIRONMENT BLOCKED |
| Video | — | none (`samples/` absent) | — | — | — | ENVIRONMENT BLOCKED |

Phase 51 feed gate on a 10 s real capture: `state=NOT EXECUTED / INVALID FEED`,
reason `BLACK/DARK (median brightness below minimum)`, brightness_median 4.12,
change_mean 0.0, changed_ratio 0.0, fps 18.54. Reproducible across sessions.

Camera system code itself: reader threads, reconnect backoff, read timeout,
stale-frame exclusion, FROZEN/low-FPS states, black-frame gate — CODE VERIFIED
by `test_camera_failure.py`, `test_camera_health_realfeed.py` (5 skips), etc.
No resource-leak defects proven; `cap.release()` present in health paths.
No destructive camera changes made.

## 7. Person Detection Audit

- Model: YOLO via `detector.py`. Default `models/yolo11n.pt` (config.py:161),
  fallback `yolov8n.pt`. Verified present on disk.
- `detector.py:287` `YOLO(model_path)` — if missing, ultralytics auto-download
  (air-gap risk). In-workspace no GPU path (CPU only).
- Bounding boxes, per-person entries + confidence; multi-person supported;
  detection cadence per-camera round-robin (`TARGET_FPS_PER_CAMERA=2`) and
  face cadence 5, phone cadence 3.
- Docs model = loaded model (yolo11n default) — consistent.
- Real feed: none (blocked). Real-world mAP/person accuracy NOT VALIDATED.

## 8. Face Recognition Audit

- `face_registry.py`: InsightFace `buffalo_l` (runtime download), embeddings
  cache `data/embeddings.pkl` (v3), rosters `data/faces/` (9 images;
  EMP001×6 + EMP002/003/004 ×1 each). EMP001 includes a `MULTIPLE_FACES`-tagged
  image (inherited from enrolment history — low risk, flagged).
- Similarity + threshold model: threshold 0.60 (confirmed), candidate 0.50,
  margin 0.06, confusability cap 0.88; candidates never demote trusted identity
  (config.py:178-182). Consecutive voting via `IDENTITY_STABILITY_FRAMES=3`.
- Registry: `employees` table EMP001–004 all active=1, enrolled=1.
- Invariant verification (all CODE VERIFIED / tests; none REAL VERIFIED):
  1. Unknown ≠ employee ✔ tests
  2. Face authority highest ✔ camera/tracker tests
  3. Weak ReID cannot override trusted face ✔ candidate guard
  4. Ambiguous face doesn't fabricate identity ✔ threshold/margin
  5. Unknown cannot become employee without evidence ✔ adoption frames
  6. No silent identity jump between people ✔ switch-after-N voting
  7. EMP003/Piyush cannot be fabricated ✔ no live data
  8. Raw embeddings not logged ✔ redaction contract
- No live face ever recognized on a real feed (blocked).

## 9. ReID / Appearance Audit

- Built-in descriptor: L2-normalised 256-d (config default dim), used in
  production default. Cache `data/appearance.pkl` v2 (CACHE_VERSION=2).
- OSNet: genuine `src/reid_osnet.py` OsnetEmbedder; variants x0_5/x1_0/ain_x1_0
  all present on disk (verify: x0_5 10.56 MB, x1_0 16.47 MB, ain 16.49 MB) and
  loadable (Phase 49 bench + Phase 51 identity test with x0_5).
- Routing: `reid.py` picks torch-OSNet → ONNX → built-in; on load failure it
  **warns and falls back to built-in** (`reid.py:127-133, 147-153`) — a silent
  downgrade risk (documented; flagged HIGH CQ).
- Production gate: `REID_GATE_RESOLVED=0`, `REID_GATE_IOU=0.30` hardcoded trial
  gate; OSNet remains opt-in; production default built-in. **No threshold
  changes, no automatic calibration write to config.py** (calibration writes
  JSON only).
- Calibration tool (`src/calibration.py`): feeder + recommender + fail-closed
  checkpoint guard (`_ensure_model_fail_closed`); dry-run only; never run on
  real crops (data/calibration absent).
- Real ReID validation: none (no real feeds; OSNet benchmarks are SIMULATED).

## 10. Tracking Audit

- Single tracker implementation (`src/tracker.py`) — SpatialTracker +
  MultiTracker + EmployeeTracker; **no second tracker, no hidden registry, no
  DeepSORT/ByteTrack parallel path** (verified by search).
- IoU 0.20 threshold, max age 3.0 s, adopt-after-3, switch-after-N identity
  voting, never-demote trusted identity.
- Unknown numbering deterministic (`Unknown-<n>` seeded per tracker).
- Pruning of stale tracks and unknown-label recycling — covered by tests
  (`test_tracker.py`, `test_tracker_regression.py`).
- Phone attribution tied to owning track (no cross-person leakage by design;
  tests). Multi-person covered (`test_multiperson_pipeline.py`).
- Camera failure: track state freeze (offline gap excluded). CODE VERIFIED
  via `test_camera_failure.py`.
- Real multi-person: none.

## 11. Employee State Audit

- FSM ACTIVE / ON_PHONE / AWAY with wall-clock timing (no frame-count
  substitution). Timers: PHONE_AFTER_SEC 5.0, AWAY_AFTER_SEC 3.0,
  PRESENCE_PATIENCE_SEC 8.0, SMOOTHING_BUFFER_SEC 5, GAP_GRACE 2.0.
- Camera-offline/ frozen → freeze timers, no state change, offline gap excluded
  from committed intervals (`tracker.py:125-149`, `main.py:1336-1339`).
- **Critical invariant "camera offline/frozen ≠ AWAY"**: CODE VERIFIED (many
  tests); NOT REAL VERIFIED (no valid feed ever). The frozen feed is REAL
  VERIFIED as a rejection event, not as an AWAY condition.
- State transitions/recovery/resets covered by tests; real live transitions
  unobserved.

## 12. Phone Detection Audit

- YOLO class 67 with contextual person-scan (`phone_detector.py`), cadence 3,
  IMGSZ 1280 (full-frame cost). phone_since/phone_last tracked per employee;
  ON_PHONE after PHONE_AFTER_SEC with gap grace.
- One-alert-per-episode enforced in FSM; alerts deduped; no per-frame alerts
  (tests).
- Attribution to owning track; unknown phone user stays Unknown (no identity
  fabrication). Camera failure freezes phone state.
- **Functional FSM correctness: CODE VERIFIED. Real phone detection accuracy
  (precision/recall): NOT VALIDATED / MODEL LIMITATION (small phones).** No
  labelled phone dataset exists. DB `security_events` shows exactly 1
  PHONE_USE event (from old offline-day feeds) — not live-sourced.

## 13. Productivity Audit

- `productivity.py` computes `score = productive / max(1, productive + phone
  + away)`; last-known per day; `analytics.py` filters `is_unknown` so Unknown
  never yields a roster row. daily/report via `reporter.py` → xlsx (8 files).
- Security events never write into productivity (semantics separation verified
  `semantics.py`; security events don't mutate productivity tables).
- Double-counting: timers owned by EmployeeTracker; committed intervals on
  transitions; offline gaps excluded — tests. Checked DB `activity_logs`
  (159 rows): EMP001 elevated ACTIVE/AWAY counts; Unknown has largest ACTIVE
  total (5,779 s) — historical/offline-era artifacts, not evidence of a
  current bug.
- 09-01 EOD entry without xlsx; 08-31 xlsx without EOD row — report
  continuity gap (LOW/MEDIUM).
- Real accuracy of productivity = NOT VALIDATED.

## 14. Security / Incident Audit

Runtime-connected and DB-backed (PARTIALLY VERIFIED, real historical rows):
- `security_engine.py` orchestrates unknown-presence / after-hours / intrusion
  / loitering / occupancy / offline→recovered / tamper; detector-blind handling
  (`DETECTOR_UNAVAILABLE`, rate-limited).
- `incidents.py` `INC-…-NNNN`, correlation ids, temporal states; `alerts.py`
  lifecycle.
- Evidence chain: `evidence.py` (OFF default; SHA-256; path-safety; orphan
  cleanup); `evidence_access.py` (audit-controlled).
- Live DB check: incidents 5 rows (all OPEN/DETECTED — CAMERA_RECOVERED INFO
  never auto-resolved, flagged); security_events 91 (60 AFTER_HOURS, 23
  CAMERA_RECOVERED, 7 CAMERA_OFFLINE, 1 PHONE_USE); anomaly_events 6 all
  `REPEATED_PATTERN_DETECTED` on UNKNOWN_ZONE, re-fired 09-05 10:50-10:54 (4
  in 5 min) — indicates anecdotal cooldown gap (MEDIUM); alert_rules/alerts/
  security_zones/evidence_files/camera_health majority empty (fine by design).
- Simulation harness isolated; scenario families bounded by tests.
- Real-time security accuracy: NOT VALIDATED.

## 15. Evidence Audit

- Code: snapshots (crops, per EVENT/continuous), SHA-256, path safety,
  machine-generated names, retention 30 d, size caps, orphan cleanup,
  audit-controlled access — CODE VERIFIED.
- Runtime: `EVIDENCE_MODE=OFF` and `evidence_files` **0 rows** — evidence has
  never actually captured anything in the live DB. NOT EXECUTED in practice.
- Privacy: verified no raw face images / raw embeddings / passwords / RTSP
  creds / SMTP creds in logs (redaction contract). `data/embeddings.pkl` +
  `appearance.pkl` persist with no retention path — privacy gap (MEDIUM).

## 16. Alert Audit

- `alerts.py`: generation, dedup, cooldown (per-rule `ALERT_COOLDOWN_SEC`),
  delivery attempts (retries 3/60 s, escalation 900 s), status lifecycle —
  CODE VERIFIED.
- Wired into runtime via security engine `safe_call` (alert evaluate +
  evidence capture paths).
- Runtime: `alerts` table has **0 rows**; SMTP creds are placeholders →
  alert delivery never exercised with a real mail server (ENVIRONMENT
  BLOCKED). Episode semantics for phone verified in tests only.

## 17. Database Audit

- SQLite at `data/database/sessions.db`, WAL, `.bak-p27` present; schema
  22 tables (activity_logs, employees, security_events, incidents,
  incident_events/notes, alert_rules, alerts, audit_log, security_zones,
  evidence_files/access, camera_config/health, detector_registry,
  camera_topology, anomaly_events, behavioral_baselines, incident_risk,
  motion_events, entry_exit_events).
- Transactions: inserts wrapped (`database.py`); WAL concurrent readers;
  backup via `backup_tool.py` WAL-safe + integrity-gated restore.
- Row audit (read-only): activity_logs 159; incidents 5 OPEN; security_events
  91; anomaly_events 6; alerts 0; evidence_files 0; motion 0; camera_health 0;
  employees 4 (all enrolled). Index usage not fully audited (no FK cascades
  for some joins; indices exist on key columns via CREATE INDEX — not
  exhaustively verified this pass).
- Timestamps consistent ISO; event→query→dashboard path present.
- Risks: incidents left OPEN indefinitely; anomaly re-fires; no biometric
  retention; report/EOD continuity gap. No live data-loss defect proven.

## 18. Authentication / RBAC Audit

Source-verified:
- `src/rbac.py` AccessGuard; `app.py` auth: `_auth_fail_closed()` runs before
  the `_auth_enabled()` shortcut (`app.py:303-314`); `_authenticate` requires
  non-empty admin password and a separate viewer password (`app.py:354-357`);
  session expiry (15 min) enforced every render; logout clears session.
- Defaults: `DASH_AUTH_ENABLED="1"` (config.py:600), admin/viewer passwords
  empty by default, `DASH_FAIL_CLOSED=0`. **In-app fail-closed only triggers
  when FAIL_CLOSED=1 and no password — see CRITICAL deploy issue.**
- Secrets: `.env` git-ignored; redacted logging; preflight rejects placeholders.
- Findings: Docker default auth-off (CRITICAL, §26/§32); compose doesn't
  forward `CCTV_DASH_FAIL_CLOSED`; no TLS by default; shared viewer password
  weak audit identity (MEDIUM/LOW).

## 19. Dashboard Audit

- `app.py`: 8 pages (Dashboard, Live Monitoring, Employees, Security,
  Reports, Analytics, Settings, Deploy/System Check); read-only DB handle
  (`PRAGMA query_only=ON`), `st.cache_data(ttl=5)`; dark SOC theme.
- Health verified (this session): `/_stcore/health` 200 "ok"; root page 200.
  No browser automation available — UI logic verified only by stub tests
  (`test_dashboard_app.py`, `test_dashboard_phase42.py`,
  `test_dashboard_auth.py`).
- Issues: Settings shows dead tunables (intrusion_persist/alerts_cooldown/
  pilot_*); live page would render black frames from broken feed (no explicit
  invalid-feed banner confirmed); 5s refresh with stale data.
- Empty/error states: present in code path for several pages (not fully
  exercised).

## 20. Reporting Audit

- Daily/employee/security reports; exports xlsx; timestamps ISO; calculations
  from DB events (final scores). `report_consistency` tests pass.
- 8 xlsx in `data/reports/` (08-31…09-12). 09-01 missing xlsx vs EOD entry;
  08-31 xlsx without EOD entry — inconsistency flagged.
- Real report correctness (live data) NOT VALIDATED; offline-era rows explain
  low volumes.

## 21. Performance Audit

Separate REAL vs SIMULATED:
- REAL: none (broken feed prevents live measurement). CPU-only host.
- SIMULATED/BENCH: Phase 45 ~268 ms full cycle / ~10 FPS on cadence-skip;
  Phase 49 OSNet init/embed ms (CPU); phase46 research JSON. These are NOT
  real performance.
- Expected bottlenecks on CPU: YOLO cadence (2 fps default), phone full-frame
  1280 scan every 3rd cycle (highest single cost), face every 5th, ReID per
  track per cadence (built-in); SQLite writes small; Streamlit poll 5 s.
  If OSNet armed, CPU cost rises. No memory-leak proof; bounded batches exist.

## 22. Test Audit

Execution (this session, fresh):
- `pytest -q` → **1043 passed, 5 skipped, 116.71 s**
- `pytest -q -W error` → **1043 passed, 5 skipped, 115.02 s**
- failures 0, errors 0, xfailed 0. 68 test files, 1048 collected.

Classification (by file survey):
- Unit: dominant above all.
- Integration: `test_e2e_pipeline.py`, `test_main.py`, `test_main_scheduler.py`,
  `test_pipeline_fix.py` (main-boot style but with injected/stub inputs).
- Regression: tracker_regression, camera_failure, dashboard_auth, rbac,
  correlation, failure_isolation, phase_ab, etc.
- Simulation: `test_virtual_camera.py`, `test_scenario.py`, `test_simulation_*`.
- Real hardware: `test_camera_health_realfeed.py` (5 tests, skipped).
- UI: dashboard (3 files), stub-driven. Security: rbac/auth/auditlog/evidence/
  redaction. Database: test_database/test_security_database/test_backup_tool.
  Model: detector/face/reid/calibration unit tests.
- Weak-point observations: several older phase tests assert "no exception",
  attribute existence, or dict-key presence without semantic checks; many use
  synthetic crops/frames; some tests exercise standalone logic rather than the
  full production main loop (main loop paths under-covered). Few tests boot the
  real camera path (skipped). No live RTSP/SMTP/GPU tests.

## 23. Phase History Cross-Check

| Phase | Intended work | Actual implementation | Tests | Real validation | Current status | Remaining issue |
| --- | --- | --- | --- | --- | --- | --- |
| 40–42 | pipeline/detection refinements | code present | yes | none | CODE VERIFIED | — |
| 43 | per-person events | code | yes | none | CODE VERIFIED | — |
| 44/44A/44B | phone+FSM | FSM + detector | yes | none (accuracy) | CODE VERIFIED | phone accuracy NOT VALIDATED |
| 45 | realism/perf | single-source FPSCAP + benches | yes | none (CPU bench) | SIMULATED code | no live perf |
| 45A | (report) | — | — | — | NOT EXECUTED evidence | doc absent |
| 46 | research bench | motion/spatio-temporal bench | yes | none | SIMULATED | anomaly cooldown gap (DB) |
| 47 | validation protocol report | code of report | — | none | NOT EXECUTED (real) | report absent from tree |
| 48 | real validation | tried index 0 → black | — | black/frozen (R/V) | ENVIRONMENT BLOCKED | same blocker |
| 49 | genuine ReID | vendored OSNet, opt-in | yes | OSNet load verified | SIMULATED (real) | runtime fallback risk |
| 50 | calibration tool | written + gated | yes | none real | CODE VERIFIED | no real crops |
| 51 | calibration hardening | done; feed gate | yes (73) | gate rejects video | CODE VERIFIED | — |
| 52 | real-camera execution | discovery + gate → reject | — | black/frozen (R/V) | ENVIRONMENT BLOCKED | checkpoint-unavailable claim contradicted |

Cross-checks flagged:
- README "753 tests" (5 places) ≠ actual 1043.
- PHASE52 "osnet_x1_0 unavailable" ≠ file present (16.47 MB) + benchmarked same day (phase49_benchmark.json).
- README capability matrix skips Phases 40–48 rows.
- VALIDATION_REPORT names EMP002 Rahul / EMP003 Priya ≠ DB EMP002 Sourabh / EMP003 Piyush.
- Referenced PHASE47–51 reports and `PHASE50_VALIDATION_PROTOCOLS` absent from tree (4 tracked docs deleted).

## 24. Documentation Audit

Stale/contradicted (see §23), plus:
- README phase rows current to 51 but missing 40–48; test counts wrong; demo
  URLs (rtspstream/wowza) — real endpoints not exercised.
- `.env.example` placeholders `CHANGE_ME` consistent with preflight rejection.
- Docker compose auth default contradicts app default (CRITICAL doc+config gap).
- `PRODUCTION_READINESS.md` etc. deleted from tree yet cited by older reports.

## 25. Dead / Orphan / Duplicate Code

- Dead config: `SOURCE`, `VIDEO_FILE_PATH`, `RTSP_URL` (utility-only),
  `ENTRY_EXIT_ENABLED`, `INTRUSION_PERSIST_SEC`, `ALERT_COOLDOWN_SEC`,
  `DEFAULT_ALERT_CHANNELS`, `NORMALIZE_SCHEMA`, `PILOT_*_INTERVAL_SEC`,
  `ROI_PRESET_FILE`.
- Orphaned: root `test_droidcam*.py`, `main_backup_before*.py`, stray
  `s -ExecutionPolicy…` file, 4 deleted docs.
- Duplicates: dual YOLO weights (intentional), `DASH_REFRESH_SEC` alias.
- Duplicate state machines: none (single tracker/FSM). Duplicate registries:
  one.
- Simulation isolated (not dead, but opts-out by design).

## 26. Security / Privacy Risk Register

| ID | Severity | Finding | Evidence | Impact | Exploitability | Remediation |
| --- | --- | --- | --- | --- | --- | --- |
| SEC-01 | CRITICAL | Docker dashboard default unauthenticated on 0.0.0.0:8501 | compose:126 `CCTV_DASH_AUTH:-0`, :103/117 vs config:600 `"1"`; FAIL_CLOSED not forwarded | Live presence/AWAY/phone/security exposure | Easy (any network peer) | Default auth on, forward FAIL_CLOSED, require passwords, TLS guide |
| SEC-02 | HIGH | Model weights auto-download (yolo11n / buffalo_l) if absent | detector.py:287; face_registry | Air-gap boot fail; DETECTOR_UNAVAILABLE flood; supply-chain vector | Medium (network) | Vendor weights / --validate-models preflight |
| SEC-03 | MEDIUM | Biometric cache no retention | config.py:203,351; main.py:619-630 | PII persists indefinitely | Low | Add retention/manual purge docs |
| SEC-04 | MEDIUM | No TLS default; shared viewer password weak audit identity | app.py:12,356 | Credential sniffing; unidentifiable viewers | Medium (network) | Reverse proxy; per-user accounts |
| SEC-05 | MEDIUM | Compose shares full data/ volume to web app + daemon | compose:118-120 | Compromise exposes DB + embeddings | Medium | Mount minimal subsets |
| SEC-06 | LOW | SMTP creds in container env visible via docker inspect | compose:73-77 | Credential exposure | Low | Secrets manager |
| SEC-07 | LOW | live_state.json plaintext with per-employee fields on disk | main.py:352-374 | Info exposure | Low | Restrict perms |
| SEC-08 | INFO | Redaction discipline overall good | auditlog contract; camera redact; notifier | (positive) | — | Keep |

## 27. Master Feature Matrix

| Feature | Implemented | Tested | Real validated | Runtime connected | Status | Evidence | Issue |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Camera capture | yes | yes | ✗ (black/frozen) | yes | ENV BLOCKED | probe | feed invalid |
| Camera health | yes | yes | partial (rejects bad feed) | yes | PARTIALLY VERIFIED | gate + tests | — |
| RTSP/multi-cam | yes | partial | ✗ | yes | ENV BLOCKED | tester | placeholders |
| Person detection (YOLO) | yes | yes | ✗ | yes | CODE VERIFIED | tests | air-gap download |
| Phone detection | yes | yes | ✗ | yes | CODE VERIFIED | tests | no labelled set |
| Face recognition | yes | yes | ✗ | yes | CODE VERIFIED | tests | runtime download |
| Employee registry | yes | yes | partial (files, no live) | yes | NOT VALIDATED | DB | multi-face img |
| Unknown tracking | yes | yes | ✗ | yes | CODE VERIFIED | tests | — |
| Spatial tracker | yes | yes | ✗ | yes | CODE VERIFIED | tests | — |
| ReID built-in | yes | yes | ✗ (sim bench) | yes | SIMULATED | bench | — |
| OSNet | yes (opt-in) | yes (load) | ✗ (sim bench) | no (off) | SIMULATED | bench | runtime fallback |
| Identity FSM | yes | yes | ✗ | yes | CODE VERIFIED | tests | — |
| Phone FSM | yes | yes | ✗ | yes | CODE VERIFIED | tests | 1 alert/ep verified |
| AWAY | yes | yes | ✗ | yes | CODE VERIFIED | tests | offline≠AWAY invariant |
| Productivity | yes | yes | ✗ | yes | CODE VERIFIED | tests | report gap |
| Security events | yes | yes | historic rows | yes | PARTIALLY VERIFIED | DB | recovery OPEN |
| Incident engine | yes | yes | historic rows | yes | PARTIALLY VERIFIED | DB | no autoclose |
| Correlation | yes | yes | ✗ | yes | CODE VERIFIED | tests | no topology |
| Evidence | yes | yes | ✗ (0 captures) | wired | NOT EXECUTED | DB empty | mode OFF |
| Alerts | yes | yes | ✗ (0 rows) | wired | ENV BLOCKED | DB empty | SMTP placeholder |
| Database | yes | yes | ✗ | yes | CODE VERIFIED | tests | minor gaps |
| RBAC/auth | yes | yes | ✗ | yes | CODE VERIFIED | tests + health | deploy default |
| Dashboard | yes | stub-tested | health 200 | yes | PARTIALLY VERIFIED | health | no browser |
| Reports | yes | yes | 8 real files | yes | PARTIALLY VERIFIED | files | EOD mismatch |
| Backups | yes | yes | 1 real backup | util | EXECUTED | file | 1 snapshot |
| Calibration | yes | yes | ✗ (dry-run only) | CLI | CODE VERIFIED | cal | no real crops |
| Simulation | yes | yes | ✗ | no (isolated) | SIMULATED | tests | — |
| Preflight | yes | yes | ✗ live | util | CODE VERIFIED | tests | model check |
| Performance | partial | bench | ✗ | n/a | SIMULATED | bench | no live |
| Deployment | partial | partial | ✗ | n/a | PARTIAL | compose | auth default |

## 28. Confirmed Working

CODE VERIFIED (test-backed): camera-health invariants; offline≠AWAY freeze;
trusted-identity protection; phone one-alert-per-episode; Unknown handling;
RBAC/auth fail-closed ordering; credential redaction; evidence integrity;
backup/restore; productivity no-double-count; reporting consistency; full
1043-test green under `-W error`.

REAL VERIFIED (live observed): index-0 camera opens via MSMF but produces only
black/frozen frames (lum 0.0/diff 0.0; gate 4.12) — the feed is correctly
rejected by the Phase 51 feed gate; indices 1–5 do not open; no RTSP/video;
Streamlit serves HTTP 200; DB read-wire intact for the above.

## 29. Confirmed Broken

No product-code defect was reproduced that causes wrong output on valid input
(in the suite, on the DB, or via the broken feed — the feed failure is
environmental, not a code bug). Issues classified near-BROKEN:

| Problem | Location | Reproduction | Impact | Sev |
| --- | --- | --- | --- | --- |
| Docker dashboard auth default OFF (open) | docker-compose.yml:126/103/117 | `docker compose up` | Unauthenticated presence exposure | CRITICAL |
| Swallowed live-state write errors | main.py:380-381 | any disk/json failure | Silent stale dashboard | HIGH |
| OSNet silent downgrade on load failure | reid.py:127-153 | corrupt/missing ckpt at load | Unnoticed identity-quality drop | HIGH |
| Model auto-download if weights absent | detector.py:287/face_registry | missing yolo11n/ buffalo_l | Boot failure/flood | HIGH |
| Anomaly repeated-pattern re-fire (no cooldown) | anomaly_engine/security_engine | real weekend AE loop | SOC queue noise | MED |
| CAMERA_RECOVERED incidents never auto-resolved (left OPEN) | incidents.py | observed DB | Polluted queue | MED |
| Report/EOD continuity mismatch (09-01/08-31) | reporter/eod | DB | Report gap | MED |

## 30. Incomplete Features

Partially implemented: evidence capture (never ON); detection registry family
statuses (future-only); motion/entry-exit tables (wired, empty); anomaly
dedup; incident auto-resolution; biometric retention; invalid-feed UI banner;
live performance measurement.
Implemented but not connected: OSNet (prod off), `ENTRY_EXIT_ENABLED`,
`MOTION_ENABLED` semantics at runtime, several dead config knobs.
Connected but not validated: phone accuracy, side/back ReID, thresholds,
two-person live, multi-camera live, SMTP delivery, Docker run, GPU.
Planned only: none documented beyond future phases.
Externally blocked: real camera/RTSP, video, labelled datasets, SMTP, GPU,
rostered on-site employee.

## 31. False-Confidence Risks

1. "753 tests" in README (understates suite) yet more above-grade confidence in
   production-readiness wording.
2. Calibration tool + gates + dry-run JSON exist ⇒ looks like calibration is
   "done" — no real crop set.
3. OSNet models present ⇒ looks production-usable — production default is
   built-in and uncalibrated.
4. Phase 49 benchmark numbers ⇒ could be read as real accuracy — SIMULATED.
5. Camera index 0 "opens" at FPS 30 ⇒ an operator doing an open-check could
   believe video is live — every frame is black (a real false-conf approach).
6. Dashboard security page shows 91 events / 5 incidents ⇒ looks like an
   operating system — all from a frozen/offline era, single broken feed.
7. Employees show enrolled/active ⇒ roster ready — never recognized live.
8. Phase 52 "checkpoint unavailable" ⇒ contradicts on-disk truth.
9. Docker open-dashboard default ⇒ the most likely real leak.
10. Swallowed write errors ⇒ operator believes status is being recorded.

## 32. Completion Analysis

Denominators are defensible only at the feature level (44-row matrix §27).
Not mathematically defensible for "project completion" as a single number.

- Implementation completeness: ~42/44 matrix features have working code +
  tests ≈ 95% (code-existence metric; explicitly not quality).
- Automated-test verification: 1043/1048 collected tests pass (99.5%); the 5
  skipped are real-hardware gates.
- Real-world validation completeness: 1/22 features that genuinely require
  live input (the broken-feed rejection) ≈ 5%. Everything else NOT VALIDATED /
  ENV BLOCKED / NOT EXECUTED.
- Deployment readiness: PARTIAL — Docker buildable but auth-default broken;
  air-gap download issue; no GPU/real SMTP/exercised run.

## 33. Production Readiness

| Category | Status | Evidence |
| --- | --- | --- |
| A. Functional completeness | PARTIAL | full corpus; dead knobs; evidence/alert dormant |
| B. Functional correctness | PARTIAL (CODE VERIFIED) | 1043 green; invariants hold in code |
| C. AI/model reliability | NOT VALIDATED | no real-world accuracy data |
| D. Real-world validation | BLOCKED | black/frozen feed only |
| E. Performance | NOT VALIDATED | CPU-only, sim benchmarks only |
| F. Security | PARTIAL | in-app good; Docker default broken; TLS absent |
| G. Data integrity | PARTIAL | WAL+backup; stale incidents/anomaly re-fire |
| H. Deployment | PARTIAL | buildable; auth default; air-gap download |
| I. Monitoring/ops | PARTIAL | preflight/tools exist; swallow errors; stale UI |
| J. Documentation | PARTIAL | stale counts/names; missing reports; contradicting claims |

## 34. Master Issue Register

| ID | Sev | Area | Issue | Evidence | Impact | Repro? | Ext blocker | Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M01 | P0 | Deploy/Security | Docker dashboard auth default OFF, binds 0.0.0.0:8501 | compose:126/103/117 vs config:600 | Exposure of live presence data | yes | no | Flip default; require pw; forward FAIL_CLOSED; TLS doc |
| M02 | P1 | Runtime | Live-state writer swallows all exceptions | main.py:380-381 | Silent stale dashboard/health | yes | no | log+degrade |
| M03 | P1 | Models | Weights auto-download if absent (yolo/buffalo_l) | detector.py:287; face_registry | Air-gap boot failure; alert flood | yes | network | Vendor / preflight model check |
| M04 | P1 | ReID | Silent OSNet→builtin downgrade on load fail | reid.py:127-153 | Unnoticed quality drop | yes | no | Make failure visible |
| M05 | P2 | Incidents | Advisory incidents left OPEN; anomaly re-fires | DB rows | SOC noise | yes | no | Auto-resolve advisory; cooldown |
| M06 | P2 | Privacy | Biometric caches no retention | config.py:203,351 | PII persists | yes | no | retention path |
| M07 | P2 | Reporting | Report/EOD continuity mismatch | eod vs xlsx | Gaps | yes | no | reconcile |
| M08 | P2 | UI | Dead settings shown in Settings; no invalid-feed banner | config.py:943-950; app | Misleading ops | yes | no | wire/hide; banner |
| M09 | P2 | Deploy | Compose env passthrough gaps (SOURCE, FAIL_CLOSED, EVIDENCE…) | compose:121-133 | Local≠container | yes | no | align |
| M10 | P2 | Perf | Phone full-frame 1280 scan cadence 3 (CPU) | config | realtime risk | yes | MODEL LIM | ROI/cadence tuning |
| M11 | P3 | Docs | README "753" ×5; matrix missing 40–48 | README vs run | Misleading | yes | no | update; CI test count |
| M12 | P3 | Docs | PHASE52 checkpoint-unavailable claim wrong | PHASE52 vs models/ | Credibility | yes | no | correct |
| M13 | P3 | Docs | Missing PHASE47-51/protocols files; stale roster names | git status, VALIDATION_REPORT | Navigability | yes | no | restore/update |
| M14 | P3 | Config | Dead settings (SOURCE, PILOT_*, etc.) | config+grep | Confusion | yes | no | wire or remove |
| M15 | P3 | Code | `main.py:956-958` unactionable "no cameras" error | main.py | Boot confusion | yes | no | early actionable check |
| M16 | P3 | Code | `reid.py:177-178` per-crop traceback logging | reid.py | Log volume | yes | no | rate-limit |
| M17 | LOW | Code | Heuristic email retry classification | notifier.py:230-241 | Minor waste | no | no | explicit non-retryable |
| M18 | LOW | House | Stray PS1 + root test_droidcam/main_backup | glob | Clutter | yes | no | remove post-audit |
| M19 | INFO | Tests | Some old assertions weak ("no exception" etc.) | sampled phase33-40 | Weak signal | yes | no | strengthen |
| M20 | INFO | Security | No TLS default; shared viewer password | app.py:12,356 | Sniff risk | env | proxy | document/limit |

## 35. Remediation Plan (NOT to be executed now)

P0:
- M01: edit `docker-compose.yml` — `CCTV_DASH_AUTH:-1`, forward
  `CCTV_DASH_FAIL_CLOSED`, require non-empty admin password; add TLS/reverse-
  proxy guidance. Tests: `test_dashboard_auth` + new compose-env test.
  Validate by `docker compose config` + auth smoke.

P1:
- M02: replace swallowed except in live-state writer with `logger.exception` +
  degrade; test with injected json error.
- M03: vendor weights or add `--validate-models` preflight fail; test missing-
  model path; validate by preflight on clean host.
- M04: make ReID fallback emit status (warning counter/state flag) and surface
  in dashboard; test corrupt-checkpoint path.

P2:
- M05: auto-resolve advisory incidents; add anomaly dedup/cooldown (test DB
  rows); M06 retention path for embeddings/appearance (test); M07 report/EOD
  reconcile; M08 hide/rewire dead settings + invalid-feed banner; M09 align
  compose env passthrough; M10 ROI scan for phone + cadence tuning.

P3:
- M11/M12/M13/M14/M15/M16/M17/M18/M19/M20 — documentation corrections, dead
  code removal after approval, assertion strengthening, TLS notes.

Dependencies: M03 (models/repo), M01/M09 need compose run access; M05-M07 need
DB access; M10/M12 need operator decision; real validation steps additionally
need usable feed + employee + datasets + SMTP.

## 36. Exact Next Phase

**Phase 54 = DEFENSIVE REMEDIATION (audit-derived, not feature-driven).**

Fix in this order: P0 dashboard-auth default (M01) → P1 error-visibility and
model-availability hardening (M02, M03, M04) → P2 SOC hygiene and privacy
(M05, M06) → then re-run the FULL audit + full test suite to confirm closure.
Only after remediation, begin a controlled validation-preparation phase
(preflight→RTSP harness→feed-gate→real calibration with a rostered employee).

Do NOT add new features before M01–M06 are closed. Do NOT enable OSNet or
change thresholds during remediation.

## 37. Final Evidence Classification

## PHASE 54 CLOSURE ADDENDUM (same day, after remediation)

Issues closed by the Phase 54 remediation pass (all verified by the
`tests/test_phase54.py` regression suite + full-suite rerun):

| ID | Sev | Action taken | Evidence |
| --- | --- | --- | --- |
| M01 | P0 | compose dashboard default now `CCTV_DASH_AUTH:-1` + forwards `CCTV_DASH_FAIL_CLOSED:-1`, 30-min session, optional passwords | docker-compose.yml resolved by `docker compose config` (auth=1, fail-closed=1); tests TestM01 |
| M02 | P1 | live-state writer logs first failure (traceback) + rate-limited repeats + recovery warning; never raises | `main.py:330-415`; TestM02 (fail→log, recover→log, rate-limit, snapshot preserved) |
| M03 | P1 | preflight model remedy corrected to yolo11n + ONNX provider check added (weights present on disk) | src/preflight.py; preflight coverage |
| M04 | P1 | ReID fallback records `requested_model` + `load_error` (warn, NOT silent) | src/reid.py:117-177; TestM04 |
| M05 | P2 | INFO-advisory incidents auto-close on CAMERA_RECOVERED (HIGH/CRITICAL never auto-closed); anomaly REPEATED_PATTERN cooldown dedup (24 h, count-growth re-fires) | incidents.py:137-175, security_engine.py, anomaly_engine.py + `latest_anomaly_for`; TestM05 (7 tests) |
| M06 | P2 | opt-in biometric-cache retention `CCTV_RETENTION_BIOMETRICS_DAYS` (enrolment images never touched) | config.py + main.py `_maybe_run_retention` + compose env; TestM06 (4 tests) |
| M07 | P2 | report/EOD reconcile: phantom `report_generated` surfaced, on-disk xlsx without state flagged; generation-left-unmarked on missing file | main.py:775-841; TestM07 (3 tests) |

Docs corrected (M11/M12/M13): README test count → **1075**, `-W error` clean;
Phases 40–48 + 52–54 matrix rows added; PHASE52 OSNet checkpoint claim
corrected via an inline Erratum (`models/osnet_x1_0_msmt17.pth` **IS present**,
16.47 MB); PHASE50 protocol file marked tree-absent; VALIDATION_REPORT roster
names corrected (EMP002 Sourabh, EMP003 Piyush).

Re-validation after remediation (this session): `pytest -q -W error` →
**1075 passed, 0 skipped, ~2.2 min** (incl. 27 new Phase 54 tests);
`docker compose config` clean and renders the secure defaults.

Still BLOCKED (unchanged): usable camera/RTSP feed, live employee session,
labelled phone dataset, real calibration crops, SMTP delivery, Docker run,
GPU. No real-world accuracy claims were made.

---

## REAL VERIFIED
- Camera index 0 opens via MSMF (640×480, nominal 30 FPS) and produces only
  black/frozen frames (lum 0.0, diff 0.0; 4.12/255 on the 10 s gate capture) —
  measured live this session, twice.
- Indices 1–5 do not open on any backend; no RTSP endpoint; no video file —
  measured live.
- Phase 51 feed gate returns `NOT EXECUTED / INVALID FEED (BLACK/DARK)` on the
  real capture.
- Streamlit service responds 200 on `/` and `/_stcore/health` — live.
- `sessions.db` contents read (employees/events/incidents/anomalies) — real
  historical rows, not generated.

## CODE VERIFIED
- 1043-test suite green under both `pytest -q` and `pytest -q -W error`
  (two fresh runs each; 5 hardware skips).
- Camera-offline≠AWAY, trusted-identity protection, phone one-alert-episode,
  candidate/confirmed face voting, Unknown handling, evidence SHA-256,
  credential redaction, RBAC fail-closed ordering.

## SIMULATED
- Phase 46 research/bench and Phase 49 OSNet bench JSON and calibration
  dry-runs — synthetic crops, no real camera. Do not treat as accuracy.

## PARTIALLY VERIFIED
- Camera-health subsystem (good-feed path never observed; bad-feed rejection
  real). Security/incident engine (code + tests + real historical rows).
  Dashboard (HTTP live, UI logic test-stubbed).

## NOT VALIDATED
- All real accuracy: YOLO person/phone metrics, InsightFace recognition,
  OSNet/ReID thresholds and margins, side/back identity, multi-person
  identity, productivity correctness, phone precision/recall, live reporting,
  live event quality.

## NOT EXECUTED
- Real calibration (no crops), live phone scenarios A–D, two-person episodes,
  AWAY-safety live, evidence capture (0 files), SMTP alert delivery, Docker
  run, GPU path, real EOD report, browser smoke.

## BROKEN
- No product-code defect producing wrong output on valid input was reproduced.
  Documented near-BROKEN: docker auth default (CRITICAL), swallowed
  live-state errors, silent OSNet fallback, weight auto-download, anomaly
  re-fire, incidents never auto-resolved, report/EOD mismatch.

## INCOMPLETE
- Evidence capture enablement, anomaly dedup/cooldown, incident
  auto-resolution, biometric retention, dead-settings cleanup, README/docs
  freshness, invalid-feed UI banner.

## ENVIRONMENT BLOCKED
- Usable camera/RTSP/video, labelled phone dataset, real calibration crops,
  working SMTP, GPU, rostered on-site employee.

## MODEL LIMITATION
- Small-phone detection, side/back ReID ambiguity, CPU inference latency for
  multi-camera real-time (documented, not defects).

## UNKNOWN
- Exact real-world accuracy of every model; live performance under load; true
  behaviour with a healthy feed; browser-rendered UI behaviour (no automation
  available).

---

### Git Safety Result

No `git add/commit/push/reset/clean/checkout` executed. Working tree untouched
by this audit. Final status (also embedded below as required):

```
 M .env.example
 D FIX_REPORT_2026-09-09.md
 D GAP_ANALYSIS_PHASE_AB.md
 D PRODUCTION_READINESS.md
 D PROJECT_FULL_SYSTEM_AUDIT.md
 M README.md
 M app.py
 M config.py
 M main.py
 M src/alerts.py
 M src/detector.py
 M src/face_registry.py
 M src/preflight.py
 M src/tracker.py
 M tests/test_dashboard_app.py
 M tests/test_multiperson_pipeline.py
 M tests/test_phase38.py
 M tests/test_phase39.py
 M tests/test_phase40.py
 M tests/test_tracker.py
?? .streamlit/
?? AUDIT_REPORT_2026-09-10.md
?? PHASE52_REAL_CAMERA_VALIDATION_REPORT_2026-09-19.md
?? PROJECT_FULL_SYSTEM_AUDIT_2026-09-19.md
?? models/
?? "s -ExecutionPolicy RemoteSigned) ; (& \357\200\242f\357\200\272synilogiccctv monitoring.venvScriptsActivate.ps1\357\200\242)"
?? src/calibration.py
?? src/phase46_benchmark.py
?? src/phase46_research.py
?? src/phase49_benchmark.py
?? src/phone_detector.py
?? src/reid.py
?? src/reid_osnet.py
?? tests/test_dashboard_auth.py
?? tests/test_dashboard_phase42.py
?? tests/test_phase41.py
?? tests/test_phase43.py
?? tests/test_phase44.py
?? tests/test_phase44a.py
?? tests/test_phase44b.py
?? tests/test_phase45.py
?? tests/test_phase46.py
?? tests/test_phase49.py
?? tests/test_phase50.py
?? tests/test_phase51.py
```

This report is the single authoritative description of the actual project
condition as of 2026-09-19. Nothing was fixed, changed, or committed. STOP —
no remediation phase has been started.