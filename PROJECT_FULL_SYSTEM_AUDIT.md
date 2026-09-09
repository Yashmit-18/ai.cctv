# PROJECT FULL SYSTEM AUDIT
## AI Office CCTV Intelligence & Security Platform

**Project root:** `cctv monitoring\`
**Audit date:** 2026-09-04
**Method:** Independent, evidence-based audit. No project source was modified during the audit. Every claim below was verified against source code (file:line), configuration, tests, or runtime behavior — not against prior phase reports (README/CHANGELOG were cross-checked only for documentation-consistency findings).

**Audit principle applied:** *code exists ≠ feature works; test passes ≠ real-world accuracy; simulation ≠ real CCTV deployment; AI model available ≠ AI capability validated.*

---

## 1. EXECUTIVE SUMMARY

The project is a mature, deeply layered Streamlit dashboard + CLI daemon SQLite system that fuses real **face recognition + person/phone object detection** with an extensive rule-based **security/incident/evidence/alert/RBAC layer**, plus a deterministic **simulation** harness for the non-AI pipeline.

**What is genuinely real and verified in this environment:**
- Two real, integrated AI models load and run **on CPU**: YOLOv8n (`models/yolov8n.pt`) for person + cell-phone detection; InsightFace `buffalo_l` for face detection + recognition.
- The **real webcam path ran end-to-end** on this machine (verified via `data/live_state.json` and a live camera probe): camera ONLINE ~3 fps, employee `EMP001` + `Unknown` tracked, ON_PHONE/AWAY telemetry recorded.
- Productively: **camera downtime does NOT create fake AWAY** in the canonical score (`src/productivity.py:149-154`); **security events do NOT contaminate productivity** (decoupled tables/engines).
- The dashboard is genuinely connected to backend/DB for live state, camera health, security/incident/evidence/alerts; its 9 tabs render without exception; camera UI leaks no RTSP credentials.

**What is NOT real in this environment (honestly reported):** no real RTSP/NVR camera, no GPU/CUDA, no SMTP delivery, no running Docker daemon. The preflight tool reports all of these honestly as WARN / `NOT EXECUTED — ENVIRONMENT LIMITATION`. **No automated test loads the real YOLO or InsightFace model** — models are mocked in tests; real-model evidence is manual webcam smoke only.

**Verdict:** **PILOT ONLY → GO WITH CONDITIONS.** The system can be used TODAY as a **supervised, advisory-pilot, webcam/CPU productivity + presence + rule-based-security tool**. It is **NOT** ready for unattended production or as a "24/7 physical CCTV AI" without real RTSP/GPU/SMTP/Docker validation, and it currently has several genuine defects (detail below).

**Headline verified defects (see Bug Register):**
1. `app.py:1041` calls `IncidentRiskScorer(...).list()` which does **not exist** (class has `top()`/`get_for()`) → Phase 33 risk panel always crashes (caught → warning).
2. `app.py:946` "Acknowledge" uses `AccessGuard.acknowledge_incident()` which sets only `acknowledged_at` and **never sets `status='ACKNOWLEDGED'`** → acknowledged incidents remain OPEN. (RBAC release path `AccessGuard.resolve/dismiss_incident` is likewise status-less but the dashboard does not use those for resolve/dismiss.)
3. **Email alert channel is not implemented** (`src/alerts.py:262-265` returns `False`), yet default channels and SMTP config exist → any `email` rule produces FAILED alerts.
4. **Dashboard auth is OFF by default** (`.env` sets no `CCTV_DASH_PASS`/`CCTV_DASH_VIEWER_PASS`; `config.py:322-323` default empty) → open dashboard, everyone = admin.
5. **`LEFT`/departure event is never emitted** (`src/security_engine.py:295-307` emits only `ARRIVED`), despite `LEFT` being a defined event type.
6. Evidence capture is gated to **HIGH+ severity only** (`src/security_engine.py:391`) even in `EVENT_ONLY` mode → MEDIUM/LOW events never get evidence.

---

## 2. OVERALL COMPLETION ASSESSMENT

| Area | Code complete | Backend-integrated | Tested (pytest) | Real-world validated | Notes |
|---|---|---|---|---|---|
| Productivity core | 95% | Yes | Yes | Webcam-validated (real run evidence) | `grace`/`break` knobs unused |
| Employee recognition | 85% | Yes | Logic-tested (mocked) | Webcam-run evidence (EMP001/Unknown) | No automated real-model regression |
| Person/phone detection | 90% | Yes | Logic-tested (mocked) | Webcam-run evidence | Only COCO person + cell phone queried |
| Tracker (identity FSM) | 90% | Yes | Yes | Webcam-run evidence | No spatial multi-object tracker (no DeepSORT/ByteTrack) |
| Security rules (rule-based) | 90% | Yes | Yes | Software/sim only | Not AI; LEFT/departure missing |
| Incident/correlation/risk | 80% | Yes | Yes | Software/sim only | Acknowledge status bug; risk-panel crash |
| Evidence | 70% | Yes | Yes | Software/sim only | HIGH+ only; no UI verify/export; no encryption |
| Alerts | 60% | Yes | Yes | Software/sim only | Email channel broken |
| Website/dashboard | 85% | Yes | Render + secrets tested | Webcam-run data reflects | Employee mutations lack RBAC/audit; no audit viewer |
| Backup/restore/retention | 85% | Yes | Yes | Backup drill passes | Retention opt-in; no FK integrity |
| Simulation harness | 95% | Yes (synthetic) | Yes (sim/stress/soak) | Simulation only (NOT real cameras) | Bypasses AI models |

---

## 3. ARCHITECTURE AUDIT

The documented pipeline is **mostly true**, with caveats:

```
Camera (webcam HERE / RTSP in target env)
  → Frame capture (src/camera.py; MultiCameraManager round-robin batch, src/camera_manager.py)
  → Detection (YOLOv8n person+phone; src/detector.py:245,288)
  → Recognition (InsightFace buffalo_l embeddings; src/detector.py:296, src/face_registry.py)
  → Tracking (identity FSM, NO spatial tracker; src/tracker.py:EmployeeTracker/MultiTracker)
  → Productivity (src/productivity.py — receives pre-classified states)
  → Security (rule-based; src/security_engine.py)
  → Incident (src/incidents.py)
  → Correlation (src/correlation.py)
  → Investigation (src/investigation.py, read-only)
  → Evidence (src/evidence.py; HIGH+ only)
  → Alerts (src/alerts.py; email broken)
  → Database (SQLite WAL; src/database.py)
  → Dashboard (app.py; reads live_state.json + DB)
```

**Stage verification:**

| Stage | Implementation | Entry point | Output → consumer | Error handling | Tests | Status |
|---|---|---|---|---|---|---|
| Capture | `src/camera.py`, `MultiCameraManager` | `main.py` loop | frames → batch | reconnect/stale isolation | `test_camera_failure.py` | RTSP NOT validated here |
| Detection | YOLOv8n (person+phone) | `ActivityDetector.detect_batch` | detections → MultiTracker | degraded no-detector retry | geometry-only (mocked) | REAL webcam smoke only |
| Recognition | InsightFace buffalo_l | `ActivityDetector` | emp_id → tracker | model-load fallback | mocked | REAL webcam smoke only |
| Tracking | EmployeeTracker FSM | `MultiTracker.process_batch` | intervals → DB | smoothing/patience | yes | software + webcam |
| Productivity | `src/productivity.py` | report/analytics | daily metrics | lunch/away handling | yes | verified formula |
| Security | `src/security_engine.py` (rules) | daemon tick | events/incidents | failure-isolated | yes | simulation |
| Incident | `src/incidents.py` | security engine | incidents | dedup | yes | simulation |
| Evidence | `src/evidence.py` | security engine (HIGH+) | files + DB | path-safety | yes | sim (OFF default) |
| Alerts | `src/alerts.py` | security engine | alerts table | retry | yes | sim (email broken) |
| Dashboard | `app.py` | streamlit | — | try/except everywhere | render + secrets | reads real live_state + DB |

**Broken/disconnected pipeline notes:**
- Acknowledge-ack station does not update incident `status` (see Bug 2).
- Risk panel crashes (see Bug 1).
- `LOG`/`departure` clearly connected but missing LEFT emission.
- Email feed end-of-day and alert email both non-functional (SMTP not implemented).
- No spatial tracking → a person who steps out of face-frame is re-identified only by face re-detection; no visual continuity across frames without face.

---

## 4. COMPONENT-BY-COMPONENT AUDIT

*(Summary; details in the dedicated phases below.)*

- **Auth/app**: `app.py`, `main.py` — daemon + dashboard; solid structure, graceful shutdown, atomic live-state writes.
- **AI models**: 2 real (YOLOv8n, InsightFace buffalo_l). Only 1 model file in `models/`; insightface auto-downloaded to user cache.
- **Camera**: `camera.py`, `camera_manager.py`, `camera_state.py` — robust FSM (offline/recovery/reconnect/stale/frozen). Real camera = webcam here.
- **Productivity**: `productivity.py` — safe denominator; camera downtime ≠ fake AWAY.
- **Security/incident/correlation**: extensive rule engine; no AI-beyond-person/face.
- **Evidence/alerts**: functional for log/dashboard; email missing; evidence OFF by default.
- **RBAC/audit**: `rbac.py`/`auditlog.py` — real enforcement on incident/evidence actions; **employee mutations bypass RBAC** in dashboard.
- **Preflight/harness**: `preflight.py`/`rtsp_harness.py` — honest tooling (verified live: reports dashboard-open, evidence-off, etc.).

---

## 5. AI MODEL INVENTORY (verified)

| Model | Path | Framework | Purpose | Classes | Loaded | Integrated | Called | Tests (model) | Real-validation | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv8n | `models/yolov8n.pt` (~6.2 MB) | Ultralytics (PyTorch) | detect person + cell phone | COCO 0 (person), 67 (cell phone) — only these 2 queried (`src/detector.py:23-27`) | Yes (`src/detector.py:163`) | Yes (daemon) | `detect_batch`/`predict` (`:245`/`:288`) | **None load real model** (mocked) | Manual webcam smoke | INTEGRATED, tested only at logic level |
| InsightFace `buffalo_l` | user cache `~/.insightface/models/buffalo_l/*.onnx` (det_10g, w600k_r50 recognition, genderage, landmarks) | insightface + onnxruntime | face detect + 512-D recognition | embedding-based (no classes) | Yes (`src/face_registry.py:101`) | Yes (daemon) | `app.get(frame)`, `identify()` (`:296`,`:305`) | **None load real** (mock `_FakeApp`) | Manual webcam smoke | INTEGRATED, tested only at logic level |

**Rules:** No other model files exist (`models/` has only `yolov8n.pt`). Detector family support in `src/detector.py` is exactly: object detection (person+phone), face detection, face recognition, phone-near-face heuristic. The detector registry (`src/detector_registry.py:29`) treats fall/fight/fire/weapon as `FUTURE_MODEL_REQUIRED` — no such models exist.

---

## 6. AI CAPABILITY MATRIX (what the PIPELINE can actually produce)

| Capability | Model / Logic | Implemented | Tested | Real-world validated | Current usability | Confidence notes / limitations |
|---|---|---|---|---|---|---|
| Person detection | YOLOv8n (COCO 0) | Yes (`detector.py:325`) | logic-only | webcam run | YES | threshold 0.4;
| Employee face recognition | InsightFace ArcFace 512-D | Yes (`face_registry.py:303`) | logic-only (mocked) | webcam (EMP001) | YES — with enrollment | threshold 0.6; needs enrolled `data/faces/*.jpg` |
| Unknown person | face not matched → "Unknown" | Yes | logic | webcam (Unknown) | YES | `security_engine._eval_unknown` |
| Phone/cell-phone detection | YOLOv8n (COCO 67) + proximity heuristic | Yes (`detector.py:111`) | logic | partial (webcam noted low phone recall) | YES — with limitations | phone near face; known sensitivity limitation (README) |
| Multiple people | person boxes per frame | Yes | logic/sim | sim | YES | dedup by identity |
| Person tracking | identity FSM (NO spatial tracker) | Yes | yes | webcam | PARTIAL | no DeepSORT/ByteTrack; face-reliant |
| Cross-camera continuity | identity dedup (`tracker.py:242`) | Yes | sim | sim-only | PARTIAL | no spatial handoff; `CROSS_CAMERA_CORRELATION_UNAVAILABLE` fallback |
| Occupancy / crowd | person-count threshold | Yes | yes | sim | YES | rule-based (`CROWD_THRESHOLD=5`) |
| Zone intrusion | point-in-polygon (`zones.py`) | Yes | yes | sim | YES | requires configured zones |
| Loitering | duration timer | Yes | yes | sim | YES | rule-based |
| After-hours | clock window | Yes | yes | sim | YES | rule-based |
| Camera offline/recovery | frame-staleness FSM | Yes | yes | webcam(unexpected) | YES | rule-based |
| Camera tamper | grayscale stats (frozen/dark/bright) | Yes | yes | sim | YES — limited | heuristic |
| Arrival (ARRIVED) | first-seen | Yes | yes | sim | YES | rule-based |
| Departure (LEFT) | — | **NO** | NO | NO | **NOT PRODUCED** | constant exists, no emission |
| ACTIVE / ON_PHONE / AWAY | tracker FSM | Yes | yes | webcam | YES | productivity semantics |
| Fall / Fight / Fire / Smoke / Weapon / PPE / Mask / Violence / Theft / Emotion / Vehicle / Plate etc. | — | **NO** | NO | NO | **NOT SUPPORTED** | no model/labels; registry marks FUTURE_MODEL_REQUIRED |

---

## 7-9. READY / LIMITED / UNSUPPORTED FEATURES (see Phase 24-27 tables too)

**Ready to use (software-validated, webcam-verified):** Employee recognition (with enrollment), Unknown detection, person detection, productivity ACTIVE/ON_PHONE/AWAY, tracking, camera health/offline/recovery, security rules (unknown/intrusion/after-hours/loitering/occupancy/tamper), incidents + correlation + risk, investigation workbench (read-only), evidence (OFF by default), dashboard, RBAC on incident/evidence actions, backup/restore, retention (opt-in), preflight + RTSP harness.

**Not supported (no model/implementation):** Fall, fire, smoke, fight, violence, weapon, PPE/helmet, mask/face-mask, theft, suspicious-behavior classification, running, sleeping, emotion, food/drinking/smoking, vehicle, plate recognition.

---

## 10. WEBSITE / DASHBOARD AUDIT

Dashboard = `app.py`, **9 tabs**: Live Overview, Live Employees, Historical, Camera Health, Security & Incidents, Reports, Employees, Settings, Deploy & System Check.

| UI Feature | UI | Backend connected | Real data | Mutation | RBAC | Audit | Empty state | Pagination | Findings |
|---|---|---|---|---|---|---|---|---|---|
| Login/auth | Yes | config | env | — | front gate | **No login audit** | — | — | **OFF by default; plaintext; config boundary** |
| Live Overview | Yes | `live_state.json`+DB | Yes | No | n/a | n/a | Yes | No | real telemetry |
| Live Employees | Yes | `live_state.json` | Yes | No | n/a | n/a | Yes | No | real |
| Historical | Yes | DB (`activity_logs`) | Yes | No | n/a | n/a | Yes | No | real |
| Camera Health | Yes | `live_state.json` + `config.CAMERAS` | Yes | No | n/a | n/a | Yes | No | **Configured/Not, no credential leak (verified)** |
| Security & Incidents | Yes | DB (events/incidents) | Yes | resolve/dismiss/ack/note | **Yes (guarded)** | Yes (guard) | Yes | hard limits | **Ack bug; risk-panel crash** |
| Reports | Yes | filesystem + notifier | Yes | download | view (guard?) | via engine | Yes | No | lists real xlsx |
| Employees | Yes | DB (`employees`) | Yes | add/edit/photo | **UI-only** | **No audit** | Yes | No | **RBAC/audit gap; no delete** |
| Settings | Yes | `config.py` | Yes | read-only | — | — | always | No | no secret leak |
| Deploy & System Check | Yes | `src.preflight` | Yes | read-only | — | — | always | No | honest status; static checklist |

**Missing from UI (backend has):** audit-log viewer, alert ack/resolve UI, evidence integrity verify UI, evidence export UI, camera/zone/alert-rule config CRUD, employee deletion, temporal intelligence surfacing, phase-37 RTSP harness button.

**Broken:** `app.py:1041` risk panel (AttributeError). Acknowledge (app.py:946) status-less.

---

## 11. SECURITY AUDIT

- **Auth:** Streamlit session gate; **off by default**; plaintext compare; no hashing/CSRF/rate-limit/session expiry. Documented as config boundary; must sit behind reverse proxy/VPN.
- **RBAC:** Real `AccessGuard` + audited denials for incident/evidence/export/config. **Gap:** employee create/update/photo mutation in dashboard is UI-hidden only — no `AccessGuard`, no audit (app.py:667-689).
- **Secrets:** **No hardcoded real secrets.** Placeholders (`CHANGE_ME`, `replace-with-*`) and public test-stream creds only. RTSP redacted (`domain.redact_url`). Verified no `log.*` leaks secrets.
- **Path traversal:** `save_enrollment_image()` (app.py:199) builds path from an employee ID that is not stripped of `..`/separators — LOW (low practical exploitability via Streamlit typed input), worth sanitizing.
- **Logging/audit:** clean; `access.denied` and lifecycle audited; login not audited.
- **Network/Docker:** dashboard port exposed, no TLS (config boundary). Daemon uses `ipc: host` in compose (reduces isolation). Non-root user in Dockerfile; health checks both services; **no `.dockerignore`** (risk of baking `.env` into image → MEDIUM).
- **Biometric:** `data/embeddings.pkl` + `data/faces/*.jpg` stored **unencrypted at rest**, filesystem-access-only (MEDIUM). No emotion/personality/per-person-risk/auto-accusation (policy respected & verified).

---

## 12. PRIVACY / BIOMETRIC AUDIT

- Face embeddings in `data/embeddings.pkl` (~4.8 KB, versioned cache) — unencrypted, no access control beyond FS. `data/` is gitignored.
- Face images in `data/faces/` — unencrypted.
- Employee data: minimal work metadata (no SSN). No emotion analysis, personality scoring, per-employee risk, or auto-misconduct accusation anywhere (verified).
- Risk scores apply to **incidents only** (explicit policy, `risk_scoring.py:10-14`).
- Retention **off by default** → data persists indefinitely unless operator enables it (GDPR concern).
- Deletion workflow for employee + embeddings not surfaced in UI.

---

## 13. DATABASE AUDIT

- 19 tables, WAL + busy_timeout + synchronous=NORMAL (WAL-safe backup claims hold).
- **No foreign-key constraints at all** (MEDIUM) → possible orphans (events/evidence/alerts referencing removed incidents).
- **Timezone inconsistency** (MEDIUM): `activity_logs.timestamp` & tracker use **local** time (`tracker.py:50`); `security_events.created_at` uses UTC (`datetime('now')`); `audit_log` uses **localtime** (`database.py:920`) → cross-table timestamps not comparable.
- Migrations: additive `PRAGMA table_info` pattern; re-entrant; no rollback (acceptable).
- Unbounded/concern queries: `list_employees` no limit; `AlertEngine.health()` loads up to 100k alerts; evidence orphan/list loads all.
- `upsert_employee` uses two non-atomic UPDATEs (crash can leave `enrolled` stale) — LOW/MEDIUM.
- Retention does not cascade to linked security/evidence rows; evidence retention deletes **whole day directories** (can delete evidence of still-open incidents) — MEDIUM.

---

## 14. EVIDENCE AUDIT

- Event-driven only, **default OFF**. Capture gated: **HIGH+ severity only** (`security_engine.py:391`) AND `EVIDENCE_MODE` (`evidence.py:111-114`) — so `EVENT_ONLY` still only captures HIGH+ (MEDIUM discrepancy).
- SHA-256 hashing + path-safety (verified) + metadata + per-file access audit (`evidence_access.py`).
- **Not encrypted at rest.** Export exists but no UI. Integrity verify exists but no UI. Orphan cleanup only clears DB rows for missing files (not reverse).
- Frames are JPEG stills (pre/post buffer), **not continuous recording** (deliberate).

---

## 15. ALERT AUDIT

- Log + dashboard channels **work**. **Email channel NOT implemented** (`alerts.py:262-265`) → email rules = FAILED alerts (HIGH).
- Retry (≤3), escalation-stale (status-only, no external notify), ack/resolve in DB.
- Alert-storm: per-rule cooldown (in-memory, lost on restart), incident-level dedup (3600s), loitering/occupancy re-fire suppression. **No global rate cap.**

---

## 16. CAMERA AUDIT

- Webcam: **IMPLEMENTED + REAL-VALIDATED** (live run evidence ~3 fps, ONLINE).
- RTSP/MultiCamera: **IMPLEMENTED + INTEGRATED** (manager, health FSM, reconnect, stale, reconnect) but **NOT real-validated** (no RTSP endpoint here).
- VirtualCamera: **SIMULATED only**, drives real downstream pipeline — NOT a real camera.
- Batching (round-robin, MAX_BATCH_SIZE), batching tested with fakes; real multi-camera/GPU batching NOT tested.

---

## 17. PERFORMANCE AUDIT (measured, not fabricated)

- No GPU/CUDA → **GPU/VRAM metrics not measurable** (NOT EXECUTED).
- Webcam real run: ~3 fps single camera CPU; global FPS 0.8 in the stale `live_state.json` (daemon throttled). This is a genuine single-camera CPU figure.
- No sustained multi-camera (1/2/5/10) physical benchmark, no wall-clock soak, no p50/p95/p99 production sampling → **NOT EXECUTED** (simulation latencies exist but are synthetic).
- Sim/stress tests assert latencies *under bound* in pytest (software level).

---

## 18. TEST AUDIT

**Live results (this environment):**
- Default: **585 passed, 0 failed, 0 skipped-of-interest** (≈29 s).
- Simulation marker: 51 passed; Stress marker: 6 passed.
- `-W error`: 583 passed, **2 failed** (`test_simulation_isolation.py:test_production_main_does_not_import_simulation`, `test_config_does_not_import_simulation`) — failures are an **unraisable ResourceWarning** ("Exception ignored in FileIO config.py") during import isolation, not a product defect. So the README's "0 warnings" is true only under the default filter; latent resource warnings exist (**LOW**).

**Quality assessment:**
- Strong unit + rule-logic coverage; good security/RBAC/database/evidence/backup tests.
- **Weakness:** EVERY AI-model test mocks the model — **no test loads the real YOLO/InsightFace**; real-model confidence is manual webcam smoke only.
- Missing: real RTSP integration, GPU/CUDA, SMTP delivery, Docker build, DB-corruption recovery, concurrent write conflicts, `embeddings.pkl` corruption, missing model file at runtime.
- Website tests cover render + secret-leak only; not full tab interactions/login/roles.
- Some overlapping secret tests across test_phase35/37/38/dashboard (minor redundancy).

---

## 19. DEPLOYMENT AUDIT

- `Dockerfile` (python:3.10-slim or CUDA 11.8, non-root `appuser`, VOLUME /app/data) + `docker-compose.yml` (ai_daemon GPU + web_dashboard CPU, health checks, persistent `./data`, stop_grace_period).
- **Deployment blockers verified:** (1) CUDA unavailable here; (2) Docker daemon not running here; (3) insightface needs MSVC Build Tools on Windows; (4) Python 3.10-3.12 required; (5) **no `.dockerignore`** → build context may bake `.env`/`.venv`/`data` into image (MEDIUM); (6) `onnxruntime-gpu` on CPU-only host conflicts without the helper fallback; (7) SMTP not wired for alert/report email.
- A fresh machine **can** deploy for webcam/CPU operation via `setup_env.ps1`; it **cannot** reach full production without real RTSP/GPU/SMTP/Docker validated.

---

## 20. BUG / ISSUE REGISTER (verified)

| # | Severity | Component | Evidence | Impact | Recommended action | Complexity |
|---|---|---|---|---|---|---|
| B1 | P1-HIGH | Dashboard risk panel | `app.py:1041` calls nonexistent `IncidentRiskScorer.list()` (class has `top()/get_for()`) | Phase 33 risk table never renders (silently fails→warning) | Change to `list_incident_risk()`/`.top()`; add render test | Low |
| B2 | P1-HIGH | Incident ack workflow | `app.py:946`→`AccessGuard.acknowledge_incident()` sets only `acknowledged_at` (`rbac.py:66-71`); `status` untouched | "Acknowledge" does not change incident status to ACKNOWLEDGED | Route ack through `IncidentEngine.acknowledge()` or set `status`; unify RBAC/engine paths; test status | Medium |
| B3 | P1-HIGH | Alerts | `alerts.py:262-265` email channel returns False; no SMTP send | email alert rules always FAILED | Implement email adapter or drop email channel from defaults | Medium |
| B4 | P1-HIGH | Auth | config default `DASH_PASS`/`VIEWER_PASS` empty; `.env` unset | Dashboard open, all users=admin | Require strong passwords + document reverse-proxy; default fail-closed | Low |
| B5 | P1-HIGH | Employee mutations | `app.py:667-689` add/edit/photo: UI-only gate, no `AccessGuard`, no audit | Backend not authoritative; changes un-audited | Guard + audit employee mutations | Low |
| B6 | P2-MED | Security/departure | `security_engine.py:295-307` emits ARRIVED only; LEFT defined never fired | No departure record | Implement LEFT on last-seen expiry | Medium |
| B7 | P2-MED | Evidence | `security_engine.py:391` HIGH+ gate even in EVENT_ONLY | Low/med events never get evidence | Decouple severity gate from EVENT_ONLY semantics | Medium |
| B8 | P2-MED | DB | timezone mix (local vs UTC) across tables (`tracker.py:50`,`database.py:920`) | Cross-table timestamps wrong/incomparable | Normalize to one tz (UTC) | Medium |
| B9 | P2-MED | DB | no FK constraints | Orphan rows possible | Add FKs/migrations | Medium |
| B10 | P2-MED | Evidence/retention | `evidence.py:232` deletes whole day dirs | Loses evidence incl. open incidents | Per-file retention | Medium |
| B11 | P2-MED | Privacy/biometric | `embeddings.pkl`,`data/faces` unencrypted | Biometric data exposure if FS compromised | Encrypt or document controls | Medium |
| B12 | P2-MED | Docker/deploy | no `.dockerignore` | build may embed `.env`/venv/data | Add `.dockerignore` | Low |
| B13 | P2-MED | RBAC | login not audited; `AccessGuard.resolve/dismiss_incident` status-less paths exist (dead) | weak audit; confusing APIs | Audit logins; fix/remove status-less guard methods | Low |
| B14 | P3-LOW | Path security | `app.py:199` employee-ID path not stripped of `..`/sep | path traversal (low practicality) | sanitize ID in `save_enrollment_image` | Low |
| B15 | P3-LOW | Productivity config | `grace_period_min`,`break_minutes` parsed but unused | knobs no-op | wire or remove | Low |
| B16 | P3-LOW | Orphan code | `src/rtsp_tester.py` unused; `roi_selector.py`,`dummy_face_enroll.py` unreferenced by app/main | dead code / confusion | prune or document | Low |
| B17 | P3-LOW | Test hygiene | resource warnings under `-W error`; overlapping secret tests | warnings latent | fix file closes; dedupe | Low |
| B18 | P3-LOW | Auth timeout | no session expiry | persistent sessions | add expiry | Low |

---

## 21. MISSING FEATURES

**MVP (must-have):** email delivery (alerts + reports — SMTP), departure/LEFT, acknowledge status fix, evidence for non-HIGH in EVENT_ONLY, RBAC+audit on employee mutations, audit-log viewer, auth fail-closed.
**Controlled pilot (should-have):** evidence verify/export UI, alert ack/resolve UI, camera/zone/alert-rule CRUD in UI, employee deletion + "right to be forgotten", real RTSP + GPU + SMTP + Docker validation, `.dockerignore`, timezone normalization, FK integrity.
**Production/enterprise (nice-to-have/next):** spatial multi-object tracker (ByteTrack/DeepSORT), per-camera evidence tuning, global alert rate limiter, notification/escalation channel beyond status, encryption at rest, session/CSRF hardening, metrics/soak dashboards.

---

## 22. REAL ENVIRONMENT VALIDATION (this machine, verified)

| Resource | Available | Tested | Notes |
|---|---|---|---|
| Webcam (index 0, 640x480) | **Yes** | **Yes (live run evidence)** | camera ONLINE ~3 fps; real states recorded |
| CPU inference | **Yes** | **Yes (models load+run on CPU)** | YOLOv8n + InsightFace buffalo_l |
| NVIDIA GPU / CUDA | **No** | No | `torch.cuda.is_available()=False`, 0 devices |
| Real RTSP/NVR camera | No | No | RTSP harness only (not executed on real endpoint) |
| SMTP server | No | No | placeholder creds; email code absent |
| Docker daemon | **No** | No | daemon not running (verified) |
| Production network | No | No | local only |

---

## 23. SDLC POSITION

- Completed: Requirements, Design, Development (core), Testing (unit+sim+stress), Deployment Preparation (Docker/compose/preflight/harness).
- Partially completed: Integration testing (real models), Production Validation.
- Remaining: Deployment (real RTSP/GPU/SMTP/Docker), Production Validation (soak), Maintenance.
- **Current stage:** **DEPLOYMENT PREPARATION → PRODUCTION VALIDATION (gate)**.

---

## 24. PRODUCTION READINESS SCORES (advisory; do not inflate)

| Dimension | /100 |
|---|---|
| Architecture | 90 |
| AI (models integrated) | 80 |
| AI (validated real-world) | 55 |
| Camera (webcam) | 85 |
| Camera (RTSP multi) | 40 |
| Productivity | 88 |
| Security (rules) | 80 |
| Incident management | 78 |
| Investigation | 85 |
| Evidence | 60 |
| Alerts | 55 |
| Analytics | 85 |
| Website/dashboard | 85 |
| RBAC | 75 |
| Privacy | 65 |
| Reliability | 82 |
| Performance | 50 |
| Deployment | 55 |
| Observability | 70 |
| Backup/restore | 85 |
| Testing | 78 |
| **OVERALL SOFTWARE READINESS** | **~80/100** |
| **OVERALL REAL-WORLD READINESS** | **~40/100** |

---

## 25. GO / NO-GO

**Can it be used TODAY?** **YES — for a supervised advisory pilot on the local webcam/CPU path**, as a productivity/presence tracker with rule-based security events, incident management, and investigation. **NOT** for unattended production or for "24/7 physical CCTV AI" until real RTSP/GPU/SMTP/Docker are validated and the P1 defects (B1–B5) are fixed.

**Classification:** **PILOT ONLY → GO WITH CONDITIONS.** Must: fix B1–B5, enable evidence + auth + retention for pilot, and complete on-site RTSP/GPU/SMTP/Docker validation before go-live.

---

## 26. TOP 10 RECOMMENDED NEXT ACTIONS

1. Fix `app.py:1041` risk-panel crash (use `list_incident_risk()`; add a render test).
2. Fix acknowledge (and RBAC release helper) so ack/set status is unified via `IncidentEngine`; add status-asserting tests.
3. Implement or disable the email alert channel; stop defaulting rules to unsupported channels.
4. Default dashboard auth to **fail-closed** (require `CCTV_DASH_PASS`), document reverse-proxy/TLS.
5. Add RBAC + audit to employee create/update/enrollment mutations.
6. Implement `LEFT`/departure emission from last-seen expiry.
7. Normalize timestamps to a single timezone (UTC) across all tables.
8. Add `.dockerignore`; ensure `.env`/`data`/`.venv` never enter the image.
9. Add real-model integration tests (load YOLO+InsightFace on a real frame) so AI regressions are caught automatically, and add environment-gate tests (RTSP/GPU/SMTP/Docker) that run on-site.
10. Surface evidence verify/export, alert ack/resolve, camera/zone/alert-rule CRUD, and an audit-log viewer in the dashboard.

---

## 27. FINAL CONCLUSION

This is a **substantial, well-engineered software system** with a genuinely working webcam/CPU path (verified live) and an extensive, correctly-decoupled security/productivity architecture. Its honesty discipline (no fabricated RTSP/GPU/SMTP/Docker results, `NOT EXECUTED — ENVIRONMENT LIMITATION` reporting, incidents-never-people risk policy) is unusually strong. However, it is **not** the "100% capable multi-AI CCTV" the feature list might imply: only person + face + phone AI exist; advanced AI (fall/fire/fight/weapon/PPE/etc.) is absent; email is broken; several P1 web/RBAC/incident defects exist; and no real client hardware has been validated. Treat it as **pilot-only, advisory, human-in-the-loop** today.

---

# WHAT IS FULLY WORKING (evidence-backed)

- **Employee face recognition + Unknown detection** — real InsightFace; live webcam run produced `EMP001` + `Unknown` states.
- **Person + cell-phone detection** — real YOLOv8n COCO 0/67.
- **Productivity ACTIVE/ON_PHONE/AWAY** — validated formula; camera downtime does not become fake AWAY; security events do not contaminate productivity.
- **Camera health/offline/recovery/stale/frozen** FSM — tested + live webcam ONLINE.
- **Rule-based security:** unknown presence, zone intrusion, after-hours, loitering, occupancy/crowd, tamper (grayscale), camera offline/recovery, arrival (ARRIVED).
- **Incident creation/correlation/risk/dedup, investigation workbench, evidence SHA-256/path-safety (when enabled + HIGH+), alerts (log/dashboard), RBAC with audited denials, audit log, WAL-safe backup/restore, opt-in retention** — genuine and tested.
- **Dashboard live state + DB connectivity**, camera UI **no credential leak**, 9-tab rendering, honest preflight.
- 585 pytest tests pass (default config).

# WHAT IS WORKING WITH LIMITATIONS

- **Email delivery (alerts + report EOD)** — NOT implemented (SMTP absent).
- **Phone-use classification** — documented YOLOv8n sensitivity limitation (often missed on webcam 640x480).
- **Evidence** — HIGH+ only, OFF by default, not encrypted, no UI verify/export.
- **Cross-camera continuity** — identity-based dedup only, not spatial; `CROSS_CAMERA_CORRELATION_UNAVAILABLE` fallback.
- **Tracking** — no spatial multi-object tracker (DeepSORT/ByteTrack); face-reliant.
- **Dashboard auth** — off by default, plaintext, config boundary only.
- **Announced ack/resolve** — RBAC guard helper paths are status-less (live dashboard resolve/dismiss use the correct engine, but acknowledge does not).

# WHAT IS ONLY SIMULATED / TEST-VALIDATED (no real-world validation)

- Multi-camera RTSP pool behavior (VirtualCamera/simulation).
- Cross-camera continuity, occupancy anomaly/short-term scaling, tamper, loitering at scale — simulation/synthetic frames only.
- Stress/soak (sim feeds). Real cameras never drive these.
- Detection/recognition accuracy (models mocked in tests; only manual webcam smoke).

# WHAT IS NOT WORKING (verified defects)

- Phase 33 risk panel (app.py:1041 AttributeError).
- Incident "Acknowledge" (does not set status=ACKNOWLEDGED).
- Email alert channel (returns FAILED).
- Dashboard open-by-default auth (security gap).
- LEFT/departure never emitted.
- Employee mutations not RBAC-guarded nor audit-logged.
- `grace_period_min` / `break_minutes` config knobs are no-ops.

# WHAT IS NOT IMPLEMENTED

- Fall / Fire / Smoke / Fight / Violence / Weapon / PPE / Helmet / Mask / Theft / Suspicious-behaviour / Running / Sleeping / Emotion / Food-Drink-Smoking / Vehicle / Plate detection.
- Email (alert + EOD report) delivery.
- Spatial multi-object tracking.
- Audit-log viewer, alert ack/resolve UI, evidence verify/export UI, camera/zone/alert-rule CRUD UI.
- Employee deletion / biometric "right to be forgotten" workflow.
- `.dockerignore`, session expiry/CSRF, encryption at rest.

# WHAT REQUIRES REAL HARDWARE (or external services)

- **Real RTSP/NVR cameras** (only webcam validated).
- **NVIDIA GPU / CUDA** (not present; CPU only here).
- **SMTP server** (not implemented/no delivery).
- **Docker daemon** (present config, daemon not running here; needs on-site build/run).
- Multi-hour wall-clock soak and sustained production resource sampling.

# WHAT AI CAN ACTUALLY DETECT RIGHT NOW

(listed only with evidence/limitations)
- **Person** — YOLOv8n COCO0; conf 0.4; webcam smoke; no automated accuracy benchmark.
- **Cell phone** — YOLOv8n COCO67 + near-face heuristic; known sensitivity limit.
- **Face** — InsightFace RetinaFace; webcam smoke.
- **Employee identity** — InsightFace ArcFace 512-D vs enrolled embeddings, threshold 0.6; webcam smoke (EMP001).
- **Unknown/person-not-recognized** — fallback of face matching.
- Rule-derived (NOT AI): intrusion, after-hours, loitering, occupancy, tamper, offline/recovery, arrival.

# WHAT AI CANNOT DETECT RIGHT NOW

Fall, fire, smoke, fight, violence, weapon, PPE, one or more masks, theft, suspicious behaviour, running, sleeping, emotion, eating/drinking/smoking, vehicles, license plates, and any fine-grained action beyond the presence/phone/face signals above.

# WHAT I CAN USE TODAY (GREEN)

- Single webcam/CPU productivity + presence tracking (ACTIVE/ON_PHONE/AWAY) with enrolled employees.
- Unknown-person alerts, zone intrusion, after-hours, loitering, occupancy, tamper, camera-offline/health (rule-based, advisory, human-reviewed).
- Incident creation/correlation/risk + investigation workbench + evidence (if enabled, HIGH+) + log/dashboard alerts.
- Dashboard live overview, history, security, reports (xlsx), camera health, preflight checks.
- Backup/restore, opt-in retention, RBAC on incident/evidence/export.
- Preflight + RTSP harness to gate on-site validation.

# WHAT I SHOULD NOT RELY ON YET (RED)

- Email alerts or scheduled email reports (not implemented).
- Acknowledge as a real status transition (bug).
- Risk-scored panel in dashboard (crash).
- Employee add/edit as RBAC+audited (UI-only today).
- Dashboard as a security boundary without reverse-proxy/TLS + strong passwords.
- Phone-use %, multi-camera, GPU performance, or cross-camera continuity as production metrics (not validated on real hardware).
- Any advanced AI (fall/fire/fight/weapon/PPE/etc.) as available.
- Unattended 24/7 operation or "GO" for physical CCTV deployment.

---

# FINAL PROJECT STATUS

- **Software Completion:** ~80%
- **Real-World Completion (webcam/CPU validated):** ~55% (of the full multi-camera+GPU+email vision)
- **Real-World Completion (client production):** ~25%
- **Production Readiness:** 55/100 (advisory; ~80/100 software-only, ~40/100 real-world)
- **Current SDLC Stage:** Deployment Preparation → Production Validation (gate)
- **Overall Verdict:** **PILOT ONLY → GO WITH CONDITIONS** (supervised advisory pilot now on webcam/CPU; fix P1 issues and validate real RTSP/GPU/SMTP/Docker before go-live)
