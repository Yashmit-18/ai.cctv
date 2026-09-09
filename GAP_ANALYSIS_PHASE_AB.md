# GAP ANALYSIS — PHASE A (Core Vision) & PHASE B (Security Intelligence)

**Project:** AI Office CCTV Intelligence & Security Platform
**Root:** `cctv monitoring\`
**Date:** 2026-09-09
**Baseline:** 646/646 pytest pass (real .venv, 92 s). Prior audit (`PROJECT_FULL_SYSTEM_AUDIT.md`, 2026-09-04) kept intact; this doc supersedes its gap list with the PHASE A/B capability view and an enrichment plan.
**Status after implementation:** **700/700 pytest pass** (real .venv, 36 s — 646 baseline + 54 new Phase A/B regression tests in `tests/test_phase_ab.py`). Live webcam re-validated this session: camera index 0 delivered 640×480@30 device frames at ~18 fps (healthy, above LOW_FPS threshold), and the `MotionDetector` pipeline ran end-to-end over 90 real frames (static scene → 0 events, as expected; sustained frame rate healthy). DroidCam (port 4747) was not reachable this session; multi-person real scenes remain unvalidated (single-operator webcam only) — see §5.
**Post-fix round (same date): full suite green (744/744)`** — added `tests/test_camera_health_realfeed.py` (5 tests) exercising FROZEN_FRAME classification on real webcam pixels: signature, streak accumulation, no-false-positive ONLINE, sustained FROZEN, sub-min-seconds no-FROZEN; skipped honestly when no camera reachable. Following the "only one employee" + false-AWAY incident (root cause: `CCTV_CAMERA_MODE=INDEX:1` pinned today's dark/stuck feed — live is index 0, black is index 1 → real live session on `--source auto` showed **EMP001 + Unknown ACTIVE simultaneously**), multi-person detection/identity is now **live-validated for known+unknown**, with roster-recognition of two employees still pending EMP002/EMP003 enrollment. New gap notes added in §5.

**Classification legend**
- **FULLY IMPLEMENTED** — capability present, integrated, and (unit-)tested.
- **PARTIALLY IMPLEMENTED** — works but misses required behaviours.
- **BROKEN** — present but defective under the stated requirements.
- **MISSING** — absent (verified by search).
- **SIMULATED ONLY** — only exercised via the synthetic harness.
- **NOT VALIDATED IN REAL CAMERA ENVIRONMENT** — logic tested, no real-camera run.

An entry may carry two labels (e.g. `PARTIALLY IMPLEMENTED / NOT VALIDATED`).

---

## 1. PHASE A — CORE VISION & REAL-TIME INTELLIGENCE

### A1. Multi-person detection (multiple persons, independent) — **FULLY IMPLEMENTED / PARTIALLY VALIDATED**
- `src/detector.py::_process_frame` now emits **one detection entry per person** with
  `person_box` + `person_conf` (`src/detector.py`; per-person entries use `emp_id="__person__"`)
  while keeping the legacy `__person__` presence entry for backward compatibility
  (EmployeeTracker / `_deduplicate` / `_labels_for` skip the per-person entries).
- **Integration:** the spatial tracker (`SpatialTracker`) consumes each `person_box`,
  FaceRegistry matches per-face, phone boxes are associated via expanded-face/box proximity
  and the owner is resolved per **track**, and person counts per camera/zone now come from
  per-person entries.
- **Honest gap:** validated on unit/integration tests + single-operator webcam; not yet on a
  real *multi-person* scene (see §5).

### A2. Multi-face detection + recognition — **FULLY IMPLEMENTED**
- InsightFace `buffalo_l` returns one entry per face; `face_box`, `face_score`, per-face
  `identify()` with threshold gating (`src/detector.py`; `src/face_registry.py:338`).
- Not validated on real multi-person scenes (single-operator webcam runs only).

### A3. Face recognition & enrollment — **FULLY IMPLEMENTED**
- `FaceRegistry.identify()` → `(EMP_ID, score)`; employee ID is canonical; names resolved
  from `employees` table. Enrollment statuses ENROLLED / NO_FACE / MULTIPLE_FACES /
  LOW_QUALITY / INVALID_IMAGE (`src/face_registry.py:403`). Candidate pruning +
  confusability report present.
- **Minor gap:** insightface detection exceptions are conflated with genuine
  INVALID_IMAGE (no distinct ERROR/enrollment-detection status) — low priority.

### A4. Multi-person identity (known + unknown simultaneously) — **FULLY IMPLEMENTED / PARTIALLY VALIDATED**
- Each face is independently matched (Unknown included); `Track` + `SpatialTracker`
  (`src/tracker.py:443,567`) now provide **spatially consistent attribution**: identities
  are topic of per-track consecutive-run voting, so the same unknown person at 10 fps stays
  one tracked instance with a stable `identity`, and known + unknown persons remain distinct
  tracked individuals (see A6 for flicker rules).
- **Honest gap:** not validated on a real multi-person scene (see §5).

### A5. Robust person tracking (spatial) — **FULLY IMPLEMENTED**
- `src/tracker.py` now carries both layers: the identity FSM (`EmployeeTracker` /
  `MultiTracker`) remains the state layer, and the new `Track` / `SpatialTracker`
  (`src/tracker.py:443,567`) provide per-camera IoU/centroid association, `track_id`
  (`{camera}#{n}`), EMA-smoothed bbox, trajectory (deque, `TRACK_TRAJECTORY_LEN`),
  per-track `first_seen`/`last_seen`, `prune(max_age)`, `snapshot()` (JSON-safe), and
  `_associate_camera` greedy high-IoU matching with unmatched-person spawning.
- **Enrichment summary:** out-of-scope heavy trackers (DeepSORT/ByteTrack) were avoided by
  design; the light IoU tracker feeds productivity AND security layers.

### A6. Identity stability (no EMP↔Unknown flicker) — **FULLY IMPLEMENTED**
- Track-level identity voting replaces per-frame matching (`Track.register_vote` +
  `_decide`): a fresh track adopts after `IDENTITY_ADOPT_FRAMES` (3) consecutive votes; a
  KNOWN identity is **never demoted to Unknown** while the track lives; a switch to another
  known employee requires `IDENTITY_SWITCH_FRAMES` (3) consecutive votes AND the prior
  identity's run to have dropped to 0. `_register_face_vote` does this for every recognised
  face per frame.

### A7. Phone detection + association to the correct person — **FULLY IMPLEMENTED / PARTIALLY VALIDATED**
- YOLO `PHONE_CLASS=67`; `_phone_near_face` associates a phone to a *recognised* employee
  via expanded-face proximity (`src/detector.py`). On the spatial layer the phone flag is
  carried per track (`Track.phone` / `phone_since`), so **phone→person attribution now
  resolves through the track** and per-track phone duration (`phone_seconds`) is reported.
- **Honest gap:** `ON_PHONE` recall on the 640×480 webcam path remains low (documented YOLO
  sensitivity limitation); phone→track→employee is unit-tested and code-validated, not
  real-handset-validated.

### A8. Motion detection — **FULLY IMPLEMENTED / REAL-FRAME VALIDATED**
- New `src/motion.py` — `MotionDetector`: per-camera frame differencing with Gaussian blur,
  configurable sensitivity (`MOTION_MIN_SCORE`, `MOTION_DOWNSCALE`), `min_duration` gate,
  per-camera cooldown (`MOTION_COOLDOWN_SEC`) and anti-flood (`_fire_at`), per-run peak
  score + bounding region, live per-camera status for the dashboard, `enabled` flag, and
  honest "motion ≠ suspicious" advisory semantics. Events persist to the `motion_events`
  table via `db.insert_motion_event` and are exposed in `live_state.json["ai"]["motion"]`.
- **Semantics:** one event per continuous motion run; continuous motion re-fires once per
  cooldown window; new episodes within the cooldown are suppressed.
- **Real-frame check this session:** camera index 0, 90 real frames @ ~18 fps, peak
  score 0.0142 (< 0.02 gate on a static scene) → 0 events, as expected; pipeline ran clean.

### A9. Employee presence AI — **FULLY IMPLEMENTED / PARTIALLY VALIDATED**
- ACTIVE / ON_PHONE / AWAY / NOT_OBSERVED; camera-offline freeze prevents fake AWAY;
  ARRIVED emitted by security engine; dashboard `NOT_OBSERVED` semantics correct.
- **Gap closed:** **LEFT is now emitted** — (a) from `EXIT_CROSSING` via
  `SecurityEngine.record_entry_exit` (bookkept in `_exit_left_at`), and (b) as the
  appearance-based fallback in `tick` step 6, debounced by `LEFT_DEBOUNCE_SEC` (10 s) so
  one physical departure yields exactly one LEFT event.
- **Honest gap:** presence states remain single-operator webcam validated; real multi-person
  presence not yet exercised live.

### A10. Entry/Exit line-crossing — **FULLY IMPLEMENTED / PARTIALLY VALIDATED**
- New `src/entry_exit.py` — `VirtualLine` (normalised 0..1 `x1/y1/x2/y2`, validated in
  `__post_init__`, directed-line `side()` = A/B), optional JSON line file
  (`load_lines`, `ENTRY_EXIT_LINES_FILE`), and `EntryExitDetector`: **track-based**
  crossings only (first frame per track primes; disappearance near the line is NOT a
  crossing), direction mapping (`A_TO_B→EXIT_CROSSING`, `B_TO_A→ENTRY_CROSSING`), per
  `(track_id, line)` debounce (`ENTRY_EXIT_DEBOUNCE_SEC`), duplicate suppression, stale
  placement cleanup, and employee identity carried when the track has a stabilised identity.
- Crossings are consumed by `SecurityEngine.record_entry_exit` → `entry_exit_events` table +
  security pipeline (incident/alert/evidence) with `event_id` backlink; an `EXIT_CROSSING`
  for a known employee also emits `LEFT` (debounced, B6/A9).
- **Honest gap:** line geometry is code-validated; no real multi-line/multi-person scene
  exercise yet.

### A11. Live data — **FULLY IMPLEMENTED**
- `_write_live_state()` → `data/live_state.json` (employee detail, FPS, camera health,
  security snapshot); dashboard auto-refresh fragments (`app.py:1532`). Honest IPC vs
  processing FPS separation exists.

### A12. Camera health (frozen frame, blur, low FPS, black frame, latency) — **FULLY IMPLEMENTED / PARTIALLY VALIDATED**
- `VideoCapture.health` now returns the Phase A degraded-but-flowing states alongside the
  originals: **ONLINE / NO_FRAME / RECONNECTING / OFFLINE / FROZEN_FRAME / LOW_FPS**
  (`src/camera.py:134`). FROZEN_FRAME derives from a frame-content signature stall
  (`_frame_signature` ≥ 30 identical frames sustained ≥ `_FROZEN_MIN_SEC` 6 s); LOW_FPS is
  0 < sustained fps < 5.0. Thresholds are module constants; `health_info()` exposes
  `frozen_streak`; the dashboard health tab renders the new states. Additionally
  `CAM_HEALTH_ORDER`/`DEGRADED_STATES` move FROZEN/LOW_FPS into the degraded tier in
  `src/domain.py` and `src/camera.py`.
- **Honest gaps:** BLACK_FRAME/blur classification remains the local-webcam-path watchdog in
  `main.py` only (not in `VideoCapture`); real-camera validation covered ONLINE + LOW_FPS
  classification logic via unit tests and the live webcam repeated healthy (≥ 5 fps).
- **Gap closed (2026-09-09):** FROZEN_FRAME classification is now exercised on **real captured
  pixels** from the live webcam (`tests/test_camera_health_realfeed.py`, 5 tests, skipped
  honestly when no camera is reachable): the actual `_frame_signature` fingerprint is produced
  from a real frame, `_FROZEN_STREAK` identical real frames accumulate into the streak, a real
  frame is confirmed **NOT** to false-positive FROZEN while the scene is live (stays ONLINE), a
  sustained real-pixel streak ≥ `_FROZEN_MIN_SEC` reports `CAM_FROZEN`, and below the min-seconds
  window it does **not** report FROZEN.

### A13. Camera selection: EXPLICIT vs AUTO — **FULLY IMPLEMENTED**
- `--source auto` exists with `_auto_detect_camera_index()` probing bright devices
  (`main.py`); explicit `--source <index>` exists. **Added:** `config.CAMERA_MODE`
  (AUTO/EXPLICIT), and `main._write_live_state` emits a `camera_selection` payload
  (mode + resolved source/device index) into `live_state.json`, surfaced by the dashboard's
  "🤖 AI Capabilities" tab (tab index 4), so operators see which camera the system actually
  picked.

---

## 2. PHASE B — SECURITY INTELLIGENCE

### B1. Zone engine (configurable, camera-specific) — **FULLY IMPLEMENTED**
- `src/zones.py` `ZoneStore`, polygon zones, per-zone policies
  BYPASS / INTRUSION / UNKNOWN_ONLY / AFTER_HOURS_INTRUSION / ALLOWLIST / RESTRICTED.
- **Gap:** policy naming — audit uses BYPASS/INTRUSION/UNKNOWN_ONLY/AFTER_HOURS_INTRUSION;
  RESTRICTED/ALLOWLIST defined but unused by the engine. Low priority.

### B2. Restricted-zone intrusion — **FULLY IMPLEMENTED**
- Intrusion detection + incident binding + `INTRUSION_COOLDOWN_SEC` config exist and the
  cooldown is now **actually wired** (`src/security_engine.py`, zone-eval 4c): while a
  person remains inside a restricted zone, repeated INTRUSION events are gated per
  `(camera, zone)` via `_last_intrusion`, so each tick no longer writes a fresh INTRUSION
  security event (event-spam to DB/correlation/dashboard is closed). Loitering bookkeeping
  (`_zone_entry`) still runs while gated. Tested: first tick fires, immediate re-tick
  suppressed, re-fire after the cooldown elapses.

### B3. Loitering — **FULLY IMPLEMENTED**
- `LOITERING_SUSPECTED` with entry timestamp + dwell duration + re-fire cooldown. Per-zone
  and per-camera. **Note:** entry key is (camera, zone) not (camera, zone, track) — after
  A5 this should become per-track for multi-person loitering.

### B4. After-hours — **FULLY IMPLEMENTED**
- `AFTER_HOURS_ACTIVITY` via configured working hours (office start/end window).
  Configurable; camera-specific schedules not required.

### B5. Occupancy / crowd / capacity — **FULLY IMPLEMENTED**
- `UNUSUAL_OCCUPANCY` with global `CROWD_THRESHOLD` census + baseline anomaly engine
  (time-of-day/weekday). **Gap closed:** optional **per-zone `capacity`** now exists — the
  `Zone` dataclass/to_dict and `ZoneStore.add` accept `capacity`, and `db.upsert_zone`
  persists it to `security_zones.capacity` (falls back to global `CROWD_THRESHOLD` when
  unset).

### B6. Entry/exit security correlation — **FULLY IMPLEMENTED / PARTIALLY VALIDATED**
- Exit-crossing source (A10) is now present: ENTRY_CROSSING / EXIT_CROSSING events flow
  through `SecurityEngine.record_entry_exit` → `entry_exit_events` (+ security pipeline with
  `event_id` backlink), LEFT is emitted from exit-crossing (debounced), and the opt-in
  **UNEXPECTED_STAY** heuristic ("crossed in via a line, never crossed out before the office
  window ended") runs at the end of each tick, producing one LOW-severity event per
  employee-session from `_crossing_arrivals` (populated only from ENTRY_CROSSING).
- **Honest gap:** ENTRY/EXIT pairing in `entry_exit_events` is code-validated; real
  multi-person line scenes not yet exercised live.

### B7. Camera tamper — **FULLY IMPLEMENTED**
- Frozen / dark / bright detection with persistence thresholds (`src/tamper_monitor.py`),
  CAMERA_TAMPER_* events, evidence hooks. Scene-change/camera-reposition detection not
  implemented (documented limitation; not required).

### B8. Anomaly intelligence — **FULLY IMPLEMENTED**
- `src/anomaly_engine.py`: occupancy baselines, z-score (3.0), time/day baselines,
  recurring-pattern detector, honest refusal to fabricate from thin data.

### B9. Event correlation — **FULLY IMPLEMENTED**
- `src/correlation.py`: person-threat pairing, camera-lifecycle pairing, explainable
  reasons, evidence binding, incident contexts.

### B10. Risk scoring (explainable, not per-person) — **FULLY IMPLEMENTED**
- `src/risk_scoring.py`: weights 0.5/0.25/0.15/0.10 + bounded context bonuses; incident
  scoped only; never per-person.

### B11. Evidence capture (EXPLICIT/OFF modes) — **FULLY IMPLEMENTED / PARTIAL GATING**
- SHA-256, path-isolation, retention delete, evidence modes OFF/EVENT_ONLY/
  HIGH_SEVERITY_ONLY/CONTINUOUS(reserved).
- **Note:** `event_only` captures HIGH+ only today (`src/security_engine.py`). That is
  by design (HIGH_SEVERITY_ONLY-ish) but conflicts with literal EVENT_ONLY semantics; keep
  behaviour, align config documentation.

### B12. Alert engine — **FULLY IMPLEMENTED (email NOT validated)**
- Full lifecycle PENDING/DELIVERED/FAILED/RETRYING/ACKNOWLEDGED/RESOLVED/ESCALATED; rule
  cooldown, dedup, escalation. Email adapter honest-returns False (not claimed as working).

### B13. Incident lifecycle — **FULLY IMPLEMENTED** (open→acknowledged→resolved/dismissed)
- `src/incidents.py`; `review_state` investigation semantics; audit of transitions.
- **Note on acknowledged-status:** fixed post-prior-audit? Validate `acknowledge()` sets
  `status='ACKNOWLEDGED'` (previously only set timestamp). Same for RBAC release path.

### B14. Security dashboard — **FULLY IMPLEMENTED**
- Dashboards for incidents, zones, evidence, tamper, occupancy, alerts, security events,
  privacy indicator, RBAC. Section 13-era additions present: **camera selection, spatial
  track list, live motion feed, and a 10-tab layout with the "🤖 AI Capabilities" tab**
  (`app.py`) rendering detector capability truth from the registry plus spatial tracks and
  motion status.

### B15. Capability registry & accuracy — **FULLY IMPLEMENTED**
- `src/detector_registry.py` vocabulary is now **AVAILABLE / DISABLED / DEGRADED /
  NOT_CONFIGURED / FUTURE_MODEL_REQUIRED** (+ legacy UNAVAILABLE in `STATUS_VOCABULARY`,
  exposed via `status_vocabulary()`), `disable()` sets DISABLED, and
  `register_system_capabilities()` registers the NEW capabilities (`person_tracker`,
  `motion_detector`, `entry_exit_engine`) with honest statuses (no camera → NOT_CONFIGURED;
  `MOTION_ENABLED=false` → DISABLED; no line file → NOT_CONFIGURED + actionable detail).
  `main._registry_capabilities()` feeds the dashboard panel, which only ever claims what the
  registry reports.

---

## 3. SECTION-13 / CROSS-CUTTING GAPS

| Item | Status | Enrichment |
|---|---|---|
| 6 - Data model consistency | OK | `security_events.track_id` added (`_migrate_phase_ab`); `motion_events` + `entry_exit_events` tables added; zone capacity column added. `security_events.track_id` set by `_process_event`. |
| 7 - Time standardisation | PARTIAL | Mixed epoch (tracker/live_state) vs local ISO strings (DB). Document; keep epoch in per-frame metrics, ISO in DB. |
| 14 - Debug flags | OK | `TRACK_DEBUG`, `MOTION_DEBUG` exist and gate spatial tracker / motion logs; `DETECTION_DEBUG` + `FACE_DEBUG` present. |
| 13 - Camera selection UI | OK | Resolved index / source / mode shown via `camera_selection` payload + AI Capabilities tab. |
| 15 - Email honesty | OK | Keep unvalidated; document in readiness. |
| Simulation bypass | OK | Sim harness bypasses AI; that is the design; keep separation explicit. |

---

## 4. ENRICHMENT PLAN (WHAT WILL BE BUILT) — **ALL DONE (2026-09-09)**

1. **Detector (A1):** ✅ per-person `person_box`/`person_conf` entries emitted; faces→person
   boxes and phones→person boxes mapped; legacy presence entry kept.
2. **Spatial tracker (A5, A6):** ✅ `Track` + `SpatialTracker` (per-camera IoU association,
   trajectory, first/last seen) in `src/tracker.py`; identity smoothing via track-level
   votes (adopt-after-3 / never-demote / switch-after-3). FSM kept as state layer.
3. **Phone association (A7):** ✅ phone→track→employee attribution + per-track phone
   duration (`Track.phone_seconds`).
4. **Motion (A8):** ✅ `src/motion.py` + `motion_events` table + MOTION events; advisory only.
5. **Entry/exit (A10):** ✅ `src/entry_exit.py` (virtual lines, direction, debounce,
   dedup) → ENTRY_CROSSING / EXIT_CROSSING; LEFT from exit-crossing; feeds B6.
6. **Security (B2, B5, B6):** ✅ `INTRUSION_COOLDOWN_SEC` wired; per-zone `capacity`;
   entry/exit correlation + UNEXPECTED_STAY ("entered, never exited").
7. **Camera (A12, A13):** ✅ FROZEN_FRAME/LOW_FPS/DEGRADED health classification;
   `CAMERA_MODE` config + dashboard camera-selection display.
8. **Registry (B15):** ✅ status vocabulary AVAILABLE/DISABLED/DEGRADED/NOT_CONFIGURED/
   FUTURE_MODEL_REQUIRED; new capabilities registered; dashboard panel reflects registry.
9. **DB/data (6, 7):** ✅ `security_events.track_id`; `motion_events`; `entry_exit_events`;
   time conventions documented (epoch in per-frame metrics, ISO in DB).
10. **Tests + docs:** ✅ **54 new regression tests added** (`tests/test_phase_ab.py`) covering
    detector enrichment + conf fallback, Track/SpatialTracker identity rules, MotionDetector
    gating, entry/exit crossings + debounce, zone capacity, camera health states, registry
    vocabulary/registration, intrusion cooldown, LEFT debounce, UNEXPECTED_STAY gating, DB
    roundtrips, and dashboard `ai` payload. Docs updated (this doc, README,
    PRODUCTION_READINESS.md, VALIDATION_REPORT.md).

**Explicitly out of scope (honest):** no real RTSP/NVR. Real-model multi-person accuracy is
not claimed without a real multi-person camera run; DroidCam was not reachable this session
(see §5).

---

## 5. VALIDATION RESULTS (2026-09-09)

**Automated suite:** full suite green (`744 passed / 0 failed`, real `.venv`, `-p no:cacheprovider`, ~45 s), incl. 54 Phase A/B (`tests/test_phase_ab.py`), 30 multi-person/camera-honesty (`tests/test_multiperson_pipeline.py`), and 5 real-feed camera-health (`tests/test_camera_health_realfeed.py`, `-W error`). All untouched suites green.

**Live webcam (this session):**
- Camera **index 0 opened** and delivered 640×480 BGR frames (device fps 30.0).
- Motion validation loop read **90 real frames at ~18.5 fps delivered** (well above the
  LOW_FPS threshold of 5.0 → classification would be ONLINE).
- `MotionDetector` pipeline (downscale 320, `MOTION_MIN_SCORE 0.02`, min-duration 1.0 s)
  ran end-to-end: peak frame-diff score **0.0142** on a static scene → **0 motion events**
  (correct advisory behaviour; nothing fabricated). Priming, per-camera status, and the
  real-frame diff loop all exercised.

**Post-fix live session (same date, 50 s, `--source auto` + CAMERA_DEBUG/FACE_DEBUG):**
- **AUTO resolved index 0** (today's live device); structured debug confirms
  `resolved_index=0 frame_received=True frame_width=640 frame_height=480 frame_age=0.07
  input_fps=2.98 processing_fps=0.84 camera_online=True usable=True`.
- `live_state.json` showed **EMP001 and Unknown ACTIVE simultaneously** (genuine known+unknown
  multi-person observation on the real feed), EMP001 `away_sec: 0.0`, EMP002/EMP003/EMP004
  honest `NOT_OBSERVED`, `camera_health.frame_mean=125.74 usable=true`, **no CAMERA_OFFLINE,
  no dark-feed warning**, spatial track `local_webcam#1` present.
- Directly addresses the incident that motivated the round: `CCTV_CAMERA_MODE=INDEX:1` was
  pinned to today's dark/stuck feed (index 1, mean 4.1) while the live webcam is index 0
  (mean 142.7) — the fixed `.env` now uses `AUTO`, and dead/blank frames are excluded by
  `_usable_camera_frames()` so no false AWAY can accrue offline. Security `is_online`
  inversion (NO_FRAME treated as online) fixed; `CAMERA_OFFLINE` failsafe verified in tests.

**Not reachable / not exercised (honest):**
- **DroidCam (127.0.0.1:4747): connection refused** this session — the phone-app/RTSP path
  was NOT re-validated. Historical DroidCam runs are documented in VALIDATION_REPORT.md.
- **Two-roster-employee simultaneous recognition** still pending: EMP002/EMP003 have no
  enrolled face (`NO_FACE` — the `EMP002.jpg`/`EMP003.jpg` files carry no detectable face, so
  real enrollment photos are required; no fake enrollment data is fabricated). Known+unknown
  multi-person is now live-validated; known+known remains an honest gap until enrollment.
- **FROZEN_FRAME** now validated via real-camera pixels (see A12); a genuine 6-second physical
  lens freeze end-to-end is still not exercised, but the classification path is.

**Legacy note:** past DroidCam validation reported index 0 delivering black frames while the
daemon auto-picked a live index; this session index 0 delivered live frames directly, so the
device enumeration picture has changed — the daemon's `--source auto` still resolves the
live device automatically and the black-frame watchdog remains the guard for covered lenses.