# PHASE 60A — OFFLINE OFFICE SIMULATION + DESK/CHAIR/EMPLOYEE CONTEXT VALIDATION REPORT

- **PHASE:** 60A — OFFLINE OFFICE SIMULATION + DESK/CHAIR/EMPLOYEE CONTEXT VALIDATION
- **STATUS:** COMPLETE
- **DATE:** 2026-09-22
- **CCTV HARDWARE:** NONE connected — this phase is 100% SOFTWARE / SIMULATED validation.

---

## OBJECTIVE

Prove that the existing office-intelligence platform correctly executes the
software-side context chain **when valid tracking observations are supplied**:

    Employee → Identity → Spatial Track → Desk/Zone → Chair/Seat →
    Office Schedule → Employee Context → Productivity/State semantics

This phase does **NOT** claim real-CCTV/NVR validation.  Physical IP CCTV/NVR is
unavailable, so every observation used here is a **deterministic TEST
observation** (never a real camera detection).  The goal is only to prove the
software correctly derives desk/chair/employee context and preserves identity /
state safety guarantees from valid, scripted spatial input.

---

## IMPLEMENTED (this phase)

- **`tests/test_phase60a.py`** (NEW) — 41 focused deterministic tests covering
  the full Phase 60A requirement matrix (zone membership, footpoint semantics,
  boundary behaviour, zone/chair/employee mapping, identity safety, office
  schedule boundaries, thresholds, camera context, multi-person, temporal
  stability, privacy, admin-config surface).
- **`src/simulation/office_context.py`** (NEW) — smallest deterministic
  simulation helper inside the existing `src/simulation` package.  Provides
  `Observation` (step/track_id/employee_id/camera_id/footpoint/timestamp),
  `footpoint_bbox`, `make_track`, `ts_for_step`, `simulate_step`.  It shapes
  test input only: it contains **no** tracking/identity/zone/schedule logic
  (that lives in the real modules) and never runs in production.
- **`src/database.py`** (MINIMUM BUG FIX) — `upsert_employee` now persists a
  first-time `display_name` on INSERT (previously INSERT silently dropped it;
  only the UPDATE path wrote it).  This is the single genuine missing-capability
  the audit surfaced for the admin-config surface (see CHANGED).

---

## ALREADY EXISTED (reused, not rewritten)

- `src/seat_zones.py` — `SeatZone`, `SeatZoneStore`, `SeatZoneTracker`
  (footpoint membership, confirm/clear hysteresis, camera freeze, neutral
  `SEAT_ASSIGNMENT_MISMATCH`).
- `src/seat_chairs.py` — `Chair`, `ChairStore`, chair-event constants,
  `load_chair_defaults`.
- `src/office_schedule.py` — `OfficeSchedule`, `parse_hhmm` (Asia/Kolkata,
  10:00–18:30, lunch 14:00–14:35).
- `src/tracker.py` — `Track.footpoint_normalized`, `EmployeeTracker`
  camera-offline freeze (never fabricates AWAY), identity-voting authority.
- `src/zones.py` — `parse_polygon` / `point_in_polygon` shared geometry.
- `src/database.py` — CRUD for `seat_zones`, `seats`/`chairs`, `seat_events`,
  `chair_events`, `employees` (incl. `set_employee_active`).
- `config.py` — `CCTV_OFFICE_*`, `CCTV_LUNCH_*`, `CCTV_OFFICE_TZ`,
  `FUTURE_AWAY/PHONE/TALKING_*`, `SEAT_*`, `CHAIR_*` knobs.
- Existing Phase 59 / 59C helpers and test seams (`tmp_db`, synthetic tracks).

---

## CHANGED

- `src/database.py` — `upsert_employee` INSERT path: add the `display_name`
  column (lazy migration) and store a supplied display_name on first insert.
  Idempotent, additive; UPDATE semantics unchanged.
- `tests/test_phase60a.py` — added (41 tests).
- `src/simulation/office_context.py` — added (simulation helper).

---

## NOT CHANGED

- Phase 57 identity thresholds (`FACE_SIMILARITY_THRESHOLD`,
  `FACE_CANDIDATE_THRESHOLD`, `FACE_MARGIN_MIN`) — untouched.
- Identity authority (face/appearance pipeline in `src/tracker` /
  `face_registry`) — untouched.
- Phase 58 camera architecture (`CameraConfig`, credential handling,
  reconnect, health states) — untouched.
- `OfficeSchedule`, `SeatZoneTracker`, `ChairStore/Chair`, `SeatZone` engine
  semantics — NOT modified (no schedule/zone/chair redesign, no second engine).
- Employee FSM semantics — untouched (healthy-absence AWAY confirmed as correct).
- `config.py` knobs — untouched (no new runtime mode, no knob changes).
- No second tracker, no second employee registry, no second schedule engine.
- No existing tests deleted, weakened, or skipped.
- No git commit / push / branch.  No repository reset.

---

## TESTS

- **Phase 60A (focused):** `python -m pytest tests/test_phase60a.py -q` →
  **41 passed** (0.70 s).  The 41 tests map to all 20 required test categories:
  1) desk zone membership, 2) footpoint calculation/usage, 3) boundary
  behaviour, 4) zone stability, 5) chair assignment, 6) unassigned chair,
  7) disabled chair, 8) UNKNOWN near assigned chair, 9) identity/chair
  mismatch, 10) employee_id stability, 11) display_name change,
  12) office hours, 13) lunch window, 14) threshold configuration,
  15) camera context mismatch, 16) camera failure no fake AWAY,
  17) empty chair no AWAY, 18) multi-person mapping, 19) track
  loss/reappearance, 20) privacy-safe logging.
- **Full suite:** `python -m pytest -q` → **1196 passed** (81 s).
- **Full suite w/ warnings as errors:** `python -m pytest -q -W error` →
  **1196 passed** (75 s).

---

## STATIC/IMPORT CHECK

- `python -m py_compile src/database.py src/simulation/office_context.py tests/test_phase60a.py` → **OK**.
- Import smoke test of every touched/changed module
  (`src.simulation.office_context`, `src.seat_zones`, `src.seat_chairs`,
  `src.office_schedule`) → **OK**.
- No new tooling stack introduced (compile/import checks only, per rule 20).

---

## CODE VERIFIED

Behaviour confirmed by source inspection and/or exercise of the real modules
(no camera needed):

- `Track.footpoint_normalized` is bbox bottom-centre; desk zones evaluate the
  footpoint, not the centroid.
- `SeatZoneTracker` confirm/clear hysteresis keeps a single noisy frame from
  flipping a stable zone; track loss shorter than `clear_frames` never resets
  the occupancy episode.
- A zone on a wrong / disabled / missing camera is never occupied.
- `ChairStore` scopes chairs by zone and camera; `assigned_employee_id` is a
  home-desk hint only.
- `OfficeSchedule` defaults (Asia/Kolkata, 10:00–18:30, lunch 14:00–14:35) and
  boundary semantics (`[start, end)` exclusive end) verified.
- `FUTURE_AWAY/PHONE/TALKING = 10 / 5 / 15 s` centralised and deterministic.
- `EmployeeTracker.process(..., camera_online=False)` freezes state (never
  fabricates AWAY); AWAY is produced only from the employee's own confirmed
  absence with a healthy camera.
- `upsert_employee` now round-trips display_name on first insert while
  `employee_id` stays the stable identity key.
- Admin config surfaces exist for display_name, employee enabled, desk
  assignment, chair assignment, office schedule, thresholds, chair-tracking
  settings.

---

## SIMULATED VALIDATION

All simulation work in this phase is label-aligned with the existing project
conventions (Phase 36/59 style): deterministic synthetic observations via
`src.simulation.office_context` over real `src.seat_zones` / `src.seat_chairs`
/ `src.office_schedule` / `src.tracker` / `src.database` modules.

Validated deterministically (simulated tracks, never real footage):

- EMP001 → D01 → C01 → CAM01 and EMP002 → D02 → C02 → CAM01 complete mapping.
- Both employees correctly mapped simultaneously; identity stable per employee.
- EMP002 in EMP001's desk/chair → EMP002 stays EMP002; only a NEUTRAL
  `SEAT_ASSIGNMENT_MISMATCH` observation is produced.
- Unknown person near an assigned chair → stays UNKNOWN (chair never bestows
  identity); Unknown + empty chair → stays Unknown / VACANT.
- Desk membership: inside, outside, epsilon boundary, disabled zone,
  invalid/empty polygon, wrong camera, unknown zone.
- Zone stability: inside→outside→inside does not disturb a stable occupancy;
  track loss/reappearance keeps one episode; >= clear_frames resets context.
- Multi-person: two in one zone (persons=2, deterministic first occupant),
  one leaving (VACATED), one returning (new episode).
- Camera-outage: zone frozen, no VACATED, no AWAY; FSM frozen, no fake AWAY.
- Empty chair: VACANT, no generated events, no AWAY anywhere.
- Office-schedule and lunch boundary timestamps (09:59 … 18:31 /
  13:59 … 14:36) behave as specified.

---

## REAL VALIDATION

**NOT RUN — physical CCTV/NVR unavailable.**

No real RTSP was opened, no NVR touched, no camera stream driven, no frame
content assessed, and no FPS / OSD / real multi-person scenario was produced
from hardware.

---

## NOT VALIDATED

- Real IP CCTV
- Real NVR
- Physical camera placement (actual desk/zone geography)
- Real FOV
- Real face pixel density
- Real lighting / occlusion behaviour
- Real multi-person CCTV behaviour
- Real desk/chair physical layout confirmation
- Real RTSP URL / credential negotiation against hardware

---

## MODEL LIMITATIONS

- Recognition accuracy, threshold tuning, margin calibration and
  unknown/known balance cannot be established without real-person footage —
  simulated identities carry no vision evidence.
- Zone/chair "fit" to a physical desk depends on real camera placement and FOV;
  the deterministic polygons here are configuration fixtures, not measured
  geometry.
- Chair/sitting-vs-standing semantics are not validated by any vision model in
  this phase — chairs are configuration children of zones only.
- Nothing in this phase measures FPS, latency, or throughput on real hardware.

---

## PRIVACY/SECURITY

- All simulated observations carry IDs / footpoint / timestamp only — no face
  embeddings, face images, person crops, or credentials by construction.
- Phase 60A privacy test asserts seat events persist only sanitized columns and
  contain none of: password / rtsp:// / http:// / embedding / face_box /
  person_box / bbox / credential.
- No secrets introduced; RTSP passwords remain never-logged (Phase 58 rule kept).
- Office schedule and chair/zone layers never relabel an occupant and never
  fabricate employee AWAY — identity and productivity semantics stay intact
  (Phase 43 protections preserved).

---

## REGRESSIONS

NONE.  Full suite: 1196 passed both normally and with `-W error` (includes the
pre-existing Phase 40–59 suites).

---

## NEXT PHASE

**PHASE 61 — ADMIN PANEL + COMPLETE OFFICE CONFIGURATION CONTROL**

Full admin surface to edit employee display names / enable state, desk-zone
assignment, chair assignment, office schedule, thresholds, and chair-tracking
settings (Phase 60A objective 17 lays the verified groundwork; the complete
panel is deliberately deferred to a dedicated phase).

---

**REPORT FILE:** `PHASE60A_OFFLINE_OFFICE_SIMULATION_VALIDATION_REPORT_2026-09-22.md`