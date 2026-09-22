# PHASE 59 REPORT — DESK / SEAT ZONE TRACKING + EMPLOYEE SEAT MAPPING

**Phase:** 59
**Date:** 2026-09-19
**Status:** IMPLEMENTED — CODE VERIFIED (SIMULATED/OFFLINE validation)
**Nature:** Additive context layer over the Phase A spatial tracker + Phase 57 identity system.

---

## 1. Executive Summary

Phase 59 adds a **desk/seat zone layer** that answers *"where is this person?"* for
spatial tracks. A desk zone is a normalised polygon attached to one camera,
optionally mapped to an employee (a *seat-map hint*). Occupancy is decided from
each track's **footpoint** (bbox bottom-centre) with temporal stability so a
single noisy frame never flips a zone.

The layer is **strictly CONTEXT ONLY**:

* it never changes the face/appearance identity pipeline (Phase 57 semantics
  untouched);
* an Unknown person in a seat stays Unknown;
* a confirmed EMP002 sitting in EMP001's seat is reported as EMP002 with a
  **neutral** `SEAT_ASSIGNMENT_MISMATCH` observation — never an accusation,
  never a relabel;
* an empty seat is a vacancy, never an invented "employee AWAY";
* a camera outage **freezes** its zones (no force-vacancy, no fake absence).

Identity authority remains exactly where it was before Phase 59.

## 2. Scope

### In
- New `src/seat_zones.py`: `SeatZone`, `SeatZoneStore`, `seed_seat_zones`,
  `SeatZoneTracker` (single engine), sanitized transition-driven events.
- `src/database.py`: `seat_zones` + `seat_events` tables, indexes, CRUD/query
  helpers (`upsert_seat_zone`, `list_seat_zones`, `get_seat_zone`,
  `delete_seat_zone`, `insert_seat_event`, `list_seat_events`).
- `config.py`: `SEAT_ZONE_TRACKING`, `SEAT_ZONES_FILE`, `SEAT_ZONE_CONFIRM_FRAMES`
  (5), `SEAT_ZONE_CLEAR_FRAMES` (8), `SEAT_EVENT_COOLDOWN_SEC` (60) + validation.
- `src/zones.py`: public `point_in_polygon` (shared geometry primitive).
- `src/tracker.py`: `Track.footpoint_normalized` + context-only `Track.zone`.
- `main.py`: engine lifecycle (init after spatial, per-cycle tick, live-state
  payload, isolated failures).
- `app.py`: read-only desk/seat panel in AI Capabilities + admin-only zone
  editor (role-gated).
- `data/seat_zones.json`: empty seed template (no hard-coded real assignments).
- `tests/test_phase59.py`: 26 deterministic SIMULATED/OFFLINE tests.

### Out (deliberately)
- No new identity system, tracker, registry, employee FSM, or productivity
  engine (hard constraint).
- No real client NVR/camera validation (none available).
- No production-ready claims.

## 3. Architecture

```
 Detector -> SpatialTracker (Phase A)              <-- unchanged
                |
                v  active Track objects (footpoint, cam)
        SeatZoneTracker.tick(tracks, usable)
                |  (enabled zones whose camera is usable this cycle)
                v
   per-zone temporal stability:
     confirm_frames consecutive inside  -> OCCUPIED
     clear_frames   consecutive absent  -> VACANT
                |
                v
   transition-driven events (neutral)  --> seat_events table
                |
                v
      snapshot()  --> live_state.json ("ai"."seat_zones")  --> dashboard
```

- **Identity fusion is one-way and advisory.** The engine reads the track's
  face-confirmed `identity` for events/claims. Assignment hints feed only the
  neutral `SEAT_ASSIGNMENT_MISMATCH` observation. There is exactly one
  `SeatZoneTracker`; no secondary identity authority exists.
- **Temporal stability** uses consecutive-frame evidence counters
  (`SEAT_ZONE_CONFIRM_FRAMES`, `SEAT_ZONE_CLEAR_FRAMES`).
- **Event cooldown** (`SEAT_EVENT_COOLDOWN_SEC`) gates repeated identity
  *reports*; transitions are never cooldown-gated.
- **Camera semantics:** zones are evaluated only when
  `usable_camera_ids ∩ zone.camera_id`; otherwise they are frozen in place.
  An all-camera outage therefore cannot vacate desks or fabricate AWAY.

## 4. Config Knobs

| Knob | Env | Default |
|---|---|---|
| `SEAT_ZONE_TRACKING` | `CCTV_SEAT_ZONE_TRACKING` | `1` |
| `SEAT_ZONES_FILE` | `CCTV_SEAT_ZONES_FILE` | `data/seat_zones.json` |
| `SEAT_ZONE_CONFIRM_FRAMES` | `CCTV_SEAT_ZONE_CONFIRM_FRAMES` | `5` |
| `SEAT_ZONE_CLEAR_FRAMES` | `CCTV_SEAT_ZONE_CLEAR_FRAMES` | `8` |
| `SEAT_EVENT_COOLDOWN_SEC` | `CCTV_SEAT_EVENT_COOLDOWN_SEC` | `60` |

All validated in `validate_config()`.

## 5. Event Vocabulary (sanitized)

| Event | Meaning |
|---|---|
| `ZONE_OCCUPIED` | zone entered OCCUPIED after confirm frames |
| `ZONE_VACATED` | zone returned to VACANT after clear frames |
| `ZONE_IDENTITY_CONFIRMED` | occupant's face-confirmed identity is KNOWN |
| `ZONE_IDENTITY_UNKNOWN` | occupant identity is Unknown |
| `SEAT_ASSIGNMENT_MISMATCH` | confirmed occupant != seat assignment (neutral) |

Events carry only IDs + neutral detail string (no raw biometrics, no RTSP
credentials, no boxes/embeddings — asserted by test).

## 6. Honesty Considerations

- `REAL_ZONE_VALIDATION = NOT_VALIDATED`. No client office NVR/camera was
  available; polygons, cadence and camera geometry are validated only through
  deterministic synthetic simulation.
- Tests are labeled SIMULATED/OFFLINE.
- An empty `data/seat_zones.json` ships; no real seat assignments are hard-coded.

## 7. Test Summary

- New: `tests/test_phase59.py` — **26 passed** (24 named tests + live-state
  integration + multi-person desk-seat swap simulation).
- Simulated scenario (offline): Person A (EMP001)-A01, Person B (EMP002)-A02
  for 300 frames; desks briefly vacated; then a swap. Verified: correct zone
  occupancy before/after, exactly two `ZONE_OCCUPIED` per desk, neutral
  `SEAT_ASSIGNMENT_MISMATCH` with `assigned=EMP002 confirmed=EMP001`, identity
  never bent by the seat map, no AWAY/fake-vacancy anywhere.
- Regressions: `pytest -q` **1139 passed** (Phase 58 baseline 1113 → +26).
- Warnings-as-errors: `pytest -q -W error` **1139 passed**.

## 8. Performance

Synthetic load, 200 ticks, 40 tracks × 40 zones (multi-row office layout):
~0.59 ms/cycle average (p99 ~1.1 ms) — negligible vs the ~400 ms/frame budget
at the configured 2.5 FPS.

## 9. Productivity / Employee FSM Regression

None. The seat layer never writes to the employee FSM, never emits AWAY, and a
camera outage freezes zones exactly like the existing "no fake AWAY" invariant.
All Phase 43/46/51/57 outage & productivity tests still pass.

## 10. Classification

| Item | Class |
|---|---|
| Desk/seat zone engine (Code) | **CODE VERIFIED** |
| Polygon + footpoint geometry | CODE VERIFIED (synthetic) |
| Temporal stability (confirm/clear hysteresis) | CODE VERIFIED |
| Seat-map + identity fusion (context only) | CODE VERIFIED |
| Unknown-seat handling | CODE VERIFIED |
| Camera-offline freeze / no fake AWAY | CODE VERIFIED |
| Neutral mismatch reporting | CODE VERIFIED |
| Event persistence + sanitization | CODE VERIFIED |
| Multi-camera zone readiness | CODE VERIFIED (synthetic multi-camera) |
| Real office validation | **NOT VALIDATED** (no client NVR) |
| Production readiness | **NOT READY** (pilot semantics preserved) |

## 11. Next Phase (60)

Await instructions. (Suggested candidates: real-RTSP desk-zone calibration
tooling; seat-map dashboard heatmap; longitudinal seat-occupancy analytics.)