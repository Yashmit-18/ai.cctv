# Phase 58 — Production RTSP / IP-CCTV / NVR Camera Layer

**Date:** 2026-09-19
**Phase:** 58
**Baseline commit:** `c2a80c915b059ca162db8e83d9b34a059dd0618d` (unchanged — **no commit/push made during Phase 58**)
**Status:** PASS_WITH_LIMITATIONS
**Real RTSP validation:** NOT_VALIDATED (no client CCTV/NVR-adjacent real feed was reachable)

---

## 1. Executive Summary

Phase 58 hardened the existing multi-camera capture layer into a production
RTSP / IP-CCTV / NVR camera layer. The architecture already had a strong
foundation (threaded `VideoCapture`, FFMPEG/TCP RTSP transport, bounded
exponential-backoff reconnect, frozen/low-FPS health, stale-frame and
dark/blank feed gating). This phase added the missing production pieces on
top of it:

* a first-class per-camera configuration model (`CameraConfig`) with source
  kind (`local | rtsp | video_file | test`), name, location, FPS target,
  enabled flag and **per-camera reconnect policy**;
* split `_USERNAME` / `_PASSWORD` env credentials so the authenticated URL is
  never stored or logged;
* a guarantee (enforced by tests) that credentials never appear in config
  output, logs, health snapshots, dashboard or reports;
* explicit `DARK_BLANK_FRAME` domain constant replacing a magic string;
* richer per-camera health telemetry (id, kind, resolution, reconnects)
  surfaced end-to-end to the live-state JSON and Streamlit camera cards;
* delivery: exactly **18 requirement tests (TEST 1–18)** plus a few focused
  sub-tests/asserts (e.g. TEST 14b, 17b) — 20 test functions total in
  `tests/test_phase58.py`, all SIMULATED/OFFLINE.
  functions total in `tests/test_phase58.py`, all SIMULATED/OFFLINE.

Full regression passed: `pytest -q` **1113 passed**, `pytest -q -W error`
**1113 passed**, AST + import smoke clean.

**Honest limitations.** No client CCTV/NVR feed was available in this
environment. Public "test" RTSP URLs were attempted but the host could not be
reached (DNS failure / no route), and even a reachable public demo feed is
**not** a client-camera validation. `REAL_RTSP_VALIDATION` is therefore
honestly **NOT_VALIDATED**, `REAL_CAMERA_MODEL = NOT_AVAILABLE`, and real
RTSP FPS / AI processing FPS remain **NOT_VALIDATED** (not measured on a real
feed). `PRODUCTION_READINESS = NOT_READY` per the existing project rule of
running exactly one phase at a time.

---

## 2. Baseline

* Prior phase (57) complete and uncommitted per its rules;
  `src/face_registry.py`, `tests/test_phase57.py`,
  `PHASE57_FACE_GALLERY_MARGIN_AUDIT_REPORT_2026-09-19.md` all still present
  and untouched.
* Baseline test count (Phase 57 final): **1093 passed** both modes.
* Camera architecture audited (Phase 58): existing `VideoCapture`,
  `MultiCameraManager`, `CameraStateManager`, `security_engine.tick`,
  `main._usable_camera_frames` dark gate, dashboard camera cards, Docker
  env pass-through.

## 3. Existing Camera Architecture (audit result)

| Layer | File | Role (pre-existing) |
|---|---|---|
| Capture engine | `src/camera.py` | `VideoCapture`: threaded reader, FFMPEG/CAP_FFMPEG + TCP, buffer=1, 4 s read timeout helper, frozen-frame (30 streak / 6 s), rolling FPS, reconnect backoff 2 s→30 s ×2.5, health: ONLINE / NO_FRAME / RECONNECTING / OFFLINE / FROZEN_FRAME / LOW_FPS |
| Pool manager | `src/camera_manager.py` | `MultiCameraManager`: per-camera thread, `get_latest_batch`, `latest_frames(max_age)` stale filter, `camera_health()` (redacted), `frame_interval()` |
| Security FSM | `src/camera_state.py` | per-camera ONLINE⇄OFFLINE, CAMERA_OFFLINE / CAMERA_RECOVERED events; **offline ≠ employee AWAY** |
| Health wiring | `main.py` | `_usable_camera_frames` gates `{ONLINE, LOW_FPS, FROZEN}` + mean-brightness; `unusable_reason="DARK_BLANK_FRAME"`; freezes states, never AWAY |
| Security engine | `src/security_engine.py` | ONLINE/LOW_FPS ⇒ online; else CAMERA_OFFLINE |
| Dashboard | `app.py` | camera cards (`_safe_cam_source` hides `://`), `tab_camera_health`, DARK gate notes |
| Docker | `Dockerfile`, `docker-compose.yml` | two services + shared data volume + `CCTV_CAM_<NN>_URL` env pass-through |
| Tooling | `src/rtsp_tester.py` | multi-stream RTSP stress tester (Phase 7) |

## 4. RTSP Architecture

* **Common source abstraction:** `VideoCapture` remains the single engine for
  local, RTSP, file and test sources — no duplicated capture hierarchy was
  introduced (the brief explicitly allowed keeping the existing abstraction).
* **Source kinds:** new `CameraSourceKind` enum
  (`local / rtsp / video_file / test`) with `infer()` for legacy `source`
  values (ints, `webcam`, `local`, `rtsp://…`, file paths). `VideoCapture`
  now carries `kind`, `camera_id` and a per-camera reconnect tuple.
* The RTSP open path is unchanged and production-tuned: `CAP_FFMPEG`,
  `rtsp_transport=tcp`, `CAP_PROP_BUFFERSIZE=1`, open/read timeouts 5 s.

## 5. Files Changed

| File | Change |
|---|---|
| `src/domain.py` | `CameraSourceKind` enum, `CameraConfig` dataclass (`display()`/`safe_dict()`/`_rtsp_host()`, never expose creds), `CAM_DARK_BLANK_FRAME` constant, `infer()` aliases |
| `config.py` | `camera_configs()` reader for `CCTV_CAM_<NN>_{URL,NAME,TYPE,ENABLED,LOCATION,FPS,RECONNECT_BASE/MAX/FACTOR,USERNAME,PASSWORD}`; module-level `CAMERA_CONFIGS`; split-credential folding |
| `src/camera.py` | `kind`, `camera_id`, per-camera `reconnect` policy, `_reconnect_params()`, resolution tracking, health_info gains id/kind/resolution/reconnects + redaction |
| `src/camera_manager.py` | accepts `configs` (dict or `CameraConfig`), skips disabled cameras, `_cfg()` normalization, keeps legacy `cameras` dict path |
| `main.py` | passes `config.CAMERA_CONFIGS` into `MultiCameraManager` (pool mode), uses `CAM_DARK_BLANK_FRAME` |
| `app.py` | camera cards show Type/Resolution/Reconnects; `CAM_DARK_BLANK_FRAME` constant; never credentials |
| `docker-compose.yml` | per-camera `_USERNAME`/`_PASSWORD` env pass-through for all 10 slots |
| `.env.example` | full per-camera configuration docs + example |
| `tests/test_phase58.py` | **NEW** — 18+ tests (TEST 1–18), SIMULATED/OFFLINE |

## 6. Configuration

* New per-slot env vars (URL unchanged; metadata optional):
  `CCTV_CAM_<NN>_URL` (existing), `NAME`, `TYPE`, `ENABLED`, `LOCATION`,
  `FPS`, `RECONNECT_BASE`, `RECONNECT_MAX`, `RECONNECT_FACTOR`,
  `USERNAME`, `PASSWORD`.
* Reconnect defaults preserve the legacy curve: base 2.0 s, max 30.0 s,
  factor 2.5; overridable per camera and hard-bounded by `_reconnect_params`
  (base ≥ 0.1 s, max ≥ base, factor ≥ 1.01).
* Empty slots are excluded; `ENABLED=0` excludes a configured slot; unknown
  `TYPE` falls back to inference (safe). Malformed URLs never crash the
  loader (TEST 2).
* `CameraConfig` is a frozen dataclass. Its only outward shapes,
  `display()` and `safe_dict()`, contain no credentials (TEST 4).

## 7. Security Model

* **Never store/log the authenticated URL.** Credentials may be supplied
  either inline in the URL (existing behaviour, redacted at every boundary)
  or via split `_USERNAME`/`_PASSWORD` env vars which are folded into the URL
  only at open time (`camera_configs()`), never written to
  `safe_dict()`, files, logs, reports or the dashboard.
* Redaction is enforced in `camera.health_info()` and
  `camera_manager.camera_health()` (`redact_url`), and in the dashboard via
  `_safe_cam_source`.
* Tests lock this in: TEST 3 (failure logs), TEST 4 (config shapes),
  TEST 18 (config + health + manager output). `grep`-level assertions ensure
  `***` redaction is present.

## 8. Reconnect Strategy

* Bounded exponential backoff, per-camera configurable, hard-capped at
  `CAMERA config.reconnect_max` (default 30 s). Sequence plateaus at the cap
  (TEST 10). Sleep uses small increments so `stop()` stays responsive.
* Repeated failures cannot busy-spin: paced by real sleeps, `reconnect_count`
  stays within a small factor of wall-clock/backoff (TEST 11), and failure
  never flips health to ONLINE (TEST 5 → CAMERA_OFFLINE / RECONNECTING).

## 9. Frame Health

* Health states unchanged plus `CAM_DARK_BLANK_FRAME` now a first-class
  domain constant: ONLINE, LOW_FPS, FROZEN_FRAME, NO_FRAME, RECONNECTING,
  OFFLINE, DARK_BLANK_FRAME (gate reason in `_usable_camera_frames`).
* Frozen feeds (same content ≥ 30 frames / 6 s) and low FPS stay usable
  (person-present monitoring), while dark/blank/offline/no-frame feeds are
  gated out and **never drive employee state** (TEST 7, 8, 9).

## 10. FPS / Latency

* Per-camera rolling 1 s FPS window; resolution captured on first frame.
* AI processing FPS is pipeline-controlled downstream (`frame_interval`,
  `TARGET_FPS_PER_CAMERA`); **not measured against a real RTSP feed** this
  phase (see §13).
* `health_info` now exposes `fps`, `frames_read`, `last_frame`,
  `reconnects`, `width`/`height`, `resolution` for the HUD/live-state.

## 11. Multi-Camera Readiness

* Pool is per-camera thread; `latest_frames(max_age)` filters stale frames;
  `get_latest_batch` bounded by `max_batch`. One dead camera cannot stall or
  starve the others (TEST 15/16). Manager supports configs dict with disabled
  camera exclusion and per-camera reconnect/kind plumbing (keeps legacy
  `cameras` path).
* Ready for 1–10 cameras; no cap regression.

## 12. Docker Readiness

* `docker-compose.yml` passes through `CCTV_CAM_<NN>_URL` plus per-camera
  `_USERNAME`/`_PASSWORD` for all 10 slots; daemon + dashboard share the
  data volume; RTSP ports are not exposed to the host. No Docker *runtime*
  validation was executed this phase (environment limitation) — Dockerfiles
  are unchanged and the changed surface is env-only.

## 13. Tests

* **New:** `tests/test_phase58.py` — 18 requirements (TEST 1–18) expressed as
  20 test functions (TEST 14/17 split into `14`+`14b`, `17`+`17b`, plus an
  import-sanity check). All SIMULATED/OFFLINE (fake `cv2.VideoCapture`,
  stubbed manager); **no real camera required**.
* **Matrix pass (selected suites):** camera-failure, camera-health-realfeed,
  camera-state, multiperson-pipeline (dark/frozen/offline semantics),
  Phase 50/51/54/57, main, dashboard-phase42, webcam-mode, phase A/B,
  security-engine — **292 passed**.
* **Full regression:** `pytest -q` **1113 passed**; `pytest -q -W error`
  **1113 passed**. Baseline was 1093; +20 from Phase 58.

## 14. Real RTSP Validation

* Attempted (bounded) public `rtsp://` streams and a client-style
  `rtsp://user:CHANGE_ME@…` placeholder to confirm transport: **both
  unreachable from this environment** (hostname resolution failed / no route;
  probe bounded to ≤ 30 s, FAILED_TO_OPEN).
* No client IP-CCTV or NVR feed was available. Per the brief, public "open
  test streams" do **not** count as real validation, and this environment
  could not even reach them.
* **Classification: `REAL_RTSP_VALIDATION = NOT_VALIDATED`.**
  `REAL_CAMERA_MODEL = NOT_AVAILABLE`; `REAL_RTSP_FPS = NOT_VALIDATED`;
  `AI_PROCESSING_FPS = NOT_VALIDATED`.
* The RTSP layer's *protocol behaviour* is verified only via simulated mocks;
  transport under a real NVR remains the top on-site acceptance item.

## 15. Performance

* No real-feed AI latency measurement is possible without a live RTSP/NVR
  source (see §14). Capture-thread overhead added this phase is negligible
  (two extra dict keys in `health_info`; one `extra` lookup per frame for
  resolution). No new per-character synchronous work was added to the hot
  loop beyond reading `frame.shape`.

## 16. Limitations

* No real RTSP/NVR/CCTV feed reachable → genuine transport validation has
  not happened (simulated only).
* Docker runtime environment not re-executed (env-only change, unchanged
  images).
* No desk/seat tracking (Phase 59 scope) — deliberately NOT implemented here.
* Camera *model* (make/NVR tenant) is intentionally configuration, not code;
  per-vendor transport quirks will surface only under a real feed.

## 17. Remaining Production Blockers

1. **Real RTSP/NVR feed validation** — must be run against actual client
   cameras before `PRODUCTION_READINESS` can move off `NOT_READY`.
2. On-site credential ingestion (correct `_USERNAME`/`_PASSWORD` / URL form)
   and verified per-camera reconnect under real network variance.
3. Phase 59: desk/seat zone tracking and employee seat mapping (next phase).

## 18. Security & Cleanliness

* Phase 58 made **no commits and no pushes**; HEAD remains `c2a80c9`.
* Phase 57 uncommitted deliverables (`src/face_registry.py`,
  `tests/test_phase57.py`, Phase 57 report) are intact.
* Credential hygiene verified by tests and by redaction at every outward
  boundary (config, camera, manager, dashboard).

## 19. Phase 58 Output (final block — required do-not-push rule)

```
PHASE: 58
STATUS: PASS_WITH_LIMITATIONS
RTSP_IMPLEMENTATION: COMPLETE
LOCAL_CAMERA_REGRESSION: PASS
RTSP_REAL_VALIDATION: NOT_VALIDATED
NVR_SUPPORT: CODE_READY
RECONNECT: VERIFIED
FRAME_HEALTH: VERIFIED
MULTI_CAMERA_READINESS: VERIFIED
SECURITY_CREDENTIAL_HANDLING: PASS
FULL_TESTS: 1113 passed
WARNINGS_AS_ERRORS: 1113 passed
REAL_CAMERA_MODEL: NOT_AVAILABLE
REAL_RTSP_FPS: NOT_VALIDATED
AI_PROCESSING_FPS: NOT_VALIDATED
PRODUCTION_READINESS: NOT_READY
NEXT_PHASE: 59 — Desk/Seat Zone Tracking and Employee Seat Mapping
```

## 20. Phase 59 Recommendation

* Implement desk/seat zone tracking and employee seat mapping.
* Reserve a mandatory first job for Phase 59: point the pool at at least one
  **real** client RTSP/NVR CI feed to close the Phase 58 `NOT_VALIDATED`
  item, then re-run the Phase 58 camera suites against it.
* Keep the `CameraConfig` model as the single on-site configuration surface;
  add per-seat/desk zones as a separate configuration class to avoid
  entangling camera-layer concerns with productivity decisioning.