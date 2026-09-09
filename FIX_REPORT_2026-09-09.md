# FIX REPORT — "Only One Employee" + False AWAY Incident

**Date:** 2026-09-09 · **Project:** AI Office CCTV Intelligence & Security Platform (`cctv monitoring\`)
**Status:** All 20 phases executed · full test suite green · real-camera multi-person (known+unknown) verified live · FROZEN_FRAME classification validated on real webcam pixels

---

## 1. Incident

- Dashboard showed: `Camera is not delivering frames: local_webcam`
- Only one employee ever appeared in live state / live table.
- Employee states were being driven to **AWAY** even though the camera was effectively dead — fabricated absence, not observed absence.

## 2. Root causes (confirmed by bounded probe + log forensics)

| # | Cause | Evidence |
|---|---|---|
| 1 | **Daemon pinned to the dead camera.** `.env` had `CCTV_CAMERA_MODE=INDEX:1`. Today index 1 is a dark/stuck feed; the live webcam is index 0. | `probe_cam2.py`: index 0 live (mean 142.7, 640×480, 30 fps), index 1 black (mean 4.1), 2–5 not open. Logs: `Stream opened: 1`; dark watchdog `mean 0.5 < 12.0 for ~15 frames`. |
| 2 | **Black frames were still fed to detection.** The watchdog only logged; the run loop kept processing the black feed → tracker created → EMP001 driven AWAY. | log 12:33:14 EMP001 tracker, then AWAY |
| 3 | **Security `is_online` inversion + `now=0` clock bug.** `is_online` treated `NO_FRAME` as online / `FROZEN` as offline; `CameraStateManager.update` used `now = now or time.time()` so a `now=0` never applied → `CAMERA_OFFLINE` never fired for the dead feed. | `CAMERA_OFFLINE` fired only 12:08/12:33 but not for the pinned run |
| 4 | **Face→person pairing order-sensitivity.** Face top-left + first-list-match could grab the wrong person in multi-person frames. | detector.py inspection |
| 5 | **Dashboard had no Unknown-person row** and no way to see per-camera usability/frame mean. | app.py inspection |

Device indices are unstable across reboots / DroidCam reconnects, so a hardcoded `INDEX` is the hazard; AUTO + usable-frame gating is the fix.

## 3. Fixes shipped

| File | Change |
|---|---|
| `.env` | `CCTV_CAMERA_MODE=INDEX:1 → AUTO` (auto-pick the live device at startup) |
| `main.py` | `_usable_camera_frames()` — frames from `{ONLINE,LOW_FPS,FROZEN}` health that are bright (mean ≥ `BLACK_FRAME_MEAN`) keep running; `OFFLINE/NO_FRAME/RECONNECTING/DARK_BLANK_FRAME` ⇒ `camera_online = False` → **all employee states freeze; no AWAY can accrue**. Health snapshot reused for loop, security, metrics, live state. Structured `CAMERA_DEBUG` lines. `_write_live_state` gains top-level `last_seen`. |
| `src/security_engine.py` | `is_online = health in ("ONLINE","LOW_FPS")` — FROZEN/NO_FRAME now drive the `CAMERA_OFFLINE` failsafe (states freeze, never AWAY). |
| `src/camera_state.py` | `now = now if now is not None else time.time()` — fixes the falsy-`0.0` bug that suppressed offline triggering. Regression test: `update(camera_id, online, now=0.0)` uses the provided clock and the 15-s trigger fires at t=16. |
| `src/detector.py` | `_containing_person`: face-centre contained in person box (raw or 1.5× expanded) + **nearest person centroid** wins — never first-match. |
| `app.py` | Live table **Unknown person row**; dead/blank camera → warning "**never counts as AWAY … states frozen**" vs FROZEN-but-flowing → info "Monitoring continues"; camera-health tab **Frame Age + Usable/Reason**; per-camera health detail expander. |
| `config.py` | `CAMERA_DEBUG` (env `CCTV_CAMERA_DEBUG`). |
| `tests/test_multiperson_pipeline.py` | **New — 30 regression tests** (N-faces → N identities, per-person tracking independence, camera-offline freeze/failsafe incl. the `now=0` clock bug, usable-frame gating, dashboard dead-vs-frozen rows, Unknown excluded from roster). |

## 4. Verification

- **Full suite green (744/744 pytest pass)** — 730 baseline + 14 new (30 multi-person + real-feed camera-health in `tests/test_camera_health_realfeed.py`; new files run with `-W error`). Touched suites (tracker, security, webcam-mode, pipeline-fix, camera-failure, virtual-camera, soak, Phase A/B, 38–40) green — no regressions.
- **Real-camera session (50 s, `--source auto`)**: `Stream opened: 0`; `CAMERA_DEBUG … resolved_index=0 frame_received=True 640×480 frame_age=0.07 input_fps=2.98 processing_fps=0.84 camera_online=True usable=True`; `live_state` showed **EMP001 + Unknown ACTIVE simultaneously** (real known+unknown multi-person), EMP001 `away_sec=0.0`, EMP002/003/004 honest `NOT_OBSERVED`, `frame_mean=125.74 usable=true`, **no CAMERA_OFFLINE, no dark-feed warning**, spatial track `local_webcam#1` present.

## 5. Honest remaining gaps (unchanged requirements)

- **ON_PHONE** — never exercised on a real handset (simulator/tests only).
- **Two-roster-employee simultaneous recognition** — EMP002/EMP003 have no enrolled face (`NO_FACE`); needs real enrollment photos. Known+unknown multi-person is now live-verified.
- **DroidCam / RTSP / IP / NVR**, GPU/CUDA, SMTP, Docker, multi-hour overnight soak, physical intrusion test — environment limitations, not executed.
- **FROZEN_FRAME** — classification now exercised on **real webcam pixels** (`tests/test_camera_health_realfeed.py`: signature, streak accumulation, no-false-positive ONLINE, sustained-FROZEN, sub-min-seconds no-FROZEN); a genuine 6-second physical lens freeze end-to-end via the daemon is still not exercised.
- No thresholds were weakened (face 0.60, conf 0.40 untouched). No output-weakening. No fake enrollment data.

## 6. Bottom line

The false-AWAY mechanism is eliminated (dead/blank camera now freezes state instead of fabricating absence, and AUTO guarantees a live device is chosen). Multi-person behaviour is validated for known+unknown on the real feed with independent per-person tracking. The platform is GO WITH CONDITIONS — the conditions above (§5) require real phone + enrollment data before unattended deployment.