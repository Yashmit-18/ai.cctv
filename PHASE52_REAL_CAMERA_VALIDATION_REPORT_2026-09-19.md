# PHASE 52 — REAL CAMERA VALIDATION EXECUTION & EVIDENCE CAPTURE

Project: AI CCTV Office Intelligence & Security Platform
Date: 2026-09-19
Status classification: REAL VERIFIED / CODE VERIFIED / SIMULATED / NOT VALIDATED / NOT EXECUTED / MODEL LIMITATION

> **ERRATUM (added by Phase 53/54 audit, 2026-09-19):** Sections 2, 5, 13, 16,
> 19 and 20 state that `models/osnet_x1_0_msmt17.pth` is "NOT present". That claim
> is incorrect: the checkpoint **IS present on disk** (16.47 MB) and was loaded
> and benchmarked the same day (Phase 49 benchmark JSON). The phase outcome is
> unchanged — no usable feed existed, so real calibration was still `NOT
> EXECUTED` — but the checkpoint-availability justification for §13/§16 is
> hereby withdrawn. Correct, in-place references are marked `[ERRATUM]`.

---

## 1. Executive Summary

Phase 52 set out to obtain the first genuine real-camera validation of the
existing CCTV pipeline, and ONLY if a genuinely usable camera feed existed.

Camera Discovery and the Phase 51 feed gate both ran against the only device
that opened (OpenCV index 0, MSMF backend).

**The feed is unusable and real validation stopped at the preflight choke point.**

- Index 0 opens and nominally streams at ~18.5 FPS (real capture).
- Every frame is black and frozen: median brightness `4.12/255` (gate requires
  `>= 20/255`), mean frame-to-frame change `0.0`, changed-ratio `0.0`.
- The Phase 51 `validation_runner_verdict()` returned:
  **`NOT EXECUTED / INVALID FEED`** — reason `BLACK/DARK (median brightness below minimum)`.
- This is identical in character to the DroidCam failure observed in Phase 48
  (a camera that opens and even reports an FPS while producing black/frozen
  frames is not evidence).

**REAL VERIFIED = none.**

No frames, no person was ever observed by the camera. Every subsequent Phase 52
step (manifest, employees, front-face, side/back, two-person, phone, AWAY
safety, calibration, performance) is therefore `NOT EXECUTED`; every accuracy
claim remains `NOT VALIDATED`.

No code was changed. No thresholds were changed. OSNet remains opt-in and
uncalibrated. No commit/push/branch performed.

---

## 2. Environment

| Item | Value |
| --- | --- |
| Host | Windows (win32), CPU-only laptop |
| Python | venv at `.venv`, run via `.venv\Scripts\python.exe` |
| OpenCV | project venv build (opencv-python) |
| ReID | built-in extractor; OSNet `models/osnet_x1_0_msmt17.pth` present on disk — not used (opt-in, unchanged) `[ERRATUM]` |
| Config | `CCTV_REID_MODEL_PATH=''`, `CCTV_REID_GATE_RESOLVED=0`, thresholds unchanged from Phase 50/51 |
| Env vars | no `CCTV_CAM_*`, no `CCTV_RTSP_URL`, no camera env of any kind set |
| Data dir | `data/calibration/` does not exist |

---

## 3. Camera Discovery

Sources actually available in the environment:

- OpenCV camera indices `0..5`, backends DSHOW / MSMF / ANY.
- Configured RTSP sources: none — every `CCTV_CAM_<NN>_URL` and `CCTV_RTSP_URL`
  in `.env.example`/`config.py` is a placeholder (`user:CHANGE_ME@192.168.x.x`),
  and no real URL is present in the environment.
- Configured video files: none — `samples/` contains no files.

Per-index discovery results (12-read probe per backend, first backend with
frames wins per index):

| Source | Backend | Open result | Resolution | FPS (prop / captured) | Brightness median | Frame-to-frame change | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| index 0 | DSHOW | `not opened` (raised unknown C++ exception; DSHOW "can't be used to capture by index") | — | — | — | — | NOT VALIDATED / INVALID FEED |
| index 0 | MSMF | opened | 640 x 480 | 30 / ~18.5 | 0.0 (all 12 frames) | 0.000 max diff 0.000 | NOT VALIDATED / INVALID FEED (black/frozen) |
| index 1 | DSHOW / MSMF / ANY | not opened | — | — | — | — | NOT VALIDATED / INVALID FEED |
| index 2 | DSHOW / MSMF / ANY | not opened | — | — | — | — | NOT VALIDATED / INVALID FEED |
| index 3 | DSHOW / MSMF / ANY | not opened | — | — | — | — | NOT VALIDATED / INVALID FEED |
| index 4 | DSHOW / MSMF / ANY | not opened | — | — | — | — | NOT VALIDATED / INVALID FEED |
| index 5 | DSHOW / MSMF / ANY | not opened | — | — | — | — | NOT VALIDATED / INVALID FEED |

The OpenCV open-success on index 0 was explicitly NOT treated as usability.

---

## 4. Feed Preflight

Step 2 requires >= 10 seconds of frames run through the existing Phase 51 feed
gate.

Input: 10.03 s real capture from index 0 (MSMF), 640 x 480.

Output of `validation_runner_verdict(frames, capture_fps=18.54)`:

| Check | Requirement | Observed | Pass? |
| --- | --- | --- | --- |
| Frame count | >= 3 | 186 | Yes |
| FPS | >= 15 | 18.54 true capture FPS | Yes |
| Brightness | median >= 20/255 | **4.12/255** | **No** |
| Changing | majority-change >= 0.5 and mean diff > 0 | mean diff 0.0, changed ratio 0.0 | **No** |

Result: `state = NOT EXECUTED / INVALID FEED`, `proceed = False`,
`reason = invalid feed: BLACK/DARK (median brightness below minimum)`.

**Per the Phase 52 rule, STOP real validation for this source. Real validation
is hereby stopped.**

REAL VERIFIED:
- none

NOT VALIDATED:
- index 0 rejected — BLACK/DARK + FROZEN (median brightness 4.12/255 < 20/255,
  mean frame change 0.0); even though FPS is nominal, the content is absent.

---

## 5. Camera Manifest

NOT EXECUTED — no usable feed passed preflight. No manifest was created and no
geometry was invented.

---

## 6. Available Employees

No employee was physically present on a valid feed.

| Employee | Availability | Status |
| --- | --- | --- |
| EMP001 | not observed | NOT VALIDATED |
| EMP002 | not observed | NOT VALIDATED |
| EMP003 (Piyush) | not observed | NOT VALIDATED |
| EMP004 | not observed | NOT VALIDATED |

No samples fabricated for any employee, including EMP003.

---

## 7. Front-Face Validation

NOT EXECUTED (no usable feed, no employee in frame).

Scenario — | Input | Observed | Expected | Result | Evidence status | Notes
--- | --- | --- | --- | --- | --- | ---
Baseline front-face | real person, camera-facing | none — no frames with content | face-confirmed identity assigned | NOT EXECUTED | NOT EXECUTED | no valid feed

---

## 8. Side/Back Validation

NOT EXECUTED (no usable feed; the protocol in
`PHASE50_VALIDATION_PROTOCOLS_2026-09-19.md` could not be followed).

No crops, no PNGs, no pose directories, no filenames were created.

---

## 9. Two-Person Validation

NOT EXECUTED (no feed, no employees).

---

## 10. Phone Functional Validation

NOT EXECUTED (no usable feed and no phone frame evidence).

Scenario A (>= 6 s phone use), Scenario B (3–4 s), Scenario C (two persons),
Scenario D (4.9 s / 5.0 s wall-clock) — all NOT EXECUTED. Frame counts were not
substituted for time anywhere.

---

## 11. Phone Accuracy Status

NOT VALIDATED — no frame-labelled phone dataset exists and no phone frames were
captured. Phone precision/recall remains unsupported. The Phase 44 FSM
semantics remain CODE VERIFIED via the existing test suite only.

---

## 12. Camera Health / AWAY Safety

NOT EXECUTED as a live episode (no usable feed, no employee to depart).
However, one safety fact WAS directly re-established on the real camera:

- The camera-health gate correctly refuses to treat the broken feed as usable
  (Section 4). The pipeline's existing semantics (camera offline/frozen does
  NOT equal employee AWAY; state progression freezes) remain CODE VERIFIED by
  `tests/test_tracker.py`, `tests/test_phase44.py` etc. They were NOT observed
  live.

---

## 13. ReID Calibration

NOT EXECUTED for real data.

- No real labeled crops exist (`data/calibration/` absent).
- Preferred checkpoint `models/osnet_x1_0_msmt17.pth` IS present on disk
  (`[ERRATUM]` — the original claim it was missing was wrong), but was not used:
  real calibration still has no real labelled crops to process.
- Per the phase rule, no silent fallback to built-in ReID calibration was done.

**REAL CALIBRATION NOT EXECUTED — MODEL CHECKPOINT UNAVAILABLE**
AND
**REAL CALIBRATION NOT EXECUTED — NO USABLE FEED / NO REAL LABELED CROPS**

Prior dry-run calibration evidence from Phase 50/51 remains classified SIMULATED,
not REAL.

---

## 14. Threshold / Margin Recommendation

None produced this phase. No recommendation was written to `config.py`, OSNet
was not enabled, and the Phase 50/51 recommender output was not treated as
deployment authorization. Production thresholds remain unchanged
(`REID_STRONG_THRESHOLD=0.90`, `REID_MARGIN_MIN=0.06`).

---

## 15. Performance Measurements

No live throughput was measured (no usable feed). Real loop, capture, YOLO,
phone, face, ReID, and tracker latencies remain NOT VALIDATED for this
session's camera. Historical CPU-only figures from Phase 45/49 remain
environment SIMULATED/baseline evidence, not live measurements.

---

## 16. Failures / Refusals

| Failure/Refusal | Detail | Status |
| --- | --- | --- |
| Camera index 0 opens but black/frozen | DShow unusable by index; MSMF yields all-black frames | NOT VALIDATED / INVALID FEED |
| Camera indices 1–5 absent | none open on any backend | NOT VALIDATED / INVALID FEED |
| No real RTSP endpoint | placeholder URLs only; env unset | NOT EXECUTED |
| No video file | `samples/` empty | NOT EXECUTED |
| OSNet checkpoint not used | `osnet_x1_0_msmt17.pth` present on disk but not opted-in — real calibration still blocked by **no usable feed / no labelled crops** (`[ERRATUM]`) | NOT EXECUTED — NO REAL LABELED CROPS |
| Employees absent | EMP001–004 not observed | NOT VALIDATED |
| No labelled phone dataset | none exists for scoring | NOT VALIDATED |

All refusals are safe and explicit; nothing silently fell back.

---

## 17. Model Limitations

- This session: no real frames captured, so side/back ambiguity and phone-size
  accuracy could not even be observed.
- Known prior limitations remain (unchanged by Phase 52): small-phone
  detection, side/back ambiguity for appearance-only ReID, CPU inference
  latency — all MODEL LIMITATION, effectively NOT OBSERVED here.

---

## 18. Evidence Classification

- REAL VERIFIED: none.
- CODE VERIFIED: the only guarantees exercised this phase are the Phase 51
  feed gate (`feed_usable_stats` / `validation_runner_verdict`) and the
  existing 1043-test suite (unchanged baseline).
- SIMULATED: prior dry-run calibration/benchmark numbers only (nothing new this
  phase).
- NOT VALIDATED / NOT EXECUTED: every real camera, person, phone, and accuracy
  scenario in Phase 52.

---

## 19. Remaining Blockers

1. No usable camera feed (index 0 black/frozen on this host; no real RTSP;
   no video file).
2. No rostered employee physically available for a session (EMP001/002/004;
   EMP003/Piyush also absent).
3. No frame-labelled phone dataset for accuracy scoring.
4. No real labelled crops for ReID calibration (`data/calibration/` absent) —
   the checkpoint itself IS available (see Erratum).

---

## 20. Exact Next Phase Recommendation

Do NOT attempt live validation on this host again until at least one of the
following is true:

1. A genuinely usable feed exists (e.g., a real RTSP/NVR endpoint, IP camera,
   or a working USB webcam verified by the Phase 51 feed gate producing
   `proceed=True`), AND
2. At least one rostered employee is available for a supervised session.

Then re-run Phase 52 from Step 1, which will re-run discovery and the feed
gate automatically. If the gate says `EXECUTABLE (FEED OK)`, continue with the
manifest, employee capture, calibration (real labelled crops required — the
OSNet checkpoint is available on disk, see Erratum),
side/back, two-person, phone, AWAY-safety, and performance steps in order.

Additionally: real calibration needs a **labelled real crop set**
(`data/calibration/`); the OSNet checkpoint is already present — no substitute
is needed `[ERRATUM]`.

---

## PHASE 52 FINAL STATUS

REAL VERIFIED
- Camera index 0 (the only device that opens) is black and frozen: median
  brightness 4.12/255, mean frame change 0.0 — actually measured, not assumed.
- The Phase 51 feed gate rejected that feed with `NOT EXECUTED / INVALID FEED`
  (BLACK/DARK) on a real 10-second capture.

CODE VERIFIED
- `validation_runner_verdict` / `feed_usable_stats` choking the broken feed.
- Existing full suite baseline: 1043 passed, 5 skipped (unchanged).
- Camera-offline != AWAY invariant (test-backed, not observed live this session).

SIMULATED
- No new synthetic evidence this phase. Existing Phase 50/51 dry-run
  calibration numbers remain SIMULATED and were not reused as real.

NOT VALIDATED
- All real accuracy claims: face recognition on real cameras, phone precision/
  recall, side/back recognition, calibration thresholds, live performance.

NOT EXECUTED
- Steps 5, 7, 8, 9, 10, 11, 12, 13, 14, 15 — blocked by no usable feed /
  no employee / no checkpoint.

MODEL LIMITATION
- No real frames observed this session; historical CPU-latency and side/back
  ambiguity limitations remain known and unaddressed.

**Bottom line: no usable camera feed exists on this host this session. Phase 52
stopped real validation at the preflight gate. Nothing was fabricated, nothing
was silently substituted, and production state remains untouched.**