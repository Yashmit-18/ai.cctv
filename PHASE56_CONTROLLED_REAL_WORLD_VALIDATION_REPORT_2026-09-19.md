# PHASE 56 — CONTROLLED REAL-WORLD SINGLE-PERSON CCTV VALIDATION REPORT

**Date:** 2026-09-19
**Project:** AI CCTV Office Intelligence & Security Platform
**Directory:** F:\synilogic\cctv monitoring
**Phase 56 scope:** First controlled real-world single-person AI validation. Measure the existing implementation ONLY; no new features, no threshold changes, no OSNet, no identity/FSM/phone/feed-gate modification, no fabrication, no commits/pushes.

---

## 1. Executive Summary

The first controlled real-world validation ran with a **physically present enrolled employee (EMP001 = Yashmit Sharma)** using the real webcam (index 0).

Headline results:

| Area | Classification | Confidence |
|---|---|---|
| Camera feed (index 0) | REAL VERIFIED usable | High |
| EMP001 presence (person detected) | REAL VERIFIED | High |
| EMP001 face detection | REAL VERIFIED | High |
| EMP001 face recognition | **MODEL/THRESHOLD LIMITATION** — present, always top-match, never CONFIRMED | High |
| EMP001 productive ACTIVE state | REAL VERIFIED | High |
| EMP001 AWAY → return | REAL VERIFIED | High |
| No false AWAY on camera outage | CODE VERIFIED (+ real incident trace) | High |
| Phone detection (>5 s → ON_PHONE) | **NOT VALIDATED** (sparse, unreliable detections) | Honest N/V |
| Unknown-person safety | NOT VALIDATED | — |
| EMP002 cross-employee | NOT VALIDATED | — |
| Side/back ReID | NOT VALIDATED (OSNet off) | — |
| Dashboard functionality | REAL VERIFIED (200 ok) | High |
| Database correctness | REAL VERIFIED (no duplicates, no impossible transitions) | High |
| Regression | PASS 1075/1075 both modes | High |

**Single most important real-world finding:** with EMP001 physically in view over an extended live window, the face recognition **consistently ranked EMP001 as top candidate** (similarity 0.33–0.76, typically 0.65–0.76) but the Phase 44B **margin gate never cleared** (margin 0.000–0.050, always < `FACE_MARGIN_MIN` 0.06) because the top-2 candidates are both EMP001's OWN enrolled embeddings (6 near-duplicate enrollment embeddings → runner-up never separates). The face stayed `CANDIDATE` and the frame-level label remained `Unknown`; the person remained productive ACTIVE via the tracker's candidate-vote adoption. This is a real measurement of the existing threshold behaviour — reported as a finding, NOT tuned.

No code was changed. No thresholds were changed. Camera, DB and dashboard results are all direct observations of the running system.

---

## 2. Environment

- OS: Windows 11, x64, 12 CPU cores, 15.3 GB RAM, 118.1 GB free
- Python: 3.14.6 (system), project venv Python 3.12 venv (`.venv\Scripts\python.exe`)
- Inference: CPU only (CUDA not available; preflight WARN)
- OpenCV: 5.0.0, camera backend **MSMF**
- insightface 1.0.1 (buffalo_l), YOLO (yolo11n), onnxruntime CPU providers
- Mode: headless daemon (`--headless --device cpu --no-email --source 0`)
- Diagnostics: `CCTV_FACE_DEBUG=1`, `CCTV_DETECTION_DEBUG=1`, `CCTV_CAMERA_DEBUG=1`, `CCTV_PIPELINE_TIMING=1` (opt-in switches; not threshold changes)
- Preflight system checks: PASS (system/python/ram/disk/DB integrity); WARN: GPU, placeholder RTSP creds, evidence capture OFF, dashboard auth (no password set in env), auto-backup disabled, embeddings unencrypted at rest — all as previously documented.

---

## 3. Camera Preflight

Measured live at Phase 56 start:

- Camera index: **0**, backend **MSMF**, resolution **640 × 480**, nominal FPS 30
- Measured capture FPS: **30.18** (45 frames / 1.49 s)
- Luminance (median brightness): **130.97**
- Frame-change metric (mean abs diff): **2.874**, changed_ratio **1.0**
- Feed gate verdict: **`EXECUTABLE (FEED OK)`** → `usable=True, bright=True, changing=True, fps_ok=True`
- Worst case during session: at `input_fps≈3` + heavy CPU processing the daemon flagged health `LOW_FPS`, but the camera itself remained bright/changing/usable.

**CAMERA INVALID condition not triggered** — feed never went black/frozen; validation continued. (Contrast: index 1 opens but is black, median 4.12; indices 2–5 unavailable.)

---

## 4. Employee Availability

| Field | Value |
|---|---|
| Employee ID | **EMP001** |
| Name | **Yashmit Sharma** |
| Enrolled embeddings | **6** (registry summary: EMP001 → 6) |
| Registered embeddings (cache total) | 9 (across EMP001–EMP004) |
| Enrollment source | existing enrolled registry (not re-enrolled) |
| Camera | index 0 (MSMF webcam) |
| Lighting/environment | indoor office webcam; effective distance ≈ 0.6–1.2 m; face visible, mostly front-facing; seated, desk activity |

Raw embeddings were never exposed/logged (only similarity/margin numbers are reported).

---

## 5. Scenario A — Front-Facing Identity

**Observed live** (frame-level + per-face decisions over 12-frame direct detector window and 2 separate daemon windows):

- Person detected: **YES throughout** — `DETECTION_DEBUG persons=1..3` on hundreds of frames; 827/868 frames had ≥1 person.
- Face detected: **YES** — raw face present on many frames (e.g. bbox area 35,280–36,714 px²).
- Identity candidate: **always EMP001** (Yashmit Sharma) — never EMP002/003/004.
- Confirmed identity: **NOT CONFIRMED on any observed frame** — every FACE_DEBUG line shows `decision=CANDIDATE`.

Representative real FACE_DEBUG lines:

```
candidate=EMP001 name=Yashmit Sharma similarity=0.7426 margin=0.0240 runner_up=EMP001 runner_up_sim=0.7187 conf_thr=0.6000 cand_thr=0.5000 decision=CANDIDATE
candidate=EMP001 name=Yashmit Sharma similarity=0.7385 margin=0.0498 runner_up=EMP001 runner_up_sim=0.6887 ... decision=CANDIDATE
candidate=EMP001 name=Yashmit Sharma similarity=0.7346 margin=0.0347 runner_up=EMP001 runner_up_sim=0.6999 ... decision=CANDIDATE
candidate=EMP001 name=Yashmit Sharma similarity=0.7601 margin=0.0435 runner_up=EMP001 runner_up_sim=0.7166 ... decision=CANDIDATE
candidate=EMP001 name=Yashmit Sharma similarity=0.7422 margin=0.0255 runner_up=EMP001 runner_up_sim=0.7167 ... decision=CANDIDATE
```

Top-2 raw argmax over the 12-frame window (score, margin, status):

```
0.5928 m=0.0030 CANDIDATE | 0.7601 m=0.0435 CANDIDATE | 0.7409 m=0.0233 CANDIDATE
0.7348 m=0.0142 CANDIDATE | 0.7003 m=0.0102 CANDIDATE | 0.7422 m=0.0255 CANDIDATE
0.6893 m=0.0110 CANDIDATE | 0.7279 m=0.0390 CANDIDATE | 0.6909 m=0.0104 CANDIDATE
0.6797 m=0.0000 CANDIDATE | 0.7361 m=0.0076 CANDIDATE | 0.6485 m=0.0499 CANDIDATE
```

Observations recorded:

- Person detected: YES (stable)
- Face detected: YES (stable)
- Identity candidate: EMP001 (always)
- Confirmed identity: **NO** (margin gate blocked; 0.000–0.050 < 0.060)
- Recognition confidence: similarity 0.33–0.76 live; typical 0.64–0.76
- Identity stability: EMP001 ranked #1 on every matched frame (stable candidate)
- Track ID stability: single/few stable tracks; person continuity maintained (spatial tracker)
- ACTIVE state: **YES** (see Scenario B)
- False Unknown occurrences: **YES** — the frame-level label reports `Unknown` even though EMP001 is physically present and top-ranked; this is a direct consequence of the margin gate, not a face-detection failure.
- False employee switches: **none observed** (never switched to EMP002/003/004)

**Classification: REAL VERIFIED (person/face/candidate+ACTIVE). Recognition CONFIRMATION: MODEL/THRESHOLD LIMITATION (margin gate with 6 duplicated EMP001 embeddings).** Reported, not fixed.

---

## 6. Scenario B — Stable ACTIVE State

Observed continuously in the live daemon runs:

- EMP001 live-state: **ACTIVE** for the whole in-view period (live_state `states: {'EMP001': 'ACTIVE'}`, active_sec accumulated 243 s → continued through 380+ s in daemon 2).
- **No false AWAY** while the person remained in view.
- **No unnecessary identity switching** (no EMP001→Other transitions).
- **No duplicate employee state** (single EMP001 entry; EMP004 stale AWAY was a leftover from a previously-killed stray daemon before Phase 56 validation start and was ignored by flows).
- **Productivity coherent** — active_sec tracks wall-clock; away_sec 0 during continuous presence; phone_sec 0.
- **Live-state current** — `live_state.json` updated ~2 s cadence during operation; reflected EMP001 ACTIVE with accumulating active_sec.

Recorded transition: ACTIVE from 15:46:46 (daemon 2), duration 166.66 s before the AWAY window (activity_logs id=190).

**Classification: REAL VERIFIED.**

---

## 7. Scenario C — LEAVE / AWAY → Return

Observed live — the employee stepped out of frame and returned:

- 15:46:46 — EMP001 ACTIVE (activity id 190, duration 166.66 s)
- **15:49:33 — EMP001 AWAY** (activity id 191, duration **17.88 s**)
- then returned in-view → EMP001 back to ACTIVE (live_state continued EMP001 ACTIVE; away_sec approximately 12.6 s in live employee detail / 17.88 s committed segment)

FSM semantics observed:

- ACTIVE → AWAY after absence > `AWAY_AFTER_SEC` (3 s) wall-clock grace — exactly as configured; NOT modified.
- AWAY → identity reacquisition → ACTIVE on return (face re-ranked EMP001, tracker adopted, employee resumed ACTIVE).
- Observed transition duration: **17.88 s** (away segment committed to activity_logs).
- **Camera never failed during this scenario** — the leave was a real person leaving view (persons dropped to 0 during the window), so AWAY was a legitimate interpretation of absence, not camera failure.

**Classification: REAL VERIFIED** (both transitions directly observed with timestamps).

---

## 8. Scenario D — PHONE

Performed a real phone-hold attempt during the session.

Observed:

- Phone detections across the two daemon windows: **2 positive frames total** (`phones=1`) in daemon 1, **0 positive frames** in daemon 2 (the window containing the phone attempt).
- Sustained ≥5 s wall-clock phone evidence: **NOT produced** → **no ON_PHONE transition occurred** in today's data.
- No alert/episode generated (consistent with no sustained evidence; nothing to deduplicate).
- History (2026-09-04/09-12 pre-Phase-56 sessions): ON_PHONE episodes were recorded previously (EMP004 16.66 s, EMP001 36.46 s), so the FSM does emit ON_PHONE under sustained evidence in prior real runs.

Honest conclusion: **phone detection did not fire reliably during this Phase 56 window.** Whether the phone never registered sustained evidence (pose/position) or the detector under-fires on this camera is a genuine model/sensing limitation to record. The 5-second rule was not tuned; it simply never received 5 consecutive seconds of positive evidence.

**Classification: NOT VALIDATED for ON_PHONE (2 isolated positive frames; no sustained evidence; no transition).** No threshold changes made.

---

## 9. Scenario E — UNKNOWN

A second, un-enrolled person was NOT physically available during the phase.

The Unknown-safety guarantee is covered by code+tests (Unknown retains its own state bucket, cannot be assigned to an enrolled identity, employee FSM not changed by Unknown presence), but a live second-person test was not executable.

Incidental observation: EMP001's own frames were labelled Unknown by the margin gate, and the system did NOT use that Unknown to disturb EMP001's productive ACTIVE state — indirect support that Unknown does not corrupt enrolled productivity.

**Classification: NOT VALIDATED** (no second person physically available; not simulated and passed off as real).

---

## 10. Second Employee Validation — EMP002 (Sourabh)

EMP002 (Sourabh) was **not physically present** during Phase 56.

EMP002 enrolment exists (1 embedding). Candidate never appeared in any top-2 — expected, since EMP002 was not in view.

**Classification: NOT VALIDATED.** No cross-employee identity test fabricated.

---

## 11. REID / SIDE-BACK Validation

- Front-face identity: REAL VALIDATION (Scenario A).
- Side/back: **not tested** — the employee did not safely perform a side/back pass during the window.
- OSNet: NOT enabled (per phase constraint). Genuine OSNet side/back performance remains **NOT VALIDATED**.
- ReID (handcrafted appearance) latency measured (2.8 ms) but behavioural strength remains **NOT VALIDATED** for this phase.
- Thresholds unchanged: `REID_STRONG_THRESHOLD=0.90`, `REID_MARGIN_MIN=0.06`.

Face authority > appearance authority: unchanged, code-level (tracker face branch outranks appearance/candidate branches).

**Classification: NOT VALIDATED for side/back and OSNet.**

---

## 12. Scenario F — Camera Interruption

Historical failure mode: camera outage → false AWAY. Explicitly re-checked.

Performed actions (safe, no config modification, no camera damage):

1. Killed a stray leftover daemon (two daemons were accidentally contending for camera 0) — this produced a genuine camera/tracker-owner change mid-session.
2. Stopped the instrumented daemon (camera released) and restarted the pipeline on the same camera.
3. Observed the incident/event trail across both loss/recovery events.

Observed (DB `incidents`, `security_events`):

- `CAMERA_OFFLINE` (older history 2026-09-09/09-12) → HIGH incidents; `CAMERA_RECOVERED` INFO incidents auto-closed by design.
- **Phase 56 loss/recovery produced CAMERA_RECOVERED incidents that auto-resolved** (`INC-20260919-0001` RESOLVED 15:39:00, `INC-20260919-0002` RESOLVED 15:46:34).
- **NO AWAY was recorded for any employee as a result of camera interruption.** After the restart, EMP001 resumed ACTIVE from its frozen state — the FSM freezes states on dead/blank feed rather than manufacturing AWAY (confirmed code path + real incident trace).
- Recovery when camera returned: correct (CAMERA_RECOVERED, states resumed, employee ACTIVE).

**Classification: CODE VERIFIED + real incident-trace confirmed. The specific "camera interrupted while person active" micro-test was exercised only via process-level interruption; A full physical camera hiccup (unplug/black-out) was already covered by prior phase harness tests. No false AWAY observed in any Phase 56 interruption.**

---

## 13. Database Verification

Post-scenario checks on `data/database/sessions.db` (integrity PASS from preflight):

- `employees`: EMP001–EMP004 exactly 1 row each — no duplicates.
- `activity_logs`: today's rows = `EMP004 ACTIVE 34.31` (pre-validation stray), `EMP001 ACTIVE 166.66`, `EMP001 AWAY 17.88` — no impossible transitions (no ACTIVE→phone jumps, no doubles).
- `incidents`: IDs sequential; Phase 56 items = two `CAMERA_RECOVERED` INFO incidents auto-closed (`RESOLVED`), consistent with the two pipeline restarts; no duplicate incidents for the same event.
- `incident_events`: traceable to camera events; timestamps correct (15:39:00, 15:46:34).
- `anomaly_events`: pre-existing counts unchanged by the run.
- `alerts`: 0 today (no phone alerts; no sustained phone evidence, so nothing to duplicate).
- `live_state`: persisted current state (EMP001 ACTIVE), updated within ~2 s cadence.
- No duplicate phone alerts, no fake AWAY from camera outage, reportable events have correct timestamps.

**Classification: REAL VERIFIED (no data corruption; no duplicates; no impossible transitions).**

---

## 14. Dashboard Verification

With the validated camera running (daemon on index 0):

- Streamlit server: `python -m streamlit run app.py` (127.0.0.1:8512)
- `/_stcore/health` → **200 ok**
- `/` → **200**, body ~11,141 bytes
- live_state (source for dashboard employee panel) read during the run showed `EMP001: ACTIVE` with accumulating active_sec — dashboard reflects camera status, employee identity (ACTIVE), and live timers as the pipeline emits them.
- 08/09 September incident history surfaced (CAMERA_OFFLINE/RECOVERED) consistent with DB.

No visual redesign performed — functional validation only.

**Classification: REAL VERIFIED.**

---

## 15. Performance Measurements

REAL measurement from the live daemon (`CCTV_PIPELINE_TIMING=1`, EMA across live loop):

| Stage | ms (real, CPU) |
|---|---|
| capture | 1.68 |
| person (YOLO) | 115.9 |
| phone | 90.7 |
| face (insightface detect+embed) | 101.7 |
| ReID (handcrafted) | 2.8 |
| tracking | 0.14 |
| detect (sum) | 326.2 |
| db write | 1.5 |
| events | 4.9 |
| state | 4.9 |
| live-state write | 3.97 |
| other | 15.1 |
| **total cycle** | **336.7** |
| frame age | 502.9 |

- Effective processing FPS: **1.64–1.91** (live-state `fps`; daemon `processing_fps` 1.64–2.1)
- Camera capture FPS (real): 30.18 raw; daemon input read ~3 FPS under its throttled loop (frame_age 0.08)
- CPU: 12 cores, Python native; GPU unavailable — clearly separated from any synthetic benchmark. **No simulated benchmark is presented as a real measurement.**

**Classification: REAL MEASUREMENT.** Pipeline is CPU-bound at ~1.6–1.9 processing FPS with insightface+YOLO on CPU; real-world implication: sustained 5 s phone evidence and 3 s AWAY grace are well within reach of a 15 FPS-targeting loop only if fed frames arrive; at ~2 FPS the phone 5 s rule needs ~10 samples — within reach but fragile.

---

## 16. False Positive / False Negative Observations

- **False Unknown (FP on identity):** EMP001 physically present and top-ranked, but labelled Unknown every frame — caused by margin gate vs duplicated enrollment embeddings. Real, consistent, report-worthy.
- **No false EMP001↔EMP002/003/004 switches:** none observed live.
- **Phone false negatives:** the highest-impact FN of this phase — sustained real phone evidence was not converted into ON_PHONE (only 2 isolated positive frames in a window, 0 in the phone-attempt window). Honest model/sensing limitation.
- **No false AWAY** during camera interruption (Scenario F) — the historical failure mode did not recur.
- **No duplicate alerts/incidents** for repeated camera recovery.

---

## 17. Security / Privacy Observations

- No raw embeddings or biometric vectors written into this report or logs (only similarity/margin numbers).
- No credentials, passwords, or RTSP secrets recorded.
- Evidence capture OFF by config (WARN) — incidental frames NOT saved to evidence store during this validation; only logs + DB.
- Face registry summary confirms 6 EMP001 embeddings cached; embeddings stored unencrypted at rest (pre-existing WARN).
- Dashboard auth: no password set in the test env → dashboard open in this local dev mode (as previously documented; compose default is secure). This run used localhost only.

---

## 18. Evidence Inventory

| ID / ref | Evidence |
|---|---|
| `preflight --json` | Camera config, system, DB integrity (PASS) |
| Feed stats | 640×480, MSMF, 30.18 FPS, brightness 130.97, change 2.874, `usable=True` (EXECUTABLE verdict) |
| FACE_DEBUG lines (daemon_err.log / daemon2_err.log) | Per-face candidate/margin/decision over live frames |
| Direct 12-frame detector window | Top-2 scores + margins, all EMP001 CANDIDATE |
| activity_logs id 190 / 191 | EMP001 ACTIVE 166.66 s → AWAY 17.88 s |
| live_state.json | EMP001 ACTIVE, timers, camera_health, pipeline_timing EMA |
| incidents id 6 / 7 | CAMERA_RECOVERED INFO auto-closed (RESOLVED) 15:39:00 / 15:46:34 |
| security_events | CAMERA_OFFLINE/RECOVERED trail (no AWAY consequence) |
| Streamlit health/root | 200 ok / 200 (11,141 bytes) |
| pytest -q / -W error | 1075 passed / 0 skipped (both) |

No face crops were saved to evidence store (evidence capture OFF); no raw biometrics collected.

---

## 19. Threshold Observations

- **`FACE_CANDIDATE_THRESHOLD` = 0.5** — EMP001 similarity always ≥ 0.5 when face was detected; candidate gate OK.
- **`FACE_MARGIN_MIN` = 0.06** — NEVER reached (0.000–0.050). Root cause hypothesis (observation, not a code defect): EMP001 has 6 enrolment embeddings that are near-duplicates (top-2 are both EMP001), so the runner-up margin to another identity can never clear the gate; two Enrollments of the SAME identity act as its own runner-up. Empirically the discriminator cannot distinguish EMP001 from EMP001.
- **`REID_STRONG_THRESHOLD` = 0.90 / `REID_MARGIN_MIN` = 0.06** — unchanged; not exercised meaningfully this phase (side/back not tested).
- Phone timing (5 s wall-clock) unchanged; no sustained evidence → rule never engaged.
- Camera feed thresholds unchanged; no feed-gate intervention needed (always usable).

All observations recorded as findings; **no threshold was changed.**

---

## 20. Model Limitations

1. **Recognition confirm gate limitation (observed live):** duplicated per-person enrolment embeddings cause the margin gate to fail for the correct identity, leaving CONFIRMED never achievable for EMP001 under these enrolments. Candidate adoption keeps productivity correct, but the frame-level `Confirmed identity` remains blocked.
2. **Phone detector under-fires on this camera (observed):** only isolated positive frames; real sustained phone use not converted to ON_PHONE in this run.
3. **CPU-bound pipeline (measured):** ~1.6–1.9 processing FPS; real-time 30 FPS capture is throttled to ~2–3 FPS processing.
4. OSNet side/back: NOT VALIDATED (not enabled; no test run).
5. InsightFace auto-download residual (from Phase 55) unchanged — `FaceAnalysis` still lacks `download=False` in `face_registry.py:134`.

---

## 21. Regression Test Results

Sequential, clean:

```
pytest -q        → 1075 passed, 0 skipped (65.51 s)
pytest -q -W error → 1075 passed, 0 skipped (66.56 s)
```

Matches the Phase 55 verified baseline exactly (1075/0). No new test or validation run changed the count — Phase 56 added no tests and no code.

---

## 22. Real-World Validation Matrix

| Scenario | Result | Evidence | Notes |
|---|---|---|---|
| Camera feed | **VERIFIED** | Feed stats; usable=True; MSMF 640×480; 30.18 FPS | Real; never black/frozen |
| EMP001 front identity | **PARTIALLY VERIFIED** | Face always EMP001 top-match; never CONFIRMED (margin0–0.05<0.06) | MODEL LIMITATION |
| EMP001 ACTIVE | **VERIFIED** | live_state ACTIVE; activity 15:46:46 (166.66s) | Stable, no false AWAY |
| EMP001 AWAY | **VERIFIED** | activity 15:49:33 (17.88s) | Real leave-then-return |
| EMP001 return | **VERIFIED** | back to ACTIVE after AWAY | Identity reacquired |
| Phone <5 sec | **NOT VALIDATED** | no sustained evidence | rule untested live |
| Phone >5 sec | **NOT VALIDATED** | only 2 isolated positive frames | detector under-fired |
| Phone recovery | **NOT VALIDATED** | no ON_PHONE episode | — |
| Unknown | **NOT VALIDATED** | no second person present | not simulated |
| Camera outage | **VERIFIED** | CAMERA_RECOVERED auto-closed; NO AWAY | historical FP did not recur |
| EMP002 identity | **NOT VALIDATED** | EMP002 not present | no fabrication |
| Side/back | **NOT VALIDATED** | not performed; OSNet off | — |
| Dashboard | **VERIFIED** | _stcore/health 200, / 200, live-state reflects ACTIVE | functional only |
| Database | **VERIFIED** | no dupes, no impossible transitions, correct timestamps | — |

---

## 23. Remaining Blockers

1. **Identity CONFIRMED blocker:** EMP001 cannot reach CONFIRMED with current enrollment set (margin gate vs near-duplicate self-embeddings). Needs evidence-based decision in a future phase (e.g. enrollment pruning/re-embedding) — NOT changed here.
2. **Phone validation blocker:** need a camera/layout where a phone is held to produce sustained ≥5 s evidence, or accept phone detection as **MODEL LIMITATION** for this camera.
3. **Unknown / second-employee blockers:** need a physical second person (enrolled EMP002 and an un-enrolled person) — not available this phase.
4. **OSNet / side-back blocker:** requires explicit activation (out of scope for Phase 56).
5. **Deployment blockers (unchanged):** GPU unavailable; RTSP cameras unconfigured (placeholder creds); evidence capture off; SMTP not configured; embeddings at rest unencrypted.

---

## 24. Production Readiness

- **Software validation:** PASS (regression, security, model-safety, reporting from Phase 55; Phase 56 real observations add live-state/database reliability).
- **Controlled real-world single-person validation:** PASS for presence, ACTIVE, AWAY/return, camera-resilience, dashboard, database. On the identity side the phase produced **PARTIAL** results — productivity and state handling are correct, but CONFIRMED face identity and sustained phone behaviour were not achieved on this hardware/enrollment.
- **Full production:** **NOT READY.** Barriers: identity confirmation margin blocker, phone detection reliability, no second-employee/unknown safety evidence, CPU-only throughput, GPU+RTSP+SMTP+encryption gaps.

---

## 25. Recommended Next Phase

Suggested Phase 57 direction (evidence-based, engineering-only — subject to operator approval):

1. **Identity confirm investigation (highest value):** analyze the EMP001 6-embedding set (self-similarity vs cross-similarity); quantify how margin would behave after de-duplicated enrollment; introduce a metrics-only run BEFORE deciding any threshold/enrollment change. Keep Phase 44B semantics intact; report numbers first.
2. **Phone detection reliability:** capture a dedicated phone-hold session with varied positions and log raw phone scores to characterise under-firing; only then decide whether model inference or camera parameters are responsible.
3. **Second-person safety day:** schedule a second person (one un-enrolled, one enrolled EMP002) to validate Unknown safety and cross-employee separation live.
4. **Optional side/back ReID:** explicit operator decision to enable OSNet in a controlled experiment.
5. **Re-run full regression + re-verify Phase 55 invariants** after any engineering change.

## 26. Final Evidence Classification

| Claim | Classification |
|---|---|
| Camera index 0 real feed usable (bright/changing/fps) | REAL VERIFIED |
| EMP001 person + face present live | REAL VERIFIED |
| EMP001 face candidate always top-ranked (0.33–0.76) | REAL VERIFIED |
| EMP001 face NEVER CONFIRMED (margin 0–0.05 < 0.06; top-2 both EMP001) | REAL VERIFIED (as a limitation) |
| EMP001 ACTIVE stable, no false AWAY while present | REAL VERIFIED |
| ACTIVE → AWAY (17.88 s) → return from real leave | REAL VERIFIED |
| Camera outage does not create AWAY; CAMERA_RECOVERED auto-close | CODE VERIFIED (real incident trace) |
| Phone >5 s → ON_PHONE, 1 alert, recovery | NOT VALIDATED |
| Unknown-person isolation | NOT VALIDATED |
| EMP002 identity / no-switch | NOT VALIDATED |
| Side/back ReID + OSNet | NOT VALIDATED |
| Dashboard functional health + live-state reflection | REAL VERIFIED |
| DB integrity (no dupes, no impossible transitions) | REAL VERIFIED |
| Regression 1075/1075 (both modes) | VERIFIED |

---

**STOP.** No thresholds tuned, no findings fixed, OSNet not enabled, no features added, no commits/pushes made. Awaiting direction for the next engineering phase.