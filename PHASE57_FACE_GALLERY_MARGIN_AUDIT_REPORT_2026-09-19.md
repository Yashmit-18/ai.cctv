# PHASE 57 — FACE GALLERY / IDENTITY MARGIN ROOT-CAUSE AUDIT + SAFE FIX

**Date:** 2026-09-19
**Author:** autonomous Phase 57 execution on the AI Office CCTV Intelligence & Security Platform

---

## 1. Executive Summary

Phase 56 real-world validation exposed a face-recognition confirmation blocker: EMP001 was
always the top-ranked identity candidate (similarity ~0.65–0.76) yet was **never CONFIRMED**
because the Phase 44B margin gate compared the top-1 and top-2 **embedding rows**, and both
rows belonged to EMP001's own enrollment set (6 embeddings). This phase audited the identity
path end-to-end, proved the root cause with the actual enrolled gallery, implemented the
smallest safe correction — **ranking employee *identities* rather than individual enrollment
embeddings** — and validated it offline, on the real gallery reconstruction, and on the real
live camera.

**Root cause: FOUND.** The face margin was computed between the best and second-best
**enrollment embeddings**, which can both belong to the *same* employee. The margin gate is
documented (and was intended) as a gap between **competing employee identities**, but the
implementation operated at the embedding-row level.

**Result:** with the corrected employee-level margin the live camera (EMP001 physically
present) produced **36 CONFIRMED frames** with margins 0.54–0.72 over a 60 s bounded window,
versus **zero** CONFIRMED in the Phase 56 window (margins 0.000–0.050). No threshold was
lowered; `FACE_CANDIDATE_THRESHOLD = 0.50` and `FACE_MARGIN_MIN = 0.06` are unchanged.

---

## 2. Baseline Commit

- Verified baseline: `c2a80c915b059ca162db8e83d9b34a059dd0618d` (`Complete Phases 40-56 CCTV intelligence platform`)
- Branch: `main`, tracking `origin/main`. Working tree was clean at baseline.
- Baseline tests: `pytest -q` 1075 passed; `pytest -q -W error` 1075 passed.

---

## 3. Phase 56 Finding

Quoted findings recorded in `PHASE56_CONTROLLED_REAL_WORLD_VALIDATION_REPORT_2026-09-19.md`:

> EMP001 was consistently top-ranked (similarity 0.33–0.76, typically 0.65–0.76) but the
> Phase 44B margin gate never cleared (margin 0.000–0.050, always < `FACE_MARGIN_MIN` 0.06)
> because the top-2 candidates are both EMP001's OWN enrolled embeddings.

Classification in Phase 56: **REAL VERIFIED** for person/face/candidate + ACTIVE productivity;
recognition **NOT CONFIRMED** reported as a MODEL/THRESHOLD LIMITATION, deliberately not fixed.

This phase (57) re-tests that classification: the "MODEL LIMITATION" label was, on deeper
inspection, inaccurate. The embedder (buffalo_l) was not failing — **the decision logic
measured the margin against the wrong baseline** (same-person gallery samples).

---

## 4. Existing Face Identity Pipeline

The complete audited path (Step 1 of the phase brief):

```
face detection            src/face_registry.py  FaceRegistry._init_insightface  (det_10g, 640)
face crop / align         insightface FaceAnalysis.app.get(frame)
embedding generation      face.normed_embedding  (512-D, L2-normalised)
gallery lookup            FaceRegistry._embeddings  (N x 512) from data/embeddings.pkl
similarity calculation    sims = self._embeddings @ query.T      (cosine for unit vectors)
ranking                   np.argmax(sims)                         [identify / identify_candidate]
candidate threshold       best_score < CANDIDATE_THRESHOLD -> UNKNOWN
margin calculation        best_score - second_score               [was: embedding row - row]
candidate consistency     tracker's register_candidate_vote / cand-runs
confirmation              status = CONFIRMED if score>=RECOG AND margin>=MARGIN_MIN
trusted identity          Track.identity (face-sourced) never demoted
identity adoption         SpatialTracker._adopt + _decide (face/candidate/appearance branches)
final employee/frame id   Track.identity -> claims -> MultiTracker -> employee FSM
```

Key files/functions:

| Component | Location |
|---|---|
| `identify_candidate` (decision) | `src/face_registry.py:453` |
| `identify` (legacy argmax) | `src/face_registry.py:379` |
| `identify_k` (diagnostics) | `src/face_registry.py:426` |
| `_employee_ranking` (NEW) | `src/face_registry.py:407` |
| Confirmation gate rule | `src/face_registry.py:516-521` |
| Thresholds | `config.py:174-201` |
| Tracker identity vote/adopt | `src/tracker.py:735-834`, `1061-1102` |
| Candidate never-demote rule | `src/tracker.py:802-809` |
| Face-vs-appearance authority | `src/tracker.py:820-823` |

### Inspected configuration constants

- `FACE_CANDIDATE_THRESHOLD = 0.50` (candidate floor, unchanged)
- `FACE_SIMILARITY_THRESHOLD = 0.60` (CONFIRM cutoff, unchanged)
- `FACE_MARGIN_MIN = 0.06` (unchanged)
- `FACE_CONFUSABILITY_MAX = 0.88` (cross-identity separability ceiling, unchanged)
- `DUPLICATE_COS = 0.985` (enrollment dedupe, unchanged)

### Gallery structure

- `data/embeddings.pkl`, cache version 3, 9 embeddings across 4 employees:
  - EMP001 → **6** embeddings (rows 0–5)
  - EMP002 → 1, EMP003 → 1, EMP004 → 1
- Employee-to-row mapping `_by_employee` built at build time and cached.

### Embedding cache

Loaded in `_load_cache` (fingerprint-guarded rebuild); `_employee_ranking` uses only
`_employee_ids` + similarity vector, so it is robust to cache rebuilds.

---

## 5. Root Cause Investigation

**Observation (Phase 56):** EMP001 = 6 enrollment embeddings. A live query scored
~0.73–0.76 against the *best* EMP001 row and ~0.69–0.72 against the *second-best* EMP001 row.
The old margin `0.76 - 0.72 = 0.04 < 0.06` → CANDIDATE. Meanwhile EMP004/EMP002/EMP003 scored
far below (real gallery cross-similarity ≤ 0.196), so the identity was actually unambiguous.

**Code-level confirmation (`face_registry.py` before fix):**

```python
sims = (self._embeddings @ query.T).flatten()
best_idx = int(np.argmax(sims))                    # best ROW
if best_score < CANDIDATE_THRESHOLD: ...
second_idx = int(np.argpartition(sims, -2)[-2])   # second-best ROW
margin = best_score - second_score                 # ROW-level margin
```

The runner-up was chosen by **embedding row index**, not by employee id — so two enrollment
samples of the same person could occupy top-1/top-2.

**Documented intent already said "identity":** `config.py:190-193` describes
`FACE_MARGIN_MIN` as the "similarity gap between best and runner-up **gallery identity**". The
implementation used the wrong denominator.

---

## 6. EMP001 Gallery Statistics

Aggregate-only, computed from `data/embeddings.pkl` with the real enrolled vectors (raw
vectors never printed/stored):

- Embedding count: **6** (rows 0–5)
- Embedding dimension: **512**
- Normalization: all 6 rows L2-normalised to norm 1.0 (min=max=1.000000)
- Pairwise same-employee similarity (6×6, off-diagonal, 30 pairs):
  - min: 0.64434, max: 0.85575, mean: 0.77388, median: 0.80397
  - ≥0.80: 16/30 pairs; ≥0.90: 0; ≥0.985 (dedupe ceiling): 0
- Verdict: **not exact duplicates**, but a compact cluster (median 0.80) — enough that a
  single live query commonly lands *both* top-2 rows inside EMP001, making the old
  embedding-level margin ~0.00–0.05.
- Multiple EMP001 embeddings occupying top-1 *and* top-2: **confirmed in 500/500** simulated
  live queries (see §9) and in the real Phase 56 debug lines.

Other employees (for completeness — **not** real-world validated):

- EMP002: 1 embedding; EMP003: 1; EMP004: 1
- Cross-employee peak similarity (max over all pairs):
  - EMP001↔EMP002: 0.08291 · EMP001↔EMP003: 0.01291 · EMP001↔EMP004: 0.09600
  - EMP002↔EMP003: 0.19610 · EMP002↔EMP004: 0.04258 · EMP003↔EMP004: −0.07989
- No cross pair reaches `FACE_CONFUSABILITY_MAX` (0.88); the registered identities *are*
  separable by face. Confusability report: **none**.

---

## 7. Current Margin Semantics (before fix) — VERIFIED

Before the change, `identify_candidate` did exactly:

```
query -> compare against every stored embedding -> sort embedding rows
      -> top-1 row -> top-2 row -> margin = top1_row_score - top2_row_score
```

**Yes**, same-employee duplicate embeddings could occupy top-1 and top-2, artificially
deflating the margin even when the identity was clear.

---

## 8. Corrected Margin Semantics (after fix) — CODE VERIFIED

`identify_candidate` (and the diagnostic `identify_k`) now do:

```
query -> compare against every stored embedding
      -> group scores by employee id, keep each employee's BEST score
      -> rank employees by that score
      -> margin = best_employee_score - second_best_employee_score
```

`_employee_ranking(sims)` is a new private method implementing the grouping:

```python
per_employee: dict[str, float] = {}
for i, eid in enumerate(self._employee_ids):
    s = float(sims[i])
    if s > per_employee.get(eid, -1.0):
        per_employee[eid] = s
return sorted(per_employee.items(), key=lambda kv: kv[1], reverse=True)
```

This is **one identity system, one gallery, one aggregation** — it does not add a second
registry/tracker/FSM. It changes only the margin/runner-up denominator.

---

## 9. Employee-Level Scoring Analysis

Two aggregation candidates were evaluated offline against the real gallery using 500 noisy
EMP001-like live proxies (unit-norm blends of EMP001 rows + Gaussian noise), before any code
was touched:

| Metric | Embedding-level (old) | Employee max (chosen) | Employee mean-of-top-3 |
|---|---|---|---|
| margin min | 0.0001 | 0.5126 | 0.4998 |
| margin max | 0.1578 | 0.7252 | 0.6861 |
| margin mean | 0.0245 | 0.6199 | 0.5990 |
| margin median | 0.0191 | 0.6202 | 0.6011 |
| CONFIRMED fraction | **6.0%** | **100%** | 100% |
| best-other-employee score (mean) | 0.059 | 0.059 | 0.059 |

For the same 500 real-gallery proxies:

- Old semantics confirmed only 6% of frames (margin < 0.06 in 94%).
- Both employee-level aggregations confirmed 100% with large margins (≥0.50), while the
  best *other* employee stayed ~0.06 — the identity was never ambiguous.
- Same-employee top-1 **and** top-2 (by row) occurred in **500/500** proxies → root cause
  reproduced exactly.

**Aggregation selected: maximum similarity per employee** — justified because (a) it is the
minimum semantical change (the CONFIRM cutoff already uses the raw max similarity), (b) it is
identical to the old code for single-embedding employees, (c) mean-of-top-k offers no safety
advantage here and would soften the leading score. One method, one pipeline.

**Offline verdict for ambiguous cases:** two genuinely similar employees (both ~0.60, margin
~0.01) still fail the gate → CANDIDATE. Verified by regression tests (§12).

---

## 10. Identity Safety Analysis

Explicit answers to the 10 Phase 57 safety questions (all checks are in the new regression
suite, §12):

1. **Can employee-level grouping accidentally increase false confirmations?**
   NO. Confirmation still requires `score >= 0.60` **and** identity margin `>= 0.06` between
   *different* employees. Two genuinely similar employees (both ~0.60) still yield margin
   ~0.01 → CANDIDATE (TEST 3).
2. **Can two genuinely similar employees become incorrectly confirmed?**
   NO. That is precisely what the margin gate rejects (TEST 3: margin 0.01 < 0.06).
3. **Does margin remain useful?**
   YES — more useful: it now measures *identity* separation, matching the documented intent.
4. **Does candidate threshold remain meaningful?**
   YES. `CANDIDATE_THRESHOLD = 0.50` unchanged; below it a face is still UNKNOWN (TEST 4).
5. **Does candidate run consistency remain useful?**
   YES. Tracker candidate-vote runs untouched (TEST 5).
6. **Does trusted identity remain protected?**
   YES. Face-sourced identities are only ever set/upgraded by CONFIRMED; candidates can only
   adopt tracks that trust nobody (TEST 5, TEST 6).
7. **Can weak ReID/appearance evidence demote a trusted face identity?**
   NO. `dt _decide` returns immediately for face-owned identities when appearance arrives
   (TEST 6; unchanged code path `src/tracker.py:820-823`).
8. **Can Unknown incorrectly become a known employee?**
   NO. Unknown faces (below 0.50) cast only Unknown votes and never invent identity
   (TEST 7).
9. **Can one employee's gallery dominate another?**
   NO under employee max: each employee contributes exactly one score, so a 6-embedding
   EMP001 cannot crowd out a 1-embedding EMP004 at the decision level. This is the anti-
   pattern the fix removes (TEST 1).
10. **Does the corrected method preserve existing safety behaviour?**
    YES. Camera outage/frozen, phone semantics, productivity semantics, and absence wall-
    clock are untouched (TEST 10). Full suite: `pytest -q` 1093 passed; `-W error` 1093
    passed.

---

## 11. Code Changes

| File | Change | Size |
|---|---|---|
| `src/face_registry.py` | Add `_employee_ranking()`; rewrite `identify_candidate` margin to employee-level; update `identify_k` to per-employee ranking; docstrings updated | +43 / −15 |
| `tests/test_phase57.py` | NEW — 18 deterministic regression tests mapping to the Phase 57 TEST 1–10 requirements | new |

Total diff: 1 source file modified (+43/−15), 1 test file added. No threshold changed, no
second system introduced. `git status` shows exactly these two paths + the report.

---

## 12. Threshold Decision

- `FACE_CANDIDATE_THRESHOLD` = **0.50** (unchanged — no evidence it is wrong)
- `FACE_MARGIN_MIN` = **0.06** (unchanged — evidence (§9) shows the employee-level margin
  is 0.51–0.73 for genuine matches and ~0.01 for genuinely ambiguous pairs, so 0.06 remains
  a correct and meaningful safety band)

The root cause was the margin *denominator* (embedding row vs employee identity), not the
threshold value. Thresholds were deliberately **not** lowered.

---

## 13. Tests Added (`tests/test_phase57.py`, 18 tests)

| Phase brief | Covered by |
|---|---|
| TEST 1 — same-employee duplicate embeddings must NOT deflate margin | `test_57_same_employee_duplicates_do_not_deflate_margin`, `test_57_same_employee_duplicates_leave_score_and_identity_intact` |
| TEST 2 — EMP001 strong vs EMP002 lower → correct employee margin | `test_57_clear_winner_employee_level_margin`, `test_57_single_employee_gallery_full_score_margin` |
| TEST 3 — two genuinely ambiguous employees stay unconfirmed | `test_57_ambiguous_employees_stay_unconfirmed`, `test_57_identical_scores_two_employees_zero_margin` |
| TEST 4 — best below candidate floor → no confirmation | `test_57_below_candidate_floor_is_unknown`, `test_57_empty_gallery_is_unknown` |
| TEST 5 — candidate consistency intact | `test_57_candidate_adopts_resolved_track_after_consistency`, `test_57_candidate_promoted_to_face_authority_on_confirmed` |
| TEST 6 — trusted identity not demoted by appearance/ReID/candidates | `test_57_trusted_face_identity_never_demoted_by_appearance`, `test_57_candidates_never_override_trusted_identity` |
| TEST 7 — Unknown remains Unknown | `test_57_unknown_face_does_not_invent_identity` |
| TEST 8 — adoption behaviour intact | `test_57_confirmed_face_eventually_adopts` |
| TEST 9 — productivity semantics unchanged | `test_57_active_through_face_and_body` |
| TEST 10 — camera offline/frozen unchanged | `test_57_camera_offline_freezes_state_never_away`, `test_57_camera_frozen_still_keeps_presence`, `test_57_offline_from_start_creates_no_fake_away` |
| Plus | `_employee_ranking` order/scale regression coverage via the registry tests above |

---

## 14. Full Regression Results — CODE VERIFIED

- `pytest -q`            → **1093 passed** (baseline 1075 + 18 new) in 136.44 s
- `pytest -q -W error`   → **1093 passed** in 127.37 s
- Focused identity suites (re-run with the change): `test_phase44b.py`,
  `test_phase44.py`, `test_multiperson_pipeline.py`, `test_phase49.py`,
  `test_phase57.py` → **158 passed**
- AST validation: 132 tracked+new `.py` files all parse clean
- Import validation: `main`, `src.face_registry`, `src.tracker`, `src.detector` import OK

No numbers fabricated; these are the actual command outputs.

---

## 15. Real Validation Results — REAL VERIFIED (live camera, single-person EMP001)

Environment: real webcam (index 0, 640×480, MSMF/DSHOW usable), same as Phase 56. Bounded
60-second headless capture with the **corrected** registry (identical gallery, same
buffalo_l embedder). EMP001 was physically present.

Aggregate 60.4 s window:

| Metric | Phase 57 (fixed) | Phase 56 (old) |
|---|---|---|
| Frames read | 210 | (window) |
| Faces processed | 101 | (window) |
| CONFIRMED frames | **36** | **0** |
| CANDIDATE | 23 | every face |
| UNKNOWN | 42 (weak/no face) | — |
| Confirmed identity | EMP001 | none |
| Confirmed score range | 0.614–0.749 | — |
| Confirmed margin range | 0.546–0.722 | 0.000–0.050 |
| Max consecutive CONFIRMED | **11** frames | 0 |
| Runner-up on confirmed frames | EMP002/EMP003/EMP004 (always a different employee) | EMP001 (itself) |

Classic confirmed line (real, formatted for privacy — no raw data):

```
confirm emp=EMP001 score=0.6512 margin=0.5602 second=EMP004   (real live frame)
confirm emp=EMP001 score=0.7492 margin=0.7220 second=EMP002
```

Candidate/Unknown frames correspond to partial/weak face presentations (score < 0.60 or no
face), consistent with expected Phase 44B behaviour. **No identity switching, no false AWAY**
observed; EMP001 tracked and productive throughout. A multi-frame consecutive run was
confirmed (max 11) — demonstrating stability.

This is **REAL VERIFIED** single-person EMP001 validation. (EMP002/EMP003/EMP004 confirmed
only as *runner-up identities* in those frames — **not** real-world validated as primary
identities; their source data is enrollment-derived. See §18.)

---

## 16. Performance Results — CODE VERIFIED

Microbenchmark (200 000 iterations, real 9-embedding/4-employee gallery):

- Old embedding-level path: **16.244 µs/call**
- New employee-level path: **12.985 µs/call** (≈ −20%, faster — grouping avoids a second
  full argpartition pass; measured, not estimated)

Scale check (hypothetical 500 embeddings / 100 employees): new path ≈ 353 µs/call vs old
≈ 55 µs/call — +0.3 ms/call, negligible against the ~435 ms face-detection stage and the
cadence-thinned (default 5-cycle) recognition path. No premature optimisation performed;
identity safety outranks FPS.

Live pipeline (bounded window): face detect mean 434.7 ms (CPU buffalo_l), embed/match +
group **0.128 ms/call**, 9 enrolled embeddings, 4 employees. Employee grouping adds no
materially measurable overhead at real gallery scale.

---

## 17. Security / Privacy Review

- No raw face embeddings or face images printed, stored in logs, or written to this report.
- Only cosine-similarity/margin/status aggregates reported (§6, §9, §15).
- No credentials or secrets exposed; no changes to `.env`, auth, or network config.
- No new camera outage / phone / productivity semantic changes.
- Report and code contain only sums, means, medians, and decision outcomes.

---

## 18. Remaining Limitations

- **MODEL LIMITATION (unchanged):** threshold semantics rely on InsightFace buffalo_l
  discriminability; undeclared visitors or look-alike employees are only as separable as the
  embedder allows (confusability ceiling 0.88 surfaces any registered near-pair; currently
  none).
- **NOT VALIDATED:** EMP002/EMP003/EMP004 as **primary** confirmed identities on live camera
  (their only appearances here were as runner-ups). Multi-person simultaneous validation,
  side/back low-quality confirmations, and long-run identity stability remain Phase 57
  scope-limits.
- **APPLIES TO APPENDIX:** `identify()` (legacy argmax) still returns raw row
  best identity — it is only used by legacy stubs/CLI and is documented as non-decision;
  `identify_candidate` is the only decision surface and is fixed.
- Phase 56's "LIMITATION" classification of the confirmation blocker is superseded/revised
  by this audit (the blocker was a decision-logic bug, not an embedder limitation); all
  other Phase 56 findings stand and are represented accurately above.
- OSNet ReID remains disabled (unchanged); appearance registry is single-embedding per
  employee and untouched.

---

## 19. Final Acceptance Decision

Phase 57 acceptance criteria:

| Criterion | Status |
|---|---|
| Root cause established or disproven | **ESTABLISHED** (embedding-level margin denominator) |
| Actual implementation path documented | YES (§4) |
| No blind threshold lowering | YES — thresholds unchanged |
| Identity margin semantics clearly defined | YES (§8) |
| Code change minimal | YES — 1 source file, +43/−15 |
| Regression tests cover the issue | YES — 18 tests, TEST 1–10 |
| Existing identity safety intact | YES (§10, §14) |
| Productivity behaviour intact | YES (TEST 9) |
| Camera outage behaviour intact | YES (TEST 10) |
| `pytest -q` passes | YES — 1093 |
| `pytest -q -W error` passes | YES — 1093 |
| AST/import checks pass | YES |
| No biometric data leaked | YES (§17) |
| Phase 56 results represented accurately | YES (§3, §15 revision note) |
| Final report exists | YES (this file) |
| Repository changes listed | YES (§11) |
| NO Git commit / push performed | **YES** — working tree intentionally dirty |

**Verdict: PASS.**

The correction is safe, evidence-backed, minimal, and measurably resolves the real-world
confirmation blocker without weakening any existing protection or threshold.

---

**Classifications used:** REAL VERIFIED (live-camera EMP001 confirmation; gallery
cross-similarity aggregates), CODE VERIFIED (pipeline trace, unit tests, thresholds,
performance), SIMULATED/OFFLINE (employee-level aggregation comparison sweep over 500 proxy
queries), NOT VALIDATED (EMP002/003/004 primary identity, multi-person, long-run stability),
MODEL LIMITATION (embedder-driven separability ceiling only).