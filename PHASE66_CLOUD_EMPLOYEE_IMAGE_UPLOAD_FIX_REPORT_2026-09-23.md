# PHASE 66 — Cloud Employee Image Upload Fix

**Report date:** 2026-09-23
**Scope:** Employee face photo upload + enrollment path only. The separately verified Phase 66B authentication/fragment-lifecycle fix is preserved and audited, not re-investigated.
**Classification markers:** REAL VERIFIED / CODE VERIFIED / SIMULATED VALIDATION / NOT VALIDATED / MODEL LIMITATION.

---

## 1. Problem

Reported production behaviour on Streamlit Cloud:

```
Admin → Employees → Add Employee → upload employee face image → "Connection lost"
```

The dashboard became unresponsive / dropped the session at the point in the
workflow where an admin uploads an employee photo and then validates it.

## 2. Actual traced upload flow

The current implementation (`app.py`) follows this exact code path:

```
st.file_uploader("Enrollment photo (JPG / PNG / BMP)",
                 type=[jpg, jpeg, png, bmp], key="enroll_upload")        app.py:2958
  └ selectbox "Assign photo to employee"                                 app.py:2961
  └ [Save photo to `data/faces/`] button                                 app.py:2966
      └ AccessGuard.require_manage_users()  (RBAC, backend-authoritative) app.py:2978
      └ save_enrollment_image(eid, upload.name, upload.getvalue())       app.py:2979-2980
          └ _validate_enrollment_bytes(data, filename)                   app.py:360-393
              └ empty check, ≤10 MB size cap, whitelisted suffix,
                magic-bytes match, cv2.imdecode, ≤24 MP pixel cap
                … NO face model (model-free by design)
          └ write data/faces/<EID><canonical-ext> — name derived ONLY from
            the employee id, never from the uploaded filename            app.py:438-444
      └ db.audit("employee.enroll_photo") + commit + _reset_enroll_job() app.py:2981-2985
  └ [Validate enrollment] button → _start_enroll_job()  → returns immediately
                                                                            app.py:2996-3004
      └ daemon thread _enroll_worker → get_shared_registry().rebuild()
        → writes ENROLL_JOB state ("done"/"error")                       app.py:495-548
      └ _render_enroll_job()  @st.fragment(run_every=2) surfaces the result
                                                                            app.py:576-599
```

Facts established while tracing:

- The **upload/save interaction is lightweight and model-free** in both the
  previous committed code (HEAD) and the current working tree.
- `.streamlit/config.toml` sets `maxUploadSize = 100` (MB); a normal JPG/PNG
  is far below every limit that matters.
- `data/faces/*` writes and the SQLite `employee.enroll_photo` audit write are
  local sub-second operations.
- No OpenCV/InsightFace import happens at app.py module level; the face model
  is only ever reached inside the enrollment worker (lazy, deferred).

## 3. Root cause determination

A CODE-LEVEL determination only (see section 4):

The heavy, blocking step sits in **"Validate enrollment"**, which is the
immediate next action an admin performs after uploading.

- At HEAD the button **synchronously** constructed `FaceRegistry()` →
  InsightFace `FaceAnalysis` initialisation (four onnx models, including
  `w600k_r50` recognition) — and on a cold Cloud instance would additionally
  **download ~300 MB (`buffalo_l`)** on first use — then rescanned
  `data/faces/` and rebuilt the embedding cache, all inside the button-click
  script run.
- On CPU that is tens of seconds to minutes of a single Streamlit script run
  that produces no forward progress for the client. Streamlit Cloud drops the
  session / websocket on such long blocking runs, and the browser surfaces
  the generic **"Connection lost"**.
- It presents at "upload → Connection lost" because the admin is on the
  upload page, one click after saving the photo, when the blocking run starts.
- Upload-size, filesystem, database, and native-dependency failure were each
  ruled out as primary causes: all are local and sub-second, and no crash was
  observed.

## 4. Important distinction

**CLOUD ROOT CAUSE NOT VERIFIED.**

No access was available to the deployed Streamlit Cloud application or its
logs during this investigation. Everything in section 3 is derived from
reading the exact code, not from Cloud telemetry. The Cloud issue is **not**
declared fixed.

## 5. Exact implementation fix

The fix removes the blocking model work from the script thread (minimum
architecture-safe change; no Celery/Redis/queues/external infrastructure):

- **`_validate_enrollment_bytes()`** — fast, model-free upload validation
  (empty, size cap, type whitelist, magic bytes, OpenCV decode, pixel cap)
  with canonical `data/faces/<EMP_ID><canonical-ext>` naming.
- **`save_enrollment_image()`** — persists validated bytes; the stored
  filename depends only on the employee id, never the uploaded name;
  idempotent (a re-save overwrites the single canonical file).
- **Background single-flight enrollment job** — `_start_enroll_job()` returns
  immediately; `_enroll_worker()` runs InsightFace on a daemon thread and
  publishes a user-safe result into `src.face_registry.ENROLL_JOB`.
- **`_render_enroll_job()`** — a `@st.fragment(run_every=2)` auto-refreshes
  the job result in the UI without any blocking run.
- **Controlled error surface** — the worker records `error_type` name plus a
  preformatted message (no traceback, no credentials, no internal paths); the
  submit path writes audit rows on success; `main()` renders a controlled
  error card instead of a raw traceback.

## 6. Authentication fix preserved

The Phase 66B auth fix is intact and unchanged by this investigation:

- `_AUTH_SESSION_KEYS` / `_clear_auth_session()` — logout and expiry clear
  every identity marker together.
- `_current_role()` — single source of truth (honours auth-disabled dev mode,
  enforces `DASH_SESSION_MINUTES`, purges expired markers); **not modified
  during this investigation**.
- Fragment auth guards on `page_live_monitoring`, `tab_live_overview`,
  `tab_security` — render nothing when no valid session exists.
- `do_auth()` keeps production fail-closed gating before the `_auth_enabled()`
  shortcut.
- Regression suite `tests/test_dashboard_auth.py` (53 tests with the Phase 66
  suite) passes in full.

## 7. Upload validation

Accepts **JPG / JPEG / PNG / BMP**. Safely handles:

- empty uploads → rejected message
- excessive file size → rejected (10 MB cap)
- unsupported/invalid extensions (e.g. `.exe`) → rejected
- corrupt/non-decodable bytes → rejected
- magic bytes not matching the stated type (e.g. JPEG bytes named `.png`) → rejected
- excessive pixel counts → rejected (24 MP cap)
- malicious filenames / path traversal — stored name is `_safe_employee_id()`
  sanitised (letters, digits, `_`, `-`) + canonical extension; the uploaded
  name is never used as a path component
- uploads are never served publicly (internal `data/faces/` files only)

## 8. Background enrollment architecture

- Enrollment is the only step that touches InsightFace.
- `get_shared_registry()` in `src.face_registry.py` holds one process-wide
  `FaceRegistry` (initialised at most once per process under `_REGISTRY_LOCK`).
- `ENROLL_JOB` (under `ENROLL_LOCK`) carries the job state across Streamlit
  reruns because module-level state in `app.py` is re-created per script run.
- `_enroll_work()` performs the pure computation (`rebuild()` + status) and
  never touches `st.*`, so it is safe on a worker thread.

## 9. Single-flight / shared-registry behaviour

- A new photograph resets the job to `idle` (`_reset_enroll_job()`), so stale
  results are never repurposed.
- `_start_enroll_job()` refuses to start a second job while one is `running`
  (single-flight); repeated Validate clicks surface "already running".
- The shared registry is initialised exactly once per process (double-checked
  lock; verified by `test_shared_registry_initialised_at_most_once`).
- Re-saving the same employee overwrites the single canonical file (no
  duplicate rows/files); different employees map to their own files.

## 10. Error handling

- Worker failures produce `ENROLL_JOB["message"]` with only the exception
  type name — no traceback, credentials, RTSP passwords, raw embeddings, or
  internal paths (verified by
  `test_enrollment_failure_is_controlled`).
- Page-level exceptions render a controlled error card (Phase 65), never a raw
  stack.
- RBAC failures and malformed input return user-safe messages and are
  audited.

## 11. Rerun / duplicate safety

- Uploaded bytes survive Streamlit reruns (browser retains the file) — a
  rerun does not lose the selection (`test_upload_survives_rerun_and_replay_creates_no_duplicate`).
- Repeated saves do not duplicate files or records
  (`test_repeat_save_is_idempotent_single_canonical_file`).
- No duplicate embeddings: near-identical faces at `COS ≥ 0.985` are
  de-duplicated by the registry; the cache fingerprint forces a rebuild when
  `data/faces/` changes.
- The model is initialised once per process, never per click/rerun.

## 12. Persistence limitation

`data/faces/`, `data/*.db`, `data/embeddings.pkl` use the Cloud container
filesystem and therefore follow the deployment's **ephemeral-storage
lifecycle**: they reset on redeploy/restart. The Employees page shows a
caption warning stating this. **This report does not claim permanent Cloud
employee enrollment**, and — per the investigation constraints — no external
object storage (S3/Cloudinary/Supabase) was introduced.

## 13. Focused tests

`tests/test_enrollment_upload.py` (29 tests) plus the updated helpers/auth
files were run: **95 passed** across `test_enrollment_upload.py`,
`test_dashboard_helpers.py`, `test_dashboard_auth.py`, `test_phase66.py`.

Covered regressions: JPG / PNG / BMP upload; JPEG→`.jpg` canonicalisation;
invalid file type; corrupt image; empty upload; magic-byte mismatch; size and
pixel caps; verbatim byte persistence; traversal-safe canonical filename;
invalid employee id; idempotent re-save; per-employee file non-collision;
statuses ENROLLED / NO_FACE / MULTIPLE_FACES / INVALID_IMAGE / LOW_QUALITY;
single-flight start; shared-registry initialised once; controlled failure
(no secrets); job reset; admin end-to-end upload; viewer cannot upload or
validate; background validate result surfaced in the real dashboard;
rerun/replay duplicate safety; blank-state warning. No tests weakened or
deleted.

## 14. Full test result

```
python -m pytest -q      → 1373 passed
```

## 15. `-W error` result

```
python -m pytest -q -W error   → 1373 passed
```

## 16. Real local Streamlit verification

Drove the actual `app.py` headlessly (`streamlit.testing.v1.AppTest`) with the
**real InsightFace `buffalo_l` model** against isolated temp storage:

- Flow A (admin → Employees → upload real face JPEG → Save → Validate):
  Save surfaced in **< 1 s**; canonical `EMP900.jpg` written verbatim
  (44,836 bytes); background job completed with the real model →
  **`{'EMP900': 'ENROLLED'}`** surfaced by the auto-refresh fragment
  (~11–16 s warm; ~47 s on the first cold model initialisation). No
  connection issue, no exception.
- Flow B (logout → login): cached enrollment state retained
  (`EMP900` ENROLLED still readable via the fast-status path).
- Flow C (blank JPEG): real model produced
  **`{'EMP901': 'NO_FACE'}`**.

Timing for the upload interaction itself never approached the block limit;
the model work is confined to the background job. No permanent employee
record was created in the repository (verification used disposable temp
storage; the one throwaway file was removed and `data/faces/` restored).

## 17. Cloud validation status

**CLOUD NOT VERIFIED.**

The deployed Streamlit Cloud app was not tested and its logs were not
inspected. Local success does not prove the Cloud issue is resolved; only a
deployed test of the upload → save → validate flow can confirm the Cloud
behaviour.

## 18. Security / RBAC

- Viewer role cannot reach upload/validate widgets **and** is denied by the
  backend guard (`AccessGuard.require_manage_users()`), with the denial
  auditable.
- Employee ids are filesystem-sanitised; uploaded filenames are never used as
  path components (no traversal).
- Errors never print secrets, RTSP URLs, embedded face data, or internal
  paths; the failure test asserts redaction of `DASH_PASS`/`CCTV_`/RTSP
  strings and `Traceback`.
- Uploaded images are never served publicly.
- Export/download and admin functions remain governed by the existing Phase 66
  RBAC; no new privileged surface was added.

## 19. REAL VERIFIED

- Local real-model upload → save → background validate → ENROLLED.
- Real-model NO_FACE for a blank image.
- Upload/save interaction completes in under a second.
- Logout/login preserves the enrollment state.
- Auth regression suite and full suite pass on this machine.

## 20. CODE VERIFIED

- Upload validation logic (magic bytes / size / pixels / decode / canonical
  naming) via unit + AppTest coverage.
- Background single-flight job and shared registry semantics via unit tests.
- Fail-closed auth path, fragment gates, `_current_role()` /
  `_clear_auth_session()` semantics via dedicated tests.
- Trace confirms the script thread never blocks on the face model.

## 21. SIMULATED VALIDATION

The per-status enrollment outcomes (ENROLLED / NO_FACE / MULTIPLE_FACES /
INVALID_IMAGE / LOW_QUALITY), the single-flight behaviour, and the controlled
failure path are exercised with a **stub registry** in the dashboard tests —
they simulate model outputs and do not run InsightFace. The real model was
exercised only for the ENROLLED and NO_FACE paths.

## 22. NOT VALIDATED

- The actual Streamlit Cloud deployment (no deploy access).
- InsightFace **first-use model download on a cold Cloud instance**.
- MULTIPLE_FACES / LOW_QUALITY / INVALID_IMAGE against the real model.
- Concurrent multi-session Cloud behaviour and the Cloud resource envelope
  (RAM headroom while `~300 MB` of onnx runtime is resident).
- Long-run behaviour of the 2-second auto-refresh fragment on Cloud.
- The CCTV daemon on Cloud (the dashboard has no live feed without it) —
  outside this fix's scope.

## 23. MODEL LIMITATION

- InsightFace `buffalo_l` (detection + 106/68 landmarks + genderage +
  `w600k_r50` recognition; stored under `~/.insightface/models/`).
- CPU-only in this environment; ~300 MB resident; first use on a host without
  the cached weights performs a runtime download (documented residual M03).
- MIN_FACE_AREA = 120×120 and duplicate threshold `COS ≥ 0.985`; low-quality
  faces are reported as LOW_QUALITY/NO_FACE and never fabricated into
  embeddings.
- Raw 512-D embeddings and similarity values are never logged.

## 24. Remaining issues

- **Cloud persistence is ephemeral** — `data/faces/`, `data/*.db`,
  `data/embeddings.pkl` reset on redeploy/restart; photos must be re-uploaded
  after a redeploy (caption now warns the user).
- The first **Validate enrollment** on Cloud still waits on the model
  download/initialisation — now in the background, so the UI stays alive, but
  the result simply takes longer.
- Enrollment state is only refreshed by Validate or by daemon startup; a fast
  "cached status" read is shown until then.
- The real Cloud incident remains **CLOUD NOT VERIFIED** until the deployed
  app is exercised end-to-end.