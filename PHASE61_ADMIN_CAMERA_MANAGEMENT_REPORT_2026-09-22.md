# PHASE 61 — ADMIN CAMERA MANAGEMENT
### Real CCTV / IP-Camera URL Configuration + Runtime Camera Control
**Date:** 2026-09-22
**Mode:** CODE VERIFIED + SIMULATED VALIDATION ONLY — no physical CCTV/NVR hardware was available in this environment, so **REAL VALIDATION WAS NOT RUN**.

---

## 1. Executive summary

Phase 61 gives the operator a **persistent, Admin-panel-driven** way to manage the
runtime camera pool: add, edit, enable/disable, test-connect and apply camera
configuration (IP CCTV / RTSP / local webcam / video file) without touching
`.env` and without a code restart. Changes are stored in SQLite, then consumed
by the **existing** `MultiCameraManager` — there is **no second camera
manager**, no second `CameraConfig` model, no second camera-health system and no
second RTSP implementation. With no admin records, a Phase 58 environment-only
deployment keeps working **verbatim** (backward compatible).

**Result:** `tests/test_phase61.py` — **33 focused tests**. Full suite —
**1229 passed**. Full suite under `-W error` — **1229 passed**.

---

## 2. What was implemented

### 2.1 Persistence (`src/database.py`)
Two new tables, strictly additive, created via `init_db`:

| Table | Purpose |
|---|---|
| `admin_cameras` | Single source of truth for admin camera records: `camera_id` (UNIQUE, stable identity), `name`, `kind` (`local`/`rtsp`/`video_file`/`test`), `enabled`, `location`, `url`, `username`, `password`, `fps_target`, `reconnect_base/max/factor`, `created_at`, `updated_at`. |
| `camera_apply_state` | Single-row (`id=1`) dashboard↔daemon handshake: `requested_revision/at`, `applied_revision/at`, `detail`. |

Older databases are migrated in place with the existing additive `_add_col`
mechanism (`_migrate_phase61`); the newest columns (`fps_target`,
`reconnect_*`, `created_at`, `updated_at`) are added when missing.

**Separation decision:** the Phase 32 `camera_config` table remains untouched —
it is reliability-annotation metadata and is never credential-bearing. Secrets
live **only** in `admin_cameras`, so no legacy serialization path (history
exports, audit rows, retainers) can ever carry a password.

### 2.2 Store + transport (`src/camera_store.py`, new module)
- `CameraRecord` — 1:1 with the Phase 58 `CameraConfig`. `password` is
  `repr=False`; `__repr__`, `display()`, `safe_dict()` and `str()` never emit it
  (`REDACTED = "********"`). URLs are shown through `redact_url` (password
  masked, host/username-safe).
- `compose_rtsp_url()` — folds split username/password into the **transport
  URL only** (percent-encoded for special chars `@ : / # % ?`; a URL that
  already embeds `user:pass@` is left untouched). The stored base URL is never
  mutated.
- `validate_camera_fields()` — **syntax only, never reachability**: `camera_id`
  charset `[A-Za-z0-9_.-]`, name required, RTSP requires `rtsp://` + netloc,
  FPS 0–240, reconnect_base ≥ 0 / reconnect_max > 0 / factor ≥ 1.0.
- `CameraStore` — `list/get/add/update/set_enabled/delete/revision/
  request_apply/apply_state`, duplicate-id rejection, partial updates that
  never rewrite the identity, **password-preserving edits** (empty/None password
  field keeps the saved one), hardened delete guarded by
  `has_camera_usage_history` (camera_health, security_events, incidents,
  seat_zones, camera_topology) — deletion is refused and safe disable
  recommended.
- `revision()` — SHA-256 over every record (incl. credentials + enabled flag);
  empty store → `""` so an env-only deployment never triggers an apply.
- `runtime_configs()` — authoritative resolution: **persistent records win;
  ids absent from the store fall back to the environment; empty store → env
  verbatim.**
- `test_camera_connection()` — bounded single-open probe through the existing
  `VideoCapture`; waits up to `timeout` s for a live connection, always
  `cap.stop()` in `finally`, returns sanitized telemetry; **never returns the
  password or a credential-bearing URL**; without hardware it honestly reports
  `ok=False`.
- `reconcile_runtime()` — idempotent daemon tick; applies when the store
  revision drifts **or** an admin requested an apply, then records
  `applied_revision/at`.

### 2.3 Runtime apply/reload (`src/camera_manager.py`)
`MultiCameraManager.apply_configs(desired)` — incremental reconciliation:
- disabled/removed cameras → **stopped/closed**;
- changed source/kind/reconnect → **closed & reopened** (no full rebuild);
- new cameras → **started**;
- unaffected cameras → **kept untouched** (same source fingerprint).

Returns a summary `{started, updated, kept, stopped}`. Pre-start `apply_configs`
just stores the configs. No change is ever made to unrelated streams.

### 2.4 Daemon wiring (`main.py`)
- The authoritative DB connection + `init_db` now happen **before** the camera
  pool is built.
- With persistent records present: `cameras`/`configs` are resolved from the
  store (env fills gaps), a startup baseline is recorded, and the loop calls
  `reconcile_runtime` on a ~2 s throttle — the dashboard’s “Apply” takes effect
  without restarting the daemon.
- With an empty store: the exact Phase 58 `.env` path is used (zero behaviour
  change).

### 2.5 Admin page (`app.py` → Admin → Cameras)
New nav item **Cameras** (`9` tabs; `test_dashboard_app.py` + `test_phase38.py`
nav-count assertions updated 8→9). `tab_admin_cameras(role)`:
- admin-gated; viewer role is read-only;
- camera list with live daemon health (OFFLINE/ONLINE/NO_FRAME/…) merged from
  the health snapshot; **credentials column shows only CONFIGURED/NONE**;
- add form (type, URL, separate username/password, location, enabled,
  FPS, reconnect) with **Test Connection** and **Add Camera**;
- edit form (identity immutable; password shown as “saved ********”, optional
  replacement), Test Connection, Save;
- enable/disable toggles (persisted, queued for apply);
- safe delete (refused when history exists, recommends disable);
- **Apply Camera Configuration** button driving the apply handshake;
- apply-status readout; **no saved password or credential URL is ever rendered**.

---

## 3. Camera admin features

- Add / edit / enable / disable / delete cameras persistently.
- Per-camera **Test Connection** (probe only — never persists, never registers).
- Runtime **Apply / Reload** — dashboard request → daemon reconcile → applies
  without restart; per-source incremental (stop/keep/reopen).
- Live health merged into the admin table from the single existing health system.
- Split-credential storage; percent-encoded transport URL; existing
  embedded-credential URLs respected untouched.

---

## 4. Storage

- SQLite `admin_cameras` + `camera_apply_state` in the existing application DB
  (`DB_PATH`), additive schemas + additive column migration for older DBs.

---

## 5. Security

- Password field is `repr=False`; masked in `repr/str/display/safe_dict`.
- URLs redacted via the existing `redact_url`; stored base URL never rewritten.
- `safe_dict` emits `password: "********"` and a `credentials: CONFIGURED/NONE`
  flag only.
- Test-connection result contains no password and a redacted URL.
- Edits preserve the saved password unless a replacement is explicitly given.
- Delete is refused for any camera with historical references.
- no new secrets in logs; `camera_store` logs id/kind only.

---

## 6. Backward compatibility (Phase 58)

- `CameraConfig`, `CameraSourceKind`, `MultiCameraManager`, `VideoCapture`,
  health states and `redact_url` are **reused**; nothing re-implemented.
- Empty store ⇒ `.env` cameras behave exactly as Phase 58.
- Persistent records only *win* for ids they manage; env fills the rest.
- Existing `camera_config` (Phase 32) metadata table untouched.
- Existing nav/tab tests updated only for count 8→9.

---

## 7. Tests

### 7.1 New focused suite
`tests/test_phase61.py` — **33 tests** (fake-capture monkeypatch at the same two
seams Phase 58 uses: `src.camera.VideoCapture`, `src.camera_manager.VideoCapture`):

- validation: RTSP scheme + netloc syntax-only; bad id / missing name / bad
  kind / FPS and reconnect bounds rejection;
- CRUD round-trip; duplicate rejection; partial updates keep other fields;
  identity never changes on edit; `None` password keeps saved one; failed update
  leaves the record intact; enable/disable toggle;
- delete guarded by history (refused) vs safe delete;
- secrecy: password never in `safe_dict`/`repr`/`str`/`display`; URL redacted;
  percent-encoding of special-char credentials; embedded credentials win;
  no password in probe result;
- resolution: empty store ⇒ env verbatim; persistent wins + env gap-fill
  (incl. disabled flag); `to_config` contract; local-without-URL ⇒ device 0;
- reconciliation: first apply then no-op; apply on revision change; stops
  disabled/starts new; keeps unchanged/reopens changed; applied revision
  matches store; `request_apply` handshake;
- honesty: probe `ok=False` without hardware; capture released on failure;
  invalid kind reported without crash;
- revision: sensitive-field changes always bump the hash; fresh per-camera
  isolation; env-only deployment never triggers an apply.

### 7.2 Full regression
- Dashboard suite (`tests/test_dashboard_app.py`): 7 passed.
- Phase 38 nav-shell test updated (8→9): passed.
- **Full suite:** 1229 passed.
- **Full suite `-W error`:** 1229 passed.

### 7.3 Static / import
- `py_compile` on all touched modules: OK.
- `import src.camera_store, src.camera_manager, src.database, config, app, main`: OK.

---

## 8. Validation discipline

- **SIMULATED VALIDATION:** done — 33 new focused tests + 1229-test regression
  validated persistence, resolution, apply/reload, redaction and honest failure
  semantics against a fake transport (`_FakeCap`) mirroring OpenCV `VideoCapture`.
- **REAL VALIDATION: NOT RUN — physical CCTV/NVR unavailable.** No live RTSP
  stream, no fixed IP camera and no NVR feed was reachable from this machine, so
  a genuine end-to-end login → `cap.read()` loop could not be exercised. The
  transport boundary is the pre-existing, Phase 58-validated `VideoCapture`;
  Phase 61 only configures and drives it. No claim of live-camera success is
  made anywhere in this report.
- Camera reachability is **never guessed**: syntax validation only; the Test
  Connection probe opens the source once and truthfully reports unavailable
  without hardware; a camera failure never fabricates employee AWAY.

---

## 9. NOT validated (with real hardware)
- Live RTSP auth + decode round-trip on hardware cameras/NVR.
- Reconnect behaviour against a genuine flapping network cam.
- Per-camera FPS override effect on a real stream.
- Concurrent multi-device bandwidth under production load.

## 10. Model limitations (no hardware)
- No live stream verification possible; OpenCV/FFmpeg backend timing and
  device-specific quirks are outside Phase 61’s control.
- `test_camera_connection` timing depends on the network; a genuinely reachable
  camera that answers slower than the `timeout` reports a false negative.
- Local webcam (`localhost:0`) behaviour is unchanged from Phase 58 and wasn’t
  re-verified with hardware.
- DVR-style “login via NVR API” credentials are supported only in so far as
  they map to a single RTSP URL; analytics/NVR microservice APIs are out of scope.

---

## 11. Regressions
None. 1229/1229 pass with and without `-W error`; nav-count assertions updated
for the one new tab.

## 12. Files touched
- `src/camera_store.py` — **new**; persistence + resolution + probe + reconcile.
- `src/database.py` — `admin_cameras`, `camera_apply_state`, `_migrate_phase61`,
  apply-state + usage-history helpers.
- `src/camera_manager.py` — `apply_configs` + `_cfg/_same_source` helpers.
- `main.py` — DB-before-cameras; store-resolved runtime config; throttled
  reconcile in the loop.
- `app.py` — Cameras nav item + `tab_admin_cameras`.
- `tests/test_phase61.py` — **new**; `tests/test_dashboard_app.py`,
  `tests/test_phase38.py` — nav count 8→9.

No git commit was made (per phase rules).