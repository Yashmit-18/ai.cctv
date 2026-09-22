# PHASE 62 — COMPLETE ADMIN CONTROL CENTER
### Centralized Employees / Desks / Chairs / Schedule / Thresholds Management
**Date:** 2026-09-22
**Mode:** CODE VERIFIED + SIMULATED VALIDATION ONLY — no physical CCTV/NVR hardware is
required for this configuration layer; **REAL hardware validation belongs to Phase 63 and was NOT run.**

---

## 1. Executive summary

Phase 62 completes the Admin Control Center: one dashboard page that gives an administrator
the **single, persistent source of truth** for office-intelligence configuration — employees
(display label, enable/disable, desk+chair assignment), desks (= seat zones), chairs, the
office/lunch schedule and the away/phone/talking threshold knobs — all validated, all
RBAC-gated, all audited, and all pushed **down into the existing runtime** so the daemon and
dashboard always agree.

Configuration flows as:

```
ADMIN PANEL  ->  settings table (existing SQLite)  ->  config module attrs  ->  existing runtime
                   \-> EmployeeStore / SeatZoneStore / ChairStore (reused, never duplicated)
```

There is **no second KV store, no session-state-only config, no duplicate store/registry/
tracker/schedule engine/camera manager**. Desk == seat zone: the existing `seat_zones` table
holds the desk polygons; no separate `desks` table was invented. With an untouched settings
table a deployment behaves **exactly** as before (env defaults), so Phases 57/58/59/59c/60A/61
are fully backward compatible (regression suite proves it).

**Result:** `tests/test_phase62.py` — **38 focused tests**. Full suite — **1267 passed**.
Full suite under `-W error` — **1267 passed**.

---

## 2. What was implemented

### 2.1 Persistent settings in the existing database (`src/database.py`, `src/admin_store.py`)
- New additive `settings` table (`key` PRIMARY KEY, `value`, `updated_at`) created by
  `init_db` inside the **existing application SQLite file** — no second KV store/database.
- `get_setting` / `set_setting` / `list_settings` helpers; `set_employee_display_name`
  (presentation-only label, `employee_id` untouched).
- New module **`src/admin_store.py`**:
  - `OfficeSettings` dataclass — timezone / office start/end / lunch start/end /
    away/phone/talking thresholds with the Phase 62 defaults
    (`Asia/Kolkata`, `10:00→18:30`, `14:00→14:35`, `10 / 5 / 15` s) mirrored 1:1 from the
    existing `config.CCTV_OFFICE_*` + `FUTURE_*` env knobs.
  - `validate_timezone` (IANA via `zoneinfo`), `validate_threshold_seconds` (finite, ≥ 0),
    `validate_office_settings` (HH:MM 24h, office start<end, lunch window **inside** office
    hours) — reuse of the existing `office_schedule.parse_hhmm`; clear `ValueError` messages
    surfaced directly in the UI.
  - `SettingsStore.get()` — persisted values merged over env defaults (backward compatible);
    `update(actor=..., **fields)` — validate → persist changed keys → audit
    `settings.update` with the changed-field list (never secrets); `apply_to_config()` —
    writes the *existing* `config` module attrs (`CCTV_OFFICE_TZ/START/END`,
    `CCTV_LUNCH_START/END`, `FUTURE_AWAY/PHONE/TALKING_THRESHOLD_SECONDS`) so the runtime’s
    ONE authoritative schedule source reflects admin edits; `snapshot()`.
  - `validate_assignment(conn, employee_id=, zone_id=, chair_id=)` — rejects a missing/
    disabled employee, a missing/disabled desk, a missing/disabled chair, and a chair that
    references an unknown zone.
- **Runtime wiring (`main.py`)**: after `init_db` the daemon applies the persisted settings
  via `SettingsStore(conn).apply_to_config()` (guarded; env defaults otherwise).

### 2.2 Employee administration (`src/domain.py`, `src/database.py`, `src/employees.py`)
- `Employee.display_name` added (optional field); `EmployeeStore.get/list` surface it;
  `set_display_name` / `db.set_employee_display_name` update the label **only**.
- `employee_id` is the stable canonical identity and is **never rewritten**; rename
  (display label) and enable/disable never delete history; duplicate ID create is rejected.
- `PERM_MANAGE_USERS` governs employee writes (existing).

### 2.3 Desk (== seat zone) administration (`src/seat_zones.py`)
- Reused the existing `SeatZoneStore`; added `update(zone_id, **fields)` (name/camera_id/
  polygon/enabled/assigned) that **never rewrites `zone_id`** and re-validates polygons
  (≥ 3 normalised points via the existing `zones.parse_polygon`).
- Add (`add`), enable/disable (`set_enabled`), assign employee (`set_assignment`) and delete
  (config only — events/history untouched) through the existing store; duplicate desk
  rejection; delete cascades child chairs (config records only).

### 2.4 Chair administration (`src/seat_chairs.py`)
- Reused `ChairStore`; added `update(chair_id, **fields)` with stable `chair_id`; a chair
  still requires a zone and a camera; enable/disable via existing `db.set_chair_enabled`;
  assignment via existing `set_assignment`; delete config-only.

### 2.5 Identity safety
- Desk/chair assignments are **context hints only**: setting a different home desk never
  converts UNKNOWN into an employee, never relabels a face-confirmed identity, and an empty
  desk/chair stays VACANT. `SEAT_ASSIGNMENT_MISMATCH` / `CHAIR_ASSIGNMENT_MISMATCH` remain
  the neutral observations they already are(`test_phase62` covers the treaty).

### 2.6 RBAC (`src/domain.py`, `src/rbac.py`)
- New permission `PERM_CONFIGURE_SETTINGS` granted to **admin only** in `ROLE_PERMISSIONS`
  (admin already had all others).
- New `AccessGuard.require_configure_settings()`; existing guards reused for employees
  (`require_manage_users`) and desks/chairs (`require_configure_zones`).
- Viewer is read-only on every admin sub-page; the **backend guard denies and the denial is
  audited** even if a viewer reaches a write path (tested for all three write groups).

### 2.7 Dashboard — Admin Control Center (`app.py`)
- Sidebar navigation grows to **10 top-level pages** with one new tab **Admin Control Center**
  (`app.py:main` routing; `_NAV`).
- In-page sub-navigation (consistent with the app’s smallest consistent alternative):
  **Overview · Cameras · Employees · Desks & Zones · Chairs · Schedule · Thresholds · System**.
- **Cameras** sub-page reuses the Phase 61 `tab_admin_cameras(role)` (no duplicate camera UI).
- Each sub-page is admin-gated, audited via `db.audit` (neutral details only — no
  passwords/embeddings/secrets), mobile-friendly, and safe under Streamlit AppTest.

### 2.8 Schedule & thresholds
- ONE authoritative schedule source: admin edits are persisted in `settings`, then
  `apply_to_config()` writes the same `config.CCTV_*` knobs the existing `OfficeSchedule`
  consumes; `main.py` re-applies at startup. Defaults unchanged (`10:00→18:30`, lunch
  `14:00→14:35`, tz `Asia/Kolkata`).
- Thresholds are the Phase 59c reserved `FUTURE_AWAY/PHONE/TALKING_THRESHOLD_SECONDS`
  knobs (10 / 5 / 15 s) now managed centrally and persisted. The FSM timers
  `AWAY_AFTER_SEC` (3.0) and `PHONE_AFTER_SEC` (5.0) are **never** touched (test asserts it).

---

## 3. Admin features delivered

| Feature | Where |
|---|---|
| Add / rename (display label) / enable-disable / assign desk+chair per employee | Admin → Employees |
| Add / edit / enable-disable / assign / delete desk (polygon + camera) | Admin → Desks & Zones |
| Add / edit / enable-disable / assign / delete chair | Admin → Chairs |
| Schedule editor (tz, office, lunch) with validation | Admin → Schedule |
| Threshold editor (away / phone / talking) with validation | Admin → Thresholds |
| Read-only overview + system state (incl. camera apply state) | Admin → Overview / System |
| Camera management reuse (Phase 61) | Admin → Cameras |
| Every admin write audited; RBAC-enforced; viewer read-only | whole page |

---

## 4. Storage

| Table | Purpose | Safety |
|---|---|---|
| `settings` (new) | centralized schedule + threshold knobs | existing SQLite; validate-before-write |
| `employees` (+`display_name` col) | roster; canonical `employee_id` | label-only rename; history intact |
| `seat_zones` (== desks) | desk polygons + camera + hint assignment | identity never rewritten |
| `chairs` | chair within a desk + hint assignment | identity never rewritten |

Enable/disable and delete are **config-only** operations: event tables
(`seat_events`, `chair_events`, security/history) are never touched.

---

## 5. Security

- Secrets are never stored in `settings`; audit rows carry neutral field names only
  (the `settings.update` audit detail contains changed field names, never values of any
  credential-bearing kind); nothing secret is rendered in the admin UI.
- RBAC is enforced **backend-authoritatively** by `AccessGuard` on every write path;
  viewer denial is audited as `access.denied`.
- No Passwords / embeddings / APIs are logged by any Phase 62 code path.

---

## 6. Backward compatibility

- Empty `settings` table → `SettingsStore.get()` returns the env defaults verbatim;
  `apply_to_config()` writes back exactly those values, so the runtime behaves as before.
- Phase 58 env-only camera model, Phase 59 desk/chair tracking, Phase 59c reserved knobs,
  Phase 60/60A simulation, and Phase 61 admin camera management are all unchanged and all
  pass the full regression suite.

---

## 7. Tests

### 7.1 New focused suite (`tests/test_phase62.py`)
**38 tests** — real APIs only (no mocks):

- Settings defaults match `config` env knobs; persistence survives DB reopen.
- `settings` table co-located in the existing SQLite (no second KV store; no `desks` table).
- Validation: bad timezone / bad HH:MM / office-end-before-start / lunch-outside-office /
  negative / non-numeric thresholds all rejected; valid full payload accepted.
- `apply_to_config()` wires the live runtime and **leaves `AWAY_AFTER_SEC` /
  `PHONE_AFTER_SEC` untouched**; OfficeSchedule consumes the persisted schedule.
- `settings.update` audits `settings.update` with a secret-free detail.
- Employee display-label rename keeps `employee_id`/`name`; empty label clears; duplicate
  create rejected.
- `SeatZoneStore.update` / `ChairStore.update` preserve identities; bad polygon rejected;
  orphan chair rejected.
- Assignment validation: valid targets pass; missing/disabled employee, missing/disabled
  desk, missing/disabled chair, chair-of-unknown-zone all rejected; valid assignment
  round-trips through stores.
- Enable/disable never deletes history (seat events preserved; employee rows kept).
- RBAC: admin allowed; security_operator/manager/viewer denied `configure_settings`;
  viewer denied users+zones+settings with audited denial.
- Identity treaty: UNKNOWN never relabelled; mismatch constants neutral.
- Dashboard: sidebar has **10** nav items incl. Admin Control Center; all 8 sub-pages render
  without exception under Streamlit AppTest; no credential text leaks into rendered error body.

### 7.2 Full regression
- `pytest -q` → **1267 passed** (1229 prior + 38 new).
- `pytest -q -W error` → **1267 passed** (warnings-as-errors clean).
- Nav-count assertions updated to 10 in `tests/test_dashboard_app.py` and
  `tests/test_phase38.py`.

### 7.3 Static / import
- `python -m py_compile` on all touched files: OK.
- Module imports (`app`, `main`, `src.admin_store`): OK.

---

## 8. Validation discipline

- **SIMULATED VALIDATION:** done — 38 new focused tests + 1267-test regression validate the
  entire admin/config pipeline against the existing stores and a real SQLite DB (no mock
  stores). The Streamlit AppTest harness exercised the real rendering paths of every admin
  sub-page.
- **REAL VALIDATION: NOT RUN.** Phase 62 is a configuration + dashboard layer; the only
  hardware-facing boundary (camera transport) is unchanged Phase 58 code consumed via the
  Phase 61 admin page. Live-camera/NVR validation is explicitly the subject of **Phase 63**.
- Reachability is never guessed; schedule/threshold inputs are validated before any write.

---

## 9. NOT validated (with real hardware)

- Live RTSP/NVR camera control from the Admin Control Center (Phase 63).
- Multi-camera behaviour under production load.

## 10. Model/scope limitations (no hardware)

- Desk/chair assignment is a **context hint**, not an identity grant; productivity claims
  remain face-confirmed only.
- Multi-employee desks/chairs are inherently unsupported by the single assignment slot.
- The thresholds knobs are authoritative configuration knobs; their consumption by future
  detection behaviours is by design (Phase 59c reserved knobs), no FSM behaviour changed here.

---

## 11. Regressions

None. Explicitly re-verified: Phase 57 face thresholds, Phase 58 camera layer, Phase 59
desk/chair tracking, Phase 59c schedule knobs, Phase 60A offline simulation, Phase 61 admin
camera management — all pass (1267 tests, incl. `-W error`).

## 12. Files touched

- `src/admin_store.py` (new) — SettingsStore / OfficeSettings / validation / assignment checks.
- `src/database.py` — `settings` schema + helpers; `set_employee_display_name`.
- `src/domain.py` — `PERM_CONFIGURE_SETTINGS`, admin grant, `Employee.display_name`.
- `src/rbac.py` — `AccessGuard.require_configure_settings`.
- `src/employees.py` — `display_name` plumbing + `set_display_name`.
- `src/seat_zones.py` / `src/seat_chairs.py` — `update()` (stable-identity edits).
- `main.py` — startup `SettingsStore(conn).apply_to_config()`.
- `app.py` — Admin Control Center tab (10 nav) + 8 sub-pages.
- `tests/test_phase62.py` (new); `tests/test_dashboard_app.py`, `tests/test_phase38.py`
  (nav count 9→10).