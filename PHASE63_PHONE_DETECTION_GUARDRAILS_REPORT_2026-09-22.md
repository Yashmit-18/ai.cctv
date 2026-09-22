# PHASE 63 — PHONE-DETECTION GUARDRAILS & CAMERA ADMIN KNOBS
### Persistent phone-pass detection tuning knobs wired end-to-end
**Date:** 2026-09-22
**Mode:** CODE VERIFIED + FULL SUITE — 1283 tests pass (16 new + 1267 pre-existing).

---

## 1. Executive summary

Phase 63 closes the phone-detection gap left open by Phase 62. Six previously env-only
phone-class detection knobs — class id, model path, resolution, cadence, evidence window
and evidence minimum — are now **persisted via the Admin Thresholds tab**, validated,
applied to the config module, and consumed live by the FSM threshold pipeline and the
phone-pass diagnostic panel.

Configuration flows as:

```
Admin Panel (Thresholds tab, "Phone detection pass" section)
   -> SettingsStore.update(...)   [src/admin_store.py]
      -> settings table           [key/value persisted]
         -> SettingsStore.apply_to_config()
            -> config.PHONE_CLASS / PHONE_MODEL_PATH / PHONE_IMGSZ
               / PHONE_DETECT_CADENCE / PHONE_EVIDENCE_WINDOW / PHONE_EVIDENCE_MIN
               [rebound ONLY when key is persisted; untouched otherwise]
               -> EmployeeTracker / MultiTracker read live per batch
```

**Backward-compatible:** an untouched settings table keeps every env default intact. An
env-tuned deployment that never opens the panel behaves identically to before Phase 63.

**Result:** 16 focused tests in `tests/test_phase63.py`. Full suite — **1283 passed**.

---

## 2. What was implemented

### 2.1 Admin store phone-pass knobs (`src/admin_store.py`)

New keys (Phase 63 only — additive):

| Key | Config attr | Default | Validator |
|-----|-------------|---------|-----------|
| `detection.phone_class` | `PHONE_CLASS` | `67` (COCO) | `validate_positive_int(>= 0)` |
| `detection.phone_model_path` | `PHONE_MODEL_PATH` | `""` (auto-resolved) | `validate_phone_model_path` (strips, 0-255) |
| `detection.phone_imgsz` | `PHONE_IMGSZ` | `1280` | `>= 32` |
| `detection.phone_cadence` | `PHONE_DETECT_CADENCE` | `3` | `>= 1` |
| `detection.phone_evidence_window` | `PHONE_EVIDENCE_WINDOW` | `5` | `>= 1` |
| `detection.phone_evidence_min` | `PHONE_EVIDENCE_MIN` | `2` | `>= 1`, `min <= window` |

- `OfficeSettings` dataclass: six new fields + `to_dict()` entries.
- `validate_phone_model_path(name, value)`: strips whitespace, empty string kept (auto-resolve).
- `validate_office_settings()`: validates all six knobs, returns validated dict.
- `apply_to_config()`: rewrites `config.PHONE_CLASS / PHONE_MODEL_PATH / PHONE_IMGSZ /
  PHONE_DETECT_CADENCE / PHONE_EVIDENCE_WINDOW / PHONE_EVIDENCE_MIN` **only when the
  corresponding key is in `keys_set()`** (never on a missing/partially-persisted key).
  Phase 62 `FUTURE_AWAY / FUTURE_PHONE / FUTURE_TALKING_THRESHOLD_SECONDS` still set
  unconditionally (preserved).

### 2.3 FSM live-threshold wiring (`src/tracker.py`)

- New helper `_fsm_seconds(name, default)` — reads `getattr(config, name, default)` live
  on each invocation, so the tracker follows post-apply_to_config rebinds.
- `EmployeeTracker.__init__(..., *, away_after_sec=None, phone_after_sec=None,
  phone_gap_grace_sec=None)` — when `None`, reads live via `_fsm_seconds`; otherwise uses
  the explicit override (admin/persisted value).
- `MultiTracker.__init__` accepts the same kwargs and forwards per-tracker.
- Absence check: `self._away_after_sec` replaces module-level `AWAY_AFTER_SEC`.
- Phone threshold crossing: `self._phone_after_sec` replaces `PHONE_AFTER_SEC`.
- Phone gap grace: `self._phone_gap_grace_sec` replaces `PHONE_GAP_GRACE_SEC`.

### 2.3 Main loop wiring (`main.py`)

- Phase 62 `_p62_store` + `_p62_keys` block rewritten: when `KEY_AWAY` or `KEY_PHONE` are
  in `keys_set()`, builds explicit `_fsm_away_sec` / `_fsm_phone_sec` and passes them
  directly to `MultiTracker(conn, away_after_sec=..., phone_after_sec=...)`.
- `_write_live_state(..., phone_pass={})` — accepts and publishes the phone-pass payload
  in `live_state["ai"]["phone_pass"]`.
- Main loop captures `phone_pass_payload` from `detector.phase44_diagnostics()["phone_pass"]`
  and passes it to `_write_live_state`.

### 2.4 Dashboard surfaces (`app.py`)

**Admin → Thresholds tab** (Phase 63 section added after Phase 62 fields):
- `p_class` (int 0..999, COCO class id)
- `p_model` (text path — empty = auto-resolve, 0..255 chars)
- `p_imgsz` (int >= 32, step 32)
- `p_cadence` (int >= 1, per-camera cycle gating)
- `p_ev` (int >= 1, evidence window)
- `p_ev_min` (int >= 1, evidence min, help text)
- On save: validates (min <= window enforced), persists, applies, success banner.
- Caption updated: "Unpersisted knobs keep env defaults."

**Admin → Overview tab:**
- Camera identity section now shows `phone-pass:` summary line: class, imgsz, cadence,
  model path, evidence min/window (from store defaults when not persisted).

**Live Monitoring → AI Capabilities tab:**
- "Phone Detection Pass (Phase 44/63)" panel: Source, Class, Imgsz, Cadence, Calls,
  Phones seen, Conf mean, Conf min — rendered as `st.dataframe` + caption.
- "Phone/Person Association (CCTV_PHONE_DIAG=1)" panel: Camera, Cycles, Persons,
  Phones, Conf mean, Matched, Unmatched — rendered as `st.dataframe` + caption.

### 2.5 Lazy ActivityDetector phone knobs (`src/detector.py`)

`ActivityDetector` now reads phone-class knobs lazily via `getattr(config, ...)` at
`__init__`, falling back to the module-level env-derived constants (`CONFIG_PHONE_CLASS`,
`PHONE_MODEL_PATH`, `PHONE_IMGSZ`, `PHONE_DETECT_CADENCE`) when the attr is missing.
`PhoneDetector` is built from the resolved values; `_phone_cadence` is set from the live
cadence and used for per-camera cycle gating. This ensures the detector picks up
`apply_to_config()` rebinds when the daemon (re)builds the detector.

---

## 3. Test coverage

`tests/test_phase63.py` — **16 tests**:

| # | Test | Verifies |
|---|------|----------|
| 1 | `test_phone_pass_defaults_match_config_env_knobs` | admin defaults == config defaults |
| 2 | `test_phone_pass_persistence_roundtrip_marks_keys` | full set persists all 6 keys |
| 3 | `test_phone_pass_update_merges_not_resets` | partial update preserves already-set knobs |
| 4 | `test_phone_pass_validation_rejects_bad_values` | bad class/imgsz/cadence/ev rejected |
| 5 | `test_validate_office_settings_accepts_full_phase63_payload` | full valid payload accepted |
| 6 | `test_apply_to_config_rebinds_only_persisted_phone_knobs` | only persisted keys rebound |
| 7 | `test_apply_to_config_empty_settings_never_touches_env_knobs` | untouched keys stay at env |
| 8 | `test_fsm_phone_threshold_explicit_param_honored` | explicit param = instant phone state |
| 9 | `test_fsm_phone_threshold_live_config_default` | live config (3.0) delayed phone state |
| 10 | `test_fsm_away_threshold_live_config_default` | live config (3.0) delayed away state |
| 11 | `test_multitracker_forwards_fsm_thresholds` | MT passes explicit thresholds to tracker |
| 12 | `test_main_wires_admin_thresholds_into_multitracker` | main wires persisted knobs into MT |
| 13 | `test_write_live_state_publishes_phone_pass` | _write_live_state includes phone_pass |
| 14 | `test_write_live_state_phone_pass_defaults_empty` | absent phone_pass → empty dict |
| 15 | `test_app_thresholds_source_has_phone_pass_section` | app.py source has Phase 63 fields |
| 16 | `test_ai_capabilities_renders_phone_pass_panel` | AppTest renders the phone-pass subheader |

Full suite — **1283 passed** (110 s).

---

## 4. Files modified

| File | Change |
|------|--------|
| `src/admin_store.py` | 6 phone keys + validators + apply_to_config rebind-only-persisted logic |
| `src/tracker.py` | `_fsm_seconds`, explicit threshold params, instance-bound thresholds |
| `src/detector.py` | Lazy `getattr(config, ...)` for phone knobs |
| `main.py` | Phase 62/63 setup → MultiTracker explicit args + `_write_live_state` phone_pass |
| `app.py` | Thresholds phone-pass section + AI Capabilities phone_pass/diag panels |
| `tests/test_phase63.py` | **NEW** — 16 tests |

---

## 5. Backward compatibility

- **Untouched settings table:** every env default preserved; behavior identical to Phase 62.
- **Env-tuned deployments:** phone knobs bound at `apply_to_config()` time only when
  `keys_set()` contains the key — absent keys never written to `config`, env untouched.
- **FSM thresholds:** `config.AWAY_AFTER_SEC` env default is **3.0** (not 10.0); admin
  `AWAY_SECONDS_DEFAULT=10.0`. The tracker uses live reads; explicit params flow only when
  the corresponding `KEY_AWAY` / `KEY_PHONE` is persisted.
- **1267 pre-existing tests:** all pass — no regressions introduced.

---

## 6. Deferred

- **Live RTSP / NVR camera control** (referenced informally in Phase 62 report as "Phase 63")
  remains deferred to a future phase pending physical hardware availability.
- **Phone model auto-download** when `PHONE_MODEL_PATH=""` (auto-resolve) depends on the
  YOLO model cache; the admin panel shows "default" when empty, no download triggered from
  the panel itself.

---

*Generated 2026-09-22 after full regression pass (1283/1283).*
