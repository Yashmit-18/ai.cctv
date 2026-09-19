"""Phase 50 offline ReID threshold-calibration utility.

Consumes captured (or later, real) employee crops keyed by employee + pose,
embeds them with any available extractor (built-in or OSNet), computes
same-person / different-person similarity distributions and best-vs-second-best
margins, scans threshold/margin operating points, and RECOMMENDS
``REID_STRONG_THRESHOLD`` / ``REID_MARGIN_MIN`` values.

Hard rules (Phase 50 / Phase 51):
  * This module NEVER modifies ``config`` or ``config.py``.  The output is a
    report/JSON that a human operator may choose to apply later.
  * Missing employees are reported, never invented.
  * Missing ``data/calibration`` yields an honest NOT VALIDATED report, never
    synthetic pass-through as if it were real data.
  * Synthetic data may only be labelled synthetic.
  * **Provenance (Phase 51):** files existing is never enough to call data
    "REAL LABELED DATA".  Only a validated ``manifest.json`` may grant that
    status; without one the report states "provenance NOT proven".
  * **Model identity (Phase 51):** calibration records the exact checkpoint,
    variant and family.  A requested OSNet checkpoint that fails to load is a
    FAIL, never a silent fallback to the built-in descriptor.

Data layout expected::

    data/calibration/
        EMP001/front/  front_001.png ...
        EMP001/left/ ...
        EMP001/right/ ...
        EMP001/45-degree/ ...
        EMP001/partial/ ...
        EMP001/back/ ...
        EMP002/...
        EMP004/...
        manifest.json            (optional; required for REAL LABELED DATA)

``EMP003/`` may be absent -- it is reported as a missing employee.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AVAILABLE_POSES = ("front", "left", "right", "45-degree", "partial", "back")
POSES_HR = {
    "front": "front",
    "left": "left",
    "right": "right",
    "45-degree": "45-degree",
    "partial": "partial",
    "back": "back",
}
DEFAULT_DATA_DIR = ROOT / "data" / "calibration"
DEFAULT_OUT = ROOT / "data" / "calibration_report.json"

#: Synthetic seed used ONLY when a caller explicitly requests synthetic crops.
SYNTH_SEED = 42

# -- Phase 51 constants ------------------------------------------------------
#: Phase 50 capture minimum per pose (a capture target, NOT statistical
#: sufficiency -- never described as such).
MIN_EXPECTED_FRAMES_PER_POSE = 8
#: Minimum embedded employees before a recommendation can be attempted.  With a
#: single identity there is no different-person distribution, so FAR is
#: unmeasurable and any "calibration" would be unsafe.
MIN_EMPLOYEES = 2
#: Minimum embedded samples per employee (leave-one-out needs a real gallery).
MIN_SAMPLES_PER_EMPLOYEE = 4
#: Feed-gate prerequisites.
MIN_FEED_FRAMES = 3
FEED_REQUIRE_FPS = 15.0
#: Manifest file inside the calibration data directory.
MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1

#: Dataset status vocabulary (strict labels).
STATUS_REAL = "REAL LABELED DATA"
STATUS_SIMULATED = "SIMULATED"
STATUS_NOT_VALIDATED = "NOT VALIDATED"
STATUS_PROVENANCE_UNPROVEN = "FILES PRESENT -- PROVENANCE NOT PROVEN"


@dataclass
class CalibrationSample:
    emp_id: str
    pose: str
    source: str          # "real" | "synthetic"
    embedding: np.ndarray | None = None


@dataclass
class Distribution:
    n: int
    mean: float
    p01: float
    p05: float
    p50: float
    p95: float
    p99: float
    min: float
    max: float

    @staticmethod
    def of(values: np.ndarray) -> "Distribution":
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return Distribution(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0)
        return Distribution(
            n=int(arr.size),
            mean=round(float(arr.mean()), 4),
            p01=round(float(np.percentile(arr, 1)), 4),
            p05=round(float(np.percentile(arr, 5)), 4),
            p50=round(float(np.percentile(arr, 50)), 4),
            p95=round(float(np.percentile(arr, 95)), 4),
            p99=round(float(np.percentile(arr, 99)), 4),
            min=round(float(arr.min()), 4),
            max=round(float(arr.max()), 4),
        )


@dataclass
class OperatingPoint:
    threshold: float
    margin: float
    far: float             # fraction of different-person pairs accepted
    frr: float             # fraction of same-person pairs rejected
    balance: float         # == far + frr (lower is better)
    far_count: int
    frr_count: int
    same_n: int
    diff_n: int


@dataclass
class CalibrationReport:
    label: str
    model_source: str
    feature_dim: int
    employees_seen: list[str] = field(default_factory=list)
    employees_missing: list[str] = field(default_factory=list)
    missing_poses: dict[str, list[str]] = field(default_factory=dict)
    empty_poses: dict[str, list[str]] = field(default_factory=dict)
    duplicates: list[dict] = field(default_factory=list)
    unsupported_files: list[dict] = field(default_factory=list)
    image_sizes: dict[str, int] = field(default_factory=dict)
    failed_files: list[dict] = field(default_factory=list)
    samples_total: int = 0
    samples_embedded: int = 0
    samples_failed: int = 0
    files_discovered_raw: int = 0
    same: Distribution | None = None
    diff: Distribution | None = None
    margins: Distribution | None = None
    operating_points: list[dict] = field(default_factory=list)
    recommended: dict | None = None
    config_unchanged: bool = True
    note: str = ""
    # Phase 51 -- provenance + model identity
    dataset_status: str = STATUS_NOT_VALIDATED
    manifest_status: str = "MISSING"
    manifest_validation: dict = field(default_factory=dict)
    checkpoint: str | None = None
    model_variant: str | None = None
    model_identity: dict = field(default_factory=dict)
    dataset_fingerprint: str | None = None


# ---------------------------------------------------------------------------
# Data discovery + embedding
# ---------------------------------------------------------------------------


def _pixel_hash(path: Path) -> str:
    """Deterministic content hash of the DECODED pixels (survives re-save)."""
    import cv2 as cv
    img = cv.imread(str(path))
    if img is None:
        return ""
    return hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()


def _file_bytes_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_data_dir(data_dir=Path | str | None,
                  expected_employees=()) -> dict:
    """Return crop survey + full data-integrity diagnostics (Phase 51).

    Deterministic: directories and images are always visited in sorted order.
    Adds to the Phase 50 scan:
      * ``employees_missing``  -- rostered IDs with no usable crops;
      * ``missing_poses``      -- employee folder exists, pose dir absent;
      * ``empty_poses``        -- pose dir exists but holds no jpg/png;
      * ``duplicates``         -- pixel-identical crops (incl. cross-employee);
      * ``unsupported_files``  -- non-jpg/png files inside pose dirs;
      * ``image_sizes``        -- ``{WxH: count}`` across all unique crops;
      * ``all_files``          -- every supported crop path (nothing silent).

    Pixel-identical duplicates are removed from ``samples`` (first occurrence
    kept); every duplicate is still reported so no file vanishes silently.
    """
    root = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    result: dict[str, dict[str, list]] = {}
    seen_hash: dict[str, Path] = {}
    duplicates: list[dict] = []
    empty_poses: dict[str, list[str]] = {}
    missing_poses: dict[str, list[str]] = {}
    unsupported_files: list[dict] = []
    unreadable_files: list[dict] = []
    image_sizes: dict[str, int] = {}
    all_files: list[str] = []
    count_raw = 0

    if not root.is_dir():
        return {"samples": {}, "employees_seen": [], "employees_missing":
                sorted(expected_employees),
                "missing_poses": {}, "empty_poses": {}, "duplicates": [],
                "unsupported_files": [], "unreadable_files": [],
                "image_sizes": {}, "all_files": [], "files_discovered_raw": 0}

    for emp_dir in sorted(root.iterdir()):
        if not emp_dir.is_dir() or emp_dir.name == "__pycache__":
            continue
        emp = emp_dir.name
        for pose in AVAILABLE_POSES:
            pose_dir = emp_dir / pose
            if not pose_dir.is_dir():
                missing_poses.setdefault(emp, []).append(pose)
                continue
            imgs = sorted(
                list(pose_dir.glob("*.jpg")) + list(pose_dir.glob("*.png")))
            other = [p for p in sorted(pose_dir.iterdir())
                     if p.name not in (MANIFEST_NAME,) and
                     p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp",
                                              ".webp", ".tiff")]
            for p in other:
                unsupported_files.append({
                    "path": str(p.relative_to(root)).replace("\\", "/"),
                    "employee": emp, "pose": pose})
            if not imgs:
                empty_poses.setdefault(emp, []).append(pose)
                continue
            keep: list[Path] = []
            for path in imgs:
                count_raw += 1
                rel = str(path.relative_to(root)).replace("\\", "/")
                img = _decode_or_none(path)
                if img is None:
                    unreadable_files.append(
                        {"path": rel, "employee": emp, "pose": pose,
                         "reason": "decode failed / corrupt"})
                    continue
                h = _pixel_hash(path)
                key = f"{path.stat().st_size}:{h}"
                if key in seen_hash:
                    duplicates.append({
                        "path": rel,
                        "duplicate_of": str(seen_hash[key].relative_to(root))
                        .replace("\\", "/"),
                        "employee": emp, "pose": pose})
                    continue
                seen_hash[key] = path
                keep.append(path)
                all_files.append(rel)
                wh = f"{img.shape[1]}x{img.shape[0]}"
                image_sizes[wh] = image_sizes.get(wh, 0) + 1
            if keep:
                result.setdefault(emp, {})[pose] = keep

    employees_seen = sorted(result)
    return {"samples": result, "employees_seen": employees_seen,
            "employees_missing": sorted(
                e for e in expected_employees if e not in employees_seen),
            "missing_poses": missing_poses, "empty_poses": empty_poses,
            "duplicates": duplicates,
            "unsupported_files": unsupported_files,
            "unreadable_files": unreadable_files,
            "image_sizes": image_sizes,
            "all_files": all_files, "files_discovered_raw": count_raw}


def _decode_or_none(path: Path):
    """Return decoded BGR ndarray or ``None`` without raising."""
    try:
        import cv2 as cv
        img = cv.imread(str(path))
        if img is None or img.size == 0:
            return None
        return img
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Manifest (Phase 51) -- provenance required for REAL LABELED DATA
# ---------------------------------------------------------------------------


def load_manifest(data_dir=Path | str | None) -> dict | None:
    """Load ``data/calibration/manifest.json`` or return ``None`` (corrupt or
    absent => ``None`` so callers never trust a broken manifest)."""
    root = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    path = root / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict) or int(data.get("manifest_version", 0)) \
            != MANIFEST_VERSION:
        return None
    return data


def validate_manifest(manifest: dict | None, samples: dict) -> dict:
    """Check a manifest REALLY proves provenance of every unique sample.

    Returns ``{"ok": bool, "status": str, "missing": [...], "mismatches":
    [...], "records_unreferenced": int, "capture": dict}``.  ``ok`` requires:
    status == REAL LABELED DATA, every sample file has a record whose
    ``employee_id`` + ``pose`` match, and no mismatches.
    """
    base = {"ok": False, "status": MANIFEST_VERSION and STATUS_NOT_VALIDATED,
            "missing": [], "mismatches": [], "records_unreferenced": 0,
            "capture": {}}
    if not manifest:
        return base
    status = str(manifest.get("dataset_status", ""))
    files = manifest.get("files", {})
    if not isinstance(files, dict):
        return dict(base, status=status)
    all_expected = sorted(
        f"{emp}/{pose}/{Path(p).name}"
        for emp, poses in samples.items()
        for pose, paths in poses.items()
        for p in paths)
    missing = [fp for fp in all_expected
               if fp.replace("\\", "/") not in files]
    mismatches = []
    for fp, rec in files.items():
        if not isinstance(rec, dict):
            continue
        parts = fp.replace("\\", "/").split("/")
        if len(parts) < 3:
            continue
        emp, pose = parts[0], parts[1]
        if rec.get("employee_id") not in (None, emp) or \
                rec.get("pose") not in (None, pose):
            mismatches.append(fp)
    records_unreferenced = len(
        [fp for fp in files if fp.replace("\\", "/")
         not in {p.replace("\\", "/") for p in all_expected}])
    ok = (status == STATUS_REAL and not missing and not mismatches)
    return {"ok": ok, "status": status, "missing": missing,
            "mismatches": mismatches,
            "records_unreferenced": records_unreferenced,
            "capture": manifest.get("capture", {})}


def dataset_fingerprint(data_dir=Path | str | None,
                        survey: dict | None = None) -> str:
    """Deterministic fingerprint of the dataset (paths + size + dimensions).

    Does not read every crop's bytes -- covers file identity and structure.
    """
    if survey is None:
        survey = scan_data_dir(data_dir)
    h = hashlib.sha256()
    for fp in sorted(survey["all_files"]):
        h.update(fp.encode("utf-8"))
    for dup in survey.get("duplicates", []):
        h.update(("dup:" + dup["path"] + "->" + dup["duplicate_of"])
                 .encode("utf-8"))
    for emp, poses in survey.get("samples", {}).items():
        for pose, paths in poses.items():
            for p in paths:
                try:
                    sz = Path(p).stat().st_size
                except OSError:
                    sz = 0
                h.update(f"{emp}/{pose}/{Path(p).name}:{sz}".encode("utf-8"))
    return h.hexdigest()


def checkpoint_identity(model_path: str) -> dict | None:
    """Filename + size + sha256 of a checkpoint (Phase 51 model identity)."""
    p = Path(model_path)
    if not p.is_file():
        return None
    return {"checkpoint": p.name,
            "family": "osnet" if p.suffix.lower() in (".pth", ".pt")
            else ("onnx" if p.suffix.lower() == ".onnx" else "unknown"),
            "bytes": p.stat().st_size,
            "sha256": _file_bytes_sha256(p)}


def embed_samples(samples: dict, extractor) -> tuple:
    """Embed every unique crop; return per-employee embeddings + stats.

    Returns ``(per_emp, stats)`` where ``per_emp`` is ``{emp: {pose: [emb]}}``
    and stats holds total/embedded/failed counts.  Unreadable crops are counted
    as failures AND recorded per-path in ``failed_files`` (never silently
    dropped, never fabricated).
    """
    per_emp: dict = {}
    total = failed = embedded = 0
    missing_poses: dict[str, list[str]] = {}
    failed_files: list[dict] = []
    for emp, poses in samples.items():
        per_emp[emp] = {}
        missing = []
        for pose in AVAILABLE_POSES:
            paths = poses.get(pose, [])
            per_emp[emp][pose] = []
            for path in paths:
                total += 1
                try:
                    import cv2 as cv
                    img = cv.imread(str(path))
                    if img is None:
                        failed += 1
                        failed_files.append(
                            {"path": str(path), "reason": "decode failed"})
                        continue
                    emb = extractor.embed(img)
                except Exception as exc:
                    failed += 1
                    failed_files.append(
                        {"path": str(path), "reason": str(exc)})
                    continue
                if emb is None:
                    failed += 1
                    failed_files.append(
                        {"path": str(path),
                         "reason": "extractor returned no embedding"})
                    continue
                per_emp[emp][pose].append(np.asarray(emb, dtype=np.float32))
                embedded += 1
            if not per_emp[emp][pose]:
                missing.append(pose)
        if missing:
            missing_poses[emp] = missing
    stats = {"samples_total": total, "samples_embedded": embedded,
             "samples_failed": failed, "missing_poses": missing_poses,
             "failed_files": failed_files}
    return per_emp, stats


def collect_employee_embeddings(per_emp: dict) -> dict:
    """Flatten ``{emp: {pose: [emb]}}`` into ``{emp: [emb]}`` in sorted order."""
    return {emp: [e for pose in AVAILABLE_POSES
                  for e in per_emp.get(emp, {}).get(pose, [])]
            for emp in sorted(per_emp)}


def _cos(a, b) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 1e-8 or nb <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pairwise_similarities(by_emp: dict) -> tuple:
    """Return ``(same_person, different_person)`` cosine arrays.

    Same-person pairs include every pose combination within one employee
    (front-vs-back etc.), which is exactly what side/back validation will feed.
    """
    same: list[float] = []
    diff: list[float] = []
    for emp, embs in by_emp.items():
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                same.append(_cos(embs[i], embs[j]))
    emps = list(by_emp)
    for a in range(len(emps)):
        for b in range(a + 1, len(emps)):
            for ea in by_emp[emps[a]]:
                for eb in by_emp[emps[b]]:
                    diff.append(_cos(ea, eb))
    return np.array(same, dtype=np.float64), np.array(diff, dtype=np.float64)


def best_vs_second_margins(by_emp: dict) -> np.ndarray:
    """Margin per probe embedding: best - second-best similarity to the gallery."""
    emps = list(by_emp)
    margins: list[float] = []
    for probe_emp, embs in by_emp.items():
        others = [e for e in emps if e != probe_emp]
        recruits: dict[str, list] = {}
        for o in others:
            recruits[o] = []
        for probe in embs:
            scored: dict[str, float] = {probe_emp: 0.0}
            for emp2, embs2 in by_emp.items():
                for g in embs2:
                    s = _cos(probe, g)
                    if emp2 in scored:
                        scored[emp2] = max(scored[emp2], s)
                    else:
                        scored[emp2] = s
            top = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
            if len(top) >= 2:
                margins.append(top[0][1] - top[1][1])
    return np.array(margins, dtype=np.float64)


# ---------------------------------------------------------------------------
# Threshold / margin analysis
# ---------------------------------------------------------------------------

def _l2normalize(a):
    arr = np.asarray(a, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(arr))
    return arr / n if n > 1e-8 else arr


def _probe_decisions(by_emp: dict, threshold: float, margin: float) -> dict:
    """Decide every sample exactly like production ``identify`` does.

    Gallery = L2-normalised mean embedding per employee, built leave-one-out
    (a sample never scores against itself).  A sample is a STRONG match to its
    own employee only when argmax == own employee, best >= threshold and
    best-second >= margin.  Returned dict carries ``strong_own``, ``wrong``
    (strong to another employee), ``weak`` counts and n.
    """
    emps = list(by_emp)
    totals: dict[str, np.ndarray] = {}
    for e in emps:
        totals[e] = np.sum(np.stack(by_emp[e]), axis=0).astype(np.float64)
    strong_own = wrong = weak = 0
    for emp, embs in by_emp.items():
        n_samp = len(embs)
        for i, probe in enumerate(embs):
            if n_samp > 1:
                gal_own = _l2normalize((totals[emp] - probe.astype(np.float64))
                                       / (n_samp - 1))
            else:
                gal_own = None
            sims: list[tuple[str, float]] = []
            for e2 in emps:
                if e2 == emp and n_samp > 1:
                    g = gal_own
                else:
                    g = totals[e2] / len(by_emp[e2])
                g = _l2normalize(g)
                sims.append((e2, float(np.dot(_l2normalize(probe), g))))
            sims.sort(key=lambda kv: kv[1], reverse=True)
            best_id, best = sims[0]
            second = sims[1][1] if len(sims) > 1 else 0.0
            m = best - second
            strong = best >= float(threshold) and m >= float(margin)
            if not strong:
                weak += 1
            elif best_id == emp:
                strong_own += 1
            else:
                wrong += 1
    return {"strong_own": strong_own, "wrong": wrong, "weak": weak,
            "n": strong_own + wrong + weak}


def scan_operating_points(by_emp: dict, threshold_grid=None,
                          margin_grid=None) -> list[OperatingPoint]:
    """Evaluate every (threshold, margin) combo via probe decisions.

    ``far`` = fraction of samples strongly matched to the WRONG employee,
    ``frr`` = fraction of same-person samples that failed (weak).  Deterministic
    ordering: threshold ascending, then margin ascending.
    """
    if threshold_grid is None:
        threshold_grid = np.round(np.arange(0.30, 0.951, 0.025), 3)
    if margin_grid is None:
        margin_grid = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.10])
    points: list[OperatingPoint] = []
    for t in threshold_grid:
        for m in margin_grid:
            dec = _probe_decisions(by_emp, float(t), float(m))
            n = max(1, dec["n"])
            points.append(OperatingPoint(
                threshold=round(float(t), 4), margin=round(float(m), 4),
                far=round(dec["wrong"] / n, 4),
                frr=round(dec["weak"] / n, 4),
                balance=round((dec["wrong"] + dec["weak"]) / n, 4),
                far_count=dec["wrong"], frr_count=dec["weak"],
                same_n=n, diff_n=n))
    return points


def recommend(by_emp: dict, safety_far_max: float = 0.01,
              frr_ceiling: float = 0.5, min_employees: int = MIN_EMPLOYEES,
              min_samples_per_employee: int = MIN_SAMPLES_PER_EMPLOYEE) -> dict:
    """Recommend (threshold, margin) minimising FRR subject to caps.

    An operating point is only eligible when it meets BOTH caps: false-accept
    <= ``safety_far_max`` AND false-reject <= ``frr_ceiling``.  A model that
    can only avoid false accepts by rejecting everything must be refused, so
    the caller never hands a degenerate config to production.  Structural
    guards refuse before any scan when the input cannot support a safe
    verdict (Phase 51):
      * fewer than ``min_employees`` embedded employees (FAR unmeasurable),
      * any employee with < ``min_samples_per_employee`` samples (a
        leave-one-out gallery cannot be trusted),
      * no valid samples at all.
    Every refusal carries an explicit ``reason``; never mutates config.
    """
    by_emp = {e: list(v) for e, v in by_emp.items()
              if e and v is not None and len(v) > 0}
    if not by_emp:
        return {"eligible": False, "threshold": None, "margin": None,
                "far_at_recommendation": None, "frr_at_recommendation": None,
                "safety_far_max": safety_far_max,
                "frr_ceiling": frr_ceiling,
                "reason": "no valid embeddings to calibrate on"}
    if len(by_emp) < max(2, int(min_employees)):
        return {"eligible": False, "threshold": None, "margin": None,
                "far_at_recommendation": None, "frr_at_recommendation": None,
                "safety_far_max": safety_far_max,
                "frr_ceiling": frr_ceiling,
"reason": (f"only {len(by_emp)} employee(s) with embeddings; "
                   f"{int(min_employees)} needed because FAR is "
                   "unmeasurable with a single identity")}
    shortages = {e: len(v) for e, v in by_emp.items()
                 if len(v) < int(min_samples_per_employee)}
    if shortages:
        return {"eligible": False, "threshold": None, "margin": None,
                "far_at_recommendation": None, "frr_at_recommendation": None,
                "safety_far_max": safety_far_max,
                "frr_ceiling": frr_ceiling,
                "reason": ("insufficient samples per employee: "
                           + ", ".join(f"{e}={n}" for e, n in
                                       sorted(shortages.items()))
                           + f" (min {int(min_samples_per_employee)} needed "
                           "for a trustworthy leave-one-out gallery)")}
    points = scan_operating_points(by_emp)
    eligible = [p for p in points if p.far <= safety_far_max
                and p.frr <= frr_ceiling]
    if not eligible:
        p0 = min(points, key=lambda p: (p.far, p.balance))
        return {"eligible": False, "threshold": None, "margin": None,
                "far_at_recommendation": None, "frr_at_recommendation": None,
                "safety_far_max": safety_far_max,
                "frr_ceiling": frr_ceiling, "reason":
                    "no operating point meets BOTH the FAR safety cap and "
                    "the FRR quality ceiling; the model cannot separate "
                    "these employees at the current sample coverage "
                    "(either it confuses them, or the only safe points "
                    "reject everything)"}
    best = min(eligible, key=lambda p: (p.frr, p.balance))
    return {"eligible": True,
            "threshold": round(float(best.threshold), 4),
            "margin": round(float(best.margin), 4),
            "far_at_recommendation": round(float(best.far), 4),
            "frr_at_recommendation": round(float(best.frr), 4),
            "same_n": best.same_n, "diff_n": best.diff_n,
            "safety_far_max": safety_far_max,
            "frr_ceiling": frr_ceiling,
            "reason": "minimised false-reject rate under the configured "
                      "false-accept safety cap and FRR quality ceiling"}


def recommended_config(model_source: str, rec: dict | None) -> dict:
    """Translate a recommendation into proposed config values (NEVER applied).

    Returns the proposed ``REID_STRONG_THRESHOLD`` / ``REID_MARGIN_MIN`` for a
    human operator to review.  This function performs no I/O and never touches
    ``config``.
    """
    if not rec or not rec.get("eligible"):
        return {"REID_STRONG_THRESHOLD": "keep-current",
                "REID_MARGIN_MIN": "keep-current",
                "model_source": model_source,
                "note": "no eligible calibrated point; keep current config"}
    t = float(rec["threshold"])
    m = float(rec["margin"])
    # Never recommend anything weaker than the current conservative floor.
    t = round(min(max(t, 0.50), 0.99), 4)
    m = round(max(m, 0.0), 4)
    return {"REID_STRONG_THRESHOLD": t, "REID_MARGIN_MIN": m,
            "model_source": model_source,
            "note": "proposed values -- apply manually after review"}


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


def build_report(*, extractor, model_path: str | None = None,
                 data_dir=None, survey=None,
                 expected_employees=("EMP001", "EMP002", "EMP003", "EMP004")) \
        -> CalibrationReport:
    """Embed the calibration directory and produce a full report.

    ``survey`` may be the dict from ``scan_data_dir`` (reused to avoid
    double discovery).  ``expected_employees`` is the rostered set whose
    absence must be reported explicitly.  Missing data => report with ``note``
    set and no distributions (never fabricates).  Phase 51: the ``REAL LABELED
    DATA`` status requires a validated manifest proving provenance -- PNG files
    existing is never sufficient.
    """
    if survey is None:
        survey = scan_data_dir(data_dir, expected_employees)
    samples = survey["samples"]
    _ensure_model_fail_closed(model_path, extractor)
    source = str(getattr(extractor, "model_source", "built-in"))
    dim = int(getattr(extractor, "feature_dim", 256))
    report = CalibrationReport(label="NOT VALIDATED -- no calibration data "
                               "present", model_source=source, feature_dim=dim)
    report.checkpoint = getattr(extractor, "checkpoint", None)
    report.model_variant = getattr(extractor, "model_variant", None)
    report.model_identity = {
        "family": source,
        "variant": report.model_variant,
        "checkpoint": report.checkpoint,
        "feature_dim": dim,
    }
    report.dataset_fingerprint = dataset_fingerprint(data_dir, survey)
    if not samples:
        report.dataset_status = STATUS_NOT_VALIDATED
        report.manifest_status = "MISSING" if load_manifest(data_dir) is None \
            else "PRESENT"
        report.employees_missing = sorted(expected_employees)
        report.note = (f"no calibration crops under {data_dir or DEFAULT_DATA_DIR}; "
                       "thresholds CANNOT be calibrated. Production config "
                       "left untouched.")
        return report

    manifest = load_manifest(data_dir)
    man_val = validate_manifest(manifest, samples)
    report.manifest_status = "MISSING" if manifest is None else "PRESENT"
    report.manifest_validation = man_val

    per_emp, stats = embed_samples(samples, extractor)
    report.samples_total = stats["samples_total"]
    report.samples_embedded = stats["samples_embedded"]
    report.samples_failed = stats["samples_failed"]
    report.failed_files = stats["failed_files"]
    report.missing_poses = stats["missing_poses"]
    report.empty_poses = survey["empty_poses"]
    report.duplicates = survey["duplicates"]
    report.unsupported_files = survey["unsupported_files"]
    report.image_sizes = survey["image_sizes"]
    report.files_discovered_raw = survey["files_discovered_raw"]
    report.employees_seen = sorted(per_emp)
    report.employees_missing = sorted(
        e for e in expected_employees if e not in per_emp or not per_emp[e])

    by_emp = collect_employee_embeddings(per_emp)
    by_emp = {e: [x for x in v if x is not None]
              for e, v in by_emp.items()}
    report.employees_seen = [e for e in sorted(by_emp) if by_emp[e]]

    if not by_emp:
        report.dataset_status = STATUS_NOT_VALIDATED
        report.recommended = recommend(by_emp)
        report.note = ("no usable embeddings (all crops corrupt/failed); "
                       "thresholds CANNOT be calibrated. Production config "
                       "left untouched.")
        return report

    if man_val["ok"]:
        report.dataset_status = STATUS_REAL
        report.label = "REAL LABELED DATA -- CODE VERIFIED"
    else:
        report.dataset_status = STATUS_PROVENANCE_UNPROVEN
        report.label = ("FILES PRESENT -- PROVENANCE NOT PROVEN "
                        "-- NOT VALIDATED")

    same, diff = pairwise_similarities(by_emp)
    marg = best_vs_second_margins(by_emp)
    report.same = Distribution.of(same)
    report.diff = Distribution.of(diff)
    report.margins = Distribution.of(marg)
    rec = recommend(by_emp)
    report.recommended = rec
    pts = scan_operating_points(by_emp)
    report.operating_points = [
        {"threshold": p.threshold, "margin": p.margin, "far": p.far,
         "frr": p.frr, "balance": p.balance}
        for p in (pts[:25] + pts[-10:])]
    report.config_unchanged = True
    report.note = "report only -- production config NOT modified"
    return report


def _ensure_model_fail_closed(model_path, extractor) -> None:
    """Phase 51: a requested OSNet/ONNX checkpoint must load or FAIL loudly.

    Production's intentional built-in fallback stays untouched, but a
    calibration that requested a checkpoint must never silently compute on
    another model.
    """
    if not model_path:
        return
    p = Path(str(model_path))
    if not p.is_file():
        raise FileNotFoundError(
            f"calibration requested checkpoint {model_path} which does not "
            "exist; refusing to run (no silent fallback)")
    wanted = "osnet" if p.suffix.lower() in (".pth", ".pt") else "onnx"
    actual = str(getattr(extractor, "model_source", ""))
    if actual != wanted:
        raise RuntimeError(
            f"calibration requested {wanted} checkpoint {p.name} but the "
            f"extractor loaded as '{actual}'; refusing to calibrate on the "
            "wrong model (no silent fallback)")


# ---------------------------------------------------------------------------
# Production safety gate (Part 7)
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    gates: dict[str, bool]
    all_gates: bool
    reasons: list[str]


_PROD_GATES = ("real_calibration_data", "thresholds_calibrated",
               "margin_calibrated", "identity_safety_tests",
               "real_camera_validation")


def production_gate(*, real_calibration_data: bool = False,
                    thresholds_calibrated: bool = False,
                    margin_calibrated: bool = False,
                    identity_safety_tests: bool = False,
                    real_camera_validation: bool = False,
                    report=None) -> GateResult:
    """Decide whether OSNet may become the PRODUCTION ReID backend.

    All five gates must be true.  This is a reporting/decision helper -- it
    does not, and cannot, change runtime behaviour by itself.

    Phase 51 provenance enforcement: when a ``CalibrationReport`` is supplied,
    ``real_calibration_data`` is FORCED to False unless the report's
    ``dataset_status`` is exactly ``REAL LABELED DATA``.  SIMULATED /
    NOT VALIDATED / provenance-not-proven artifacts therefore can never satisfy
    the real-data gate even if every boolean is passed True.
    """
    if report is not None and \
            getattr(report, "dataset_status", None) != STATUS_REAL:
        real_calibration_data = False
    gates = {
        "real_calibration_data": bool(real_calibration_data),
        "thresholds_calibrated": bool(thresholds_calibrated),
        "margin_calibrated": bool(margin_calibrated),
        "identity_safety_tests": bool(identity_safety_tests),
        "real_camera_validation": bool(real_camera_validation),
    }
    all_gates = all(gates.values())
    reasons = [g for g in _PROD_GATES if not gates[g]]
    return GateResult(gates=gates, all_gates=all_gates, reasons=reasons)


# ---------------------------------------------------------------------------
# Cadence recommendation (Part 11)
# ---------------------------------------------------------------------------

_BUILTIN_CADENCE = {"cadence": "every-cycle",
                    "reason": "built-in descriptor is cheap (~ms) and runs on "
                              "every eligible tick"}
_OSNET_POLICY = {
    "cadence": "unresolved / face-unavailable / periodic refresh",
    "reason": ("OSNet (~77-123 ms CPU/embed) must NOT run continuously for "
               "face-resolved stable employees; run only when the track has no "
               "trusted identity or when its face run went quiet, plus a slow "
               "periodic refresh. The event-driven gate "
               "(CCTV_REID_GATE_RESOLVED=1) removes the provably-useless "
               "inference on face-resolved boxes.")}


def cadence_recommendation(model_source: str, embed_ms: float) -> dict:
    if model_source == "built-in":
        return dict(_BUILTIN_CADENCE, model_source="built-in",
                    embed_ms=round(float(embed_ms), 2))
    return dict(_OSNET_POLICY, model_source=model_source,
                embed_ms=round(float(embed_ms), 2))


# ---------------------------------------------------------------------------
# Real-camera feed gate (Part 10)
# ---------------------------------------------------------------------------


def feed_usable_stats(frames, *, brightness_min: float = 20.0,
                      change_min: float = 0.0,
                      require_fps: float | None = FEED_REQUIRE_FPS,
                      capture_fps: float | None = None,
                      min_frames: int = MIN_FEED_FRAMES) -> dict:
    """Return a verdict on whether a captured feed may be used for validation.

    ``frames`` is a list of BGR ndarrays in capture order.  ``bright`` requires
    the median frame brightness to be at least ``brightness_min`` (a black
    DroidCam feed measures ~4).  ``changing`` requires the mean absolute
    difference between consecutive frames to exceed ``change_min`` AND that at
    least half of the transitions changed (so a feed that freezes mid-capture
    -- "abrupt freeze after healthy frames" -- is rejected, not just a fully
    frozen feed).  ``fps`` is enforced only when provided.
    """
    if not frames or len(frames) < max(2, int(min_frames)):
        return {"usable": False, "bright": False, "changing": False,
                "fps_ok": bool(require_fps is None),
                "insufficient_frames": True,
                "brightness_median": None, "change_mean": None,
                "changed_ratio": 0.0, "fps": None}
    grays = [np.asarray(f, dtype=np.uint8) for f in frames]
    brights = [float(g.mean()) for g in grays]
    brightness_median = float(np.median(brights))
    diffs = []
    for i in range(1, len(grays)):
        d = np.abs(grays[i].astype(np.int16) - grays[i - 1].astype(np.int16))
        diffs.append(float(d.mean()))
    change_mean = float(np.mean(diffs)) if diffs else 0.0
    changed_ratio = (float(sum(1 for d in diffs if d > float(change_min)))
                     / len(diffs)) if diffs else 0.0
    bright = brightness_median >= float(brightness_min)
    changing = bool(diffs) and change_mean > float(change_min) \
        and changed_ratio >= 0.5
    fps_ok = True
    if require_fps is not None:
        fps_ok = capture_fps is not None and float(capture_fps) >= require_fps
    return {"usable": bool(bright and changing and fps_ok),
            "bright": bool(bright), "changing": bool(changing),
            "fps_ok": bool(fps_ok), "insufficient_frames": False,
            "brightness_median": round(brightness_median, 2),
            "change_mean": round(change_mean, 3),
            "changed_ratio": round(changed_ratio, 3),
            "fps": float(capture_fps) if capture_fps is not None else None}


def validation_runner_verdict(frames, *, capture_fps, require_fps=None,
                              brightness_min: float = 20.0,
                              change_min: float = 0.0,
                              min_frames: int = MIN_FEED_FRAMES) -> dict:
    """Phase 51 guard for a FUTURE real-camera validation runner.

    A camera opening successfully is meaningless (Phase 48: DroidCam opened
    and streamed "30 FPS" while every frame was black/frozen).  This returns an
    explicit decision state: ``EXECUTABLE (FEED OK)`` or ``NOT EXECUTED /
    INVALID FEED`` with the exact reason string.  When the verdict is not
    executable the runner MUST NOT produce crops, identity, phone, or accuracy
    results -- this function is the single choke point.
    """
    if require_fps is None:
        require_fps = FEED_REQUIRE_FPS
    stats = feed_usable_stats(frames, brightness_min=brightness_min,
                              change_min=change_min,
                              require_fps=require_fps,
                              capture_fps=capture_fps, min_frames=min_frames)
    if not stats["usable"]:
        reason = None
        if stats.get("insufficient_frames"):
            reason = (f"invalid feed: insufficient frames "
                      f"(need >= {int(min_frames)}); cameras that open but "
                      "spit garbage are not evidence")
        elif not stats["bright"]:
            reason = "invalid feed: BLACK/DARK (median brightness below minimum)"
        elif not stats["changing"]:
            reason = "invalid feed: FROZEN (no content change across frames)"
        elif not stats["fps_ok"]:
            reason = "invalid feed: LOW FPS (below required nominal FPS)"
        return {"state": "NOT EXECUTED / INVALID FEED", "proceed": False,
                "reason": reason or "invalid feed (unknown reason)",
                "stats": stats}
    return {"state": "EXECUTABLE (FEED OK)", "proceed": True,
            "reason": None, "stats": stats}


# ---------------------------------------------------------------------------
# Synthetic dry-run (SIMULATED) -- never passed off as real data
# ---------------------------------------------------------------------------


def synthetic_employees(employees=("EMP001", "EMP002", "EMP004"),
                        samples_per_pose: int = 3) -> dict:
    """Deterministic synthetic BGR crops ``{emp: {pose: [crop]}}`` (SIMULATED).

    Used only to demonstrate the pipeline end-to-end and to test the tool.
    These are SYNTHETIC crops, never real employee imagery, and must never be
    used to calibrate production thresholds.
    """
    outfits = {
        "EMP001": (60, 60, 210), "EMP002": (200, 90, 40),
        "EMP004": (60, 170, 60),
    }
    per_emp: dict = {e: {} for e in employees}
    for emp in employees:
        shirt = outfits[emp]
        for pose in AVAILABLE_POSES:
            pose_crops = []
            # Deterministic seed independent of PYTHONHASHSEED randomization.
            base = sum(ord(ch) for ch in emp + pose)
            for k in range(samples_per_pose):
                seed = base * 1000 + k * 101
                rng = np.random.default_rng(seed)
                crop = rng.integers(15, 45, (256, 128, 3), dtype=np.uint8)
                crop[0:160, :] = shirt
                if pose in ("left", "right"):
                    crop[:160, :64, :] = np.clip(
                        crop[:160, :64, :].astype(np.int16) + 40,
                        0, 255).astype(np.uint8)
                if pose == "back":
                    crop[0:160, :] = (35, 90, 90)
                if pose == "partial":
                    crop[0:160, 32:96, :] = np.clip(
                        crop[0:160, 32:96, :].astype(np.int16) + 60,
                        0, 255).astype(np.uint8)
                pose_crops.append(np.ascontiguousarray(crop))
            per_emp[emp][pose] = pose_crops
    return per_emp


def run_calibration(*, model_path: str = "", data_dir=None, out_file=None,
                    dry_run_synthetic: bool = False) -> CalibrationReport:
    """CLI/entrypoint driver: embed + analyse + write JSON.  Never touches config."""
    from src.reid import AppearanceExtractor
    extractor = AppearanceExtractor(model_path=str(model_path))
    _ensure_model_fail_closed(model_path, extractor)
    if dry_run_synthetic:
        crops = synthetic_employees()
        per_emp: dict = {}
        for emp, poses in crops.items():
            per_emp[emp] = {}
            for pose, imgs in poses.items():
                per_emp[emp][pose] = [e for c in imgs
                                      if (e := extractor.embed(c)) is not None]
        report = CalibrationReport(
            label="SIMULATED synthetic dry-run -- NOT real calibration data",
            model_source=str(extractor.model_source),
            feature_dim=int(extractor.feature_dim))
        report.dataset_status = STATUS_SIMULATED
        report.checkpoint = getattr(extractor, "checkpoint", None)
        report.model_variant = getattr(extractor, "model_variant", None)
        report.model_identity = {
            "family": str(extractor.model_source),
            "variant": report.model_variant,
            "checkpoint": report.checkpoint,
            "feature_dim": int(extractor.feature_dim),
        }
        by_emp = {e: [x for pose in AVAILABLE_POSES for x in per_emp[e][pose]]
                  for e in per_emp}
        report.employees_seen = sorted(by_emp)
        report.samples_embedded = sum(len(v) for v in by_emp.values())
        report.samples_total = report.samples_embedded
        report.dataset_fingerprint = dataset_fingerprint(
            data_dir, scan_data_dir(data_dir))
        same, diff = pairwise_similarities(by_emp)
        marg = best_vs_second_margins(by_emp)
        report.same = Distribution.of(same)
        report.diff = Distribution.of(diff)
        report.margins = Distribution.of(marg)
        rec = recommend(by_emp)
        report.recommended = rec
        report.note = ("SIMULATED dry-run only; thresholds from this run are "
                       "NEVER valid for production.")
        save_report(report, out_file)
        return report
    survey = scan_data_dir(data_dir, ("EMP001", "EMP002", "EMP003", "EMP004"))
    report = build_report(extractor=extractor, model_path=model_path,
                          data_dir=data_dir, survey=survey)
    save_report(report, out_file)
    return report


def save_report(report: CalibrationReport, out_file=None):
    target = Path(out_file) if out_file else DEFAULT_OUT
    target.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "label": report.label,
        "dataset_status": report.dataset_status,
        "manifest_status": report.manifest_status,
        "manifest_validation": report.manifest_validation,
        "model_source": report.model_source,
        "model_identity": report.model_identity,
        "checkpoint": report.checkpoint,
        "model_variant": report.model_variant,
        "feature_dim": report.feature_dim,
        "dataset_fingerprint": report.dataset_fingerprint,
        "employees_seen": report.employees_seen,
        "employees_missing": report.employees_missing,
        "missing_poses": report.missing_poses,
        "empty_poses": report.empty_poses,
        "duplicates": report.duplicates,
        "unsupported_files": report.unsupported_files,
        "failed_files": report.failed_files,
        "image_sizes": report.image_sizes,
        "files_discovered_raw": report.files_discovered_raw,
        "samples_total": report.samples_total,
        "samples_embedded": report.samples_embedded,
        "samples_failed": report.samples_failed,
        "same_person": _dist_dict(report.same),
        "different_person": _dist_dict(report.diff),
        "margins": _dist_dict(report.margins),
        "recommended": report.recommended,
        "operating_points_sample": report.operating_points,
        "config_unchanged": report.config_unchanged,
        "note": report.note,
    }
    target.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return target


def _dist_dict(dist: Distribution | None) -> dict | None:
    if dist is None:
        return None
    return {"n": dist.n, "mean": dist.mean, "p01": dist.p01, "p05": dist.p05,
            "p50": dist.p50, "p95": dist.p95, "p99": dist.p99,
            "min": dist.min, "max": dist.max}


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="", help="OSNet .pth path (or empty)")
    ap.add_argument("--data", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--dry-run-synthetic", action="store_true",
                    help="SIMULATED pipeline demo (never real calibration)")
    args = ap.parse_args()
    try:
        report = run_calibration(model_path=args.model, data_dir=args.data,
                                 out_file=args.out,
                                 dry_run_synthetic=args.dry_run_synthetic)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"FAILED -- {exc}")
        print("production config: NOT modified")
        sys.exit(1)
    print(f"status: {report.dataset_status}")
    print(f"model_source={report.model_source} dim={report.feature_dim}")
    print(f"model_identity={report.model_identity}")
    print(f"label: {report.label}")
    print(f"dataset_fingerprint={report.dataset_fingerprint}")
    print(f"employees_seen={report.employees_seen}")
    print(f"employees_missing={report.employees_missing}")
    print(f"missing_poses={report.missing_poses}")
    print(f"empty_poses={report.empty_poses}")
    print(f"duplicates={len(report.duplicates)} "
          f"unsupported={len(report.unsupported_files)} "
          f"failed={len(report.failed_files)}")
    if report.same and report.diff:
        print(f"same-person p50={report.same.p50} (n={report.same.n})")
        print(f"diff-person p50={report.diff.p50} (n={report.diff.n})")
        print(f"recommended={report.recommended}")
    print(f"note: {report.note}")
    print("production config: NOT modified")


if __name__ == "__main__":
    _cli()