"""Dynamic face registry using insightface embeddings.

Scans ``data/faces/`` for employee images (e.g. ``EMP001.jpg``), extracts
512-D face embeddings, and caches them to ``data/embeddings.pkl`` for fast
startup.

Production hardening (Phase 4)
-----------------------------
* Multiple images per employee are supported -- each image may add one
  embedding; near-duplicate embeddings are de-duplicated so a noisy set of
  photos does not skew the match.
* Embeddings are validated for dimension/finiteness; bad/corrupted images
  are skipped with a clear warning rather than crashing the build.
* A minimum face size keeps tiny, low-quality crops out of the registry.
* Cache is versioned so stale caches from older schema versions are rebuilt.

Usage
-----
  python -m src.face_registry      # rebuild registry + print status
  FaceRegistry()                   # normal import + cache load
"""

import logging
import os
import pickle
import re
import sys
from pathlib import Path

import cv2
import numpy as np

from config import (EMBEDDINGS_FILE, FACE_CANDIDATE_THRESHOLD, FACE_MODEL,
                    FACE_SIMILARITY_THRESHOLD, FACES_DIR)
import config  # noqa: E402  (module-level settings incl. FACE_CONFUSABILITY_MAX)

logger = logging.getLogger("cctv.face_registry")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FACES_DIR = Path(FACES_DIR) if not isinstance(FACES_DIR, Path) else FACES_DIR
EMBEDDINGS_FILE = Path(EMBEDDINGS_FILE) if not isinstance(EMBEDDINGS_FILE, Path) else EMBEDDINGS_FILE

FACE_ANALYSIS_MODEL = FACE_MODEL
RECOGNITION_THRESHOLD = FACE_SIMILARITY_THRESHOLD  # CONFIRM cutoff
CANDIDATE_THRESHOLD = FACE_CANDIDATE_THRESHOLD     # tentative/candidate floor
MARGIN_MIN = float(getattr(config, "FACE_MARGIN_MIN", 0.06))
EMBEDDING_DIM = 512
CACHE_VERSION = 3

# Face-decision statuses (Phase 44B).  A per-frame decision is one of these:
#   CONFIRMED -- best score >= RECOGNITION_THRESHOLD and margin >= MARGIN_MIN.
#   CANDIDATE -- best score >= CANDIDATE_THRESHOLD but not CONFIRMED; may aid
#                adoption on tracks with no trusted identity, never demotes.
#   UNKNOWN   -- below the candidate floor (or malformed embedding).
RECOG_CONFIRMED = "CONFIRMED"
RECOG_CANDIDATE = "CANDIDATE"
RECOG_UNKNOWN = "UNKNOWN"

# Registry tuning
MIN_FACE_AREA = 120 * 120       # skip tiny/low quality crops
DUPLICATE_COS = 0.985           # embeddings this close to an existing one are skipped

# Per-file enrollment statuses (dashboard / CLI)
ENROLLED = "ENROLLED"
MULTIPLE_FACES = "MULTIPLE_FACES"
NO_FACE = "NO_FACE"
INVALID_IMAGE = "INVALID_IMAGE"
LOW_QUALITY = "LOW_QUALITY"

# Multi-sample enrollment convention.  Additional photos of the same employee
# are named ``EMP001_2.jpg``, ``EMP001_3.jpg`` ... and are aggregated into the
# same employee id (``EMP001``) as additional embeddings.  Only a numeric
# ``_<n>`` suffix is stripped; a bare ``EMP001.jpg`` still maps to ``EMP001``.
_MULTI_SAMPLE_RE = re.compile(r"^(.*)_(\d+)$")


def _employee_id_from_stem(stem: str) -> str:
    """Map a face-file stem to its canonical employee id.

    * ``EMP001`` -> ``EMP001``
    * ``EMP001_2`` -> ``EMP001``  (multi-sample enrichment)
    * ``EMP01`` -> ``EMP01``      (unchanged, not EMP001)
    """
    m = _MULTI_SAMPLE_RE.match(stem)
    return m.group(1) if m else stem


def _providers() -> list[str]:
    """Return onnxruntime providers in order of preference.

    Only requests CUDA when torch detects an available GPU, otherwise the
    CUDA probe can stall on CPU-only installs.
    """
    try:
        import torch
        cuda = torch.cuda.is_available()
    except Exception:
        cuda = False
    providers = []
    if cuda:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


class FaceRegistry:
    """Manages known face embeddings and performs identity matching.

    On construction, loads cached embeddings from ``data/embeddings.pkl``
    if available, otherwise scans ``data/faces/`` and builds the cache.
    """

    def __init__(self, threshold: float = RECOGNITION_THRESHOLD,
                 detect_size: int = 640):
        self.threshold = threshold
        self._app = None
        self._employee_ids: list[str] = []
        self._embeddings: np.ndarray | None = None  # shape (N, 512)
        self._by_employee: dict[str, list[int]] = {}   # emp_id -> row indices
        self._file_status: dict[str, str] = {}         # file stem -> status
        self._names: dict[str, str] | None = None      # lazy DB name cache

        self._init_insightface(detect_size=detect_size)
        self._load_or_build()

    # ------------------------------------------------------------------
    # InsightFace initialisation
    # ------------------------------------------------------------------

    def _init_insightface(self, detect_size: int = 640):
        from insightface.app import FaceAnalysis

        logger.info("Initialising insightface FaceAnalysis (model=%s) ...", FACE_ANALYSIS_MODEL)
        self._app = FaceAnalysis(
            name=FACE_ANALYSIS_MODEL,
            providers=_providers(),
        )
        self._app.prepare(ctx_id=0, det_size=(detect_size, detect_size))
        logger.info("insightface FaceAnalysis ready (providers=%s).", _providers())

    # ------------------------------------------------------------------
    # Cache load / build
    # ------------------------------------------------------------------

    def _load_or_build(self):
        if EMBEDDINGS_FILE.exists():
            try:
                self._load_cache()
                logger.info("Loaded %d face embeddings from cache.",
                            len(self._employee_ids))
                self._log_summary()
                return
            except Exception as exc:
                logger.warning("Cache load failed (%s); rebuilding.", exc)

        self._build_from_images()
        self._log_summary()

    def _log_summary(self):
        """Startup diagnostic: one line per enrolled employee (id + name +
        embedding count) and any face files that did not contribute.

        Emits names and counts ONLY -- never raw embeddings or similarity
        values, so this stays safe for production logs.
        """
        if not self._by_employee:
            logger.info("Face registry summary: no employees enrolled.")
            return
        names = _employee_names()
        for eid in sorted(self._by_employee):
            logger.info(
                "Face registry summary: %s (%s) -> %d embedding(s)",
                eid, names.get(eid) or "(no name in DB)", len(self._by_employee[eid]))
        skipped = [k for k, s in self._file_status.items()
                   if _employee_id_from_stem(k) not in self._by_employee]
        if skipped:
            logger.warning(
                "Face registry summary: %d face file(s) registered no usable "
                "embedding (likely junk/duplicate/non-canonical): %s",
                len(skipped), sorted(skipped))

    def _load_cache(self):
        with open(EMBEDDINGS_FILE, "rb") as f:
            data = pickle.load(f)
        if data.get("version", 1) != CACHE_VERSION:
            raise ValueError("stale embeddings cache version; rebuilding")
        # Fingerprint guard (Phase 40): if a face image was added, removed or
        # modified since the cache was written, rebuild instead of serving the
        # stale cache -- otherwise a newly enrolled face would never appear.
        disk_fp = self._disk_fingerprint()
        cached_fp = data.get("file_fingerprint") or {}
        if disk_fp != cached_fp:
            raise ValueError("face files changed since cache was built; rebuilding")
        self._employee_ids = data["employee_ids"]
        embs = np.array(data["embeddings"], dtype=np.float32)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        if len(self._employee_ids) != len(embs):
            raise ValueError("cache id/embedding pairing corrupted; rebuilding")
        if embs.shape[-1] != EMBEDDING_DIM:
            raise ValueError("embedding dimension mismatch; rebuilding")
        self._embeddings = embs
        self._by_employee = data.get("by_employee", {})
        if not self._by_employee:
            # Rebuild mapping for legacy caches
            self._by_employee = {}
            for i, eid in enumerate(self._employee_ids):
                self._by_employee.setdefault(eid, []).append(i)
        self._file_status = data.get("file_status", {})

    def _embed_is_useful(self, emb: np.ndarray) -> bool:
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim != 1 or emb.shape[0] != EMBEDDING_DIM:
            return False
        if not np.all(np.isfinite(emb)):
            return False
        return True

    @staticmethod
    def _image_paths():
        if not FACES_DIR.exists():
            return {}
        return {
            p.name: (p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(FACES_DIR.iterdir())
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        }

    def _disk_fingerprint(self) -> dict:
        """Fingerprint of the current faces directory contents.

        Keys are filenames; values are (size, mtime_ns).  Used to detect any
        added/removed/modified enrollment image so the cache is rebuilt rather
        than serving a stale snapshot after a new face is saved.
        """
        return self._image_paths()

    def _build_from_images(self):
        FACES_DIR.mkdir(parents=True, exist_ok=True)

        employee_ids: list[str] = []
        embeddings: list[np.ndarray] = []
        by_employee: dict[str, list[int]] = {}
        file_status: dict[str, str] = {}

        for img_path in sorted(FACES_DIR.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue

            emp_id = _employee_id_from_stem(img_path.stem)  # e.g. "EMP001", "EMP001_2" -> EMP001
            file_status[img_path.stem] = self._register_one(img_path, emp_id,
                                                            employee_ids, embeddings,
                                                            by_employee)

        if not embeddings:
            logger.warning("No usable faces registered. Place images in %s", FACES_DIR)
            self._employee_ids = []
            self._embeddings = np.empty((0, EMBEDDING_DIM), dtype=np.float32)
            self._by_employee = {}
        else:
            self._employee_ids = employee_ids
            self._embeddings = np.stack(embeddings, axis=0)
            self._by_employee = by_employee

        self._file_status = file_status

        bad = [s for s in file_status.values()
               if s not in (ENROLLED, MULTIPLE_FACES)]
        if bad:
            logger.info("Enrollment status: %s",
                        {k: file_status[k] for k in file_status if file_status[k] in bad})
        self._save_cache()
        conf = self.confusability_report()
        if conf:
            logger.warning(
                "Identities cannot be reliably separated by face: %s",
                conf,
            )

    def _register_one(self, img_path: Path, emp_id: str,
                      employee_ids: list[str], embeddings: list[np.ndarray],
                      by_employee: dict[str, list[int]]) -> str:
        """Extract one embedding from ``img_path``.

        Appends to the shared build lists on success and returns the
        per-file status string.
        """
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Cannot read image (corrupt?): %s", img_path.name)
            return INVALID_IMAGE

        try:
            faces = self._app.get(img)
        except Exception as exc:
            logger.warning("Detection failed for %s: %s", img_path.name, exc)
            return INVALID_IMAGE

        if not faces:
            logger.warning("No face detected in %s; skipping.", img_path.name)
            return NO_FACE

        if len(faces) > 1:
            logger.warning("Multiple faces in %s; using the largest.", img_path.name)

        # Pick the largest face; validate embedding quality.
        best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        area = (best.bbox[2] - best.bbox[0]) * (best.bbox[3] - best.bbox[1])
        if area < MIN_FACE_AREA:
            logger.warning("Face too small in %s (area=%.0f); skipping.",
                           img_path.name, area)
            return LOW_QUALITY
        embedding = best.normed_embedding.astype(np.float32)
        if not self._embed_is_useful(embedding):
            logger.warning("Invalid embedding from %s; skipping.", img_path.name)
            return LOW_QUALITY

        # De-duplicate near-identical embeddings (multiple pics of same face)
        embs = np.stack(embeddings) if embeddings else np.empty((0, EMBEDDING_DIM), np.float32)
        if len(embs):
            sims = embs @ embedding
            if sims.max() >= DUPLICATE_COS:
                logger.info("Duplicate embedding in %s; already enrolled.", img_path.name)
                return ENROLLED  # face known, no extra row stored

        employee_ids.append(emp_id)
        embeddings.append(embedding)
        by_employee.setdefault(emp_id, []).append(len(employee_ids) - 1)
        logger.info("Registered %s from %s", emp_id, img_path.name)
        return MULTIPLE_FACES if len(faces) > 1 else ENROLLED

    def _save_cache(self):
        EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(
                {
                    "version": CACHE_VERSION,
                    "employee_ids": self._employee_ids,
                    "embeddings": self._embeddings.tolist() if self._embeddings is not None else [],
                    "by_employee": self._by_employee,
                    "file_status": self._file_status,
                    "file_fingerprint": self._disk_fingerprint(),
                },
                f,
            )
        logger.info("Saved %d embeddings to %s", len(self._employee_ids), EMBEDDINGS_FILE)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def app(self):
        """The underlying insightface ``FaceAnalysis`` instance."""
        return self._app

    @property
    def employee_ids(self) -> list[str]:
        return list(self._employee_ids)

    @property
    def enrolled_employee_ids(self) -> list[str]:
        """Unique employee ids that have at least one usable embedding."""
        return sorted(self._by_employee.keys())

    @property
    def num_registered(self) -> int:
        """Number of individual embeddings (may exceed unique employees)."""
        return len(self._employee_ids)

    @property
    def num_enrolled_employees(self) -> int:
        return len(self._by_employee)

    def rebuild(self):
        """Force a full rebuild from images on disk."""
        self._build_from_images()

    def identify(self, face_embedding: np.ndarray) -> tuple[str, float]:
        """Match a face embedding against the known database.

        Returns
        -------
        (employee_id, score)
            ``employee_id`` is ``"Unknown"`` if no match exceeds threshold.
        """
        if self._embeddings is None or len(self._employee_ids) == 0:
            return "Unknown", 0.0

        query = face_embedding.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.shape[-1] != EMBEDDING_DIM or not np.isfinite(query).all():
            return "Unknown", 0.0

        sims = self._embeddings @ query.T
        sims = sims.flatten()

        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score < self.threshold:
            return "Unknown", best_score

        return self._employee_ids[best_idx], best_score

    def _employee_ranking(self, sims: np.ndarray) -> list[tuple[str, float]]:
        """Collapse per-row similarities into one score per employee.

        ``sims`` is a 1-D array parallel to ``_employee_ids`` (one value per
        enrolled embedding).  Each employee id keeps its BEST embedding score
        (max aggregation -- the cleanest separation, identical to the raw
        argmax when an employee has a single enrollment, see Phase 57 Step 4),
        and the resulting ``[(employee_id, score), ...]`` list is sorted
        descending.  This makes the margin a quantity between distinct
        employee *identities* instead of between two enrollment samples of the
        same person -- the Phase 57 root-cause fix.
        """
        per_employee: dict[str, float] = {}
        for i, eid in enumerate(self._employee_ids):
            s = float(sims[i])
            if s > per_employee.get(eid, -1.0):
                per_employee[eid] = s
        return sorted(per_employee.items(), key=lambda kv: kv[1], reverse=True)

    def identify_k(self, face_embedding: np.ndarray, k: int = 2) -> list[tuple[str, float]]:
        """Top-``k`` matching employees by similarity, descending.

        Unlike :meth:`identify`, this returns the RAW argmax identity even when
        it falls below the recognition threshold -- so diagnostics can show
        *who* was nearly recognised (e.g. ``("EMP004", 0.57)`` with decision
        UNKNOWN).  ``identify`` remains the single source of truth for the
        recognition decision; ``identify_k`` is for FACE_DEBUG / diagnostic
        tools only.  Each employee id appears at most once (its best embedding
        wins), so the runner-up is a genuinely different employee -- never a
        duplicate sample of the leader (Phase 57).  Never prints raw
        embeddings.
        """
        k = max(1, int(k))
        if self._embeddings is None or len(self._employee_ids) == 0:
            return [("Unknown", 0.0)]

        query = face_embedding.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.shape[-1] != EMBEDDING_DIM or not np.isfinite(query).all():
            return [("Unknown", 0.0)]

        sims = (self._embeddings @ query.T).flatten()
        ranked = self._employee_ranking(sims)
        return ranked[:k]

    def identify_candidate(self, face_embedding: np.ndarray) -> dict:
        """Phase 44B gated face decision (CONFIRMED / CANDIDATE / UNKNOWN).

        Two-tier decision that separates a *tentative match* from a *usable
        recognition*:

        * CONFIRMED -- best score >= ``RECOGNITION_THRESHOLD`` (0.60) AND the
          margin to the runner-up identity >= ``MARGIN_MIN`` (0.06).  The
          identity is trustworthy enough to adopt/switch a track.
        * CANDIDATE -- best score >= ``CANDIDATE_THRESHOLD`` (0.50) but not
          CONFIRMED (too weak, or the gallery cannot separate the top two for
          this frame).  Callers may use it to *adopt* a track that has no
          trusted identity yet, but never to demote or switch a trusted id.
        * UNKNOWN  -- below the candidate floor (or malformed/empty input).

        Returns a dict::

          {"emp_id": str, "score": float, "margin": float, "second_id": str|None,
           "status": "CONFIRMED"|"CANDIDATE"|"UNKNOWN"}

        ``emp_id``/``second_id`` are gallery ids; on UNKNOWN ``emp_id`` is
        ``"Unknown"`` and ``second_id`` is ``None``.  ``margin`` is the gap
        between the best score of the leading EMPLOYEE and the best score of
        the runner-up EMPLOYEE (per-employee max aggregation, Phase 57) --
        never between two enrollment samples of the same person.  For a
        single-employee gallery the margin carries the full best score.  Raw
        embeddings are never returned.
        """
        if self._embeddings is None or len(self._employee_ids) == 0:
            return {"emp_id": "Unknown", "score": 0.0, "margin": 0.0,
                    "second_id": None, "status": RECOG_UNKNOWN}

        query = face_embedding.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.shape[-1] != EMBEDDING_DIM or not np.isfinite(query).all():
            return {"emp_id": "Unknown", "score": 0.0, "margin": 0.0,
                    "second_id": None, "status": RECOG_UNKNOWN}

        sims = (self._embeddings @ query.T).flatten()

        # Phase 57 root cause fix: rank EMPLOYEE identities, not individual
        # enrollment embeddings.  Group the per-row similarities by employee
        # id (each employee keeps its best embedding score), then measure the
        # margin between the two best DIFFERENT employees.  Previously the
        # top-2 could be two enrollment samples of the same person (EMP001 has
        # 6 embeddings), giving an artificially tiny margin and never
        # clearing FACE_MARGIN_MIN even though the identity was clear.
        ranked = self._employee_ranking(sims)
        best_id, best_score = ranked[0]

        if best_score < CANDIDATE_THRESHOLD:
            return {"emp_id": "Unknown", "score": round(best_score, 4),
                    "margin": 0.0, "second_id": None, "status": RECOG_UNKNOWN}

        if len(ranked) > 1:
            second_id, second_score = ranked[1]
            margin = best_score - second_score
        else:
            second_score = 0.0
            margin = best_score
            second_id = None

        status = RECOG_CONFIRMED
        if best_score < RECOGNITION_THRESHOLD or margin < MARGIN_MIN:
            status = RECOG_CANDIDATE
        return {"emp_id": best_id, "score": round(best_score, 4),
                "margin": round(margin, 4), "second_id": second_id,
                "status": status}

    def employee_name(self, employee_id: str) -> str:
        """Human name for ``employee_id`` from the employees DB (never the
        filename).  Lazily cached; empty string when DB lookup fails."""
        if self._names is None:
            self._names = _employee_names()
        return self._names.get(employee_id, "")

    def confusability_report(self) -> list[dict]:
        """Cross-employee similarity pairs that are too close to separate.

        Returns ``[{employee_a, employee_b, similarity}]`` sorted descending,
        including only distinct employee ids whose mutual best similarity
        reaches the configurable ceiling.  Empty list = the enrolled
        identities are pairwise distinguishable by face.
        """
        if self._embeddings is None or not self._by_employee:
            return []
        ceiling = float(getattr(config, "FACE_CONFUSABILITY_MAX", 0.88))
        shown: set[tuple[str, str]] = set()
        pairs: list[dict] = []
        ids = sorted(self._by_employee)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                idx_a = self._by_employee[a]
                idx_b = self._by_employee[b]
                if not idx_a or not idx_b:
                    continue
                sims = self._embeddings[idx_a] @ self._embeddings[idx_b].T
                peak = float(np.max(sims))
                key = tuple(sorted((a, b)))
                if key in shown:
                    continue
                if peak >= ceiling:
                    shown.add(key)
                    pairs.append({"employee_a": a, "employee_b": b,
                                  "similarity": round(peak, 4)})
        pairs.sort(key=lambda p: p["similarity"], reverse=True)
        return pairs

    def scan_status(self) -> dict[str, str]:
        """Per-file enrollment status keyed by image stem (e.g. ``EMP001``)."""
        return dict(self._file_status)

    def employee_status(self) -> dict[str, str]:
        """Aggregate status per employee id.

        A file marked ENROLLED wins; otherwise the worst seen status is
        reported (NO_FACE / LOW_QUALITY / INVALID_IMAGE take precedence so
        a broken photo is surfaced).
        """
        from collections import Counter
        order = {ENROLLED: 0, MULTIPLE_FACES: 0, NO_FACE: 1,
                 LOW_QUALITY: 1, INVALID_IMAGE: 1}
        agg: dict[str, str] = {}
        for file_stem, status in self._file_status.items():
            # multi-sample stems (EMP001_2) collapse to the canonical emp id
            eid = _employee_id_from_stem(file_stem)
            if status not in order:
                status = INVALID_IMAGE
            cur = agg.get(eid)
            if cur is None or order.get(status, 1) > order.get(cur, 1):
                agg[eid] = status
        # Statuses for employees that enrolled but whose photos were clean
        for eid in self.enrolled_employee_ids:
            agg.setdefault(eid, ENROLLED)
        return agg


# ------------------------------------------------------------------
# CLI: rebuild registry / scan diagnostic
# ------------------------------------------------------------------

def _employee_names() -> dict[str, str]:
    """Load {employee_id: name} from the SQLite DB (best-effort).

    Names always come from the employees table -- never derived from the face
    filename.  If the DB is unavailable the map is empty (fall back to the id).
    """
    try:
        from src import database as db
        conn = db.get_connection(db.DB_PATH)
        try:
            return {r["employee_id"]: (r.get("name") or "").strip()
                    for r in db.list_employees(conn)}
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except Exception:
        return {}


def enrollment_table() -> list[dict]:
    """Return one row per enrolled-face file with human-readable fields.

    Columns: employee_id, name, file, status, has_embedding.  Employee names
    come from the employees table, never from the filename.  ``has_embedding``
    reflects whether a usable embedding actually exists, so a NO_FACE/LOW_QUALITY
    file is never reported as recognised.
    """
    names = _employee_names()
    registry = FaceRegistry()
    status = registry.employee_status()
    embedded = set(registry.enrolled_employee_ids)
    rows: list[dict] = []
    for eid in sorted(status):
        rows.append({
            "employee_id": eid,
            "name": names.get(eid) or "",
            "file": f"{eid}.jpg",
            "status": status[eid],
            "has_embedding": "YES" if eid in embedded else "NO",
        })
    return rows, registry


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Face registry diagnostic")
    parser.add_argument("--scan", action="store_true",
                        help="print a per-employee enrollment table")
    parser.add_argument("--rebuild", action="store_true",
                        help="force a full rebuild from data/faces/")
    parser.add_argument("--diagnose", metavar="EMPLOYEE_ID",
                        help="print a safe per-employee diagnostic (never raw "
                             "embeddings): enrollment, embedding validity, "
                             "self-match, best/runner-up IDs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    if args.diagnose:
        eid = args.diagnose
        registry = FaceRegistry()
        names = _employee_names()
        status = registry.employee_status().get(eid, "NOT_ENROLLED")
        idx = registry._by_employee.get(eid, [])
        print("=" * 60)
        print("Employee :", eid)
        print("Name     :", names.get(eid) or "(no name in DB)")
        print("Enrollment:", status)
        if not idx:
            print("Embedding: NONE — employee has no usable embedding")
            sys.exit(0)
        canonical = FACES_DIR / f"{eid}.jpg"
        src_name = canonical.name if canonical.exists() else f"{eid}.jpg (missing)"
        print("Source   :", src_name)
        img = cv2.imread(str(canonical))
        nfaces = 0
        if img is None:
            print("Face count: UNREADABLE IMAGE")
        else:
            faces = registry.app.get(img)
            nfaces = len(faces)
            print("Face count:", nfaces if nfaces else "NONE",
                  "(MULTIPLE_FACES detected)" if nfaces > 1 else "")
        embs = np.asarray(registry._embeddings, dtype=np.float32)
        rows = np.asarray(idx)
        fin = bool(np.isfinite(embs[rows]).all()) if len(embs) else False
        dim = embs.shape[-1] if len(embs) else 0
        print("Embedding: %s" % ("VALID" if fin and dim == EMBEDDING_DIM else "INVALID"))
        print("Dimension:", dim if len(embs) else "-")
        if img is not None and nfaces:
            best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            new_emb = best.normed_embedding.astype(np.float32)
            sims = (embs[rows] @ new_emb).flatten()
            print("Self-match (re-embed canonical image vs stored):",
                  "%.4f" % float(sims.max()) if len(sims) else "-")
            top = registry.identify_k(new_emb, k=2)
            print("identify_k on canonical re-embed :",
                  ", ".join(f"{i}({s:.4f})" for i, s in top))
            best_id, best_score = registry.identify(new_emb)
            print("identify() on canonical re-embed :", best_id,
                  "(score %.4f, threshold %.4f)" % (best_score, registry.threshold))
        else:
            print("Self-match: cannot compute (no readable face in image)")
        cross = []
        for other in sorted(k for k in registry._by_employee if k != eid):
            o_rows = np.asarray(registry._by_employee[other])
            if len(o_rows) and len(rows):
                cross.append((other, float((embs[o_rows] @ embs[rows].T).max())))
        if cross:
            print("Stored cross-identity max sim     :",
                  ", ".join(f"{o}({s:.4f})" for o, s in cross))
        conf = registry.confusability_report()
        if conf:
            print("CONFUSABILITY WARNING             :", conf)
        else:
            print("Confusability report              : none (identities separable)")
        print("=" * 60)
        sys.exit(0)

    if args.rebuild:
        registry = FaceRegistry()
        registry.rebuild()
    elif args.scan:
        rows, registry = enrollment_table()
        if not rows:
            print("No face files found in %s" % FACES_DIR)
        else:
            w_id = max(len("Employee ID"), *(len(r["employee_id"]) for r in rows))
            w_name = max(len("Employee Name"), *(len(r["name"]) or 1 for r in rows))
            print(f"{'Employee ID':<{w_id}}  {'Employee Name':<{w_name}}  "
                  f"{'File':<12}  {'Status':<14}  Embedding")
            print("-" * (w_id + w_name + 12 + 14 + 12))
            for r in rows:
                name = r["name"] or "(no name in DB)"
                print(f"{r['employee_id']:<{w_id}}  {name:<{w_name}}  "
                      f"{r['file']:<12}  {r['status']:<14}  {r['has_embedding']}")
            print()
            print(f"{registry.num_enrolled_employees} employee(s) with embeddings, "
                  f"{registry.num_registered} embedding(s) total.")
    else:
        registry = FaceRegistry()
        print(f"Registry ready: {registry.num_enrolled_employees} employee(s) registered "
              f"({registry.num_registered} embeddings).")
        print(f"Enrolled ids: {registry.enrolled_employee_ids}")
        status = registry.employee_status()
        if status:
            width = max(len(k) for k in status)
            print("\nPer-employee enrollment status:")
            print(f"{'Employee':<{width}}  Status")
            print("-" * (width + 10))
            for eid in sorted(status):
                print(f"{eid:<{width}}  {status[eid]}")
