"""Dummy face enrollment tool (Phase 8).

Generates placeholder employee images in ``data/faces/`` and attempts to
build ``data/embeddings.pkl`` so the pipeline can be exercised before real
employee photos are available.

NOTE
----
Random-noise / blank images do **not** contain faces, so insightface will
find no faces to embed.  This tool therefore has two goals:

1. Provision the ``data/faces/`` directory and image files so the image
   layout/enrollment flow is verifiable end-to-end.
2. Exercise ``FaceRegistry.rebuild()`` and confirm ``data/embeddings.pkl``
   is created *without crashing*, even if it ends up with zero registered
   employees (a registry of size 0 is valid; every identity returns
   ``"Unknown"`` at threshold 0.6).

For meaningful recognition later, replace these dummies with real
``EMPxxx.jpg`` photos (one face each, the largest face is enrolled).

Usage
-----
    python -m src.dummy_face_enroll            # 3 employees by default
    python -m src.dummy_face_enroll --count 5
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import FACES_DIR  # noqa: E402

logger = logging.getLogger("cctv.dummy_face_enroll")

FACES_DIR = Path(FACES_DIR) if not isinstance(FACES_DIR, Path) else FACES_DIR
IMAGE_SIZE = (640, 480)  # w, h


def _make_random_face_image(seed: int):
    """Return a deterministic random-noise BGR image (no real face)."""
    rng = np.random.default_rng(seed)
    # Slightly warm skin-toned base with a large random-texture overlay.
    noise = rng.integers(60, 120, size=(IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.uint8)
    base = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), (90, 140, 170), dtype=np.uint8)
    img = cv2.addWeighted(base, 0.35, noise, 0.65, 0)
    # Place a lighter oval "face" region to loosely mimic enrollment framing.
    overlay = img.copy()
    cv2.ellipse(overlay, (IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2),
                (140, 180), 0, 0, 360, (120, 180, 210), -1)
    img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
    return img


def generate_dummy_faces(count: int = 3, force: bool = False) -> list[Path]:
    """Write ``EMP001.jpg ... EMP{count}.jpg`` to ``data/faces/``.

    Existing files for the requested IDs are left untouched unless
    ``force`` is True (to avoid clobbering real enrollments).
    """
    FACES_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for i in range(1, count + 1):
        emp_id = f"EMP{i:03d}"
        path = FACES_DIR / f"{emp_id}.jpg"
        if path.exists() and not force:
            logger.info("Skipping existing %s (use --force to overwrite).", path.name)
            continue
        img = _make_random_face_image(seed=i)
        cv2.imwrite(str(path), img)
        written.append(path)
        logger.info("Wrote %s (%dx%d)", path.name, IMAGE_SIZE[0], IMAGE_SIZE[1])

    return written


def rebuild_registry():
    """Invoke FaceRegistry.rebuild() and confirm embeddings.pkl is produced."""
    try:
        from src.face_registry import FaceRegistry, EMBEDDINGS_FILE
    except Exception as exc:
        logger.error("FaceRegistry import failed: %s", exc)
        logger.warning(
            "insightface / onnxruntime may not be installed. "
            "Run setup_env.ps1 first. Skipping registry rebuild."
        )
        return

    registry = FaceRegistry()
    registry.rebuild()
    if EMBEDDINGS_FILE.exists():
        logger.info("embeddings.pkl generated -> %s", EMBEDDINGS_FILE)
        logger.info("Registered employees: %s", registry.employee_ids or "(none)")
    else:
        logger.warning("embeddings.pkl was not created on disk.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Dummy face enrollment tool")
    parser.add_argument("--count", type=int, default=3, help="Number of dummy employees.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing EMP images.")
    parser.add_argument("--no-rebuild", action="store_true",
                        help="Only write images; skip FaceRegistry.rebuild().")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")

    written = generate_dummy_faces(count=args.count, force=args.force)
    if not written and not args.force:
        logger.info("No new images written (existing files present).")

    if not args.no_rebuild:
        rebuild_registry()

    logger.info("Done. Replace dummy EMP images with real photos before production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
