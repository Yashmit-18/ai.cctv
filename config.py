"""Central configuration for CCTV Employee Productivity Tracker.

Sensitive credentials (SMTP passwords, RTSP passwords) are loaded from
environment variables or a ``.env`` file at the project root via
``python-dotenv``.  See ``.env.example`` for the full template.
"""

import logging
import os
import sys
from pathlib import Path

# ------------------------------------------------------------------
# .env loader (python-dotenv) -- safe to call multiple times
# ------------------------------------------------------------------
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except ImportError:
    pass  # python-dotenv not installed; rely on OS env vars only

PROJECT_ROOT = Path(__file__).resolve().parent

logger = logging.getLogger("cctv.config")

# ============================================================
# INPUT SOURCE
# ------------------------------------------------------------
# Choose one source.  Can be overridden via CLI --source flag.
SOURCE = os.getenv("CCTV_SOURCE", "webcam")

VIDEO_FILE_PATH = str(PROJECT_ROOT / "samples" / "test_video.mp4")

RTSP_URL = os.getenv(
    "CCTV_RTSP_URL",
    "rtsp://user:CHANGE_ME@192.168.1.100:554/stream1",
)

# In the live HUD window: resize the feed to width for consistent display/ROI.
DISPLAY_WIDTH = 960

# ============================================================
# MULTI-CAMERA  (Phase 7 -- high-density orchestration)
# ------------------------------------------------------------
# Ordered mapping of camera_id -> RTSP URL.  The system manages a pool of
# concurrent streams and processes them in round-robin batches.
#
# URLs are read from environment variables named:
#     CCTV_CAM_01_URL ... CCTV_CAM_10_URL
# (see .env.example).  Any cameras whose env var is unset/empty are omitted
# automatically, so the pool scales to however many cameras you enable.
_NUM_CAMERAS = 10
_CAMERAS: dict[str, str] = {}
for _i in range(1, _NUM_CAMERAS + 1):
    _var = f"CCTV_CAM_{_i:02d}_URL"
    _url = os.getenv(_var, "").strip()
    if _url:
        _CAMERAS[f"cam_{_i:02d}"] = _url

# Pre-populate with sane defaults so the project runs out-of-the-box when no
# .env is present, while still honouring any explicitly set env vars above.
# Placeholder URLs use explicit CHANGE_ME markers (never real credentials).
if not _CAMERAS:
    _CAMERAS = {
        "cam_01": os.getenv("CCTV_CAM_01", "rtsp://user:CHANGE_ME@192.168.1.101:554/stream1"),
        "cam_02": os.getenv("CCTV_CAM_02", "rtsp://user:CHANGE_ME@192.168.1.102:554/stream1"),
        "cam_03": os.getenv("CCTV_CAM_03", "rtsp://user:CHANGE_ME@192.168.1.103:554/stream1"),
        "cam_04": os.getenv("CCTV_CAM_04", "rtsp://user:CHANGE_ME@192.168.1.104:554/stream1"),
        "cam_05": os.getenv("CCTV_CAM_05", "rtsp://user:CHANGE_ME@192.168.1.105:554/stream1"),
        "cam_06": os.getenv("CCTV_CAM_06", "rtsp://user:CHANGE_ME@192.168.1.106:554/stream1"),
        "cam_07": os.getenv("CCTV_CAM_07", "rtsp://user:CHANGE_ME@192.168.1.107:554/stream1"),
        "cam_08": os.getenv("CCTV_CAM_08", "rtsp://user:CHANGE_ME@192.168.1.108:554/stream1"),
        "cam_09": os.getenv("CCTV_CAM_09", "rtsp://user:CHANGE_ME@192.168.1.109:554/stream1"),
        "cam_10": os.getenv("CCTV_CAM_10", "rtsp://user:CHANGE_ME@192.168.1.110:554/stream1"),
    }

CAMERAS: dict[str, str] = _CAMERAS

# Maximum camera batch size per GPU inference call (limits VRAM usage).
MAX_BATCH_SIZE = int(os.getenv("CCTV_MAX_BATCH_SIZE", "4"))

# Round-robin frame-skip: with MAX_BATCH_SIZE cameras per batch we target
# roughly TARGET_FPS_PER_CAMERA frames-per-second per camera (0 = every frame).
TARGET_FPS_PER_CAMERA = int(os.getenv("CCTV_TARGET_FPS", "2"))

# Phase 45: single-local-source runs (``main.py --source <index>``) are not
# bound by the multi-camera round-robin budget.  Phase 45 measured ~268 ms per
# full cycle and ~10 FPS on cadence-skip cycles on this CPU, so the stock
# 2-fps cap wasted half of every loop on a sleep.  ``effective_target_fps()``
# raises the cap to this value for single local sources ONLY and always honours
# an explicit ``CCTV_TARGET_FPS`` (or ``CCTV_LOCAL_TARGET_FPS``).
LOCAL_SOURCE_DEFAULT_TARGET_FPS = int(os.getenv("CCTV_LOCAL_TARGET_FPS", "4"))


def effective_target_fps(*, local_source: bool, env=os.environ) -> int:
    """Effective loop FPS cap for this run (Phase 45).

    ``main.py --source <index>`` (single local camera) uses the faster
    single-source cap unless the operator explicitly set ``CCTV_TARGET_FPS``;
    every other configuration keeps ``TARGET_FPS_PER_CAMERA``.  Pure and
    testable; the daemon calls it once at startup.
    """
    if local_source and "CCTV_TARGET_FPS" not in env:
        return LOCAL_SOURCE_DEFAULT_TARGET_FPS
    if local_source and "CCTV_TARGET_FPS" in env:
        return int(env["CCTV_TARGET_FPS"])
    return TARGET_FPS_PER_CAMERA

# A frame older than this (seconds) is treated as STALE by the batch API
# (``latest_frames``), which governs the daemon's ``camera_online`` signal.
# This prevents a dead camera that still holds its last frame from keeping
# employees "online" and accidentally fabricating AWAY/ACTIVE time.
FRAME_STALE_SEC = float(os.getenv("CCTV_FRAME_STALE_SEC", "10"))

# Seconds we wait for a local (webcam) source to CONNECT during startup
# before printing an actionable error and exiting.  Opening a webcam can
# take several seconds on first access, so this is intentionally generous.
STARTUP_FRAME_WAIT_SEC = float(os.getenv("CCTV_STARTUP_FRAME_WAIT", "15"))


def build_cameras() -> dict[str, str]:
    """Re-read ``CCTV_CAM_<NN>_URL`` env vars and rebuild the camera map.

    Useful when the environment changes at runtime (e.g. a new .env loaded).
    Returns a fresh dict without mutating the module global.
    """
    cameras: dict[str, str] = {}
    for _i in range(1, _NUM_CAMERAS + 1):
        _var = f"CCTV_CAM_{_i:02d}_URL"
        _url = os.getenv(_var, "").strip()
        if _url:
            cameras[f"cam_{_i:02d}"] = _url
    return cameras


# ============================================================
# PHASE 58 -- production camera configuration model
# ------------------------------------------------------------
# Extension of the multi-camera pool.  In addition to the URL env var:
#     CCTV_CAM_<NN>_URL
# each camera slot supports optional richer configuration:
#     CCTV_CAM_<NN>_NAME       human-readable name (default = camera id)
#     CCTV_CAM_<NN>_TYPE       source kind: local | rtsp | video_file | test
#     CCTV_CAM_<NN>_ENABLED    0/1 (default 1; disabled slots are excluded)
#     CCTV_CAM_<NN>_LOCATION   free-text location (e.g. "Lobby", "NVR ch.3")
#     CCTV_CAM_<NN>_FPS        per-camera processing target (0 = system)
#     CCTV_CAM_<NN>_RECONNECT_BASE / _MAX / _FACTOR  (seconds; bounded)
#     CCTV_CAM_<NN>_USERNAME   optional split credentials (never logged or
#                              displayed; used only to build the URL)
#     CCTV_CAM_<NN>_PASSWORD
# URLs may carry credentials (rtsp://user:pass@host/stream) OR split creds may
# be supplied separately -- either way credentials are never written to logs,
# reports, the dashboard, or this config's outward-facing shapes.
# ============================================================
from src.domain import CameraConfig, CameraSourceKind  # noqa: E402


def _parse_camera_flag(raw: str | None, default: bool) -> bool:
    raw = (raw or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


def _parse_reconnect(value: str | None, default: float) -> float:
    try:
        v = float((value or "").strip())
        if v > 0:
            return v
    except (TypeError, ValueError):
        pass
    return default


def camera_configs() -> dict[str, CameraConfig]:
    """Load the full per-camera configuration model (Phase 58).

    Every configured camera slot yields a ``CameraConfig``.  A slot is
    included (enabled) when it has a usable URL (or split host URL) and is
    not explicitly disabled.  Disabled / empty slots are omitted so the
    pool scales with however many cameras are really on.

    Credential handling: a URL that already embeds ``user:pass@`` is kept
    as-is (the config display/serialization never writes it out raw);
    ``_USERNAME``/``_PASSWORD`` split creds are folded into the URL exactly
    once (if not already present).  Either route keeps secrets out of logs.
    """
    cameras: dict[str, CameraConfig] = {}
    for _i in range(1, _NUM_CAMERAS + 1):
        _pad = f"{_i:02d}"
        _url = os.getenv(f"CCTV_CAM_{_pad}_URL", "").strip()
        _enabled_raw = os.getenv(f"CCTV_CAM_{_pad}_ENABLED", "").strip()
        _kind_raw = os.getenv(f"CCTV_CAM_{_pad}_TYPE", "").strip().lower()
        _username = os.getenv(f"CCTV_CAM_{_pad}_USERNAME", "").strip()
        _password = os.getenv(f"CCTV_CAM_{_pad}_PASSWORD", "").strip()
        _name = os.getenv(f"CCTV_CAM_{_pad}_NAME", "").strip()
        _location = os.getenv(f"CCTV_CAM_{_pad}_LOCATION", "").strip()

        if not _url and _username:
            # Split-credential mode without a URL -- cannot infer a target.
            _url = ""
        if not _url:
            if _enabled_raw and _parse_camera_flag(_enabled_raw, True):
                continue  # explicitly enabled but unusable --> omitted here.
            continue

        _kind = CameraSourceKind.RTSP
        if _kind_raw:
            for _k in CameraSourceKind:
                if _k.value == _kind_raw:
                    _kind = _k
                    break
        else:
            _kind = CameraSourceKind.infer(_url)

        # Fold split creds into the URL exactly once (only for RTSP).
        if _kind is CameraSourceKind.RTSP and _username:
            _rest = _url.split("://", 1)[-1]
            _authority = _rest.split("/", 1)[0] if "/" in _rest else _rest
            if "@" not in _authority:
                _creds = f"{_username}:{_password}" if _password else _username
                _url = _url.replace(f"://{_authority}", f"://{_creds}@{_authority}", 1)

        _fps_raw = os.getenv(f"CCTV_CAM_{_pad}_FPS", "").strip()
        _fps = 0.0
        try:
            _fps = float(_fps_raw) if _fps_raw else 0.0
        except ValueError:
            _fps = 0.0

        cameras[f"cam_{_pad}"] = CameraConfig(
            camera_id=f"cam_{_pad}",
            name=_name or f"cam_{_pad}",
            kind=_kind,
            url=_url,
            username=_username,
            password=_password,
            enabled=True,
            location=_location,
            fps_target=_fps,
            reconnect_base=_parse_reconnect(
                os.getenv(f"CCTV_CAM_{_pad}_RECONNECT_BASE", ""), 2.0),
            reconnect_max=_parse_reconnect(
                os.getenv(f"CCTV_CAM_{_pad}_RECONNECT_MAX", ""), 30.0),
            reconnect_factor=_parse_reconnect(
                os.getenv(f"CCTV_CAM_{_pad}_RECONNECT_FACTOR", ""), 2.5),
        )
    return cameras


CAMERA_CONFIGS: dict[str, CameraConfig] = camera_configs()

# ============================================================
# DESK ROIs  (multi-desk, x/y/w/h in video-frame pixel coords)
# ------------------------------------------------------------
MULTI_DESK_ROIS = {
    "EMP001": {"x": 100, "y": 80, "w": 400, "h": 300},
}

DESK_ROI = {
    "employee_id": "EMP001",
    "x": 100,
    "y": 80,
    "w": 400,
    "h": 300,
}

ROI_PRESETS_FILE = str(PROJECT_ROOT / "data" / "roi_presets.json")
ROI_PRESET_FILE = ROI_PRESETS_FILE  # legacy alias

# ============================================================
# DETECTION
# ------------------------------------------------------------
# Phase 46: default upgraded from yolov8n.pt to yolo11n.pt.  Local benchmark
# (2026-09-12, synthetic frames, CODE VERIFIED / SIMULATED) showed YOLO11n
# beats YOLOv8n on every measured axis: mAP 39.5 vs 37.3, model size 5.61 MB
# vs 6.55 MB, phone-only @1280 252 ms vs 285 ms, person+phone @640 83.9 ms vs
# 82.9 ms (comparable).  yolov8n.pt is retained in models/ as a fallback.
MODEL_PATH = str(PROJECT_ROOT / "models" / "yolo11n.pt")
CONF_THRESHOLD = 0.4

# ============================================================
# CAPTURE  (frames processed per second)
# ------------------------------------------------------------
FRAME_SKIP = 10
TARGET_FPS = 3

# ============================================================
# FACE RECOGNITION  (Phase 6 -- dynamic identification)
# ------------------------------------------------------------
FACE_MODEL = "buffalo_l"
# CONFIRM cutoff.  FACE_SIMILARITY_THRESHOLD is the confirmation threshold:
# a match at/above it WITH a clear margin becomes a CONFIRMED identity that
# may adopt and switch tracks.
FACE_SIMILARITY_THRESHOLD = 0.6
# Phase 44B -- CANDIDATE floor.  Matches in [FACE_CANDIDATE_THRESHOLD,
# FACE_SIMILARITY_THRESHOLD) are tentative: they may ADOPT a track that has no
# trusted identity yet (after temporal consistency), but a candidate can NEVER
# overwrite (demote) an already trusted identity and can never switch the
# track's identity.  Below the candidate floor the face is Unknown -- it is a
# candidate, not a recognition decision.  Real DroidCam baseline (Phase 44B):
# every enrolled employee self-matches at 1.0000 and the worst cross-identity
# similarity is 0.1961 (EMP002<->EMP003); all identities are separable, so 0.50
# keeps a wide safety band while still granting weak partial/side poses a
# chance on tracks that do not trust anyone yet.
FACE_CANDIDATE_THRESHOLD = float(
    os.getenv("CCTV_FACE_CANDIDATE_THRESHOLD", "0.50"))
# Minimum similarity gap between best and runner-up gallery identity for a
# CONFIRMED decision.  A frame scoring EMP003(0.61) / EMP004(0.60) is NOT
# confirmed -- for that frame the gallery cannot tell the two identities
# apart, so the frame is reported CANDIDATE instead of picking a winner.
FACE_MARGIN_MIN = float(os.getenv("CCTV_FACE_MARGIN_MIN", "0.06"))
# Cross-identity confusability ceiling.  When two DIFFERENT enrolled
# employees share a face-similarity at or above this, face recognition alone
# cannot separate them reliably (e.g. very similar-looking relatives).  The
# registry logs a warning and the enrollment surface surfaces it, so an
# operator is never silently given two indistinguishable identities.
FACE_CONFUSABILITY_MAX = float(
    os.getenv("CCTV_FACE_CONFUSABILITY_MAX", "0.88"))
FACES_DIR = str(PROJECT_ROOT / "data" / "faces")
EMBEDDINGS_FILE = str(PROJECT_ROOT / "data" / "embeddings.pkl")

# Detection resolution for insightface face analysis.  640 is the
# accuracy-optimal default; dropping to 320 roughly doubles CPU face
# throughput at a slight recall cost.  0 = insightface default.
FACE_DETECT_SIZE = int(os.getenv("CCTV_FACE_DETECT_SIZE", "640"))

# Emit a structured FACE_DEBUG line per detected face (candidate, similarity,
# threshold, decision) and a per-frame DETECTION_DEBUG count.  Off by default
# to keep production logs clean; enabled via CCTV_FACE_DEBUG=1.
FACE_DEBUG = os.getenv("CCTV_FACE_DEBUG", "0") == "1"

# Live-recognition size floor (pixels): insightface occasionally returns tiny
# ~40-70 px box artefacts whose embeddings are garbage; identifying them
# polluted the Unknown counter with false "+1 unknown" increments.  Real faces
# in the working 640x480 feed are tens of thousands of pixels (measured EMP001
# ~70k-99k px).  This is a size guard on LIVE recognition only -- it does not
# change the similarity threshold and does not affect enrollment (which uses
# MIN_FACE_AREA).  0 disables the guard.
LIVE_MIN_FACE_AREA = int(os.getenv("CCTV_LIVE_MIN_FACE_AREA", "7200"))

# Part G (Phase 43) -- face-recognition cadence.  InsightFace ``app.get`` is
# the most expensive stage of the pipeline (it dominates CPU a single-thread
# frame).  ``FACE_DETECT_CADENCE`` N means each camera runs face
# detection/recognition once every N processed cycles while YOLO person+phone
# still runs every cycle.  Phase 44A: default raised 1 -> 5 (real DroidCam
# measurement: ``app.get`` = ~516 ms/frame on CPU vs ~87 ms YOLO; recognition
# every frame was the ~0.9 FPS bottleneck).  With cadence 5 the per-cycle
# recognition cost amortizes to ~103 ms while the spatial person layer still
# delivers every body's presence every cycle, so the employee FSM and YOLO
# presence are never degraded -- only identity re-recognition is thinned,
# at the cost of slightly slower identity adoption/refresh.  Set to 1 to
# restore the legacy "faces every frame" behaviour.
FACE_DETECT_CADENCE = int(os.getenv("CCTV_FACE_DETECT_CADENCE", "5"))

# Part F (Phase 43) -- per-stage pipeline timing (opt-in, daemon only).
# When enabled the daemon records milliseconds per stage (capture / YOLO +
# recognition / state / live-state write / total) and publishes them into the
# live_state payload under ``ai.timing`` so dashboard lag can be attributed to
# the right stage instead of guessed at.  Off by default: the hot path stays
# zero-overhead for the common case.
PIPELINE_TIMING = os.getenv("CCTV_PIPELINE_TIMING", "0") == "1"

# Part C (Phase 43) -- per-cycle phone/person association diagnostics
# (opt-in).  When enabled the detector records per camera: person detections,
# phone detections, phone confidence distribution, phone box sizes, how many
# phone boxes associated to a person, and which frames ran face recognition.
# Published under ``ai.phone_diag``.  Honest measurement -- a diagnosis step,
# never a silent behaviour change.
PHONE_DIAG = os.getenv("CCTV_PHONE_DIAG", "0") == "1"

# A local webcam whose mean frame brightness stays below this (0-255) is
# treated as dark/blank (e.g. DroidCam not connected, lens covered).  The
# daemon warns loudly instead of silently reporting "nobody present".
BLACK_FRAME_MEAN = float(os.getenv("CCTV_BLACK_FRAME_MEAN", "12"))

# ============================================================
# PHONE DETECTOR  (Phase 44 -- dedicated phone pass)
# ------------------------------------------------------------
# Root cause (Phase 44): YOLOv8n's COCO ``cell phone`` class is weak on small
# phones in CCTV frames.  The fix is NOT a global confidence drop -- it is a
# dedicated, cadence-gated, higher-resolution phone pass plus per-track
# temporal evidence.  Configuration:
#   CCTV_PHONE_MODEL_PATH  Absolute path to a dedicated (ideally fine-tuned)
#                          phone YOLO model.  Empty/"" = reuse the base
#                          person model.  If the file does not exist the
#                          system falls back to the base model and logs a
#                          warning (never silently dies).
#   CCTV_PHONE_CLASS       Class id to treat as "phone".  67 = COCO cell phone
#                          (base model).  A fine-tuned single-class phone
#                          model commonly exposes its class as 0.
#   CCTV_PHONE_IMGSZ       Input resolution of the dedicated phone pass.
#                          1280 helps small phones; costs more CPU.
#   CCTV_PHONE_DETECT_CADENCE  Run the phone pass every N processed cycles per
#                          camera (face cadence analog).  Default 3 keeps the
#                          pass cheap while 5-second phone use stays reliably
#                          observable at ~2+ fps.  Never < 1.
#   CCTV_PHONE_EVIDENCE_WINDOW / CCTV_PHONE_EVIDENCE_MIN
#                          Per-track hysteresis: a frame's phone report counts
#                          only when this frame is a phone detection AND at
#                          least ``_MIN`` phone detections occurred in the
#                          trailing ``_WINDOW`` observations.  A single random
#                          false positive therefore can never start a phone
#                          episode, while genuinely intermittent detections of
#                          a real phone still accumulate evidence.
PHONE_MODEL_PATH = os.getenv("CCTV_PHONE_MODEL_PATH", "").strip()
PHONE_CLASS = int(os.getenv("CCTV_PHONE_CLASS", "67"))
PHONE_IMGSZ = int(os.getenv("CCTV_PHONE_IMGSZ", "1280"))
PHONE_DETECT_CADENCE = int(os.getenv("CCTV_PHONE_DETECT_CADENCE", "3"))
PHONE_EVIDENCE_WINDOW = int(os.getenv("CCTV_PHONE_EVIDENCE_WINDOW", "5"))
PHONE_EVIDENCE_MIN = int(os.getenv("CCTV_PHONE_EVIDENCE_MIN", "2"))

# ============================================================
# PERSON RE-IDENTIFICATION  (Phase 44 -- appearance identity)
# ------------------------------------------------------------
# A fully profile/back-facing person has no face, so InsightFace cannot
# identify them.  APPEARANCE identity is a conservative, secondary signal
# used ONLY to resolve a track that face recognition cannot identify.  It can
# never outrank a face identity and is never adopted from weak/ambiguous
# evidence (see the tracker identity-policy table in src/tracker.py).
#   CCTV_REID_ENABLED          Master switch (default: on).
#   CCTV_REID_MODEL_PATH       Path to a trained person-ReID model.  A
#                              ``*.pth``/``*.pt`` file is loaded as a trained
#                              **OSNet** torch model (Release 49; MIT, inputs
#                              256x128, embeddings 512-dim); a ``*.onnx`` file
#                              is loaded through onnxruntime with the same
#                              OSNet preprocessing.  Empty/"" or unloadable =
#                              built-in lightweight appearance descriptor
#                              (histogram + spatial grid + edge/silhouette;
#                              CPU-cheap, license-free).  A model file is NEVER
#                              downloaded automatically -- supply it locally.
#   CCTV_REID_CADENCE          Extract appearance embeddings every N processed
#                              cycles per camera; never < 1.
#   CCTV_REID_IMGSZ            Crop size for the built-in descriptor.
#   CCTV_REID_STRONG_THRESHOLD Minimum cosine similarity for a STRONG match.
#   CCTV_REID_MARGIN_MIN       Margin over the second-best employee required
#                              for a STRONG match (ambiguity guard).
#   CCTV_REID_ADOPT            Consecutive STRONG appearance votes required to
#                              adopt a candidate identity on an unresolved
#                              track.
#   CCTV_REID_EMBEDDING_DIM    Built-in descriptor embedding dimension (an
#                              OSNet torch path always uses its trained 512).
#   CCTV_REID_GATE_RESOLVED    OPT-IN event-driven cadence (default 0 = off).
#                              When on, the appearance pass skips person boxes
#                              that overlap a spatial track already resolved to
#                              a face/trusted identity in the previous cycle --
#                              those tracks cannot be changed by appearance
#                              votes, so their inference is pure waste.  Kept
#                              OFF by default because the real-camera latency
#                              measurements that justify it are still pending.
REID_ENABLED = os.getenv("CCTV_REID_ENABLED", "1") == "1"
REID_MODEL_PATH = os.getenv("CCTV_REID_MODEL_PATH", "").strip()
REID_CADENCE = int(os.getenv("CCTV_REID_CADENCE", "3"))
REID_IMGSZ = int(os.getenv("CCTV_REID_IMGSZ", "128"))
REID_STRONG_THRESHOLD = float(os.getenv("CCTV_REID_STRONG_THRESHOLD", "0.90"))
REID_MARGIN_MIN = float(os.getenv("CCTV_REID_MARGIN_MIN", "0.06"))
REID_ADOPT_FRAMES = int(os.getenv("CCTV_REID_ADOPT", "5"))
REID_EMBEDDING_DIM = int(os.getenv("CCTV_REID_DIM", "256"))
REID_GATE_RESOLVED = os.getenv("CCTV_REID_GATE_RESOLVED", "0") == "1"
#: Minimum IoU for a person box to count as "covered" by a resolved track box
#: when ``CCTV_REID_GATE_RESOLVED=1``.
REID_GATE_IOU = 0.30
# Enrollment images: full-body / side-pose appearance samples per employee
# (``data/reid/EMP001.jpg``, ``EMP001_2.jpg`` ...).  If this directory has no
# images the registry falls back to the face-enrollment images
# (``data/faces``) so the feature works out of the box; only compressed
# embeddings are cached/stored -- never raw person images.
REID_DIR = str(PROJECT_ROOT / "data" / "reid")
APPEARANCE_EMBEDDINGS_FILE = str(PROJECT_ROOT / "data" / "appearance.pkl")

# ============================================================
# TRACKER
# ------------------------------------------------------------
SMOOTHING_BUFFER_SEC = 5

# ------------------------------------------------------------------
# PERSISTENCE / IDENTITY STABILITY  (Phase 6)
# ------------------------------------------------------------------
# Number of consecutive observations required before a recognition
# change (e.g. EMP001 -> Unknown -> EMP002) is committed.  0 disables.
IDENTITY_STABILITY_FRAMES = 3

# How long (seconds) a recognised employee may be unobserved before we
# consider them absent rather than just "camera missed a frame".  This
# is *not* the same as the AWAY smoothing -- it guards between-batch gaps.
PRESENCE_PATIENCE_SEC = 8.0

# ------------------------------------------------------------
# WALL-CLOCK STATE THRESHOLDS  (Phase 41)
# Employee presence/FSM transitions must follow real elapsed seconds, not
# frame counts -- the pipeline is CPU-bound and FPS varies widely.
AWAY_AFTER_SEC = float(os.getenv("CCTV_AWAY_AFTER_SEC", "3.0"))
PHONE_AFTER_SEC = float(os.getenv("CCTV_PHONE_AFTER_SEC", "5.0"))
# A phone-use run keeps counting across brief detection gaps shorter than
# this (single missed frames must not reset the episode timer).
PHONE_GAP_GRACE_SEC = float(os.getenv("CCTV_PHONE_GAP_GRACE_SEC", "2.0"))

# ============================================================
# CAMERA SELECTION MODE  (Phase A -- EXPLICIT vs AUTO)
# ------------------------------------------------------------
# How the live camera source is resolved:
#   "AUTO"     -- probe video device indices and pick the first usable one
#                (runtime probing; equivalent to ``main.py --source auto``).
#   "EXPLICIT" -- use the configured/requested index exactly as given
#                (equivalent to ``main.py --source <index>`` or the default
#                camera pool).  Never silently substitutes a different device.
# The daemon overrides this from the ``--source`` flag; the dashboard displays
# the *resolved* mode, source and index from ``live_state.json``.
CAMERA_MODE = os.getenv("CCTV_CAMERA_MODE", "AUTO").upper()

# ============================================================
# SPATIAL TRACKER  (Phase A -- A5/A6)
# ------------------------------------------------------------
# Minimum IoU to associate a detection box with an existing track.
TRACK_IOU_THRESHOLD = float(os.getenv("CCTV_TRACK_IOU", "0.20"))
# A track with no detection for longer than this is considered gone.
TRACK_MAX_AGE_SEC = float(os.getenv("CCTV_TRACK_MAX_AGE", "3.0"))
# Identity votes (consecutive) required to adopt an identity on a fresh track.
IDENTITY_ADOPT_FRAMES = int(os.getenv("CCTV_IDENTITY_ADOPT", "3"))
# Identity votes (consecutive) + the previously adopted identity must have
# dropped out entirely before a KNOWN identity is switched (anti-flicker rule).
IDENTITY_SWITCH_FRAMES = int(os.getenv("CCTV_IDENTITY_SWITCH", "3"))
# Trajectory length kept per track (normalized centroid samples).
TRACK_TRAJECTORY_LEN = int(os.getenv("CCTV_TRACK_TRAJECTORY", "120"))
# Emit structured TRACK_DEBUG lines (track create/update/drop, identity votes).
TRACK_DEBUG = os.getenv("CCTV_TRACK_DEBUG", "0") == "1"

# ============================================================
# MOTION DETECTION  (Phase A -- A8)
# ------------------------------------------------------------
MOTION_ENABLED = os.getenv("CCTV_MOTION_ENABLED", "0") == "1"
# Fraction of changed pixels (0..1) required to start/continue a motion event.
MOTION_MIN_SCORE = float(os.getenv("CCTV_MOTION_MIN_SCORE", "0.02"))
# Motion must persist at least this long (seconds) before an event fires.
MOTION_MIN_DURATION_SEC = float(os.getenv("CCTV_MOTION_MIN_DURATION", "1.0"))
# Minimum gap between successive MOTION events per camera.
MOTION_COOLDOWN_SEC = float(os.getenv("CCTV_MOTION_COOLDOWN", "30"))
# Working width for frame differencing (display latency vs sensitivity).
MOTION_DOWNSCALE = int(os.getenv("CCTV_MOTION_DOWNSCALE", "320"))
MOTION_DEBUG = os.getenv("CCTV_MOTION_DEBUG", "0") == "1"

# ============================================================
# ENTRY / EXIT LINE CROSSINGS  (Phase A -- A10)
# ------------------------------------------------------------
ENTRY_EXIT_ENABLED = os.getenv("CCTV_ENTRY_EXIT_ENABLED", "0") == "1"
ENTRY_EXIT_LINES_FILE = str(PROJECT_ROOT / "data" / "entry_exit_lines.json")
# Once a track crosses a line in one direction it cannot re-fire on the same
# (line, track, direction) until this many seconds pass.
ENTRY_EXIT_DEBOUNCE_SEC = float(os.getenv("CCTV_ENTRY_EXIT_DEBOUNCE", "30"))
# Opt-in heuristic (B6): emit UNEXPECTED_STAY when an employee ARRIVED via a
# crossing but no EXIT_CROSSING was seen before the office window ends.
UNEXPECTED_STAY_ENABLED = os.getenv("CCTV_UNEXPECTED_STAY_ENABLED", "0") == "1"
# LEFT from an EXIT_CROSSING suppresses the disappearance-derived LEFT within
# this window (seconds) so one departure produces one LEFT event.
LEFT_DEBOUNCE_SEC = float(os.getenv("CCTV_LEFT_DEBOUNCE", "10"))

# Structured per-face / per-frame debug (existing) plus motion/track gating.
DETECTION_DEBUG = os.getenv("CCTV_DETECTION_DEBUG", "0") == "1"

# ============================================================
# DESK / SEAT ZONE TRACKING  (Phase 59)
# ------------------------------------------------------------
# Desk/seat zones annotate which desk a spatial track occupies.  A seat is
# CONTEXT ONLY: it never becomes an identity authority.  Identity continues to
# follow the face/appearance pipeline (safe: unknown stays unknown, EMP002 in
# A01 is never relabelled EMP001).
SEAT_ZONE_TRACKING = os.getenv("CCTV_SEAT_ZONE_TRACKING", "1") == "1"
# Optional first-run seeding file (same pattern as security_zones.json).
SEAT_ZONES_FILE = os.getenv("CCTV_SEAT_ZONES_FILE",
                            str(PROJECT_ROOT / "data" / "seat_zones.json"))
# Consecutive frames a footpoint must remain inside a zone before the zone is
# marked OCCUPIED (temporal stability -- a single noisy frame never flips it).
SEAT_ZONE_CONFIRM_FRAMES = int(os.getenv("CCTV_SEAT_ZONE_CONFIRM_FRAMES", "5"))
# Consecutive frames with no person in the zone before it is marked VACANT.
SEAT_ZONE_CLEAR_FRAMES = int(os.getenv("CCTV_SEAT_ZONE_CLEAR_FRAMES", "8"))
# Minimum seconds between repeated identity/mismatch events per zone (events
# are transition-driven; this gates re-identification noise).
SEAT_EVENT_COOLDOWN_SEC = float(os.getenv("CCTV_SEAT_EVENT_COOLDOWN_SEC", "60"))

# Structured per-camera loop diagnostics (source, mode, resolved index, frame
# age/fps, usability decision).  Off by default; enabled via CCTV_CAMERA_DEBUG=1.
CAMERA_DEBUG = os.getenv("CCTV_CAMERA_DEBUG", "0") == "1"

# ============================================================
# WORK SCHEDULE / PRODUCTIVITY  (Phase 2B)
# ------------------------------------------------------------
# Expected working window used for productivity scoring and reporting.
# Times are local 24h strings.  Lunch is treated as an unpaid (excluded)
# window; camera downtime/unobserved time is NOT charged to the employee.
WORK_SCHEDULE = {
    "start": os.getenv("CCTV_WORK_START", "09:00"),
    "end": os.getenv("CCTV_WORK_END", "18:00"),
    "grace": int(os.getenv("CCTV_GRACE_MIN", "15")),        # tolerated lateness
    "lunch": os.getenv("CCTV_LUNCH", "12:00-13:00"),         # unpaid window
    "breaks": int(os.getenv("CCTV_BREAKS_MIN", "30")),       # additional paid breaks
}

# ============================================================
# SECURITY / INCIDENT ENGINE  (Phase 31)
# ------------------------------------------------------------
# The security layer is additive and explicit: every detector uses
# persistence thresholds + confidence + cooldown + evidence + human-review.
# Nothing here changes employee productivity semantics.
# ------------------------------------------------------------
SECURITY_ENABLED = os.getenv("CCTV_SECURITY_ENABLED", "1") == "1"

SECURITY_ZONES_FILE = str(PROJECT_ROOT / "data" / "security_zones.json")

# Office-hours used by AFTER_HOURS_ACTIVITY (defaults to the working window).
SECURITY_OFFICE_START = os.getenv("CCTV_OFFICE_START", WORK_SCHEDULE["start"])
SECURITY_OFFICE_END = os.getenv("CCTV_OFFICE_END", WORK_SCHEDULE["end"])

# -- Unknown-presence rules ------------------------------------------
# mode: immediate | after_seconds | restricted_only | off_hours_only | off
SECURITY_UNKNOWN_MODE = os.getenv("CCTV_UNKNOWN_ALERT_MODE", "after_seconds")
SECURITY_UNKNOWN_AFTER_SEC = float(os.getenv("CCTV_UNKNOWN_ALERT_AFTER_SEC", "10"))
SECURITY_UNKNOWN_COOLDOWN_SEC = float(os.getenv("CCTV_UNKNOWN_COOLDOWN_SEC", "60"))
SECURITY_UNKNOWN_SUPPRESS_REPEATS = os.getenv("CCTV_UNKNOWN_SUPPRESS_REPEATS", "1") == "1"

# -- Camera offline / tamper -----------------------------------------
# Frames stopped delivering for this long -> CAMERA_OFFLINE event.
OFFLINE_TRIGGER_SEC = float(os.getenv("CCTV_OFFLINE_TRIGGER_SEC", "15"))
# A tamper condition must persist this long before it is reported.
TAMPER_PERSIST_SEC = float(os.getenv("CCTV_TAMPER_PERSIST_SEC", "30"))
TAMPER_DARK_MEAN = float(os.getenv("CCTV_TAMPER_DARK_MEAN", "30"))
TAMPER_BRIGHT_MEAN = float(os.getenv("CCTV_TAMPER_BRIGHT_MEAN", "235"))
TAMPER_FROZEN_DIFF = float(os.getenv("CCTV_TAMPER_FROZEN_DIFF", "0.5"))
TAMPER_LOW_STD = float(os.getenv("CCTV_TAMPER_LOW_STD", "5.0"))
TAMPER_ENABLE_DARK = os.getenv("CCTV_TAMPER_ENABLE_DARK", "1") == "1"

# -- Intrusion / zones -----------------------------------------------
INTRUSION_PERSIST_SEC = float(os.getenv("CCTV_INTRUSION_PERSIST_SEC", "5"))
INTRUSION_COOLDOWN_SEC = float(os.getenv("CCTV_INTRUSION_COOLDOWN_SEC", "600"))

# -- Detector health / fail-closed (Phase 31 hardening) ---------------
# Rate limit (seconds) for recurring DETECTOR_UNAVAILABLE events emitted
# while the person-detection model is down.  Floor is 5s.
SECURITY_DETECTOR_COOLDOWN_SEC = float(
    os.getenv("CCTV_SECURITY_DETECTOR_COOLDOWN_SEC", "60"))

# -- Event correlation (Phase 32) -------------------------------------
# Temporal window (seconds) within which different-but-related events are
# grouped into one incident (e.g. Unknown + intrusion + after-hours).
CORRELATION_WINDOW_SEC = float(os.getenv("CCTV_CORRELATION_WINDOW_SEC", "300"))

# -- Phase 33: advisory intelligence (temporal / anomaly / risk / topology) --
# Temporal intelligence thresholds (transparent, documented).
TEMPORAL_REPEAT_THRESHOLD = int(os.getenv("CCTV_TEMPORAL_REPEAT", "5"))
TEMPORAL_ESCALATE_COUNT = int(os.getenv("CCTV_TEMPORAL_ESCALATE_COUNT", "3"))
TEMPORAL_SILENCE_DAYS = int(os.getenv("CCTV_TEMPORAL_SILENCE_DAYS", "1"))

# Anomaly engine (advisory; INSUFFICIENT_DATA until enough samples).
ANOMALY_MIN_SAMPLES = int(os.getenv("CCTV_ANOMALY_MIN_SAMPLES", "5"))
ANOMALY_ZSCORE = float(os.getenv("CCTV_ANOMALY_ZSCORE", "3.0"))

# Risk scoring for incidents (never people).
RISK_ENABLED = os.getenv("CCTV_RISK_ENABLED", "1") == "1"
# Phase 34: add transparent context factors (zone sensitivity, after-hours,
# evidence availability, confidence) to incident risk as a bounded bonus.
RISK_CONTEXT_FACTORS = os.getenv("CCTV_RISK_CONTEXT_FACTORS", "1") == "1"

# Alert intelligence: incident-level dedup window (seconds).
INCIDENT_ALERT_DEDUP_SEC = int(os.getenv("CCTV_INCIDENT_ALERT_DEDUP_SEC", "3600"))

# Camera reliability scoring.
RELIABILITY_FPS_TARGET = int(os.getenv("CCTV_RELIABILITY_FPS_TARGET", "10"))

# -- Loitering / occupancy -------------------------------------------
LOITERING_SEC = float(os.getenv("CCTV_LOITERING_SEC", "120"))
CROWD_THRESHOLD = int(os.getenv("CCTV_CROWD_THRESHOLD", "5"))
OCCUPANCY_COOLDOWN_SEC = float(os.getenv("CCTV_OCCUPANCY_COOLDOWN_SEC", "900"))

# -- Evidence (event-driven only; default OFF) -----------------------
EVIDENCE_MODE = os.getenv("CCTV_EVIDENCE_MODE", "OFF").upper()
EVIDENCE_DIR = str(PROJECT_ROOT / "data" / "evidence")
EVIDENCE_PRE_SEC = float(os.getenv("CCTV_EVIDENCE_PRE_SEC", "5"))
EVIDENCE_POST_SEC = float(os.getenv("CCTV_EVIDENCE_POST_SEC", "10"))
EVIDENCE_BUF_FRAMES = int(os.getenv("CCTV_EVIDENCE_BUF_FRAMES", "45"))
EVIDENCE_RETENTION_DAYS = int(os.getenv("CCTV_EVIDENCE_RETENTION_DAYS", "30"))
EVIDENCE_MAX_MB = int(os.getenv("CCTV_EVIDENCE_MAX_MB", "2048"))

# -- Alerts (default channels / cooldown) ----------------------------
ALERT_COOLDOWN_SEC = float(os.getenv("CCTV_ALERT_COOLDOWN_SEC", "300"))
DEFAULT_ALERT_CHANNELS = os.getenv("CCTV_ALERT_CHANNELS", "dashboard,log")

# -- Audit ------------------------------------------------------------------
AUDIT_RETENTION_DAYS = int(os.getenv("CCTV_AUDIT_RETENTION_DAYS", "365"))

# ============================================================
# BACKUP  (Phase 31 -- WAL-safe, default OFF)
# ------------------------------------------------------------
BACKUP_DIR = str(PROJECT_ROOT / "data" / "backups")
AUTO_BACKUP_DAILY = os.getenv("CCTV_AUTO_BACKUP_DAILY", "0") == "1"
BACKUP_HOUR = int(os.getenv("CCTV_BACKUP_HOUR", "3"))

# ============================================================
# PERSISTENT STORES  (Phase 5)
# ------------------------------------------------------------
# Normalise a ``date`` column on each activity row for fast, correct
# date filtering (avoids LIKE-prefix bugs and enables true day splits).
NORMALIZE_SCHEMA = True

# ============================================================
# PILOT MODE  (Phase 37 -- controlled advisory pilot)
# ------------------------------------------------------------
# Pilot mode enables verbose operational metrics and makes failures
# visible without disabling any security, RBAC, audit, or privacy
# controls.  It is explicitly opt-in and safe for production data.
#
# Pilot mode MUST NOT:
#   - disable security
#   - bypass RBAC
#   - bypass audit
#   - enable continuous recording
#   - automatically accuse employees
#   - modify productivity rules
# ============================================================
PILOT_MODE = os.getenv("CCTV_PILOT_MODE", "0") == "1"
PILOT_METRICS_INTERVAL_SEC = int(os.getenv("CCTV_PILOT_METRICS_INTERVAL", "30"))
PILOT_HEALTH_LOG_INTERVAL_SEC = int(os.getenv("CCTV_PILOT_HEALTH_LOG", "60"))

# ============================================================
# DATA RETENTION  (Phase 20)
# ------------------------------------------------------------
# Retention (in days) is only applied by an explicit maintenance call --
# the daemon never auto-deletes without ``CCTV_ENABLE_RETENTION=1``.
RETENTION_DAYS = int(os.getenv("CCTV_RETENTION_DAYS", "365"))
RETENTION_DAYS_LOGS = int(os.getenv("CCTV_RETENTION_LOGS_DAYS", "365"))
RETENTION_DAYS_REPORTS = int(os.getenv("CCTV_RETENTION_REPORTS_DAYS", "365"))
ENABLE_RETENTION = os.getenv("CCTV_ENABLE_RETENTION", "0") == "1"
# Phase 54 -- privacy (M06).  Rebuildable biometric caches
# (``data/embeddings.pkl``, ``data/appearance.pkl``) are pruned only when
# BOTH ``CCTV_ENABLE_RETENTION=1`` AND this window is > 0.  The enrollment
# images under ``data/faces/`` / ``data/reid/`` are the source of truth and are
# NEVER deleted by retention -- only the derived mean-embedding caches aged
# beyond the window are removed (they can be rebuilt from images).
RETENTION_DAYS_BIOMETRICS = int(os.getenv("CCTV_RETENTION_BIOMETRICS_DAYS", "0"))

# ============================================================
# DASHBOARD  (Phases 9/10/11)
# ------------------------------------------------------------
DASHBOARD_REFRESH_SEC = int(os.getenv("CCTV_DASH_REFRESH", "5"))
DASH_REFRESH_SEC = DASHBOARD_REFRESH_SEC  # short alias
# Simple auth (best-effort; production should sit behind a reverse proxy).
DASH_AUTH_ENABLED = os.getenv("CCTV_DASH_AUTH", "1") == "1"
DASH_USERNAME = os.getenv("CCTV_DASH_USER", "admin")
DASH_ADMIN_PASS = os.getenv("CCTV_DASH_PASS", "")   # empty disables auth
DASH_VIEWER_PASS = os.getenv("CCTV_DASH_VIEWER_PASS", "")  # view-only role
DASH_ROLE_VIEWER = "viewer"
# Production fail-closed guard (Phase 39, B4; hardened default in Phase 64B).
# When this is enabled ("1", the secure default) and dashboard auth is enabled
# but NO admin/viewer password is configured, the dashboard refuses to grant
# access (fail closed) and the preflight check reports a blocking FAIL.  This
# prevents a bare/cloud deployment from silently running fully open as admin.
# Local development keeps working: explicit ``CCTV_DASH_PASS`` /
# ``CCTV_DASH_VIEWER_PASS`` enable the normal login flow, and ``CCTV_DASH_AUTH=0``
# re-enables the documented open dev mode on a trusted network.
DASH_FAIL_CLOSED = os.getenv("CCTV_DASH_FAIL_CLOSED", "1") == "1"
# Session expiry (Phase 39, B18).  After this many minutes a logged-in session
# is invalidated and the user must sign in again.  A value <= 0 disables expiry
# (keeps the previous persistent-session behavior for local development).
DASH_SESSION_MINUTES = float(os.getenv("CCTV_DASH_SESSION_MINUTES", "15"))
# Read-query cache TTLs (Phase 64D).  These only bound how often the dashboard
# re-runs its read-only metric queries on a fresh short-lived connection.  The
# live-snapshot file read is NEVER cached, so live status always stays fresh at
# DASH_REFRESH_SEC.  The day-aggregate defaults to 15 s (it feeds the KPIs and
# productivity chart) and the full-range aggregate to 60 s (heaviest query,
# used by Analytics/Reports).  Both are read-only historical aggregates; admin
# mutations already invalidate the metrics caches on the paths that changed.
DASH_DAY_METRICS_TTL_SEC = int(os.getenv("CCTV_DASH_DAY_METRICS_TTL", "15"))
DASH_RANGE_METRICS_TTL_SEC = int(os.getenv("CCTV_DASH_RANGE_METRICS_TTL", "60"))

# ============================================================
# EMAIL RETRY  (Phase 13)
# ------------------------------------------------------------
EMAIL_MAX_RETRIES = int(os.getenv("CCTV_EMAIL_RETRIES", "3"))
EMAIL_RETRY_DELAY_SEC = float(os.getenv("CCTV_EMAIL_RETRY_DELAY", "10.0"))
EMAIL_TIMEOUT_SEC = float(os.getenv("CCTV_EMAIL_TIMEOUT", "30.0"))

# ============================================================
# DATABASE / REPORT
# ------------------------------------------------------------
DB_PATH = str(PROJECT_ROOT / "data" / "database" / "sessions.db")
REPORT_OUTPUT_DIR = str(PROJECT_ROOT / "data" / "reports")


def _eod_hour_from_env() -> int | None:
    """EOD report hour (0-23).  Empty / ``none`` / ``off`` disables."""
    raw = os.getenv("CCTV_EOD_HOUR", "19").strip().lower()
    if raw in ("", "none", "off"):
        return None
    try:
        return int(raw)
    except ValueError:
        return None


EOD_REPORT_HOUR = _eod_hour_from_env()  # 24h; None disables scheduled reports

# ============================================================
# LOGGING
# ------------------------------------------------------------
LOG_DIR = str(PROJECT_ROOT / "data" / "logs")
LOG_FILE = os.path.join(LOG_DIR, "app.log")


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logger to write to both stderr and a rolling log file.

    Returns the application logger (``"cctv"``).
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers on repeated calls.
    if not getattr(root, "_cctv_configured", False):
        root.handlers.clear()

        # Stream handler (stderr)
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(level)
        sh.setFormatter(fmt)
        root.addHandler(sh)

        # File handler (rotating at 5 MB, keep 3 backups)
        try:
            from logging.handlers import RotatingFileHandler
            fh = RotatingFileHandler(
                LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
            )
            fh.setLevel(level)
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except OSError:
            pass  # non-critical: run without file logging

        root._cctv_configured = True  # type: ignore[attr-defined]

    return logging.getLogger("cctv")


# ============================================================
# SMTP / EMAIL  (credentials from env / .env)
# ------------------------------------------------------------
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD", "")
RECIPIENT_EMAILS = os.getenv("RECIPIENT_EMAILS", "")  # comma-separated


def parse_recipients(raw: str | None = None) -> list[str]:
    """Return a list of email addresses from a comma-separated string."""
    value = raw or RECIPIENT_EMAILS
    return [e.strip() for e in value.split(",") if e.strip()]


# ============================================================
# LOADER HELPERS
# ============================================================

def load_desk_rois() -> dict[str, dict[str, int]]:
    """Return {emp_id: {"x","y","w","h"}, ...} preferring roi_presets.json."""
    import json

    if os.path.exists(ROI_PRESETS_FILE) and os.path.getsize(ROI_PRESETS_FILE) > 0:
        try:
            with open(ROI_PRESETS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data:
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {k: dict(v) for k, v in MULTI_DESK_ROIS.items()}


def load_desk_roi() -> dict:
    """Legacy single-desk helper for backward compatibility."""
    rois = load_desk_rois()
    emp_id = next(iter(rois))
    desk = dict(rois[emp_id])
    desk["employee_id"] = emp_id
    return desk


# ============================================================
# CONFIG VALIDATION  (Phase 22)
# ============================================================

def _parse_hhmm(value: str) -> int:
    """Parse ``HH:MM`` into minutes-from-midnight, raising ValueError."""
    h, m = value.split(":")
    hour, minute = int(h), int(m)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"invalid time {value!r}")
    return hour * 60 + minute


def validate_config(quiet: bool = True) -> list[str]:
    """Validate key numeric/config values; return a list of problems.

    Clears problems in logs; caller may fail-fast on return value.  Never
    raises for normal misconfigurations -- returns descriptive messages.
    """
    problems: list[str] = []
    log = logger.warning

    if MAX_BATCH_SIZE < 1:
        problems.append(f"CCTV_MAX_BATCH_SIZE must be >= 1 (got {MAX_BATCH_SIZE})")
    if TARGET_FPS_PER_CAMERA < 0:
        problems.append(f"CCTV_TARGET_FPS must be >= 0 (got {TARGET_FPS_PER_CAMERA})")
    if LOCAL_SOURCE_DEFAULT_TARGET_FPS < 1:
        problems.append("CCTV_LOCAL_TARGET_FPS must be >= 1 "
                        f"(got {LOCAL_SOURCE_DEFAULT_TARGET_FPS})")
    if SMOOTHING_BUFFER_SEC < 0:
        problems.append(f"SMOOTHING_BUFFER_SEC must be >= 0 (got {SMOOTHING_BUFFER_SEC})")
    if AWAY_AFTER_SEC < 0:
        problems.append(f"CCTV_AWAY_AFTER_SEC must be >= 0 (got {AWAY_AFTER_SEC})")
    if PHONE_AFTER_SEC < 0:
        problems.append(f"CCTV_PHONE_AFTER_SEC must be >= 0 (got {PHONE_AFTER_SEC})")
    if PHONE_GAP_GRACE_SEC < 0:
        problems.append(f"CCTV_PHONE_GAP_GRACE_SEC must be >= 0 (got {PHONE_GAP_GRACE_SEC})")
    if FRAME_STALE_SEC <= 0:
        problems.append(f"CCTV_FRAME_STALE_SEC must be > 0 (got {FRAME_STALE_SEC})")
    if STARTUP_FRAME_WAIT_SEC < 0:
        problems.append(f"CCTV_STARTUP_FRAME_WAIT must be >= 0 (got {STARTUP_FRAME_WAIT_SEC})")
    if FACE_DETECT_SIZE < 0:
        problems.append(f"CCTV_FACE_DETECT_SIZE must be >= 0 (got {FACE_DETECT_SIZE})")
    if LIVE_MIN_FACE_AREA < 0:
        problems.append(f"CCTV_LIVE_MIN_FACE_AREA must be >= 0 (got {LIVE_MIN_FACE_AREA})")

    # Working schedule sanity
    try:
        start = _parse_hhmm(WORK_SCHEDULE["start"])
        end = _parse_hhmm(WORK_SCHEDULE["end"])
        if start >= end:
            problems.append(f"WORK_SCHEDULE start ({WORK_SCHEDULE['start']}) must be < end ({WORK_SCHEDULE['end']})")
    except (KeyError, ValueError) as exc:
        problems.append(f"WORK_SCHEDULE start/end invalid: {exc}")

    lunch = WORK_SCHEDULE.get("lunch", "")
    if isinstance(lunch, str) and lunch:
        try:
            a, b = lunch.split("-", 1)
            if _parse_hhmm(a) >= _parse_hhmm(b):
                problems.append(f"WORK_SCHEDULE lunch invalid range: {lunch}")
        except (ValueError, TypeError):
            problems.append(f"WORK_SCHEDULE lunch invalid: {lunch}")

    if EOD_REPORT_HOUR is not None and not (0 <= EOD_REPORT_HOUR <= 23):
        problems.append(f"EOD_REPORT_HOUR must be 0-23 or None (got {EOD_REPORT_HOUR})")

    # Phase 31 security settings
    if SECURITY_UNKNOWN_MODE not in (
        "immediate", "after_seconds", "restricted_only", "off_hours_only", "off"
    ):
        problems.append(f"CCTV_UNKNOWN_ALERT_MODE invalid: {SECURITY_UNKNOWN_MODE}")
    if TAMPER_PERSIST_SEC < 0 or INTRUSION_PERSIST_SEC < 0 \
            or LOITERING_SEC < 0 or OFFLINE_TRIGGER_SEC <= 0:
        problems.append("security persistence/offline thresholds must be positive")
    try:
        _parse_hhmm(SECURITY_OFFICE_START)
        _parse_hhmm(SECURITY_OFFICE_END)
    except (ValueError, KeyError):
        problems.append("SECURITY office hours invalid (CCTV_OFFICE_START/END)")
    if EVIDENCE_MODE not in ("OFF", "EVENT_ONLY", "HIGH_SEVERITY_ONLY", "CONTINUOUS"):
        problems.append(f"CCTV_EVIDENCE_MODE invalid: {EVIDENCE_MODE}")
    if not (0 <= BACKUP_HOUR <= 23):
        problems.append(f"CCTV_BACKUP_HOUR must be 0-23 (got {BACKUP_HOUR})")

    # Phase 33 -- advisory intelligence settings
    if TEMPORAL_REPEAT_THRESHOLD < 1:
        problems.append(f"CCTV_TEMPORAL_REPEAT must be >= 1 (got {TEMPORAL_REPEAT_THRESHOLD})")
    if TEMPORAL_ESCALATE_COUNT < 1:
        problems.append(f"CCTV_TEMPORAL_ESCALATE_COUNT must be >= 1 (got {TEMPORAL_ESCALATE_COUNT})")
    if TEMPORAL_SILENCE_DAYS < 0:
        problems.append(f"CCTV_TEMPORAL_SILENCE_DAYS must be >= 0 (got {TEMPORAL_SILENCE_DAYS})")
    if ANOMALY_MIN_SAMPLES < 1:
        problems.append(f"CCTV_ANOMALY_MIN_SAMPLES must be >= 1 (got {ANOMALY_MIN_SAMPLES})")
    if ANOMALY_ZSCORE <= 0:
        problems.append(f"CCTV_ANOMALY_ZSCORE must be > 0 (got {ANOMALY_ZSCORE})")
    if INCIDENT_ALERT_DEDUP_SEC < 0:
        problems.append(f"CCTV_INCIDENT_ALERT_DEDUP_SEC must be >= 0 (got {INCIDENT_ALERT_DEDUP_SEC})")
    if RELIABILITY_FPS_TARGET < 1:
        problems.append(f"CCTV_RELIABILITY_FPS_TARGET must be >= 1 (got {RELIABILITY_FPS_TARGET})")

    # Phase 37 -- pilot mode settings
    if PILOT_METRICS_INTERVAL_SEC < 1:
        problems.append(f"CCTV_PILOT_METRICS_INTERVAL must be >= 1 (got {PILOT_METRICS_INTERVAL_SEC})")
    if PILOT_HEALTH_LOG_INTERVAL_SEC < 1:
        problems.append(f"CCTV_PILOT_HEALTH_LOG must be >= 1 (got {PILOT_HEALTH_LOG_INTERVAL_SEC})")

    # Phase 35 -- hardening: threshold ranges, retention, camera URLs, SMTP.
    _pct = lambda name, v: not (0.0 <= v <= 1.0)
    if _pct("CONF", CONF_THRESHOLD):
        problems.append(f"CCTV_CONF_THRESHOLD must be in [0.0, 1.0] (got {CONF_THRESHOLD})")
    if _pct("FACE", FACE_SIMILARITY_THRESHOLD):
        problems.append(f"CCTV_FACE_THRESHOLD must be in [0.0, 1.0] (got {FACE_SIMILARITY_THRESHOLD})")
    if not (0.0 <= FACE_CANDIDATE_THRESHOLD <= FACE_SIMILARITY_THRESHOLD):
        problems.append(
            "CCTV_FACE_CANDIDATE_THRESHOLD must be in "
            f"[0.0, FACE_SIMILARITY_THRESHOLD={FACE_SIMILARITY_THRESHOLD}] "
            f"(got {FACE_CANDIDATE_THRESHOLD})")
    if FACE_MARGIN_MIN < 0:
        problems.append(f"CCTV_FACE_MARGIN_MIN must be >= 0 (got {FACE_MARGIN_MIN})")
    if CROWD_THRESHOLD < 1:
        problems.append(f"CCTV_CROWD_THRESHOLD must be >= 1 (got {CROWD_THRESHOLD})")
    if EVIDENCE_RETENTION_DAYS < 0:
        problems.append(f"CCTV_EVIDENCE_RETENTION_DAYS must be >= 0 (got {EVIDENCE_RETENTION_DAYS})")
    if AUDIT_RETENTION_DAYS < 0:
        problems.append(f"CCTV_AUDIT_RETENTION_DAYS must be >= 0 (got {AUDIT_RETENTION_DAYS})")

    # Camera sources must not be empty/whitespace (a silent-empty source yields
    # a camera that can never come online). Duplicate sources are tolerated.
    for cid, url in CAMERAS.items():
        if not url or not str(url).strip():
            problems.append(f"camera {cid}: source URL is empty")

    # SMTP completeness -- only when a sender address is supplied (email is
    # effectively disabled otherwise) and retries are configured sanely.
    if SENDER_EMAIL:
        if not SMTP_SERVER or not str(SMTP_SERVER).strip():
            problems.append("SMTP_SERVER is empty while SENDER_EMAIL is set")
        if SMTP_PORT < 1 or SMTP_PORT > 65535:
            problems.append(f"SMTP_PORT out of range (got {SMTP_PORT})")

    # Dashboard auth fail-closed (Phase 39, B4): when production fail-closed is
    # requested but no password is set, surface it so deployment cannot start
    # silently open.
    if DASH_FAIL_CLOSED and DASH_AUTH_ENABLED \
            and not (DASH_ADMIN_PASS or DASH_VIEWER_PASS):
        problems.append(
            "CCTV_DASH_FAIL_CLOSED=1 requires a dashboard password "
            "(CCTV_DASH_PASS / CCTV_DASH_VIEWER_PASS); dashboard is blocked")

    # Phase 44 -- dedicated phone pass + person ReID settings
    if PHONE_IMGSZ < 32:
        problems.append(f"CCTV_PHONE_IMGSZ must be >= 32 (got {PHONE_IMGSZ})")
    if PHONE_DETECT_CADENCE < 1:
        problems.append(
            f"CCTV_PHONE_DETECT_CADENCE must be >= 1 (got {PHONE_DETECT_CADENCE})")
    if PHONE_EVIDENCE_WINDOW < 1 or PHONE_EVIDENCE_MIN < 1:
        problems.append("CCTV_PHONE_EVIDENCE_WINDOW/MIN must be >= 1")
    if PHONE_EVIDENCE_MIN > PHONE_EVIDENCE_WINDOW:
        problems.append("CCTV_PHONE_EVIDENCE_MIN must be <= CCTV_PHONE_EVIDENCE_WINDOW")
    if REID_CADENCE < 1:
        problems.append(f"CCTV_REID_CADENCE must be >= 1 (got {REID_CADENCE})")
    if REID_IMGSZ < 16:
        problems.append(f"CCTV_REID_IMGSZ must be >= 16 (got {REID_IMGSZ})")
    if not (0.0 <= REID_STRONG_THRESHOLD <= 1.0):
        problems.append(
            f"CCTV_REID_STRONG_THRESHOLD must be in [0,1] (got {REID_STRONG_THRESHOLD})")
    if REID_MARGIN_MIN < 0:
        problems.append(f"CCTV_REID_MARGIN_MIN must be >= 0 (got {REID_MARGIN_MIN})")
    if REID_ADOPT_FRAMES < 1:
        problems.append(f"CCTV_REID_ADOPT must be >= 1 (got {REID_ADOPT_FRAMES})")
    if not (0.0 < REID_GATE_IOU <= 1.0):
        problems.append(
            f"CCTV_REID_GATE_IOU must be in (0,1] (got {REID_GATE_IOU})")

    # Phase 59 -- desk/seat zone tracking settings
    if SEAT_ZONE_CONFIRM_FRAMES < 1:
        problems.append(
            f"CCTV_SEAT_ZONE_CONFIRM_FRAMES must be >= 1 (got {SEAT_ZONE_CONFIRM_FRAMES})")
    if SEAT_ZONE_CLEAR_FRAMES < 1:
        problems.append(
            f"CCTV_SEAT_ZONE_CLEAR_FRAMES must be >= 1 (got {SEAT_ZONE_CLEAR_FRAMES})")
    if SEAT_EVENT_COOLDOWN_SEC < 0:
        problems.append(
            f"CCTV_SEAT_EVENT_COOLDOWN_SEC must be >= 0 (got {SEAT_EVENT_COOLDOWN_SEC})")

    for p in problems:
        log("CONFIG: %s", p)
    return problems


def runtime_config_snapshot() -> dict:
    """Return a validated, env-resolved snapshot of key tunables.

    Used by the Configuration UI (Phase 33, C2) for a restart-safe, read-only
    summary of what the current process actually runs with.  It is a plain
    dict of primitive values only (safe to render/serialize); never includes
    credentials.
    """
    return {
        "input_source": SOURCE,
        "security_enabled": SECURITY_ENABLED,
        "evidence_mode": EVIDENCE_MODE,
        "evidence_retention_days": EVIDENCE_RETENTION_DAYS,
        "correlation_window_sec": CORRELATION_WINDOW_SEC,
        "temporal_repeat_threshold": TEMPORAL_REPEAT_THRESHOLD,
        "face_candidate_threshold": FACE_CANDIDATE_THRESHOLD,
        "face_margin_min": FACE_MARGIN_MIN,
        "temporal_escalate_count": TEMPORAL_ESCALATE_COUNT,
        "temporal_silence_days": TEMPORAL_SILENCE_DAYS,
        "anomaly_min_samples": ANOMALY_MIN_SAMPLES,
        "anomaly_zscore": ANOMALY_ZSCORE,
        "risk_enabled": RISK_ENABLED,
        "risk_context_factors": RISK_CONTEXT_FACTORS,
        "incident_alert_dedup_sec": INCIDENT_ALERT_DEDUP_SEC,
        "reliability_fps_target": RELIABILITY_FPS_TARGET,
        "pilot_mode": PILOT_MODE,
        "pilot_metrics_interval_sec": PILOT_METRICS_INTERVAL_SEC,
        "pilot_health_log_interval_sec": PILOT_HEALTH_LOG_INTERVAL_SEC,
        "offline_trigger_sec": OFFLINE_TRIGGER_SEC,
        "tamper_persist_sec": TAMPER_PERSIST_SEC,
        "intrusion_persist_sec": INTRUSION_PERSIST_SEC,
        "loitering_sec": LOITERING_SEC,
        "unknown_mode": SECURITY_UNKNOWN_MODE,
        "alerts_cooldown_sec": ALERT_COOLDOWN_SEC,
        "retention_days": RETENTION_DAYS,
        "retention_days_biometrics": RETENTION_DAYS_BIOMETRICS,
    }


# ======================================================================
# Phase 59b -- OFFICE SCHEDULE + FUTURE THRESHOLDS  (additive knobs)
# ======================================================================
# Purely additive Phase 59b knobs.  They let an admin define:
#   1) the OFFICE-MONITORING window (when the scheduler considers the office
#      in session for desk/chair context) and
#   2) reserved FUTURE_* threshold knobs used by a LATER engine phase.
# Nothing here relabels anyone, changes identity, or touches the legacy
# WORK_SCHEDULE / SECURITY_OFFICE_* constants above.
# ======================================================================

CCTV_OFFICE_START = os.getenv("CCTV_OFFICE_START", "10:00")
CCTV_OFFICE_END = os.getenv("CCTV_OFFICE_END", "18:30")
CCTV_LUNCH_START = os.getenv("CCTV_LUNCH_START", "14:00")
CCTV_LUNCH_END = os.getenv("CCTV_LUNCH_END", "14:35")
CCTV_OFFICE_TZ = os.getenv("CCTV_OFFICE_TZ", "Asia/Kolkata")

FUTURE_AWAY_THRESHOLD_SECONDS = float(os.getenv("CCTV_FUTURE_AWAY_SECONDS", "10"))
FUTURE_PHONE_THRESHOLD_SECONDS = float(os.getenv("CCTV_FUTURE_PHONE_SECONDS", "5"))
FUTURE_TALKING_THRESHOLD_SECONDS = float(os.getenv("CCTV_FUTURE_TALKING_SECONDS", "15"))


def _validate_phase59b():
    """Additive validation for Phase 59b knobs only.  Raises ValueError when
    a knob is misconfigured; never re-validates or touches legacy knobs."""
    if FUTURE_AWAY_THRESHOLD_SECONDS < 0.0:
        raise ValueError("FUTURE_AWAY_THRESHOLD_SECONDS must be >= 0")
    if FUTURE_PHONE_THRESHOLD_SECONDS < 0.0:
        raise ValueError("FUTURE_PHONE_THRESHOLD_SECONDS must be >= 0")
    if FUTURE_TALKING_THRESHOLD_SECONDS < 0.0:
        raise ValueError("FUTURE_TALKING_THRESHOLD_SECONDS must be >= 0")


_validate_phase59b()


# ======================================================================
# Phase 59c -- CHAIR CONTEXT KNOBS (additive; config surface only)
# ======================================================================
# Phase 59c is the CHAIR layer for the Phase 59b office-schedule foundation.
# Chairs are CONTEXT-ONLY children of a Phase 59b desk/seat zone: they
# describe *where* someone sits, never *who* they are.
#
# Identity rule (identical to seat_zones.py's rule):
#   * ``assigned_employee_id`` is a HOME-DESK hint ONLY -- it is NEVER an
#     identity authority and never relabels an occupant.
#   * A strongly-confirmed EMP002 sitting in EMP001's chair stays EMP002
#     (only a neutral CHAIR_ASSIGNMENT_MISMATCH context observation is
#     produced).
#   * An Unknown face in a chair stays Unknown; an empty chair stays VACANT
#     (a chair never invents a fake "Away").
#   * Chairs are DISABLED by default (``CCTV_CHAIRS_ENABLED`` unset -> 0)
#     and seed only from an OPTIONAL ``CHAIRS_FILE`` when present.
# ----------------------------------------------------------------------
CHAIR_TRACKING_ENABLED = os.getenv("CCTV_CHAIRS_ENABLED", "0") == "1"
CHAIRS_FILE            = os.getenv("CCTV_CHAIRS_FILE",
                                   str(PROJECT_ROOT / "data" / "chairs.json"))
CHAIR_CONFIRM_FRAMES   = int(os.getenv("CCTV_CHAIR_CONFIRM_FRAMES", "2"))
CHAIR_CLEAR_FRAMES     = int(os.getenv("CCTV_CHAIR_CLEAR_FRAMES", "3"))
CHAIR_EVENT_COOLDOWN_SEC = float(os.getenv("CCTV_CHAIR_EVENT_COOLDOWN_SEC", "30"))
