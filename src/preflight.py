"""Client environment pre-flight validation (Phase 37).

Validates that the deployment target has all required components before
starting the CCTV platform.  Intended to be run before the first pilot
session.

Usage
-----
    python -m src.preflight              # full check
    python -m src.preflight --verbose    # detailed output
    python -m src.preflight --json       # machine-readable JSON
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import platform
import shutil
import sqlite3
import sys
from pathlib import Path

logger = logging.getLogger("cctv.preflight")


class CheckResult:
    """Single pre-flight check result."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"

    def __init__(self, category: str, name: str, status: str,
                 detail: str = "", remedy: str = ""):
        self.category = category
        self.name = name
        self.status = status
        self.detail = detail
        self.remedy = remedy

    def to_dict(self) -> dict:
        d = {"category": self.category, "name": self.name, "status": self.status}
        if self.detail:
            d["detail"] = self.detail
        if self.remedy:
            d["remedy"] = self.remedy
        return d

    def __repr__(self):
        icon = {"PASS": "[OK]", "WARN": "[!!]", "FAIL": "[XX]", "SKIP": "[--]"}.get(self.status, "[??]")
        return f"{icon} {self.category}/{self.name}: {self.detail}" if self.detail else f"{icon} {self.category}/{self.name}"


def check_system() -> list[CheckResult]:
    """OS, Python, CPU, RAM, disk."""
    results = []

    results.append(CheckResult("SYSTEM", "os", CheckResult.PASS,
                               detail=f"{platform.system()} {platform.release()}"))
    results.append(CheckResult("SYSTEM", "python", CheckResult.PASS,
                               detail=f"{sys.version.split()[0]}"))

    try:
        import os as _os
        cpu_count = _os.cpu_count() or 0
        if cpu_count >= 4:
            results.append(CheckResult("SYSTEM", "cpu_cores", CheckResult.PASS,
                                       detail=str(cpu_count)))
        elif cpu_count >= 2:
            results.append(CheckResult("SYSTEM", "cpu_cores", CheckResult.WARN,
                                       detail=str(cpu_count),
                                       remedy="4+ cores recommended for multi-camera"))
        else:
            results.append(CheckResult("SYSTEM", "cpu_cores", CheckResult.FAIL,
                                       detail=str(cpu_count),
                                       remedy="At least 2 CPU cores required"))
    except Exception:
        results.append(CheckResult("SYSTEM", "cpu_cores", CheckResult.SKIP,
                                   detail="could not detect"))

    try:
        if platform.system() == "Windows":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulonglong = ctypes.c_ulonglong
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", c_ulonglong),
                            ("ullAvailPhys", c_ulonglong),
                            ("ullTotalPageFile", c_ulonglong),
                            ("ullAvailPageFile", c_ulonglong),
                            ("ullTotalVirtual", c_ulonglong),
                            ("ullAvailVirtual", c_ulonglong),
                            ("ullAvailExtendedVirtual", c_ulonglong)]
            mem = MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(mem)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
            total_gb = mem.ullTotalPhys / (1024**3)
        else:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_kb = int(line.split()[1])
                        total_gb = total_kb / (1024**2)
                        break
                else:
                    total_gb = 0

        if total_gb >= 8:
            results.append(CheckResult("SYSTEM", "ram", CheckResult.PASS,
                                       detail=f"{total_gb:.1f} GB"))
        elif total_gb >= 4:
            results.append(CheckResult("SYSTEM", "ram", CheckResult.WARN,
                                       detail=f"{total_gb:.1f} GB",
                                       remedy="8+ GB recommended for multi-camera + GPU"))
        else:
            results.append(CheckResult("SYSTEM", "ram", CheckResult.FAIL,
                                       detail=f"{total_gb:.1f} GB",
                                       remedy="Minimum 4 GB RAM required"))
    except Exception:
        results.append(CheckResult("SYSTEM", "ram", CheckResult.SKIP,
                                   detail="could not detect"))

    try:
        disk_free = shutil.disk_usage(".").free / (1024**3)
        if disk_free >= 10:
            results.append(CheckResult("SYSTEM", "disk_free", CheckResult.PASS,
                                       detail=f"{disk_free:.1f} GB"))
        elif disk_free >= 2:
            results.append(CheckResult("SYSTEM", "disk_free", CheckResult.WARN,
                                       detail=f"{disk_free:.1f} GB",
                                       remedy="10+ GB recommended for evidence + backups"))
        else:
            results.append(CheckResult("SYSTEM", "disk_free", CheckResult.FAIL,
                                       detail=f"{disk_free:.1f} GB",
                                       remedy="Insufficient disk space"))
    except Exception:
        results.append(CheckResult("SYSTEM", "disk_free", CheckResult.SKIP,
                                   detail="could not detect"))

    return results


def check_gpu() -> list[CheckResult]:
    """CUDA availability, GPU model, memory."""
    results = []

    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            driver = torch.version.cuda or "unknown"
            results.append(CheckResult("GPU", "cuda", CheckResult.PASS,
                                       detail=f"{gpu_name} ({gpu_mem:.1f} GB, CUDA {driver})"))
        else:
            results.append(CheckResult("GPU", "cuda", CheckResult.WARN,
                                       detail="CUDA not available; using CPU fallback",
                                       remedy="Install CUDA toolkit + nvidia-container-toolkit for GPU inference"))
    except ImportError:
        results.append(CheckResult("GPU", "cuda", CheckResult.SKIP,
                                   detail="torch not installed"))
    except Exception as exc:
        results.append(CheckResult("GPU", "cuda", CheckResult.WARN,
                                   detail=str(exc)))

    return results


def check_camera_config() -> list[CheckResult]:
    """RTSP capability and camera configuration."""
    results = []

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import config
        cameras = config.CAMERAS
        if cameras:
            cam_list = ", ".join(cameras.keys())
            results.append(CheckResult("CAMERA", "configured", CheckResult.PASS,
                                       detail=f"{len(cameras)} cameras: {cam_list}"))
            for cam_id, url in cameras.items():
                if "admin:password@" in str(url) or "CHANGE_ME" in str(url).upper():
                    results.append(CheckResult("CAMERA", cam_id, CheckResult.WARN,
                                               detail="placeholder credentials",
                                               remedy="Set real RTSP credentials in .env"))
                elif not url or not str(url).strip():
                    results.append(CheckResult("CAMERA", cam_id, CheckResult.FAIL,
                                               detail="empty URL"))
                else:
                    from src.domain import redact_url
                    results.append(CheckResult("CAMERA", cam_id, CheckResult.PASS,
                                               detail=redact_url(url)))
        else:
            results.append(CheckResult("CAMERA", "configured", CheckResult.WARN,
                                       detail="No cameras configured",
                                       remedy="Set CCTV_CAM_<NN>_URL in .env"))
    except Exception as exc:
        results.append(CheckResult("CAMERA", "configured", CheckResult.SKIP,
                                   detail=str(exc)))

    return results


def check_models() -> list[CheckResult]:
    """YOLO and InsightFace model availability."""
    results = []

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import config
        model_path = Path(config.MODEL_PATH)
        if model_path.is_file():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            results.append(CheckResult("MODELS", "yolo", CheckResult.PASS,
                                       detail=f"{model_path.name} ({size_mb:.1f} MB)"))
        else:
            results.append(CheckResult("MODELS", "yolo", CheckResult.FAIL,
                                       detail=f"Not found: {model_path}",
                                       remedy="Run: python -c 'from ultralytics import YOLO; YOLO(\"yolo11n.pt\")'"))
    except Exception as exc:
        results.append(CheckResult("MODELS", "yolo", CheckResult.SKIP,
                                   detail=str(exc)))

    # Phase 46: ONNX Runtime provider detection (ReID / InsightFace).
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        results.append(CheckResult("MODELS", "onnx_providers", CheckResult.PASS,
                                   detail=", ".join(providers)))
    except ImportError:
        results.append(CheckResult("MODELS", "onnx_providers", CheckResult.WARN,
                                   detail="onnxruntime not installed",
                                   remedy="pip install onnxruntime"))
    except Exception as exc:
        results.append(CheckResult("MODELS", "onnx_providers", CheckResult.SKIP,
                                   detail=str(exc)))

    try:
        import insightface
        results.append(CheckResult("MODELS", "insightface", CheckResult.PASS,
                                   detail=f"insightface {insightface.__version__}"))
    except ImportError:
        results.append(CheckResult("MODELS", "insightface", CheckResult.FAIL,
                                   detail="insightface not installed",
                                   remedy="pip install insightface onnxruntime"))
    except Exception as exc:
        results.append(CheckResult("MODELS", "insightface", CheckResult.SKIP,
                                   detail=str(exc)))

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import config
        faces_dir = Path(config.FACES_DIR)
        embeddings_file = Path(config.EMBEDDINGS_FILE)
        if embeddings_file.is_file():
            results.append(CheckResult("MODELS", "embeddings", CheckResult.PASS,
                                       detail=f"{embeddings_file.name} exists"))
        else:
            results.append(CheckResult("MODELS", "embeddings", CheckResult.WARN,
                                       detail="No embeddings file; enroll employees before pilot",
                                       remedy="Use dashboard enrollment or dummy_face_enroll.py"))
    except Exception as exc:
        results.append(CheckResult("MODELS", "embeddings", CheckResult.SKIP,
                                   detail=str(exc)))

    return results


def check_database() -> list[CheckResult]:
    """Database writability and integrity."""
    results = []

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import config
        db_path = Path(config.DB_PATH)
        db_dir = db_path.parent
        db_dir.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        result = conn.execute("PRAGMA integrity_check").fetchone()[0]
        conn.close()

        if result == "ok":
            results.append(CheckResult("DATABASE", "integrity", CheckResult.PASS,
                                       detail=str(db_path)))
        else:
            results.append(CheckResult("DATABASE", "integrity", CheckResult.FAIL,
                                       detail=result,
                                       remedy="Restore from backup or recreate database"))
    except Exception as exc:
        results.append(CheckResult("DATABASE", "integrity", CheckResult.FAIL,
                                   detail=str(exc)))

    return results


def check_evidence() -> list[CheckResult]:
    """Evidence directory writability."""
    results = []

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import config
        evidence_dir = Path(config.EVIDENCE_DIR)
        evidence_dir.mkdir(parents=True, exist_ok=True)

        test_file = evidence_dir / ".preflight_test"
        test_file.write_text("preflight", encoding="utf-8")
        test_file.unlink()

        results.append(CheckResult("EVIDENCE", "writable", CheckResult.PASS,
                                   detail=str(evidence_dir)))
    except Exception as exc:
        results.append(CheckResult("EVIDENCE", "writable", CheckResult.FAIL,
                                   detail=str(exc),
                                   remedy="Check file permissions on data/evidence/"))

    return results


def check_smtp() -> list[CheckResult]:
    """SMTP configuration (connectivity not tested without explicit flag)."""
    results = []

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import config

        sender = config.SENDER_EMAIL
        server = config.SMTP_SERVER
        recipients = config.parse_recipients()

        if sender and server and recipients:
            results.append(CheckResult("SMTP", "configured", CheckResult.PASS,
                                       detail=f"server={server}, sender={sender}, recipients={len(recipients)}"))
        elif not sender:
            results.append(CheckResult("SMTP", "configured", CheckResult.SKIP,
                                       detail="SENDER_EMAIL not set; email disabled",
                                       remedy="Set SENDER_EMAIL in .env to enable email alerts"))
        else:
            results.append(CheckResult("SMTP", "configured", CheckResult.WARN,
                                       detail="Incomplete SMTP config",
                                       remedy="Set SMTP_SERVER, SENDER_EMAIL, RECIPIENT_EMAILS"))
    except Exception as exc:
        results.append(CheckResult("SMTP", "configured", CheckResult.SKIP,
                                   detail=str(exc)))

    return results


def check_docker() -> list[CheckResult]:
    """Docker availability."""
    results = []

    docker_path = shutil.which("docker")
    if docker_path:
        results.append(CheckResult("DOCKER", "available", CheckResult.PASS,
                                   detail=docker_path))
    else:
        results.append(CheckResult("DOCKER", "available", CheckResult.SKIP,
                                   detail="docker not found on PATH",
                                   remedy="Install Docker + NVIDIA Container Toolkit"))

    compose_path = shutil.which("docker-compose") or shutil.which("docker")
    if compose_path:
        results.append(CheckResult("DOCKER", "compose", CheckResult.PASS,
                                   detail=compose_path))
    else:
        results.append(CheckResult("DOCKER", "compose", CheckResult.SKIP,
                                   detail="docker compose not available"))

    return results


def check_security() -> list[CheckResult]:
    """Security configuration readiness."""
    results = []

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import config

        if config.SECURITY_ENABLED:
            results.append(CheckResult("SECURITY", "engine", CheckResult.PASS,
                                       detail="enabled"))
        else:
            results.append(CheckResult("SECURITY", "engine", CheckResult.WARN,
                                       detail="disabled",
                                       remedy="Set CCTV_SECURITY_ENABLED=1 for pilot"))

        if config.EVIDENCE_MODE != "OFF":
            results.append(CheckResult("SECURITY", "evidence", CheckResult.PASS,
                                       detail=f"mode={config.EVIDENCE_MODE}"))
        else:
            results.append(CheckResult("SECURITY", "evidence", CheckResult.WARN,
                                       detail="evidence capture OFF",
                                       remedy="Set CCTV_EVIDENCE_MODE=EVENT_ONLY for pilot"))

        dash_user = config.DASH_USERNAME
        dash_pass = config.DASH_ADMIN_PASS
        if dash_pass:
            results.append(CheckResult("SECURITY", "dashboard_auth", CheckResult.PASS,
                                       detail=f"user={dash_user}"))
        elif config.DASH_FAIL_CLOSED and config.DASH_AUTH_ENABLED:
            results.append(CheckResult("SECURITY", "dashboard_auth", CheckResult.FAIL,
                                       detail="fail-closed: no password set and "
                                              "CCTV_DASH_FAIL_CLOSED=1; dashboard blocked",
                                       remedy="Set CCTV_DASH_PASS in .env"))
        else:
            results.append(CheckResult("SECURITY", "dashboard_auth", CheckResult.WARN,
                                       detail="no password set; dashboard is open",
                                       remedy="Set CCTV_DASH_PASS in .env"))

        if config.AUTO_BACKUP_DAILY:
            results.append(CheckResult("SECURITY", "auto_backup", CheckResult.PASS,
                                       detail="enabled"))
        else:
            results.append(CheckResult("SECURITY", "auto_backup", CheckResult.WARN,
                                       detail="disabled",
                                       remedy="Set CCTV_AUTO_BACKUP_DAILY=1 for pilot"))

        # B11 -- biometric data at rest. Face images + embeddings are stored
        # unencrypted; production must protect them via filesystem ACLs and
        # keep them out of images/git (see .dockerignore) and behind the
        # authenticated dashboard / reverse proxy / VPN.
        results.append(CheckResult(
            "SECURITY", "biometric_at_rest", CheckResult.WARN,
            detail="face embeddings/images stored unencrypted at rest",
            remedy="Keep data/ off images+git (see .dockerignore); restrict FS "
                   "and reverse-proxy/VPN the dashboard"))
    except Exception as exc:
        results.append(CheckResult("SECURITY", "config", CheckResult.SKIP,
                                   detail=str(exc)))

    return results


def check_config_validation() -> list[CheckResult]:
    """Run the existing config validation."""
    results = []

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import config
        problems = config.validate_config(quiet=True)
        if not problems:
            results.append(CheckResult("CONFIG", "validation", CheckResult.PASS,
                                       detail="no problems"))
        else:
            for p in problems:
                results.append(CheckResult("CONFIG", "validation", CheckResult.FAIL,
                                           detail=p))
    except Exception as exc:
        results.append(CheckResult("CONFIG", "validation", CheckResult.SKIP,
                                   detail=str(exc)))

    return results


def check_pilot_mode() -> list[CheckResult]:
    """Verify pilot mode configuration."""
    results = []

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import config
        pilot = getattr(config, "PILOT_MODE", False)
        if pilot:
            results.append(CheckResult("PILOT", "mode", CheckResult.PASS,
                                       detail="enabled"))
        else:
            results.append(CheckResult("PILOT", "mode", CheckResult.WARN,
                                       detail="disabled",
                                       remedy="Set CCTV_PILOT_MODE=1 for pilot verbose metrics"))
    except Exception as exc:
        results.append(CheckResult("PILOT", "mode", CheckResult.SKIP,
                                   detail=str(exc)))

    return results


def run_all_checks() -> list[CheckResult]:
    """Run all pre-flight checks and return results."""
    all_results = []
    all_results.extend(check_system())
    all_results.extend(check_gpu())
    all_results.extend(check_camera_config())
    all_results.extend(check_models())
    all_results.extend(check_database())
    all_results.extend(check_evidence())
    all_results.extend(check_smtp())
    all_results.extend(check_docker())
    all_results.extend(check_security())
    all_results.extend(check_config_validation())
    all_results.extend(check_pilot_mode())
    return all_results


def print_results(results: list[CheckResult], verbose: bool = False):
    """Print results in human-readable format."""
    categories: dict[str, list[CheckResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    print("=" * 70)
    print("  CCTV Platform -- Client Environment Pre-Flight Check")
    print("=" * 70)
    print()

    totals = {"PASS": 0, "WARN": 0, "FAIL": 0, "SKIP": 0}
    for cat, checks in categories.items():
        print(f"[{cat}]")
        for c in checks:
            totals[c.status] = totals.get(c.status, 0) + 1
            print(f"  {c}")
            if verbose and c.remedy:
                print(f"         remedy: {c.remedy}")
        print()

    print("-" * 70)
    print(f"  PASS: {totals['PASS']}  WARN: {totals['WARN']}  "
          f"FAIL: {totals['FAIL']}  SKIP: {totals['SKIP']}")
    print()

    if totals["FAIL"] > 0:
        print("  VERDICT: BLOCKED -- resolve FAIL items before pilot")
    elif totals["WARN"] > 0:
        print("  VERDICT: READY WITH WARNINGS -- review WARN items")
    else:
        print("  VERDICT: READY -- all checks passed")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="CCTV Platform Pre-Flight Check")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show remedies for warnings/failures")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    args = parser.parse_args()

    results = run_all_checks()

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        print_results(results, verbose=args.verbose)

    has_fail = any(r.status == CheckResult.FAIL for r in results)
    sys.exit(1 if has_fail else 0)


if __name__ == "__main__":
    main()
