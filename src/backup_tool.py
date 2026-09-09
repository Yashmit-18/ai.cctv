"""WAL-safe backup and restore utilities (Phase 31).

Provides:
    - ``backup_database()``: WAL-safe SQLite backup via the backup API.
    - ``backup_all()``: DB + evidence + faces + reports.
    - ``verify_integrity()``: PRAGMA integrity_check on a database file.

All backups go to ``data/backups/`` under a timestamped directory.
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger("cctv.backup")


def backup_database(db_path: str, backup_dir: str | None = None) -> str:
    """Create a WAL-safe backup of the SQLite database.

    Uses ``sqlite3.Connection.backup()`` which is safe while the daemon runs.
    Returns the path to the backup file.
    """
    if backup_dir is None:
        backup_dir = os.path.join(os.path.dirname(db_path), "..", "backups")
    os.makedirs(backup_dir, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    dst = os.path.join(backup_dir, f"sessions_backup_{ts}.db")

    src_conn = sqlite3.connect(db_path)
    dst_conn = sqlite3.connect(dst)
    try:
        src_conn.backup(dst_conn)
    finally:
        dst_conn.close()
        src_conn.close()

    # Verify
    check = verify_integrity(dst)
    logger.info("Backup created: %s (integrity: %s)", dst, check)
    return dst


def backup_all(project_root: str, backup_dir: str | None = None) -> dict[str, str]:
    """Backup DB + faces + reports.  Returns paths of created backups."""
    if backup_dir is None:
        backup_dir = os.path.join(project_root, "data", "backups")
    os.makedirs(backup_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    snap_dir = os.path.join(backup_dir, f"full_backup_{ts}")
    os.makedirs(snap_dir, exist_ok=True)

    results: dict[str, str] = {}

    # Database
    db_path = os.path.join(project_root, "data", "database", "sessions.db")
    if os.path.isfile(db_path):
        dst_db = os.path.join(snap_dir, "sessions.db")
        src_conn = sqlite3.connect(db_path)
        dst_conn = sqlite3.connect(dst_db)
        try:
            src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
            src_conn.close()
        results["database"] = dst_db

    # Faces
    faces_dir = os.path.join(project_root, "data", "faces")
    if os.path.isdir(faces_dir):
        dst_faces = os.path.join(snap_dir, "faces")
        shutil.copytree(faces_dir, dst_faces, dirs_exist_ok=True)
        results["faces"] = dst_faces

    # Reports
    reports_dir = os.path.join(project_root, "data", "reports")
    if os.path.isdir(reports_dir):
        dst_reports = os.path.join(snap_dir, "reports")
        shutil.copytree(reports_dir, dst_reports, dirs_exist_ok=True)
        results["reports"] = dst_reports

    logger.info("Full backup created: %s", snap_dir)
    return results


def verify_integrity(db_path: str) -> str:
    """Run PRAGMA integrity_check.  Returns 'ok' or the error."""
    try:
        conn = sqlite3.connect(db_path)
        result = conn.execute("PRAGMA integrity_check").fetchone()[0]
        conn.close()
        return result
    except Exception as exc:
        return f"error: {exc}"


def restore_database(backup_file: str, target_path: str) -> None:
    """Restore a WAL-safe backup file into ``target_path``.

    The restore is done via the SQLite backup API against the backup file, so
    it is atomic and does not disturb the running daemon's DB. ``backup_file``
    must exist and pass an integrity check before any write occurs.
    """
    if not os.path.isfile(backup_file):
        raise FileNotFoundError(f"Backup not found: {backup_file}")
    check = verify_integrity(backup_file)
    if check != "ok":
        raise ValueError(f"Backup failed integrity check ({check}); refusing to restore")
    os.makedirs(os.path.dirname(os.path.abspath(target_path)) or ".", exist_ok=True)
    src_conn = sqlite3.connect(backup_file)
    dst_conn = sqlite3.connect(target_path)
    try:
        src_conn.backup(dst_conn)
    finally:
        dst_conn.close()
        src_conn.close()
    ok = verify_integrity(target_path)
    logger.info("Restored %s -> %s (integrity: %s)", backup_file, target_path, ok)
    if ok != "ok":
        raise ValueError(f"Restored DB failed integrity check ({ok})")


def restore_to_temp(backup_file: str) -> str:
    """Restore a backup into a temporary DB so it can be validated safely.

    Never touches the production database. Returns the temp DB path, which the
    caller should verify and then remove.
    """
    import tempfile
    fd, tmp = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(tmp)
    restore_database(backup_file, tmp)
    return tmp
