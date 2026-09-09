"""Camera topology / location graph (Phase 33, A3).

Administratively-configured adjacency between cameras (``from_camera ->
to_camera``).  It is **admin-configured, never auto-inferred from footage** --
inferring physical adjacency from video could fabricate cross-camera claims.
The topology is consumed by cross-camera correlation and coverage analysis as
an *optional hint*; if it is not configured, correlation simply reports
``CROSS_CAMERA_CORRELATION_UNAVAILABLE``.

This module is read-mostly: it exposes config mutation (guarded elsewhere by
RBAC) and honest graph queries.
"""

from __future__ import annotations

import logging

from src import database as db

logger = logging.getLogger("cctv.topology")

RELATIONS = ("NEXT", "SAME_ZONE", "OVERLOOKS", "ALIAS")


class CameraTopology:
    """Admin-configured camera adjacency graph."""

    def __init__(self, conn):
        self._conn = conn

    # -- mutation (RBAC-guarded by caller) ------------------------------
    def add_edge(self, from_camera: str, to_camera: str, *,
                 relation: str = "NEXT", note: str = "") -> None:
        rel = relation if relation in RELATIONS else "NEXT"
        db.upsert_camera_topology(self._conn, from_camera, to_camera, rel, note)
        logger.info("[TOPOLOGY] edge %s -> %s (%s)", from_camera, to_camera, rel)

    def set_edge_enabled(self, edge_id: int, enabled: bool) -> None:
        db.set_camera_topology_enabled(self._conn, edge_id, enabled)

    def remove_edge(self, edge_id: int) -> None:
        db.delete_camera_topology(self._conn, edge_id)

    # -- reads ----------------------------------------------------------
    def edges(self, *, enabled_only: bool = True) -> list[dict]:
        return db.list_camera_topology(self._conn, enabled_only=enabled_only)

    def neighbors(self, camera_id: str) -> list[str]:
        """Direct neighbours of a camera (both directions)."""
        return db.incident_adjacent_cameras(self._conn, camera_id)

    def is_adjacent(self, camera_a: str, camera_b: str) -> bool:
        return camera_b in self.neighbors(camera_a)

    def connected_cameras(self) -> set[str]:
        """All cameras that appear in any topology edge."""
        out: set[str] = set()
        for e in self.edges(enabled_only=True):
            out.add(e.get("from_camera") or "")
            out.add(e.get("to_camera") or "")
        return {c for c in out if c}

    def summary(self) -> dict:
        edges = self.edges(enabled_only=True)
        return {
            "edge_count": len(edges),
            "connected_cameras": sorted(self.connected_cameras()),
            "relations": sorted({e.get("relation") or "NEXT" for e in edges}),
        }
