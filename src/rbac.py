"""Role-based access control enforcement (Phase 32 hardening).

Wraps the domain permission vocabulary into an ``AccessGuard`` bound to one
operator session.  Every guarded action checks the caller's role and, on
denial, writes an auditable ``access.denied`` event.  This is a defense-in-depth
UX/application guard -- real enforcement still belongs behind a reverse
proxy/VPN as noted in the domain constants.
"""

from __future__ import annotations

from src import database as db
from src.domain import (
    role_has_permission,
    PERM_VIEW,
    PERM_INVESTIGATE,
    PERM_ACKNOWLEDGE,
    PERM_RESOLVE,
    PERM_DISMISS,
    PERM_EXPORT,
    PERM_VIEW_EVIDENCE,
    PERM_DELETE_EVIDENCE,
    PERM_CONFIGURE_CAMERAS,
    PERM_CONFIGURE_ZONES,
    PERM_CONFIGURE_ALERTS,
    PERM_MANAGE_USERS,
    PERM_CONFIGURE_SETTINGS,
)
from src.incidents import IncidentEngine


class AccessDenied(Exception):
    """Raised when a caller lacks permission for a guarded action."""

    def __init__(self, permission: str, actor: str):
        super().__init__(f"permission denied for '{permission}' (actor={actor!r})")
        self.permission = permission
        self.actor = actor


class AccessGuard:
    """Enforces role permissions for a single operator session."""

    def __init__(self, conn, *, actor: str = "anon", role: str = "viewer"):
        self._conn = conn
        self.actor = actor
        self.role = role

    # -- core -----------------------------------------------------------
    def can(self, permission: str) -> bool:
        return role_has_permission(self.role, permission)

    def require(self, permission: str, *, resource: str | None = None) -> None:
        """Raise ``AccessDenied`` (and audit) if the caller lacks permission."""
        if not self.can(permission):
            db.audit(self._conn, "access.denied", actor=self.actor,
                     resource=resource or permission, detail=f"permission={permission}")
            raise AccessDenied(permission, self.actor)

    # -- helpers --------------------------------------------------------
    def _audit(self, action: str, resource: str | None, detail: str | None = None) -> None:
        db.audit(self._conn, action, actor=self.actor,
                 resource=resource, detail=detail)

    # -- incident lifecycle --------------------------------------------
    # These delegate the canonical status/timestamp mutation to IncidentEngine
    # (single source of truth), while AccessGuard owns authorization and the
    # operator-facing audit label.  Each helper emits exactly one audit entry.
    def acknowledge_incident(self, incident_id: str) -> bool:
        self.require(PERM_ACKNOWLEDGE, resource=incident_id)
        return IncidentEngine(self._conn).acknowledge(
            incident_id, actor=self.actor)

    def resolve_incident(self, incident_id: str) -> bool:
        self.require(PERM_RESOLVE, resource=incident_id)
        ok = IncidentEngine(self._conn).resolve(
            incident_id, actor=self.actor, _no_audit=True)
        if ok:
            self._audit("incident.resolve", incident_id)
        return ok

    def dismiss_incident(self, incident_id: str) -> bool:
        self.require(PERM_DISMISS, resource=incident_id)
        ok = IncidentEngine(self._conn).dismiss(
            incident_id, actor=self.actor, _no_audit=True)
        if ok:
            self._audit("incident.dismiss", incident_id)
        return ok

    def set_review_state(self, incident_id: str, state: str) -> None:
        self.require(PERM_DISMISS, resource=incident_id)
        db.set_incident_review_state(self._conn, incident_id, state, actor=self.actor)
        self._audit("incident.review_state", incident_id, detail=state)

    # -- alerts ---------------------------------------------------------
    def ack_alert(self, alert_id: int) -> None:
        self.require(PERM_ACKNOWLEDGE, resource=f"alert:{alert_id}")
        db.acknowledge_alert(self._conn, alert_id)
        self._audit("alert.acknowledge", f"alert:{alert_id}")

    def resolve_alert(self, alert_id: int) -> None:
        self.require(PERM_RESOLVE, resource=f"alert:{alert_id}")
        db.resolve_alert(self._conn, alert_id)
        self._audit("alert.resolve", f"alert:{alert_id}")

    # -- evidence / export ---------------------------------------------
    def require_evidence_view(self, resource: str | None = None) -> None:
        self.require(PERM_VIEW_EVIDENCE, resource=resource)
        self._audit("evidence.view", resource)

    def require_export(self, resource: str | None = None) -> None:
        self.require(PERM_EXPORT, resource=resource)
        self._audit("export", resource)

    def delete_evidence(self, evidence_id: int) -> None:
        self.require(PERM_DELETE_EVIDENCE, resource=f"evidence:{evidence_id}")
        db.delete_evidence_file_id(self._conn, evidence_id)
        self._audit("evidence.delete", f"evidence:{evidence_id}")

    # -- configuration --------------------------------------------------
    def require_configure_cameras(self) -> None:
        self.require(PERM_CONFIGURE_CAMERAS)

    def require_configure_zones(self) -> None:
        self.require(PERM_CONFIGURE_ZONES)

    def require_configure_alerts(self) -> None:
        self.require(PERM_CONFIGURE_ALERTS)

    def require_manage_users(self) -> None:
        self.require(PERM_MANAGE_USERS)

    def require_configure_settings(self) -> None:
        self.require(PERM_CONFIGURE_SETTINGS)

    # -- read gate ------------------------------------------------------
    def require_view(self) -> None:
        self.require(PERM_VIEW)

    def require_investigate(self) -> None:
        self.require(PERM_INVESTIGATE)
