"""Employee management model (Phase 4/5).

Wraps the ``employees`` table (see :mod:`src.database`) and the domain
:class:`~src.domain.Employee` with an optional per-employee work schedule.
Used by the daemon, reporter, dashboard and tests as the single source of
employee metadata truth.
"""

from __future__ import annotations

import logging

from src import database as db
from src.domain import Employee, WorkSchedule

logger = logging.getLogger("cctv.employees")


class EmployeeStore:
    """Thin, dependency-light employee CRUD over a SQLite connection.

    The caller owns the connection (daemon thread or read-only dashboard
    connection).  Writes are guarded so harmless metadata operations do not
    fail when the DB is briefly locked.
    """

    def __init__(self, conn, schedule_defaults: dict | None = None):
        self._conn = conn
        self._default_schedule = WorkSchedule.from_config(
            schedule_defaults if schedule_defaults is not None else {}
        ) if schedule_defaults is not None else None

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def get(self, employee_id: str) -> Employee | None:
        row = db.get_employee(self._conn, employee_id)
        if not row:
            return None
        return Employee(
            employee_id=row["employee_id"],
            name=row.get("name", ""),
            department=row.get("department", ""),
            designation=row.get("designation", ""),
            active=bool(row.get("active", 1)),
            enrolled=bool(row.get("enrolled", 0)),
            schedule=self._default_schedule,
        )

    def list(self, include_inactive: bool = True) -> list[Employee]:
        return [
            Employee(
                employee_id=r["employee_id"],
                name=r.get("name", ""),
                department=r.get("department", ""),
                designation=r.get("designation", ""),
                active=bool(r.get("active", 1)),
                enrolled=bool(r.get("enrolled", 0)),
                schedule=self._default_schedule,
            )
            for r in db.list_employees(self._conn, include_inactive=include_inactive)
        ]

    def known_ids(self) -> set[str]:
        return {e.employee_id for e in self.list()}

    def schedule_for(self, employee_id: str) -> WorkSchedule:
        emp = self.get(employee_id)
        if emp and emp.schedule:
            return emp.schedule
        if self._default_schedule:
            return self._default_schedule
        return WorkSchedule()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def upsert(self, employee_id: str, *, name: str | None = None,
               department: str | None = None, designation: str | None = None,
               active: bool = True) -> bool:
        return db.upsert_employee(
            self._conn, employee_id, name=name, department=department,
            designation=designation, active=active,
        )

    def set_active(self, employee_id: str, active: bool):
        db.set_employee_active(self._conn, employee_id, active)

    def set_enrolled(self, employee_id: str, enrolled: bool):
        db.set_employee_enrolled(self._conn, employee_id, enrolled)

    def remove(self, employee_id: str):
        db.delete_employee(self._conn, employee_id)

    # ------------------------------------------------------------------
    # Reconciliation helpers
    # ------------------------------------------------------------------
    def ensure_known(self, employee_ids: list[str], enrolled_ids: list[str]):
        """Create placeholder rows for any discovered employees not yet in DB.

        Prevents the dashboard from showing EMP/Unknown rows that have no
        metadata record.  ``Unknown`` ids are never persisted as employees.
        """
        known = self.known_ids()
        for eid in employee_ids:
            if not eid or eid in ("Unknown", "UNKNOWN"):
                continue
            if eid not in known:
                self.upsert(eid)
                logger.info("Auto-registered employee row: %s", eid)
        for eid in enrolled_ids:
            if not eid or eid in ("Unknown", "UNKNOWN"):
                continue
            if eid not in self.known_ids():
                self.upsert(eid)
                logger.info("Auto-registered employee row: %s", eid)
            self.set_enrolled(eid, True)

    def sync_enrolled(self, enrolled_ids: list[str]):
        """Reconcile DB ``enrolled`` flags with the face registry.

        Employees in the registry are marked enrolled; registered-but-not-
        enrolled rows (e.g. after an image is removed) are un-marked.  Only
        rows that already exist are touched; users may always re-enroll by
        adding a face photo and restarting.
        """
        known = self.known_ids()
        wanted = {eid for eid in enrolled_ids
                  if eid and eid not in ("Unknown", "UNKNOWN")}
        for eid in sorted(wanted):
            if eid not in known:
                self.upsert(eid)
                logger.info("Auto-registered employee row: %s", eid)
            self.set_enrolled(eid, True)
        for eid in known:
            if eid not in wanted:
                self.set_enrolled(eid, False)
