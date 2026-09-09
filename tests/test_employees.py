"""Tests for EmployeeStore (metadata CRUD + reconciliation)."""

import pytest

from src.employees import EmployeeStore
from src.domain import UNKNOWN_ID, WorkSchedule


@pytest.fixture
def store(tmp_db):
    return EmployeeStore(tmp_db, {"start": "09:00", "end": "18:00"})


def test_upsert_then_get(store):
    store.upsert("EMP001", name="Alice", department="IT")
    emp = store.get("EMP001")
    assert emp.name == "Alice" and emp.department == "IT"
    assert emp.schedule is not None
    assert emp.schedule.start_min == 540


def test_list_and_remove(store):
    store.upsert("EMP001")
    store.upsert("EMP002")
    assert {e.employee_id for e in store.list()} == {"EMP001", "EMP002"}
    store.remove("EMP001")
    assert store.get("EMP001") is None


def test_schedule_for_defaults(store):
    s = store.schedule_for("EMP999")
    assert isinstance(s, WorkSchedule)
    assert s.end_min == 1080


def test_ensure_known_creates_placeholders(store):
    store.ensure_known(["EMP005", UNKNOWN_ID], enrolled_ids=[])
    assert store.get("EMP005") is not None
    assert store.get(UNKNOWN_ID) is None     # Unknown never persisted


def test_ensure_known_marks_enrolled(store):
    store.ensure_known([], enrolled_ids=["EMP006"])
    emp = store.get("EMP006")
    assert emp is not None and emp.enrolled is True


def test_known_ids_utility(store):
    store.upsert("EMP001")
    assert "EMP001" in store.known_ids()
    assert "EMP002" not in store.known_ids()