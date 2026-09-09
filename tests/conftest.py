import sqlite3

import pytest

from config import DB_PATH
from src.database import get_connection, init_db
from src.domain import WorkSchedule

TEST_SCHEDULE = WorkSchedule(
    start_min=9 * 60,
    end_min=18 * 60,
    grace_period_min=15,
    lunch_min=(12 * 60, 13 * 60),
    break_minutes=30,
)


@pytest.fixture
def tmp_db(tmp_path):
    """A fresh, migrated database inside a temp dir."""
    path = tmp_path / "test.db"
    conn = get_connection(str(path))
    init_db(conn)
    yield conn
    try:
        conn.close()
    except sqlite3.Error:
        pass


@pytest.fixture
def schedule() -> WorkSchedule:
    return TEST_SCHEDULE