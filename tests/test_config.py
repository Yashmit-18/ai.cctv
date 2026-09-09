"""Tests for configuration loading + ``validate_config``."""

import importlib

import pytest

import config as _cfg


@pytest.fixture
def reloaded(monkeypatch):
    """Reload the module after env tweaks; auto-clean afterwards."""
    mods = {}

    def _reload(changes: dict):
        for k, v in changes.items():
            if v is None:
                monkeypatch.delenv(k, raising=False)
            else:
                monkeypatch.setenv(k, v)
        return importlib.reload(_cfg)

    return _reload


def test_validate_config_clean_with_friendly_values(reloaded):
    cfg = reloaded({
        "CCTV_WORK_START": "09:00",
        "CCTV_WORK_END": "18:00",
        "CCTV_LUNCH": "12:00-13:00",
        "CCTV_DASH_AUTH": "0",
    })
    assert cfg.validate_config() == []


def test_validate_config_flags_bad_hours(reloaded):
    cfg = reloaded({"CCTV_WORK_START": "99:99", "CCTV_WORK_END": "25:00"})
    assert cfg.validate_config(), "invalid working hours must be reported"


def test_validate_config_flags_work_start_after_end(reloaded):
    cfg = reloaded({"CCTV_WORK_START": "18:00", "CCTV_WORK_END": "09:00"})
    assert any("start" in p and "end" in p for p in cfg.validate_config())


def test_validate_config_flags_reversed_lunch(reloaded):
    cfg = reloaded({"CCTV_LUNCH": "13:00-12:00"})
    assert cfg.validate_config(), "reversed lunch window must be flagged"


def test_work_schedule_parsed_from_env(reloaded):
    cfg = reloaded({"CCTV_WORK_START": "10:00", "CCTV_WORK_END": "17:00"})
    assert cfg.WORK_SCHEDULE["start"] == "10:00"
    assert cfg.WORK_SCHEDULE["end"] == "17:00"


def test_email_retry_env_defaults(reloaded):
    cfg = reloaded({})
    assert cfg.EMAIL_MAX_RETRIES >= 1
    assert cfg.EMAIL_RETRY_DELAY_SEC >= 0


def test_schedule_values_survive_reload(reloaded):
    cfg = reloaded({"CCTV_WORK_START": "10:00", "CCTV_WORK_END": "17:00"})
    assert cfg.WORK_SCHEDULE["start"] == "10:00"
    # reloading again with defaults must not corrupt the schedule
    cfg2 = reloaded({"CCTV_WORK_START": "09:00"})
    assert cfg2.WORK_SCHEDULE["start"] == "09:00"


def test_eod_hour_env_default_19(reloaded):
    cfg = reloaded({})
    assert cfg.EOD_REPORT_HOUR == 19


def test_eod_hour_env_disabled_by_none(reloaded):
    cfg = reloaded({"CCTV_EOD_HOUR": "none"})
    assert cfg.EOD_REPORT_HOUR is None


def test_eod_hour_env_disabled_by_empty(reloaded):
    cfg = reloaded({"CCTV_EOD_HOUR": ""})
    assert cfg.EOD_REPORT_HOUR is None


def test_eod_hour_env_zero_is_valid(reloaded):
    cfg = reloaded({"CCTV_EOD_HOUR": "0"})
    assert cfg.EOD_REPORT_HOUR == 0
    assert cfg.validate_config() == []


def test_eod_hour_env_ignores_junk(reloaded):
    cfg = reloaded({"CCTV_EOD_HOUR": "banana"})
    assert cfg.EOD_REPORT_HOUR is None
    assert cfg.validate_config() == []