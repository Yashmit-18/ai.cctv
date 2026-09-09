"""Tests for Phase 32 failure isolation / resilience."""

import time

import pytest

from src.failure_isolation import (
    CircuitBreaker, safe_call, chunked,
)


def test_breaker_closed_until_threshold():
    b = CircuitBreaker(failure_threshold=3)
    assert b.allow() is True
    b.record_failure(); b.record_failure()
    assert b.allow() is True          # 2 < 3
    b.record_failure()                # 3 >= 3 -> opens
    assert b.allow() is False
    assert b.state == "OPEN"


def test_breaker_recovers_after_cooldown():
    b = CircuitBreaker(failure_threshold=1, cooldown_sec=0.01)
    b.record_failure()
    assert b.allow() is False
    time.sleep(0.02)
    assert b.allow() is True           # half-open probe
    assert b.state == "HALF_OPEN"
    b.record_success()
    assert b.state == "CLOSED"


def test_breaker_wrap_ok():
    b = CircuitBreaker(failure_threshold=2)
    ok, value = b(lambda: 42)
    assert ok is True and value == 42
    assert b.state == "CLOSED"


def test_breaker_wrap_failure_fast_fail():
    b = CircuitBreaker(failure_threshold=1)
    def _boom():
        raise ValueError("boom")
    ok, exc = b(_boom)
    assert ok is False and isinstance(exc, ValueError)
    # circuit now open -> fast-fail without invoking
    ok2, exc2 = b(_boom)
    assert ok2 is False and isinstance(exc2, RuntimeError)


def test_safe_call_returns_exc_not_raise():
    def _boom():
        raise KeyError("nope")
    ok, exc = safe_call(_boom)
    assert ok is False and isinstance(exc, KeyError)


def test_safe_call_timeout():
    def _slow():
        time.sleep(2)
        return "late"
    ok, exc = safe_call(_slow, timeout_sec=0.05)
    assert ok is False and isinstance(exc, TimeoutError)


def test_safe_call_success():
    ok, value = safe_call(lambda: "hi")
    assert ok is True and value == "hi"


def test_chunked_batches():
    batches = list(chunked(range(10), size=3))
    assert [sum(1 for _ in b) for b in batches] == [3, 3, 3, 1]
