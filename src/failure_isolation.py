"""Failure isolation & resilience (Phase 32).

Keeps one failing subsystem from taking down the whole security engine:

* ``CircuitBreaker`` -- opens after N consecutive failures, then stays open
  (fast-fail) for a cooldown before a half-open probe.
* ``safe_call`` -- runs a callable, catching/logging exceptions so callers can
  continue.  Optional timeout via a worker thread guard.
* ``BoundedProcessor``/``chunked`` -- apply work in bounded batches so a burst
  of events cannot exhaust memory or stall the tick loop.

These helpers contain the blast radius: evidence capture, a notifier, a
detector, or analytics can fail independently without aborting processing.
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger("cctv.resilience")


class CircuitBreaker:
    """Trip-and-recover state machine around a fallible operation."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

    def __init__(self, *, failure_threshold: int = 5,
                 cooldown_sec: float = 30.0):
        self.failure_threshold = max(1, failure_threshold)
        self.cooldown_sec = max(0.0, cooldown_sec)
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._state = self.CLOSED

    @property
    def state(self) -> str:
        if self._state == self.OPEN and self._opened_at is not None:
            if (time.time() - self._opened_at) >= self.cooldown_sec:
                self._state = self.HALF_OPEN
        return self._state

    def allow(self) -> bool:
        """True if a call may proceed (not fast-fail)."""
        st = self.state
        if st == self.OPEN:
            return False
        if st == self.HALF_OPEN:
            return True  # single probe
        return True

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._opened_at = None
        self._state = self.CLOSED

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._state = self.OPEN
            self._opened_at = time.time()
            logger.warning("[breaker] %s opened after %d failures",
                           self, self._consecutive_failures)

    def __call__(self, fn, *args, **kwargs):
        """Guard a callable; returns (ok, value/exception)."""
        if not self.allow():
            return False, RuntimeError(
                f"circuit open; fast-fail for scale={self}")
        try:
            value = fn(*args, **kwargs)
            self.record_success()
            return True, value
        except Exception as exc:  # noqa: BLE001 - containment boundary
            self.record_failure()
            return False, exc

    def __repr__(self):
        return f"CircuitBreaker[{self.state}](failures={self._consecutive_failures})"


def safe_call(fn, *args, timeout_sec: float | None = None, **kwargs):
    """Run ``fn(*args, **kwargs)``; never raises.  Returns ``(ok, result)``.

    On exception (or timeout) returns ``(False, exc)`` after logging.  The
    optional timeout is enforced with a non-daemon worker so a hung subsystem
    cannot block the tick loop indefinitely.
    """
    if timeout_sec is None:
        try:
            return True, fn(*args, **kwargs)
        except Exception as exc:  # containment boundary
            logger.exception("[isolation] call failed")
            return False, exc

    result: dict = {"done": False, "value": None, "error": None}

    def _run():
        try:
            result["value"] = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            result["error"] = exc
        finally:
            result["done"] = True

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout_sec)
    if not result["done"]:
        logger.warning("[isolation] call timed out after %.2fs", timeout_sec)
        return False, TimeoutError(f"timed out after {timeout_sec}s")
    if result["error"] is not None:
        return False, result["error"]
    return True, result["value"]


def chunked(seq, size: int = 100):
    """Yield ``seq`` in fixed-size chunks (bounded batching)."""
    size = max(1, size)
    batch: list = []
    for item in seq:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
