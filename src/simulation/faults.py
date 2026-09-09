"""Deterministic fault injection for simulation/stress tests (Phase 36).

Faults are explicit, isolated, opt-in, and impossible to trigger from normal
production runtime.  The production daemon and dashboard never import this
module; only simulation/stress tests and the ``SimulationRunner`` do.

Supported fault injection:

* camera faults -- delegated to ``VirtualCamera`` (dark/bright/noisy/frozen/
  disconnect) via the scenario plan or explicit schedule.
* detector exception / timeout / model-load failure -- a wrapper that raises
  (or emits a sentinel error) so the *real* Pipeline callers' fail-safe
  (the daemon's try/except around ``detect_batch`` / ``security.tick``) can be
  exercised.  The production code path is exercised exactly as it is at run
  time; only the detector's output is replaced.
* alert delivery failure -- a notifier stub that fails on demand for
  ``AlertEngine`` retry/escalation testing.

Use ``FaultInjector`` as the single opt-in gateway.  Nothing here writes to a
production database.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class FaultAlreadyComplete(RuntimeError):
    """Raised when a fault is requested that has already fully fired."""


@dataclass
class FaultInjector:
    """Deterministic, explicit fault injection (simulation/tests only)."""

    # (camera, start_step, end_step, kind) -> hand off to VirtualCamera
    camera_faults: list[tuple[str, int, int, str]] = field(default_factory=list)
    # detector raises on these steps
    detector_fail_steps: set[int] = field(default_factory=set)
    # model-load failure: detector reports unavailable until given step
    model_load_fail_until: int | None = None
    # alerts delivery fails while True
    alerts_fail: bool = False
    _fired: set[int] = field(default_factory=set)

    def register_camera_fault(self, cam: str, start: int, end: int,
                              kind: str) -> None:
        self.camera_faults.append((cam, int(start), int(end), kind))

    def mark_step_faults_delivered(self, step: int) -> None:
        for s in [x for x in self.detector_fail_steps if x <= step]:
            self._fired.add(s)

    @property
    def fired_sentinels(self) -> list[str]:
        return [f"detector_fail_{s}" for s in sorted(self._fired)]


def failing_detector(step: int):
    """Return a detector-isolation sentinel for a step (never a real result).

    Raises ``DetectorFaultSentinel`` so tests can assert the *caller's*
    fail-safe ran, without fabricating any AI detection.
    """

    class DetectorFaultSentinel(RuntimeError):
        pass

    return DetectorFaultSentinel(f"injected detector failure at step {step}")


class AlertDelivererStub:
    """Deterministic alert-delivery notifier for retry/escalation tests.

    ``should_fail`` is the single opt-in switch; when True deliveries raise
    so ``AlertEngine``'s retry path is exercised.  When False they succeed.
    """

    def __init__(self, *, should_fail: bool = False, fail_times: int = 0):
        self.should_fail = bool(should_fail)
        self.fail_times = int(fail_times)
        self.succeeded = 0
        self.failed = 0

    def deliver(self, alert: dict) -> bool:
        if self.should_fail or self.fail_times > 0:
            self.fail_times -= 1
            self.failed += 1
            raise RuntimeError("injected SMTP/notifier delivery failure")
        self.succeeded += 1
        return True