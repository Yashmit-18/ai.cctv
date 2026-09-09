"""Structured, machine-readable metrics and simulation report (Phase 36).

Provides honest percentile helpers (p50/p95/p99) and a ``SimulationReport``
dataclass that captures what a run actually produced (scenario, seed, cameras,
events/incidents/alerts/evidence, errors, measured latency + resource metrics,
final state).  Simulation reports are clearly distinguishable from real
operational data and are only produced when a simulation explicitly requests
them.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

EPS = 1e-9


def describe(values: list[float]) -> dict[str, float]:
    """Return min/max/avg/p50/p95/p99 for a sample (lenient on empty/None)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return {"min": 0.0, "max": 0.0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    s = sorted(vals)
    n = len(s)

    def pct(p: float) -> float:
        idx = min(n - 1, int((p / 100.0) * (n - 1) + EPS))
        return float(s[idx])

    avg = sum(s) / n
    return {
        "min": float(s[0]),
        "max": float(s[-1]),
        "avg": round(avg, 6),
        "p50": round(pct(50), 6),
        "p95": round(pct(95), 6),
        "p99": round(pct(99), 6),
    }


@dataclass
class SimulationReport:
    """Structured outcome of a simulation run."""

    scenario: str
    seed: int
    num_cameras: int
    duration_sec: float
    fps_target: int
    steps: int = 0
    processed_frames: int = 0
    dropped_frames: int = 0
    faults_injected: list[str] = field(default_factory=list)
    events_generated: int = 0
    incidents_generated: int = 0
    alerts_generated: int = 0
    evidence_generated: int = 0
    errors: list[str] = field(default_factory=list)
    reconnect_count: int = 0
    resource: dict[str, Any] = field(default_factory=dict)
    latency: dict[str, dict[str, float]] = field(default_factory=dict)
    final_state: dict[str, Any] = field(default_factory=dict)
    completed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


class Timer:
    """Lightweight wall-clock timer for honest latency measurement."""

    def __init__(self) -> None:
        self._start: float | None = None

    def __enter__(self) -> "Timer":
        self._start = time.time()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._start = None

    def elapsed_ms(self) -> float:
        if self._start is None:
            return 0.0
        return (time.time() - self._start) * 1000.0