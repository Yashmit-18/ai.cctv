"""Virtual-line entry/exit crossing detection (Phase A -- A10).

Detects when a *spatial person track* crosses a configured virtual line on a
camera and maps the crossing direction to an entry/exit event.  Crucially:

* it is **track-based** -- the trajectory spans many frames, so an
  appearance/disappearance near the line is NOT a crossing;
* **no identity is inferred from a single frame** -- the ``track_id`` is tied
  to the (already identity-smoothed) spatial track from
  :class:`~src.tracker.SpatialTracker`;
* one physical crossing yields exactly one event (per track+line in one
  direction), so a person pacing back and forth records alternating ENTRY and
  EXIT events instead of a duplicate flood.

Line definition (JSON, normalised 0..1 coords)
----------------------------------------------
.. code-block:: json

    [
      {"name": "main_door", "cam": "local_webcam",
       "x1": 0.5, "y1": 0.0, "x2": 0.5, "y2": 1.0,
       "mapping": {"A_TO_B": "EXIT_CROSSING", "B_TO_A": "ENTRY_CROSSING"}}
    ]

``A``/``B`` are the two sides of the directed line ``(x1,y1) -> (x2,y2)``
(left-of = ``A``, right-of = ``B``).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field

from config import ENTRY_EXIT_DEBOUNCE_SEC, TRACK_DEBUG
from src.domain import ENTRY_CROSSING, EXIT_CROSSING
from src.tracker import Track

logger = logging.getLogger("cctv.entry_exit")

_VALID_EVENTS = (ENTRY_CROSSING, EXIT_CROSSING)


def _norm(v) -> float:
    f = float(v)
    if not (0.0 <= f <= 1.0):
        raise ValueError(f"coordinate out of range [0,1]: {f}")
    return f


@dataclass
class VirtualLine:
    name: str
    cam: str
    x1: float = 0.5
    y1: float = 0.0
    x2: float = 0.5
    y2: float = 1.0
    mapping: dict = field(default_factory=lambda: {
        "A_TO_B": EXIT_CROSSING, "B_TO_A": ENTRY_CROSSING,
    })

    def __post_init__(self):
        self.x1, self.y1, self.x2, self.y2 = (
            _norm(self.x1), _norm(self.y1), _norm(self.x2), _norm(self.y2),
        )
        if self.x1 == self.x2 and self.y1 == self.y2:
            raise ValueError(f"line {self.name!r} endpoints must differ")
        for k, v in self.mapping.items():
            if v not in _VALID_EVENTS:
                raise ValueError(
                    f"line {self.name!r} maps to invalid event {v!r}")

    def side(self, x: float, y: float) -> str:
        """Left-of-directed-line = ``A``, right-of = ``B``."""
        vx, vy = self.x2 - self.x1, self.y2 - self.y1
        wx, wy = x - self.x1, y - self.y1
        return "A" if (vx * wy - vy * wx) >= 0 else "B"

    def event_for(self, transition: str) -> str | None:
        return self.mapping.get(transition)


def load_lines(path: str) -> list[VirtualLine]:
    """Read virtual lines from a JSON file (optional; [] when absent)."""
    if not path or not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read entry-exit lines %s: %s", path, exc)
        return []
    rows = data.get("lines") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return []
    lines: list[VirtualLine] = []
    for cfg in rows:
        try:
            lines.append(VirtualLine(
                name=str(cfg.get("name", "line")),
                cam=str(cfg.get("cam", "")),
                x1=cfg.get("x1", 0.5), y1=cfg.get("y1", 0.0),
                x2=cfg.get("x2", 0.5), y2=cfg.get("y2", 1.0),
                mapping=cfg.get("mapping") or {
                    "A_TO_B": EXIT_CROSSING, "B_TO_A": ENTRY_CROSSING,
                },
            ))
        except (ValueError, TypeError) as exc:
            logger.warning("Skipping bad entry-exit line %r: %s",
                           cfg.get("name"), exc)
    return lines


class EntryExitDetector:
    """Track-based line crossing detection with direction + debounce."""

    def __init__(self, lines: list[VirtualLine] | None = None,
                 debounce_sec: float = ENTRY_EXIT_DEBOUNCE_SEC,
                 debug: bool | None = None):
        self._lines_by_cam: dict[str, list[VirtualLine]] = {}
        for line in lines or []:
            self._lines_by_cam.setdefault(line.cam, []).append(line)
        self._debounce = float(debounce_sec)
        self._debug = TRACK_DEBUG if debug is None else bool(debug)
        # track_id -> (cx, cy) last centroid
        self._prev: dict[str, tuple[float, float]] = {}
        # (track_id, line.name) -> (last_fire_epoch, event_type)
        self._fired: dict[tuple, tuple] = {}
        logger.info(
            "EntryExitDetector initialised with %d line(s) on %d camera(s)",
            sum(len(v) for v in self._lines_by_cam.values()), len(self._lines_by_cam),
        )

    def has_lines(self) -> bool:
        return bool(self._lines_by_cam)

    def process(self, tracks: list[Track],
                ts: float | None = None) -> list[dict]:
        """Evaluate all tracks against their camera's virtual lines.

        Returns a list of crossing events::

            {"event_type": "EXIT_CROSSING"|"ENTRY_CROSSING",
             "camera": ..., "line": ..., "direction": "A_TO_B"|"B_TO_A",
             "track_id": ..., "employee_id": ...|None,
             "confidence": float, "timestamp": ISO}
        """
        if not self._lines_by_cam:
            return []
        now = time.time() if ts is None else ts
        events: list[dict] = []

        live_ids: set[str] = set()
        for t in tracks:
            live_ids.add(t.track_id)
            cx, cy = t.centroid_normalized
            prev = self._prev.get(t.track_id)
            self._prev[t.track_id] = (cx, cy)
            if prev is None:
                continue  # prime only
            for line in self._lines_by_cam.get(t.cam, []):
                side_now = line.side(cx, cy)
                side_prev = line.side(*prev)
                if side_now == side_prev:
                    continue
                transition = f"{side_prev}_TO_{side_now}"
                ev_type = line.event_for(transition)
                if ev_type is None:
                    continue
                key = (t.track_id, line.name)
                last_fire = self._fired.get(key)
                if last_fire is not None and now - last_fire[0] < self._debounce:
                    continue
                self._fired[key] = (now, ev_type)
                events.append({
                    "event_type": ev_type,
                    "camera": t.cam,
                    "line": line.name,
                    "direction": transition,
                    "track_id": t.track_id,
                    "employee_id": t.identity if t.identity != "Unknown" else None,
                    "confidence": round(t.confidence, 3),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S",
                                               time.localtime(now)),
                })
                if self._debug:
                    logger.info(
                        "TRACK_DEBUG crossing track=%s line=%s dir=%s -> %s",
                        t.track_id, line.name, transition, ev_type,
                    )

        # Forget placements of tracks that are gone (no stale primitives).
        for tid in list(self._prev):
            if tid not in live_ids:
                self._prev.pop(tid, None)
        return events