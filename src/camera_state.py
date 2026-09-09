"""Per-camera offline/recovered security FSM (Phase 31).

Transitions:
    ONLINE -> OFFLINE  (no frames for OFFLINE_TRIGGER_SEC)
    OFFLINE -> ONLINE  (frames resume -> CAMERA_RECOVERED event)

An offline camera is **never** treated as employee AWAY or intrusion.
Camera health remains a separate concern from the employee tracker FSM.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from src.domain import CAMERA_OFFLINE, CAMERA_RECOVERED

logger = logging.getLogger("cctv.camera_state")


@dataclass
class _CamState:
    camera_id: str
    online: bool = True
    offline_since: float | None = None
    total_downtime_sec: float = 0.0
    offline_event_fired: bool = False


class CameraStateManager:
    """Tracks per-camera online/offline transitions.

    Call ``update(camera_id, is_online)`` each tick.  Returns events when
    transitions occur.
    """

    def __init__(self, offline_trigger_sec: float = 15.0):
        self._trigger = offline_trigger_sec
        self._cams: dict[str, _CamState] = {}

    def update(self, camera_id: str, is_online: bool, now: float | None = None) -> list[dict]:
        """Feed the latest camera health state.

        Returns a list of event dicts (0, 1, or 2 entries on rare edge cases)
        to be persisted by the caller.
        """
        now = now if now is not None else time.time()
        state = self._cams.get(camera_id)
        if state is None:
            state = _CamState(camera_id=camera_id)
            self._cams[camera_id] = state

        events: list[dict] = []

        if is_online:
            if not state.online:
                # Recovery
                downtime = 0.0
                if state.offline_since is not None:
                    downtime = now - state.offline_since
                    state.total_downtime_sec += downtime
                state.online = True
                state.offline_since = None
                state.offline_event_fired = False
                logger.info("[SECURITY] CAMERA_RECOVERED camera=%s downtime=%.1fs",
                            camera_id, downtime)
                events.append({
                    "event_type": CAMERA_RECOVERED,
                    "camera": camera_id,
                    "duration_seconds": round(downtime, 1),
                    "confidence": 1.0,
                })
            # else: already online, no event
        else:
            if state.online:
                # Just went offline
                state.online = False
                state.offline_since = now
            if (not state.offline_event_fired
                    and state.offline_since is not None
                    and (now - state.offline_since) >= self._trigger):
                downtime = now - state.offline_since
                state.total_downtime_sec += downtime
                state.offline_event_fired = True
                logger.info("[SECURITY] CAMERA_OFFLINE camera=%s after=%.1fs",
                            camera_id, downtime)
                events.append({
                    "event_type": CAMERA_OFFLINE,
                    "camera": camera_id,
                    "duration_seconds": round(downtime, 1),
                    "confidence": 1.0,
                })

        return events

    def is_online(self, camera_id: str) -> bool:
        state = self._cams.get(camera_id)
        return state.online if state else True

    def downtime(self, camera_id: str) -> float:
        state = self._cams.get(camera_id)
        return state.total_downtime_sec if state else 0.0

    def all_states(self) -> dict[str, dict]:
        return {
            cid: {
                "online": s.online,
                "offline_since": s.offline_since,
                "total_downtime_sec": round(s.total_downtime_sec, 1),
            }
            for cid, s in self._cams.items()
        }
