"""Virtual office simulation for deterministic multi-camera, load, failure and
long-run validation (Phase 36).

Simulation-only module.  It is NEVER imported by production runtime code
(``main.py``, ``app.py``).  It provides deterministic, opt-in virtual cameras
and scripted scenarios that feed the *real* camera-pipeline interface
(``latest_frames``/``detect_batch``/``process_batch``/``security.tick``) so the
existing detection, tracking, security, incident, evidence and alert machinery
is exercised end-to-end without touching real cameras or the real database.

Virtual cameras are NOT real RTSP cameras.  Real RTSP/GPU/SMTP/Docker
validation remains a future client-environment task.
"""

from __future__ import annotations