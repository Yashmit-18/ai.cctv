# PHASE63 STREAMLIT DEPLOYMENT COMPATIBILITY REPORT — 2026-09-22

## 1. Original deployment error

The deployment environment was failing during startup with:

- Python: 3.14.7 initially, then moved to Python 3.12
- Error during `main.py` import:
  - `ImportError: libGL.so.1: cannot open shared object file: No such file or directory`
- Traceback path:
  - `main.py` -> `import cv2` -> `cv2/__init__.py` -> OpenCV native import failed because Linux GUI libs were missing

## 2. Exact root cause

The root cause was a Linux deployment incompatibility caused by the project depending on the GUI-enabled OpenCV wheel for a Streamlit Cloud runtime that does not ship the Linux desktop GUI library stack. The runtime was also pinned to a GPU ONNX package that is not appropriate for a headless cloud environment without NVIDIA CUDA.

This was not a redesign issue and did not require changing the local Windows CCTV runtime. The local runtime requires real camera and GUI behavior, while the cloud dashboard only needs headless-safe imports and a CPU-compatible runtime path.

## 3. Python version

- Verified project environment: Python 3.12.6
- Status: CODE VERIFIED

## 4. OpenCV package/version

- Installed and verified: `opencv-python-headless` 5.0.0.93
- Status: CODE VERIFIED

## 5. ONNX Runtime package/version

- Installed and verified: `onnxruntime` 1.29.0
- Status: CODE VERIFIED

## 6. Linux dependency requirements

The Linux cloud/runtime issue was solved by removing the GUI dependency on the desktop OpenCV build and switching the ONNX package to the CPU runtime variant.

Required compatibility behavior:

- Use headless OpenCV for cloud/dashboard startup
- Avoid requiring `libGL.so.1` on Linux
- Keep local Windows CCTV runtime behavior intact
- Do not enforce a physical camera for dashboard-only startup

## 7. Cloud startup behavior

The dashboard startup path is intentionally decoupled from the CCTV daemon. The app loads as a Streamlit dashboard without requiring camera runtime to render the admin/configuration UI.

Status: LOCAL VERIFIED

Cloud verification was not performed on the real Streamlit Community Cloud service in this environment, so the exact cloud remote deployment remains unverified at runtime.

## 8. Local startup behavior

Local Windows execution remains supported for the existing local CCTV/AI runtime and the daemon path (`main.py`). The fix did not redesign the architecture or disable the CCTV subsystem; it only removs the GUI/OpenCV mismatch from the headless/cloud path.

Status: LOCAL VERIFIED

## 9. Files modified

- `requirements.txt`
- `main.py`
- `PHASE63_STREAMLIT_DEPLOYMENT_COMPATIBILITY_REPORT_2026-09-22.md`

## 10. Tests executed

Executed:

1. `python --version`
2. `python -c "import cv2, numpy, torch; ..."`
3. `python -m pytest -q`
4. `python -m pytest -q -W error`
5. `python -m streamlit run app.py --server.headless true`

## 11. Test results

- `pytest -q`: 1283 passed in 194.07s
- `pytest -q -W error`: 1283 passed in 180.21s
- Streamlit local startup: started successfully on `http://localhost:8502`

Status summary:

- CODE VERIFIED
- LOCAL VERIFIED
- CLOUD VERIFIED: NOT VALIDATED
- ENVIRONMENT LIMITATION: yes, real Streamlit Cloud service is not available in this session
- MODEL LIMITATION: none for this fix

## 12. Streamlit local verification

Verified locally with:

- `python -m streamlit run app.py --server.headless true`

Observed result:

- Uvicorn server started successfully
- Local URL served successfully on port 8502
- Dashboard loaded without a physical camera requirement

## 13. Cloud deployment status

The code-level compatibility issue has been addressed for Linux/headless deployment compatibility, but the actual remote Streamlit Community Cloud deployment was not launched from this environment.

Status: NOT VALIDATED

## 14. Remaining limitations

- Real remote Streamlit Community Cloud runtime is not available from this session, so we cannot claim actual cloud deployment success.
- The local CCTV daemon still depends on webcam/RTSP/video-capable runtime when used on a real machine.
- The project is not a cloud-only app; it is a hybrid local CCTV + dashboard platform, and the cloud/headless path is intentionally limited to dashboard/startup compatibility.

## Status matrix

- CODE VERIFIED
- LOCAL VERIFIED
- CLOUD VERIFIED: NOT VALIDATED
- ENVIRONMENT LIMITATION: active
- MODEL LIMITATION: none
