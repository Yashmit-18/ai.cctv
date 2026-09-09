# =====================================================================
# CCTV Employee Productivity Tracker -- Docker Image
# =====================================================================
# Two base-image options are supported via --build-arg BASE_IMAGE:
#
#   CUDA (GPU inference, recommended):
#     docker build --build-arg BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 .
#
#   CPU-only (smaller image):
#     docker build --build-arg BASE_IMAGE=python:3.10-slim .
#
# Requires Docker + NVIDIA Container Toolkit for GPU passthrough.
# See README.md for full setup instructions.
# =====================================================================

ARG BASE_IMAGE=python:3.10-slim

FROM ${BASE_IMAGE}

# ---------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------
# OpenCV needs libGL and GLib. ffmpeg is optional but useful for RTSP work.
# When the CUDA image is used these apt packages are also available.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        ffmpeg \
        && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------
# Working directory
# ---------------------------------------------------------------------
WORKDIR /app

# ---------------------------------------------------------------------
# Python dependencies (cached layer--install before copying source)
# ---------------------------------------------------------------------
COPY requirements.txt .

# Torch is large; --no-cache-dir keeps the image lean.
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------
# Application source
# ---------------------------------------------------------------------
COPY . .

# ---------------------------------------------------------------------
# Persistent data volume (created inline; see docker-compose for mapping)
# ---------------------------------------------------------------------
VOLUME ["/app/data"]

# ---------------------------------------------------------------------
# Non-root user for security
# ---------------------------------------------------------------------
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

# ---------------------------------------------------------------------
# Default command (overridden by docker-compose per-service)
# ---------------------------------------------------------------------
CMD ["python", "main.py", "--headless"]
