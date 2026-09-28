# syntax=docker/dockerfile:1
# Multi-stage build: uv installs the project into a venv, then only the venv
# and the CUDA userland land in the runtime image.
#
# The server also runs fine on CPU (onnxruntime falls back automatically when
# the container has no GPU); pass --gpus all to use the NVIDIA GPU.

FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS builder

# git is needed to resolve the scyllasband git dependency
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Install dependencies first so source edits don't invalidate the layer.
# --frozen requires a committed uv.lock (regenerate with `uv lock` locally).
# README.md is required by hatchling (pyproject references it).
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY src ./src
# --no-editable: install the project as a wheel into the venv (an editable
# install would .pth-point at /app/src, which is not copied to the runtime).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# ---------------------------------------------------------------------------
# CUDA 13.0 + cuDNN 9 (what onnxruntime-gpu >= 1.30 is built against; CUDA 13
# uses stable unversioned sonames). 24.04 also matters: the builder's bookworm
# Python needs glibc >= 2.36 (jammy has 2.35).
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04 AS runtime

# libgomp1 is required by onnxruntime; ca-certificates for HF downloads.
# Ubuntu 24.04 images may already ship a UID 1000 user; reuse it if present.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && (getent passwd 1000 > /dev/null || useradd --create-home --uid 1000 app)

WORKDIR /app
# The venv's bin/python is an absolute symlink into the builder's
# interpreter (/usr/local/bin/python3), so the interpreter must come along.
COPY --from=builder /usr/local /usr/local
COPY --from=builder --chown=1000:1000 /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    SCYLLASBAND_MODEL_DIR=/data/models \
    HF_HOME=/data/hf

RUN mkdir -p /data/models /data/hf && chown -R 1000:1000 /data
USER 1000
VOLUME /data

EXPOSE 10200

# First start downloads the ~500 MB model into /data/models; give it room.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15m --retries=5 \
    CMD ["python", "-m", "scyllasbandv2_wyoming.healthcheck"]

ENTRYPOINT ["scyllasbandv2-wyoming"]
CMD ["--uri", "tcp://0.0.0.0:10200"]
