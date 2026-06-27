#!/usr/bin/env bash
# ==============================================================================
# Environment profile: container
#
# Generic OCI/Docker container with GPU passthrough (e.g. nvidia-container-
# runtime or --device=/dev/kfd).  We are already inside an isolated runtime,
# so there is no Docker daemon to talk to.  Inference servers are launched
# as native processes rather than nested containers.
#
# Callers may override these before sourcing:
#   SHARED_ROOT   — root of the shared storage mount (default /workspace/data)
#   HF_HOME       — HF model cache directory (default $SHARED_ROOT/huggingface)
#
# This profile is sourced by vendor-specific profiles (runpod.sh, etc.) that
# set SHARED_ROOT before sourcing this file.
# ==============================================================================

# Run inference servers as native processes — no Docker inside Docker.
export NATIVE=1

# uv: persistent cache + point uv run at the persistent venv so Python
# invocations (load_recipe.py, write_provenance.py, vllm serve, etc.)
# all resolve to the same environment regardless of shell activation state.
export UV_CACHE_DIR="/workspace/.uv-cache"
export VIRTUAL_ENV="/workspace/venv"
mkdir -p "$UV_CACHE_DIR" 2>/dev/null || true
if [ ! -L "${HOME:-/root}/.cache/uv" ]; then
  mkdir -p "${HOME:-/root}/.cache" 2>/dev/null || true
  rm -rf "${HOME:-/root}/.cache/uv" 2>/dev/null || true
  ln -sfn "$UV_CACHE_DIR" "${HOME:-/root}/.cache/uv" 2>/dev/null || true
fi

: "${SHARED_ROOT:=/workspace/data}"
export SHARED_ROOT

: "${HF_HOME:=${SHARED_ROOT}/huggingface}"
export HF_HOME
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_XET_HIGH_PERFORMANCE=1

# Ensure /mnt/data resolves to SHARED_ROOT so scripts that hardcode /mnt/data
# keep working without modification.
if [ ! -L /mnt/data ] && [ ! -e /mnt/data ]; then
  mkdir -p /mnt 2>/dev/null || true
  ln -sfn "$SHARED_ROOT" /mnt/data 2>/dev/null || true
elif [ -L /mnt/data ] && [ "$(readlink /mnt/data)" != "$SHARED_ROOT" ]; then
  ln -sfn "$SHARED_ROOT" /mnt/data 2>/dev/null || true
fi

# Create cache directories if they don't exist yet.
mkdir -p "${HF_HOME}" "${SHARED_ROOT}/torch_compile_cache" 2>/dev/null || true
