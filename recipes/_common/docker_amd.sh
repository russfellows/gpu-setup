#!/usr/bin/env bash
# Standard AMD ROCm docker-run flag bundle.
# Sourced by sweep.sh. Sets the array AMD_DOCKER_FLAGS.
#
# Mirrors the device pass-through pattern that AMD GPU containers expect:
# /dev/kfd for the compute driver, /dev/dri for display/render, plus the
# render+video groups so the container's processes can talk to them.

AMD_DOCKER_FLAGS=(
  --device=/dev/kfd
  --device=/dev/dri
  --group-add=video
  --group-add=render
  --ipc=host
  --network=host
  --cap-add=SYS_PTRACE
  --security-opt=seccomp=unconfined
  --shm-size=64g
)
