# gpu-setup

Bootstrap and configure bare-metal GPU server instances from a fresh Ubuntu 24.04 install.

## Assumptions

The **only** things assumed in advance:

- OS: **Ubuntu 24.04 LTS**
- One or more datacenter GPUs installed (NVIDIA or AMD)
- Root / sudo access

Everything else — driver state, CUDA/ROCm presence, kernel headers, container
toolkit — is **detected at runtime**. If a working stack is already present,
the scripts skip reinstall rather than risk breaking a working Neo-Cloud image.

## Layout

```
bootstrap.sh                 # top-level entrypoint: detect GPUs, dispatch
scripts/
  lib/common.sh              # shared bash helpers
  common/
    setup_prereqs.sh         # vendor-neutral: build tools, uv, gh, hf CLI
    setup_storage.sh         # opt-in: discover NVMe, build mdraid + XFS, mount /mnt/data
    setup_docker.sh          # Docker CE + (NVIDIA hosts) nvidia-container-toolkit
    setup_hf_env.sh          # /etc/profile.d/huggingface.sh + HF cache on /mnt/data
    pull_serving_images.sh   # pre-pull vLLM / Triton / TRT-LLM / SGLang images
  nvidia/
    setup_nvidia.sh          # CUDA + open driver + fabric manager + container toolkit
    verify_nvidia.sh         # post-reboot health checks
  amd/
    setup_amd_rocm.sh        # amdgpu-install + ROCm
    verify_amd.sh            # post-reboot HIP compute test
recipes/                     # benchmark sweep recipes — see recipes/README.md
  run_recipe.sh              # CLI entrypoint:  ./recipes/run_recipe.sh <model> <variant>
  _common/                   # harness, loader, vendor docker-flag bundles, bench shim
  gpt-oss-120b/              # OpenAI GPT-OSS 120B
  kimi-k2.6/                 # Moonshot Kimi-K2.6 (MXFP4 on AMD, NVFP4 on NVIDIA)
  qwen3-next-80b/            # Qwen3-Next-80B-A3B-Instruct-FP8
docs/
  archive/                   # original drafts kept for reference
```

## Usage

```bash
git clone https://github.com/russfellows/gpu-setup.git
cd gpu-setup

# 1. Dry-run: shows what was detected and the planned steps.
sudo ./bootstrap.sh

# 2. Execute. Runs common prereqs + the matching vendor setup.
sudo ./bootstrap.sh --yes

# 3. Reboot, then run the vendor verify script.
sudo reboot
sudo ./scripts/nvidia/verify_nvidia.sh   # or scripts/amd/verify_amd.sh
```

### Detection-first behavior

Each vendor script runs a health check before installing:

- **NVIDIA**: `nvidia-smi` works, `nvcc` reports the right CUDA version, and
  Fabric Manager is active when NVSwitches are present.
- **AMD**: `rocm-smi` works, `hipcc` works, and `/opt/rocm/.info/version`
  matches the requested ROCm version.

If the stack passes, the script exits without changes. Override with
`FORCE_REINSTALL=1`.

### Environment variables

| Variable          | Default     | Meaning                                          |
|-------------------|-------------|--------------------------------------------------|
| `PYTHON_VERSION`  | `3.12`      | Python version installed via `uv`                |
| `CUDA_VERSION`    | `13-3`      | apt suffix for `cuda-toolkit-<ver>` (NVIDIA)     |
| `ROCM_VERSION`    | `7.2.4`     | ROCm release (AMD)                               |
| `ROCM_DEB_BUILD`  | `7.2.4.70204-1` | amdgpu-install deb build suffix              |
| `FORCE_REINSTALL` | `0`         | Set `1` to skip the "already healthy" shortcut   |
| `ASSUME_YES`      | `1`         | Non-interactive apt                              |

Set `CUDA_VERSION=auto` or `ROCM_VERSION=auto` to accept whatever is already installed.

### Manual override

```bash
# Force the NVIDIA path even if detection is ambiguous
sudo ./bootstrap.sh --yes --vendor nvidia

# Skip common prereqs (e.g., you already ran them)
sudo ./bootstrap.sh --yes --skip-common
```

## Storage setup (opt-in, separate)

Most fresh GPU boxes ship with multiple unused NVMe devices. The optional
[`scripts/common/setup_storage.sh`](scripts/common/setup_storage.sh) script
discovers them, builds an mdraid array sized to the disk count, formats it
with XFS aligned to the RAID geometry, and mounts it at `/mnt/data`.

It is intentionally **not** wired into `bootstrap.sh` — storage is destructive
and conceptually independent from the GPU stack. Run it explicitly when you
want it.

```bash
# Dry-run: prints what was detected, why anything was excluded, and the exact
# commands it would run. Default mode — totally safe.
sudo ./scripts/common/setup_storage.sh

# Once you're happy with the plan:
sudo ./scripts/common/setup_storage.sh --execute
```

### RAID level by device count

| Disks | Layout                                                       |
|-------|--------------------------------------------------------------|
| 1     | No RAID — XFS straight on the device                         |
| 2     | RAID-1 mirror                                                |
| 3     | RAID-1 (2 active) + 1 hot spare                              |
| 4     | RAID-10                                                      |
| 5     | RAID-10 (4 active) + 1 hot spare                             |
| N ≥ 6 | RAID-10 over the largest even count ≤ N; any odd leftover is a spare |

### Safety screen

A device is included **only** if every check passes; any tripwire excludes it:

- Not the device that hosts `/`
- Not currently mounted (anywhere, any partition)
- No filesystem / LVM / MD / swap signature (`wipefs -n` is empty)
- No existing partitions
- Not already a member of an md array (`/proc/mdstat`, `mdadm --examine`)
- Not active swap
- Not in `$EXCLUDE_DEVICES`
- Size ≥ `STORAGE_MIN_GB`

Every device is printed with the reason it was kept or skipped, so you can
audit the decision before running with `--execute`.

### Storage env vars

| Variable           | Default     | Meaning                                                 |
|--------------------|-------------|---------------------------------------------------------|
| `STORAGE_MOUNT`    | `/mnt/data` | Mount point                                             |
| `STORAGE_MIN_GB`   | `100`       | Minimum device size considered (excludes tiny boot NVMe) |
| `STORAGE_RAID_NAME`| `data`      | md array name (becomes `/dev/md/<name>`)                |
| `EXCLUDE_DEVICES`  | (empty)     | Extra devices to skip, space-separated absolute paths   |
| `CHUNK_KB`         | `512`       | mdadm chunk size (KiB)                                  |

### XFS geometry choices

- 4 KiB block size, 4 KiB sector size (NVMe is 4K LBA).
- 2 GiB internal log — helps metadata throughput during large model downloads.
- For RAID-10: `-d su=<chunk>,sw=<active/2>` aligns allocation to stripe geometry.
- Mount options: `defaults,noatime,nodiratime,largeio,inode64,allocsize=16m,logbufs=8,logbsize=256k`.

> Larger XFS block sizes (>4 KiB) require kernel large-block-size support that
> is still maturing on Linux. We stick with 4 KiB; the AI-workload benefit
> comes from stripe alignment + `largeio` + `allocsize=16m`, not from block size.

## Serving stack setup (opt-in)

Three independent helpers in [scripts/common/](scripts/common/) get the host
ready to run containerized inference engines. None of them are wired into
`bootstrap.sh` — run them when you want them.

```bash
# Docker CE + (NVIDIA hosts) nvidia-container-toolkit. Adds you to docker group.
sudo ./scripts/common/setup_docker.sh

# /etc/profile.d/huggingface.sh so HF_HOME=/mnt/data/huggingface for every user.
# Refuses to run if /mnt/data isn't mounted.
sudo ./scripts/common/setup_hf_env.sh

# Pre-pull vLLM / Triton / TRT-LLM / SGLang images for the detected vendor.
sudo ./scripts/common/pull_serving_images.sh             # dry-run on first call
DRY_RUN=1 ./scripts/common/pull_serving_images.sh        # show plan
```

After authenticating with `hf auth login` (and optionally `gh auth login`),
the host is ready to run recipes.

## Benchmark recipes

[`recipes/`](recipes/) holds reproducible benchmark sweeps for specific
LLMs on specific serving stacks. A recipe is one TOML file describing the
model and one or more variants — `(vendor, stack)` combinations like
`amd_vllm` or `nvidia_trtllm`. A unified runner reads the TOML, launches
the matching container with the vendor-standard flag bundle, waits for the
server, runs a sweep of `(TP × ISL/OSL × concurrency)` via the bench
client, and tears the container down between combinations.

```bash
# What's available
./recipes/run_recipe.sh --list

# Use the recipe's built-in defaults
./recipes/run_recipe.sh gpt-oss-120b amd_atom

# Override the matrix
./recipes/run_recipe.sh qwen3-next-80b nvidia_vllm \
    --tp 1,2,4 --shapes "1000,100 5000,500" --conc "16 32 64"

# See the exact plan without running anything
./recipes/run_recipe.sh kimi-k2.6 amd_vllm --dry-run
```

Results land under `$HOME/results/<model>/<variant>/<timestamp>/`:

- `provenance.json` — image digest, base image (for builds), build args, sweep
  matrix, full recipe.toml snapshot, host/GPU info. Required reading for any
  apples-to-apples comparison across runs.
- `summary.csv` — one row per `(TP, ISL, OSL, CONC)` combo
- `<model>_<variant>_tp*_isl*_osl*_c*.json` — bench client output per combo
- `server_tp*_isl*_osl*_c*.log` — server stdout per combo

For full recipe authoring docs (TOML schema, custom Dockerfile builds,
runtime-config JSON injection, `extra_files` mounts, `@TP@`/`@ISL@`/`@OSL@`
/`@CONC@` placeholders) see [recipes/README.md](recipes/README.md).

## Python tooling

Where Python is needed, scripts use [uv](https://github.com/astral-sh/uv) — no
pre-installed Python environment is assumed. The HuggingFace CLI is installed
as a `uv tool` rather than into any project venv.

## License

GNU Affero General Public License v3.0 — see [LICENSE](LICENSE).
