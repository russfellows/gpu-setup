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
  common/setup_prereqs.sh    # vendor-neutral: build tools, uv, gh, hf CLI
  nvidia/
    setup_nvidia.sh          # CUDA + open driver + fabric manager + container toolkit
    verify_nvidia.sh         # post-reboot health checks
  amd/
    setup_amd_rocm.sh        # amdgpu-install + ROCm
    verify_amd.sh            # post-reboot HIP compute test
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

## Python tooling

Where Python is needed, scripts use [uv](https://github.com/astral-sh/uv) — no
pre-installed Python environment is assumed. The HuggingFace CLI is installed
as a `uv tool` rather than into any project venv.

## License

GNU Affero General Public License v3.0 — see [LICENSE](LICENSE).
