# gpu-setup

Bootstrap and configure bare-metal GPU server instances from a fresh Ubuntu 24.04 install.

## Overview

This repo provides scripts to take a bare-metal system (physical or cloud-hosted) from a
minimal Ubuntu 24.04 image to a fully configured GPU compute environment. The only
assumptions are:

- OS: Ubuntu 24.04 LTS
- GPU(s) present (NVIDIA)
- Root or sudo access

Everything else is detected and handled at runtime.

## Repository Layout

```
scripts/    # Setup and configuration scripts (primarily bash)
docs/       # Reference documentation and setup guides
```

## Usage

Clone this repo onto the target machine and run the setup scripts from `scripts/`.
Scripts are designed to be run in order, but are also safe to re-run individually.

```bash
git clone https://github.com/russfellows/gpu-setup.git
cd gpu-setup
```

## Python Scripts

Where Python tooling is needed, scripts use [uv](https://github.com/astral-sh/uv) for
dependency management and execution — no pre-installed Python environment is assumed.

## License

GNU Affero General Public License v3.0 — see [LICENSE](LICENSE).
