# CLAUDE.md — gpu-setup

Operational guidance for Claude Code sessions working inside this repo.
Repo-level rules; the parent `~/Code/CLAUDE.md` (prime directives:
no unapproved code edits, no unauthorized pushes) still applies on top.

## What this repo is

Bootstrap and benchmark tooling for bare-metal GPU servers running
Ubuntu 24.04. Two halves:

1. **`scripts/`** — gets a fresh host from "stock Ubuntu install" to "ready
   to run containerized inference." Vendor-aware (NVIDIA / AMD), all
   detection-first (skip reinstall if a healthy stack already exists),
   no destructive defaults.
2. **`recipes/`** — TOML-driven benchmark sweeps. One TOML per model
   describing the model + N variants `(vendor, stack)`. The harness loads
   the TOML, launches the matching container, runs a sweep of
   `(TP × ISL/OSL × concurrency)`, captures provenance + per-combo results.

## Runbook for a fresh GPU box

Assume: bare-metal or cloud bare-metal Ubuntu 24.04, GPUs physically present,
no software installed beyond the OS.

```bash
# 1. Clone (or copy) the repo onto the target.
git clone https://github.com/russfellows/gpu-setup.git
cd gpu-setup

# 2. Dry-run bootstrap. Detects NVIDIA vs AMD, prints what it would do.
sudo ./bootstrap.sh

# 3. If the plan looks right, execute. Runs common prereqs + vendor setup.
sudo ./bootstrap.sh --yes

# 4. Reboot (required: DKMS module load for AMD; open driver + fabric mgr for NVIDIA).
sudo reboot

# 5. After reboot, verify the GPU stack.
sudo ./scripts/nvidia/verify_nvidia.sh   # or scripts/amd/verify_amd.sh

# 6. Bulk NVMe storage (opt-in — destructive). Dry-run first.
sudo ./scripts/common/setup_storage.sh           # prints plan only
sudo ./scripts/common/setup_storage.sh --execute # actually does it

# 7. Container layer.
sudo ./scripts/common/setup_docker.sh
sudo ./scripts/common/setup_hf_env.sh           # writes /etc/profile.d/huggingface.sh
exec $SHELL -l                                  # re-source so HF_HOME / docker group take effect
hf auth login                                   # authenticate with Hugging Face

# 8. (Optional) pre-pull serving images.
./scripts/common/pull_serving_images.sh

# 9. List + dry-run a recipe to confirm the path is clear.
./recipes/run_recipe.sh --list
./recipes/run_recipe.sh qwen3-next-80b nvidia_vllm --dry-run
```

## Hard invariants — do not violate

Every script in this repo follows these rules. Future additions must too.

1. **Detection-first.** Vendor setup scripts check whether a healthy stack
   already exists (driver loaded, `nvcc`/`hipcc` works, fabric manager
   active for NVSwitches) and exit without changes if so. Override only
   with `FORCE_REINSTALL=1`. Many cloud images ship working stacks — do
   not clobber them.
2. **Dry-run by default for anything destructive.** `bootstrap.sh` and
   `setup_storage.sh` print their plan and exit unless given `--yes` (or
   `--execute` for storage). The recipe runner supports `--dry-run` for
   the same reason. New destructive scripts must follow this pattern.
3. **Multi-check safety on block devices.** `setup_storage.sh` runs every
   candidate NVMe through 8 checks (root device, mount status,
   wipefs signatures, partitions, mdadm membership, swap usage,
   user exclusion list, size threshold). Adding new device-touching code
   must reuse this screen, not bypass it.
4. **Pinned versions and digests over floating tags** when reproducibility
   matters. The default Kimi `amd_vllm` variant pins both the base image
   digest and an AITER commit. The `amd_vllm_nightly` and `amd_vllm_quark`
   variants intentionally use floating tags — that's their whole purpose.
   Document the trade-off when adding either kind.
5. **Provenance is mandatory.** Every sweep writes `provenance.json` to
   the results directory: image ref + digest, base image (for builds),
   build args, sweep matrix, full recipe.toml snapshot, host + GPU info.
   This is how runs compare apples-to-apples across weeks. Do not skip it.
6. **No YAML in the repo.** YAML-expecting tools (e.g. `trtllm-serve
   --extra_llm_api_options`) accept JSON because JSON is a strict subset
   of YAML. Put config in `[variants.<name>.runtime_config]` in the TOML;
   the harness materializes JSON at run time. If a tool genuinely cannot
   accept JSON, ask before adding YAML.
7. **No references to external/internal source documents.** The recipes
   were distilled from material that should not be cited or attributed.
   Recipes read as first-person authored configs. Do not add "see doc X"
   or "as recommended by Y" attributions.

## Recipe authoring conventions

When the user asks for a new recipe:

1. **One TOML per model**, all variants inside (`recipes/<model>/recipe.toml`).
   Schema is documented in `recipes/README.md` and enforced by
   `recipes/_common/load_recipe.py`.
2. **Pin every `image`.** `:latest` is never acceptable in a checked-in
   recipe. Use a digest or a specific tag.
3. **Use `@TP@`, `@ISL@`, `@OSL@`, `@CONC@` placeholders** in
   `server_args`. No arithmetic — if a value needs derivation, pick a safe
   upper bound and hardcode it.
4. **Custom Dockerfile builds** go in `recipes/<model>/Dockerfile.<variant>`
   referenced from `[variants.<name>.build]`. Parameterize with `ARG`s so
   one Dockerfile can serve multiple variants (see Kimi for an example).
5. **Provide reasonable sweep defaults** in `[recipe.defaults]`. The
   CLI overrides them, but a recipe that needs flags to do anything is
   a broken recipe.
6. **Add a short `README.md`** per model: what the model is, what each
   variant differs in, any host-specific caveats (NUMA pinning, gated
   repo access, etc.). Link to the HF model card.

## Results go to `$HOME/results/...`, NOT `/mnt/data/...`

`/mnt/data` is scratch — it can disappear. Sweep results are valuable and
live with the user. The harness already does this; don't change it.

## Common pitfalls

- **`docker info` failing as the user**: they need to be in the `docker`
  group AND have logged out + back in (or run `newgrp docker`).
  `setup_docker.sh` adds them; the shell session has to be refreshed.
- **`hf` not on PATH**: `setup_prereqs.sh` installs it via `uv tool
  install`, which puts it in `~/.local/bin`. New shells pick that up via
  `~/.profile`; existing shells need `exec $SHELL -l`.
- **`/mnt/data` not mounted — NVMe drives are the root FS**: Some hosts
  (e.g. TensorWave MI355X nodes) pre-configure all NVMe drives as a RAID-0
  root filesystem. `setup_storage.sh` correctly declines to touch them.
  `setup_hf_env.sh` auto-detects this (root has >= 500 GB free) and falls
  back to `/data/huggingface` on root instead of erroring. No action needed.
- **NUMA mismatches on AMD**: the `qwen3-next-80b/amd_vllm_numa` variant
  bakes in cpuset values for a *specific* reference host (GPUs 0–3 on
  NUMA 0, GPUs 4–7 on NUMA 1). On a different topology those values
  mis-pin and skew results. Run `rocm-smi --showtopo` and verify before
  trusting `amd_vllm_numa` numbers on a new box.
- **Float-tag drift**: the Kimi `amd_vllm_nightly` and `amd_vllm_quark`
  variants pull from floating tags by design. `provenance.json` captures
  the resolved digest each run — read it before comparing two runs.

## What the user has explicitly said

(Things to internalize, not re-ask each session.)

- **TOML only — no YAML, ever.** Repo is YAML-free as a hard rule.
- **No attribution to source docs**, internal or external. The recipes
  must read as first-person authored.
- **Container-only for serving stacks** — no native pip/uv installs of
  vLLM / Triton / SGLang / TRT-LLM on the host.
- **Results in `$HOME/results`**, never under `/mnt/data`.
- **Pinned base images are the default** for AMD recipes; floating-tag
  variants exist as opt-ins for direct comparison against vendor guidance.
