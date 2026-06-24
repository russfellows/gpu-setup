# Recipes

Reproducible benchmark sweeps for specific LLMs on specific serving stacks.

A **recipe** is one TOML file describing a model and one or more *variants* —
each variant is a `(vendor, stack)` combination such as `amd_vllm` or
`nvidia_trtllm`. The runner reads the TOML, launches the server in a
container, waits for it, runs the bench client via `docker exec`, and tears
the container down between combos.

## Quick start

```bash
# List what's available
./recipes/run_recipe.sh --list

# Run a recipe with its built-in default sweep
./recipes/run_recipe.sh gpt-oss-120b amd_atom

# Override the matrix from the command line
./recipes/run_recipe.sh qwen3-next-80b nvidia_vllm \
    --tp 1,2,4 \
    --shapes "1000,100 5000,500 10000,1000" \
    --conc "4 8 16 32 64 128 256"

# Dry-run (prints the plan and exits — no containers launched)
./recipes/run_recipe.sh kimi-k2.6 amd_vllm --dry-run
```

## Prerequisites

These must be in place before running any recipe. See
[docs/QUICKSTART.md](../docs/QUICKSTART.md) for verification commands.

1. **GPU drivers + ROCm or CUDA** — run `scripts/<vendor>/setup_*.sh` then reboot.
2. **Docker** — `scripts/common/setup_docker.sh`. Your user must be in the `docker`
   group (log out and back in after install).
3. **Storage + HF environment** — `scripts/common/setup_hf_env.sh`. Sets
   `HF_HOME=/mnt/data/huggingface` system-wide and writes
   `/etc/profile.d/huggingface.sh`. Source it or open a new shell.
4. **HF authentication** — `hf auth login`. Recipes forward `HF_TOKEN` from
   `$HF_TOKEN_PATH` into the container automatically.

## Results layout

Each run creates a timestamped directory:

```
$HOME/results/<model>/<variant>/<timestamp>/
    provenance.json                                          # image digest, build args, sweep matrix, recipe snapshot, host/GPU info
    summary.csv                                              # one row per (TP, ISL, OSL, CONC) combo: status + result filename
    server_tp<TP>_isl<ISL>_osl<OSL>_c<CONC>.log            # serving container stdout per combo
    <model>_<variant>_tp<TP>_isl<ISL>_osl<OSL>_c<CONC>.json  # bench client output per combo
```

`provenance.json` is the authoritative record for apples-to-apples comparisons —
read it before comparing two runs that used different images or build args.

Results go to `$HOME/results/`, never to `/mnt/data/` (which is scratch storage
that may not persist).

## Directory layout

```
recipes/
  run_recipe.sh              # unified CLI entrypoint
  README.md                  # this file
  _common/
    load_recipe.py           # reads recipe.toml, emits bash variable assignments
    sweep.sh                 # sweep harness (sourced by run_recipe.sh)
    docker_amd.sh            # standard AMD docker-run flag bundle
    docker_nvidia.sh         # standard NVIDIA docker-run flag bundle
    bench_client.sh          # bench client shim (vllm bench serve / atom)
    write_provenance.py      # writes provenance.json after each run
  <model-name>/
    recipe.toml              # model + all variants defined here
    README.md                # human-readable notes: variants, caveats, sweep matrix
    Dockerfile.<variant>     # optional: only when a variant needs a custom image build
```

## Recipe TOML schema

Each model has exactly one `recipe.toml`. All variants live inside it.

### Top-level `[recipe]` table

```toml
[recipe]
model_name  = "my-model"              # slug used in result paths
model_id    = "org/my-model"          # default HF model ID (variants may override)
description = "..."
```

### `[recipe.defaults]` — sweep matrix defaults

```toml
[recipe.defaults]
sweep_tp           = [1, 2, 4]
sweep_isl_osl      = ["1024,1024", "1024,8192", "8192,1024"]
sweep_conc         = [4, 8, 16, 32, 64, 128, 256]
random_range_ratio = 0.9
ready_timeout_s    = 1800
```

CLI flags (`--tp`, `--shapes`, `--conc`) override these at run time.

### `[variants.<name>]` — one block per variant

Required fields:

| Field | Example | Meaning |
|---|---|---|
| `vendor` | `"amd"` or `"nvidia"` | Selects the docker flag bundle |
| `stack` | `"vllm"`, `"atom"`, `"trtllm"`, `"sglang"`, `"triton"` | Selects default bench tool and ready marker |
| `image` | `"vllm/vllm-openai-rocm:nightly-<digest>"` | Docker image; **must be pinned**, no `:latest` |
| `server_entrypoint` | `"vllm serve"` | Command run as PID 1 inside the container |
| `server_args` | `["model-id", "--port", "8000", ...]` | Full argument list; use `@TP@`, `@ISL@`, `@OSL@`, `@CONC@` as placeholders |

Optional fields:

| Field | Default | Meaning |
|---|---|---|
| `model_id` | recipe-level `model_id` | Per-variant model identifier (use when AMD/NVIDIA use different HF repos) |
| `port` | `8000` | Port the server listens on |
| `ready_marker` | stack-default | Log string that signals server is ready |
| `bench_tool` | `"vllm"` (or `"atom"` for ATOM stack) | Which bench client to use |
| `docker_flags` | `[]` | Extra docker run flags (e.g. `--cpuset-cpus`, `--cpuset-mems`) |
| `bench_extra_args` | `[]` | Extra args forwarded to the bench client (e.g. `["--trust-remote-code"]`) |
| `extra_files` | `[]` | Files in the recipe dir to mount at `/recipe/<basename>` (read-only) |

### `[variants.<name>.env]` — environment variables

```toml
[variants.amd_vllm.env]
VLLM_ROCM_USE_AITER = "1"
HIP_VISIBLE_DEVICES = "0,1,2,3"
```

Every key-value pair is passed to the container as `-e KEY=VALUE`.

### `[variants.<name>.build]` — custom Docker builds

When a variant needs a customised image (e.g. to bake in a tuning CSV):

```toml
[variants.amd_vllm.build]
dockerfile = "Dockerfile.amd_vllm"   # relative to the recipe directory
context    = "."
tag        = "gpu-setup/my-model-amd:local"
build_args = { BASE_IMAGE = "vllm/vllm-openai-rocm:nightly-<digest>" }
```

The harness builds the image once and reuses it on subsequent runs. When
`build` is present, `image` serves as the base for the `FROM` line only.

### `[variants.<name>.runtime_config]` — YAML-tool config as JSON

Some serving stacks (e.g. `trtllm-serve`) accept config via a YAML file.
Put the config as a TOML table; the harness serializes it to JSON (valid
YAML) at run time and mounts it where the tool expects it:

```toml
[variants.nvidia_trtllm.runtime_config]
# ...fields here...

[variants.nvidia_trtllm]
runtime_config_path = "/path/in/container/config.json"
```

No YAML files are committed to this repo.

### `[variants.<name>.defaults]` — variant-level sweep overrides

A variant can override the recipe-level defaults (e.g. to lock to TP=4):

```toml
[variants.amd_vllm_numa.defaults]
sweep_tp = [4]
```

## Writing a new recipe

1. Create `recipes/<model-name>/recipe.toml`. Copy an existing one as a template.
2. **Pin every `image`** — `:latest` is not allowed in checked-in recipes.
   Use a digest or a specific version tag.
3. **Use `@TP@`, `@ISL@`, `@OSL@`, `@CONC@` in `server_args`** — never
   hardcode the tensor-parallel size or sequence lengths.
4. Provide sensible **sweep defaults** in `[recipe.defaults]`. A recipe that
   requires CLI flags to do anything is a broken recipe.
5. If a variant needs a custom image, put the build definition in
   `[variants.<name>.build]` and the Dockerfile in the same directory.
6. Add `recipes/<model-name>/README.md`: what the model is, variant differences,
   NUMA/gated-repo/hardware caveats.
7. Test with `--dry-run` before committing.
