# Recipes

Reproducible benchmark sweeps for specific LLMs on specific serving stacks.

A **recipe** is one model + one serving stack + one vendor's hardware. The
runner sweeps across a configurable matrix of tensor-parallel sizes, input
and output sequence lengths, and concurrency levels — launching the server
in a container, waiting for it to come up, running the bench client via
`docker exec`, and tearing the container down between combinations.

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

These must be in place before running any recipe:

1. **GPU drivers + ROCm or CUDA** — `scripts/<vendor>/setup_*.sh`.
2. **Docker** — `scripts/common/setup_docker.sh`. Your user must be in the
   `docker` group (you may need to log out and back in).
3. **Bulk storage mounted at `/mnt/data`** — `scripts/common/setup_storage.sh`.
4. **HF environment configured** — `scripts/common/setup_hf_env.sh`. Sets
   `HF_HOME=/mnt/data/huggingface` system-wide; recipes mount that into
   the serving container at `/root/.cache/huggingface`.
5. **HF authentication** — `hf auth login` as your user. Gated models also
   need `HF_TOKEN` exported.

The serving image is pulled lazily by the recipe if not already present.
You can also pre-pull the standard set with
`scripts/common/pull_serving_images.sh`.

## Results layout

Each run creates:

```
$HOME/results/<model>/<variant>/<timestamp>/
    summary.csv                    # one row per (TP, ISL, OSL, CONC) combo
    server_tp<TP>_isl<ISL>_osl<OSL>_c<CONC>.log    # serving stack log per combo
    <model>_<variant>_tp<TP>_isl<ISL>_osl<OSL>_c<CONC>.json    # bench result per combo
```

`$HOME/results` was chosen because `/mnt/data` is conceptually scratch
storage that may disappear; sweep results are valuable and belong with
the user.

## Directory layout

```
recipes/
  run_recipe.sh                # unified CLI entrypoint
  README.md                    # this file
  _common/
    sweep.sh                   # the sweep harness (sourced by every variant)
    docker_amd.sh              # standard AMD docker-run flag bundle
    docker_nvidia.sh           # standard NVIDIA docker-run flag bundle
    bench_client.sh            # bench client shim (vllm bench serve / atom)
  <model-name>/
    README.md                  # human-readable notes for this model
    <variant>.sh               # one file per (vendor, stack) variant
```

## What a recipe variant script contains

Each `<variant>.sh` declares a small set of variables and then calls
`run_sweep`. The harness handles container lifecycle, ready-wait,
results capture, and CSV summary.

Required variables (set by the recipe):

| Variable        | Meaning                                                       |
|-----------------|---------------------------------------------------------------|
| `MODEL_NAME`    | Slug used in paths (e.g. `gpt-oss-120b`)                      |
| `VARIANT_NAME`  | E.g. `amd_atom`, `nvidia_vllm`                                |
| `VENDOR`        | `amd` or `nvidia`                                             |
| `STACK`         | `vllm`, `atom`, `trtllm`, `sglang`, `triton`                  |
| `IMAGE`         | Docker image (pinned tag)                                     |
| `MODEL_ID`      | Model identifier (HF id or in-container path)                 |
| `SERVER_CMD`    | Array — full command line for the server. Use `@TP@` where the tensor-parallel size goes. |

Optional defaults the recipe may set (CLI overrides win):

| Variable             | Default                                |
|----------------------|----------------------------------------|
| `SWEEP_TP`           | `1`                                    |
| `SWEEP_ISL_OSL`      | `1024,1024`                            |
| `SWEEP_CONC`         | `4 8 16 32 64 128 256`                 |
| `RANDOM_RANGE_RATIO` | `1.0`                                  |
| `READY_MARKER`       | `Application startup complete`         |
| `READY_TIMEOUT_S`    | `1800`                                 |
| `BENCH_TOOL`         | `vllm` (`atom` for ATOM stack)         |
| `PORT`               | `8000`                                 |
| `EXTRA_DOCKER_ENV`   | (empty array)                          |
| `EXTRA_DOCKER_FLAGS` | (empty array — e.g. NUMA pinning)      |

## Writing a new recipe

1. Create `recipes/<model-name>/<variant>.sh` (copy an existing one as a
   template).
2. Pin the image tag — `:latest` is forbidden in checked-in recipes
   because runs must be reproducible.
3. Use `@TP@` in `SERVER_CMD` instead of hardcoding the tensor-parallel size.
4. Provide reasonable sweep defaults that match what you usually run.
5. Add a short `recipes/<model-name>/README.md` describing what the model
   is, what variants exist, and any caveats (NUMA pinning, gated repo,
   nightly image, etc).
6. Test with `--dry-run` first.
