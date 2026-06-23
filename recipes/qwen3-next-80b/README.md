# qwen3-next-80b

[`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct-FP8)
— 80B total parameters / 3B active, hybrid GDN + MoE, FP8-quantized.

## Variants

| Variant            | Vendor | Stack | Notes                                                                      |
|--------------------|--------|-------|----------------------------------------------------------------------------|
| `amd_vllm`         | AMD    | vLLM  | No NUMA pinning. Sweeps TP 1, 2, 4. Default.                              |
| `amd_vllm_numa`    | AMD    | vLLM  | **TP=4 only**, NUMA-pinned to NUMA 0 (reference host). Reproduces source. |
| `amd_atom`         | AMD    | ATOM  | `rocm/atom-dev:vllm-v0.22.0-nightly_20260617`                              |
| `nvidia_vllm`      | NVIDIA | vLLM  | `vllm/vllm-openai:v0.22.1`                                                 |

All four serve an OpenAI-compatible endpoint on port 8000. `amd_atom` runs
its bench client via `atom.benchmarks.benchmark_serving`; the other three
use `vllm bench serve`.

### `amd_vllm_numa` — source-faithful reference

This variant locks `sweep_tp = [4]` and bakes in the cpuset/HIP_VISIBLE_DEVICES
values that match the reference dual-socket MI355X host described in the
source guidance:

| TP | HIP_VISIBLE_DEVICES | `--cpuset-cpus`    | `--cpuset-mems` |
|----|---------------------|--------------------|-----------------|
| 4  | `0,1,2,3`           | `0-63,128-191`     | `0`             |

If your host has a different topology, run `rocm-smi --showtopo` and edit
the values in `recipe.toml` (under `[variants.amd_vllm_numa.env]` and
`docker_flags`) before using this variant — the cpuset will otherwise
mis-pin and skew results.

For TP=1 or TP=2 reference configurations, use `amd_vllm` and pass the
relevant cpuset/HIP_VISIBLE_DEVICES values yourself; the reference values
are different per TP (see the table further down).

## Default sweep matrix

- TP: 1, 2, 4
- ISL,OSL: 1000/100, 5000/500, 10000/1000
- Concurrency: 4, 8, 16, 32, 64, 128, 256

The 100k-ISL long-context sweep from the source guidance is omitted from the
default because it dominates runtime; add it when you want it:

```bash
./recipes/run_recipe.sh qwen3-next-80b amd_vllm \
    --shapes "1000,100 5000,500 10000,1000 100000,1000" \
    --conc "1 2 4 8 16 32 64"
```

## NUMA pinning (AMD)

On a dual-socket MI355X box, pin the container to the NUMA node local to
the GPUs you're using. Get the GPU-to-NUMA mapping with:

```bash
rocm-smi --showtopo
```

Then add `--cpuset-cpus` and `--cpuset-mems` to the container's run flags
by editing `[variants.<name>.docker_flags]` in `recipe.toml`, e.g.:

```toml
[variants.amd_vllm]
docker_flags = [
  "--cpuset-cpus=0-63,128-191",
  "--cpuset-mems=0",
  "-e", "HIP_VISIBLE_DEVICES=0,1,2,3",
  "-e", "CUDA_VISIBLE_DEVICES=0,1,2,3",
]
```

Reference values for the topology described in the source guidance (GPUs
0–3 on NUMA 0, GPUs 4–7 on NUMA 1):

| TP | HIP_VISIBLE_DEVICES | NUMA node | `--cpuset-cpus`     | `--cpuset-mems` |
|----|---------------------|-----------|---------------------|-----------------|
| 1  | `6`                 | 1         | `64-127,192-255`    | `1`             |
| 2  | `4,5`               | 1         | `64-127,192-255`    | `1`             |
| 4  | `0,1,2,3`           | 0         | `0-63,128-191`      | `0`             |

These are host-specific; verify against your own `rocm-smi --showtopo`
before using them.

## Async scheduling on AMD vLLM

The `amd_vllm` variant uses `--async-scheduling` because it's required for
TP > 1. If you sweep TP = 1 specifically, set `--no-async-scheduling`
instead (edit `server_args` in `recipe.toml`). For mixed TP sweeps, leave
async scheduling on — TP = 1 still works, it's just not the configuration
the source guidance recommends for that case.

## NVIDIA target architecture

`vllm/vllm-openai:v0.22.1` defaults to whatever `TORCH_CUDA_ARCH_LIST` is
baked into the image. For Blackwell (B200) the source guidance used 10.0;
add `TORCH_CUDA_ARCH_LIST = "10.0"` to `[variants.nvidia_vllm.env]` if
you need to force it.
