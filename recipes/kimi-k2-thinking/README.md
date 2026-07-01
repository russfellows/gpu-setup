# kimi-k2-thinking

Moonshot Kimi K2 Thinking, served via NVIDIA's own TensorRT-LLM release
container using their published curated config — this recipe answers "what
does NVIDIA engineering consider fully optimized on their own GPUs," not
"same model as the vLLM sweep." The model differs from
[kimi-k2.6](../kimi-k2.6): `nvidia/Kimi-K2.6-NVFP4` is only confirmed
supported on vLLM per its own HF card; `nvidia/Kimi-K2-Thinking-NVFP4` is the
only Kimi checkpoint NVIDIA has an official TensorRT-LLM deployment guide for.

Guide: https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/deployment-guide/deployment-guide-for-kimi-k2-thinking-on-trtllm.md

## Why no Triton

NVIDIA's own guide runs `trtllm-serve` directly with no Triton front end.
Triton only entered the picture for the kimi-k2.6 comparison because it
mirrors the AMD-side Triton+vLLM axis — it isn't part of what NVIDIA
considers the optimized config for this engine. Add a Triton-fronted variant
later if you specifically need a serving-layer (not raw-engine) comparison.

## Variants

| Variant          | Vendor | Stack     | Model                                                                                   | Notes                                    |
|------------------|--------|-----------|------------------------------------------------------------------------------------------|-------------------------------------------|
| `nvidia_trtllm`  | NVIDIA | TensorRT-LLM | [`nvidia/Kimi-K2-Thinking-NVFP4`](https://huggingface.co/nvidia/Kimi-K2-Thinking-NVFP4) | Uses NVIDIA's curated `kimi-k2-thinking.yaml`, baked into the release image. 8-way EP + 8-way DP, needs 8 GPUs. |

## Prerequisites: this needs a different pod image

Unlike the vLLM variants (installed into `/workspace/venv` on top of a
generic dev container), `trtllm-serve` and its matched CUDA/TensorRT stack
ship only in NVIDIA's release container — there's no reliable pip route on
this driver (TensorRT-LLM's pip wheel currently wants CUDA Toolkit 13.1;
this repo has run on CUDA 13.0 nodes). You must relaunch the RunPod pod
itself from that image before running this recipe:

- **Image**: `nvcr.io/nvidia/tensorrt-llm/release:1.2.1`
  (CUDA 12.8.1 — safe on any driver ≥ 525; well under the CUDA-13.1/driver-590
  cliff that affects newer Triton Inference Server tags, so it isn't a
  concern for this image specifically, but re-check `nvidia-smi` on
  whatever node you use).
- **Ports**: expose `8000` (HTTP, OpenAI-compatible — same port convention as
  the vLLM variants).
- **Start command**: set explicitly — `bash -c "sleep infinity"` is enough;
  the recipe harness launches `trtllm-serve` itself once you run it. (If you
  want to sanity-check the server by hand first, see NVIDIA's guide for the
  direct `trtllm-serve ... --config ...` invocation and the `/health` check.)
- **Volume**: keep `/workspace` attached to the same persistent network
  volume as before — it survives the image swap, so `gpu-setup`, HF model
  cache, and prior results all carry over unchanged.
- **Env vars**: forward `HF_TOKEN` (from `$HF_TOKEN_PATH` on your current
  pod) so the harness can pull the gated model; `HF_HOME` should still
  resolve to `/workspace/data/huggingface` via `environments/runpod.sh`.
- **Shared memory**: NVIDIA's guide flags host OOM risk during weight
  loading and recommends a large `/dev/shm` (`tmpfs:/dev/shm:size=640G` in
  their `docker run` example). Check RunPod's shm-size setting for the pod
  template you use — if it's not configurable, watch the server log for OOM
  on first launch and downsize `--config` overrides if needed.

## Known risk: `uv run` and the wrong venv

`environments/container.sh` unconditionally sets `VIRTUAL_ENV=/workspace/venv`
(the vLLM venv from the kimi-k2.6 pod), and native-mode server launch always
wraps the command in `uv run --no-project`. `trtllm-serve` is a system
binary in the release container, not something in that venv, so this
*should* still resolve via PATH — but it hasn't been verified end to end
after the pod swap. If the server fails to start with a "command not found"
or wrong-interpreter error, try clearing `VIRTUAL_ENV` before running the
recipe:

```bash
env -u VIRTUAL_ENV ./recipes/run_recipe.sh kimi-k2-thinking nvidia_trtllm --dry-run
```

## Default sweep matrix

- TP: 8 (informational only — fixed by the curated YAML, not a live sweep
  knob; see the comment in `recipe.toml`)
- ISL,OSL: 1024/1024, 1024/8192, 8192/1024 (same shapes as kimi-k2.6, for a
  shape-behavior reference point)
- Concurrency: 4, 8, 16, 32, 64, 128, 256

Override per-run:
```bash
./recipes/run_recipe.sh kimi-k2-thinking nvidia_trtllm --conc "16 32 64"
```

## Caveats

- Needs 8 GPUs. If your node has fewer, this recipe won't run as-is — the
  parallelism is fixed inside NVIDIA's curated YAML, not something
  `server_args` can override without editing that file.
- `nvidia/Kimi-K2-Thinking-NVFP4` requires Blackwell (NVFP4 has no Hopper
  kernel support).
- Not a same-model comparison to `kimi-k2.6` — see the top of this file.
