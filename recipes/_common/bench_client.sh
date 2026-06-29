#!/usr/bin/env bash
# Bench client shim. Sourced by sweep.sh; provides `run_bench`.
#
# Most serving stacks (vLLM, SGLang, Triton-Inference-Server's OpenAI front,
# TRT-LLM via trtllm-serve) expose an OpenAI-compatible /v1/completions
# endpoint, so a single client tool — `vllm bench serve` — works against all
# of them. ATOM ships its own equivalent (atom.benchmarks.benchmark_serving)
# with the same flags.
#
# The client is invoked INSIDE the running server container via `docker exec`
# so we don't need any of these packages installed on the host.
#
# Required env when calling run_bench:
#   CONTAINER_NAME, MODEL_ID, BENCH_TOOL (vllm|atom), HOST, PORT,
#   ISL, OSL, CONC, RANDOM_RANGE_RATIO, RESULTS_HOST_DIR, RESULT_FILENAME
# Optional:
#   BENCH_EXTRA_ARGS (array)

run_bench() {
  # Ensure BENCH_EXTRA_ARGS is always an array, even if the caller left it unset.
  BENCH_EXTRA_ARGS=("${BENCH_EXTRA_ARGS[@]+"${BENCH_EXTRA_ARGS[@]}"}")
  local num_warmups=$(( CONC * 2 ))
  local num_prompts=$(( CONC * 10 ))

  local -a cmd
  case "$BENCH_TOOL" in
    vllm)
      cmd=(vllm bench serve)
      ;;
    atom)
      cmd=(python3 -m atom.benchmarks.benchmark_serving)
      ;;
    *)
      err "Unknown BENCH_TOOL='$BENCH_TOOL' (expected vllm|atom)"
      return 2
      ;;
  esac

  # In docker mode results land at /results (mounted from $RESULTS_DIR).
  # In native mode we write directly to $RESULTS_DIR on the host.
  local _result_dir
  if [ "${NATIVE:-0}" = "1" ]; then
    _result_dir="$RESULTS_DIR"
  else
    _result_dir="/results"
  fi

  # shellcheck disable=SC2054  # commas are inside argument values, not array separators
  cmd+=(
    --model="$MODEL_ID"
    --backend=vllm
    --base-url="http://${HOST}:${PORT}"
    --dataset-name=random
    --random-input-len="$ISL"
    --random-output-len="$OSL"
    --random-range-ratio="${RANDOM_RANGE_RATIO:-1.0}"
    --num-prompts="$num_prompts"
    --num-warmups="$num_warmups"
    --max-concurrency="$CONC"
    --request-rate=inf
    --ignore-eos
    --save-result
    --result-dir="$_result_dir"
    --result-filename="$RESULT_FILENAME"
    --percentile-metrics=ttft,tpot,itl,e2el
    --metric-percentiles=25,50,75,90,95,99
  )
  if [ "${#BENCH_EXTRA_ARGS[@]}" -gt 0 ]; then
    cmd+=("${BENCH_EXTRA_ARGS[@]}")
  fi

  log "Bench: ISL=$ISL OSL=$OSL CONC=$CONC -> $RESULT_FILENAME"
  if [ "${NATIVE:-0}" = "1" ]; then
    PYTHONUNBUFFERED=1 uv run --no-project "${cmd[@]}"
  else
    docker exec "$CONTAINER_NAME" "${cmd[@]}"
  fi
}
