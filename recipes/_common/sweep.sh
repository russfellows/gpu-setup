#!/usr/bin/env bash
# ==============================================================================
# Recipe sweep harness.
#
# Sourced by run_recipe.sh after load_recipe.py has eval'd the recipe TOML.
# Provides one entrypoint, `run_sweep`, which:
#   1. Builds the image from EXTRA_BUILD_DOCKERFILE if declared, else pulls
#      IMAGE if not already local.
#   2. Creates a timestamped results dir under $HOME/results/...
#   3. Iterates the (TP) x (ISL,OSL) x (CONC) matrix.
#   4. For each combo: launches a container with the vendor-standard flag
#      bundle + recipe-supplied extras + EXTRA_FILES mounted at /recipe/,
#      waits for the server, runs the bench client via docker exec, tears
#      the container down.
#   5. Aggregates a summary.csv.
#
# Variables sweep.sh expects (populated by load_recipe.py from the TOML):
#   MODEL_NAME, VARIANT_NAME, VENDOR, STACK, IMAGE, MODEL_ID, RECIPE_DIR
#   SERVER_CMD (array, with @TP@/@ISL@/@OSL@/@CONC@ placeholders)
#   PORT (int)
#   EXTRA_DOCKER_ENV (array, -e KEY=VAL pairs)
#   EXTRA_DOCKER_FLAGS (array)
#   EXTRA_FILES (array — files in RECIPE_DIR to mount at /recipe/<basename>)
#   Optional: BENCH_TOOL, READY_MARKER, READY_TIMEOUT_S, RANDOM_RANGE_RATIO
#   Optional (when build is declared): EXTRA_BUILD_DOCKERFILE,
#     EXTRA_BUILD_CONTEXT, EXTRA_BUILD_TAG, BASE_IMAGE
#   Sweep defaults: SWEEP_TP_DEFAULT, SWEEP_ISL_OSL_DEFAULT, SWEEP_CONC_DEFAULT
#   Sweep CLI overrides: SWEEP_TP, SWEEP_ISL_OSL, SWEEP_CONC (set by CLI parser)
# ==============================================================================

# Sourced — don't `set -e` globally; let the caller decide.

_SWEEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../scripts/lib/common.sh
source "${_SWEEP_DIR}/../../scripts/lib/common.sh"
# shellcheck source=docker_amd.sh
source "${_SWEEP_DIR}/docker_amd.sh"
# shellcheck source=docker_nvidia.sh
source "${_SWEEP_DIR}/docker_nvidia.sh"
# shellcheck source=bench_client.sh
source "${_SWEEP_DIR}/bench_client.sh"

_default_ready_marker() {
  case "$1" in
    vllm|sglang|atom|trtllm) echo "Application startup complete" ;;
    triton)                   echo "Started GRPCInferenceService" ;;
    *)                        echo "Application startup complete" ;;
  esac
}

_default_bench_tool() {
  case "$1" in
    atom) echo "atom" ;;
    *)    echo "vllm" ;;
  esac
}

_require_var() {
  local n="$1"
  if [ -z "${!n:-}" ]; then
    err "Required variable '$n' not set."
    return 1
  fi
}

# Substitute @TP@, @ISL@, @OSL@, @CONC@ in an array.
_subst_placeholders() {
  local tp="$1" isl="$2" osl="$3" conc="$4"; shift 4
  local x out=()
  for x in "$@"; do
    x="${x//@TP@/$tp}"
    x="${x//@ISL@/$isl}"
    x="${x//@OSL@/$osl}"
    x="${x//@CONC@/$conc}"
    out+=("$x")
  done
  printf '%s\n' "${out[@]}"
}

run_sweep() {
  # ---------- Validate ----------
  for v in MODEL_NAME VARIANT_NAME VENDOR STACK IMAGE MODEL_ID; do
    _require_var "$v" || return 2
  done
  if [ -z "${SERVER_CMD+x}" ] || [ "${#SERVER_CMD[@]}" -eq 0 ]; then
    err "SERVER_CMD array is empty."
    return 2
  fi

  # ---------- Resolve sweep matrix (CLI > TOML default > hardcoded fallback) ----------
  : "${SWEEP_TP:=${SWEEP_TP_DEFAULT:-1}}"
  : "${SWEEP_ISL_OSL:=${SWEEP_ISL_OSL_DEFAULT:-1024,1024}}"
  : "${SWEEP_CONC:=${SWEEP_CONC_DEFAULT:-4 8 16 32 64 128 256}}"
  : "${RANDOM_RANGE_RATIO:=0.9}"
  : "${READY_MARKER:=$(_default_ready_marker "$STACK")}"
  : "${READY_TIMEOUT_S:=1800}"
  : "${BENCH_TOOL:=$(_default_bench_tool "$STACK")}"
  : "${PORT:=8000}"
  : "${DRY_RUN:=0}"
  HOST="${HOST:-localhost}"

  IFS=$' \t\n' read -r -a _TP_ARR    <<< "$SWEEP_TP"
  IFS=$' \t\n' read -r -a _ISLOSL_ARR<<< "$SWEEP_ISL_OSL"
  IFS=$' \t\n' read -r -a _CONC_ARR  <<< "$SWEEP_CONC"

  # ---------- Vendor flag bundle ----------
  local -a VENDOR_FLAGS
  case "$VENDOR" in
    amd)    VENDOR_FLAGS=("${AMD_DOCKER_FLAGS[@]}") ;;
    nvidia) VENDOR_FLAGS=("${NVIDIA_DOCKER_FLAGS[@]}") ;;
    *) err "Unknown VENDOR='$VENDOR'"; return 2 ;;
  esac

  # ---------- HF cache mount ----------
  local HF_HOST_DIR="${HF_HOME:-/mnt/data/huggingface}"
  if [ ! -d "$HF_HOST_DIR" ]; then
    if [ "$DRY_RUN" = "1" ]; then
      warn "HF cache dir $HF_HOST_DIR does not exist (would be created on real run)."
    else
      warn "HF cache dir $HF_HOST_DIR does not exist — creating."
      mkdir -p "$HF_HOST_DIR" \
        || die "Could not create $HF_HOST_DIR. Run scripts/common/setup_hf_env.sh first, or set HF_HOME to a writable path."
    fi
  fi
  local -a HF_MOUNT=(-v "${HF_HOST_DIR}:/root/.cache/huggingface")

  # ---------- Torch compile cache mount ----------
  # Inductor/cudagraph compilation artifacts are cached under
  # ~/.cache/torch inside the container. Mounting a persistent host
  # directory means compiled graphs survive container restarts, avoiding
  # recompilation on every run (which can take 30+ minutes for large MoE
  # models with use_inductor_graph_partition=true).
  local TORCH_CACHE_HOST="${HF_HOST_DIR}/../torch_compile_cache"
  TORCH_CACHE_HOST="$(cd "$(dirname "$TORCH_CACHE_HOST")" && pwd)/$(basename "$TORCH_CACHE_HOST")"
  if [ "$DRY_RUN" != "1" ]; then
    mkdir -p "$TORCH_CACHE_HOST"
    chmod 1777 "$TORCH_CACHE_HOST"
  fi
  local -a TORCH_CACHE_MOUNT=(-v "${TORCH_CACHE_HOST}:/root/.cache/torch")

  local -a HF_TOKEN_ENV=()
  local _hf_token="${HF_TOKEN:-}"
  if [ -z "$_hf_token" ] && [ -f "${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}" ]; then
    _hf_token="$(cat "${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}")"
  fi
  [ -n "$_hf_token" ] && HF_TOKEN_ENV=(-e "HF_TOKEN=${_hf_token}")

  # ---------- Extra files mount: RECIPE_DIR/<file> -> /recipe/<basename> ----------
  local -a FILE_MOUNTS=()
  if [ "${#EXTRA_FILES[@]}" -gt 0 ] && [ -n "${RECIPE_DIR:-}" ]; then
    for f in "${EXTRA_FILES[@]}"; do
      local src="${RECIPE_DIR}/${f}"
      local base
      base="$(basename "$f")"
      if [ ! -e "$src" ]; then
        err "extra_files entry not found: $src"
        return 2
      fi
      FILE_MOUNTS+=(-v "${src}:/recipe/${base}:ro")
    done
  fi

  # ---------- Results dir ----------
  local TS
  TS="$(date +%Y%m%d_%H%M%S)"
  local USER_HOME
  USER_HOME="$(getent passwd "${SUDO_USER:-$USER}" | cut -d: -f6)"
  : "${RESULTS_DIR:=${USER_HOME}/results/${MODEL_NAME}/${VARIANT_NAME}/${TS}}"
  if [ "$DRY_RUN" != "1" ]; then
    mkdir -p "$RESULTS_DIR"
  fi
  log "Results dir: $RESULTS_DIR"

  # ---------- Runtime config: materialize the [runtime_config] TOML table as
  # JSON in the results dir, mount it where the server expects it. JSON is
  # valid YAML, so YAML-expecting tools (trtllm-serve etc.) accept it.
  if [ -n "${RUNTIME_CONFIG_JSON:-}" ] && [ -n "${RUNTIME_CONFIG_PATH:-}" ]; then
    if [ "$DRY_RUN" != "1" ]; then
      printf '%s' "$RUNTIME_CONFIG_JSON" > "${RESULTS_DIR}/runtime_config.json"
      log "Wrote runtime config: ${RESULTS_DIR}/runtime_config.json -> ${RUNTIME_CONFIG_PATH}"
    else
      log "(dry-run) would write runtime config to ${RESULTS_DIR}/runtime_config.json -> ${RUNTIME_CONFIG_PATH}"
    fi
    FILE_MOUNTS+=(-v "${RESULTS_DIR}/runtime_config.json:${RUNTIME_CONFIG_PATH}:ro")
  fi

  # ---------- Build (optional) or pull ----------
  if [ -n "${EXTRA_BUILD_DOCKERFILE:-}" ]; then
    local df_path="${RECIPE_DIR}/${EXTRA_BUILD_DOCKERFILE}"
    local ctx="${RECIPE_DIR}/${EXTRA_BUILD_CONTEXT:-.}"
    if [ ! -f "$df_path" ]; then
      err "Dockerfile not found: $df_path"
      return 2
    fi
    if [ "$DRY_RUN" = "1" ]; then
      log "(dry-run) would build $IMAGE from $df_path"
      if [ -n "${EXTRA_BUILD_ARGS+x}" ] && [ "${#EXTRA_BUILD_ARGS[@]}" -gt 0 ]; then
        log "(dry-run) build args: ${EXTRA_BUILD_ARGS[*]}"
      fi
    elif docker image inspect "$IMAGE" >/dev/null 2>&1; then
      ok "Built image already present: $IMAGE (skipping rebuild)"
    else
      log "Building $IMAGE from $df_path ..."
      docker build -f "$df_path" -t "$IMAGE" \
        ${EXTRA_BUILD_ARGS[@]+"${EXTRA_BUILD_ARGS[@]}"} \
        "$ctx" \
        || { err "Build failed."; return 3; }
    fi
  else
    if [ "$DRY_RUN" = "1" ]; then
      log "(dry-run) would ensure image present: $IMAGE"
    elif ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
      log "Pulling $IMAGE ..."
      docker pull "$IMAGE" || { err "Pull failed."; return 3; }
    fi
  fi

  # ---------- Plan ----------
  local total=$(( ${#_TP_ARR[@]} * ${#_ISLOSL_ARR[@]} * ${#_CONC_ARR[@]} ))
  cat <<EOF

==============================================================================
 SWEEP PLAN
==============================================================================
 Model       : $MODEL_ID
 Variant     : $VARIANT_NAME ($VENDOR / $STACK)
 Image       : $IMAGE
 TP sizes    : ${_TP_ARR[*]}
 ISL,OSL     : ${_ISLOSL_ARR[*]}
 Concurrency : ${_CONC_ARR[*]}
 Bench tool  : $BENCH_TOOL
 Ready mark  : $READY_MARKER
 Port        : $PORT
 Results     : $RESULTS_DIR
 Combos      : $total
EOF
  [ "${#EXTRA_FILES[@]}" -gt 0 ] && echo " Extra files : ${EXTRA_FILES[*]}  (mounted at /recipe/)"
  [ "${#EXTRA_DOCKER_ENV[@]}" -gt 0 ] && echo " Extra env   : ${EXTRA_DOCKER_ENV[*]}"
  echo "=============================================================================="
  echo

  if [ "$DRY_RUN" = "1" ]; then
    log "DRY_RUN=1 — exiting without running."
    return 0
  fi

  # ---------- Provenance ----------
  # Records exact image digest, base image (for builds), build args, sweep
  # matrix, the full recipe.toml content, and host/GPU info. Apples-to-apples
  # result comparisons across runs depend on this.
  if [ "$DRY_RUN" = "1" ]; then
    log "(dry-run) would write ${RESULTS_DIR}/provenance.json"
  else
  PROV_RESULTS_DIR="$RESULTS_DIR" \
  PROV_TIMESTAMP="$TS" \
  PROV_MODEL_NAME="$MODEL_NAME" \
  PROV_VARIANT_NAME="$VARIANT_NAME" \
  PROV_MODEL_ID="$MODEL_ID" \
  PROV_VENDOR="$VENDOR" \
  PROV_STACK="$STACK" \
  PROV_IMAGE="$IMAGE" \
  PROV_BASE_IMAGE="${BASE_IMAGE:-}" \
  PROV_DOCKERFILE="${EXTRA_BUILD_DOCKERFILE:+${RECIPE_DIR}/${EXTRA_BUILD_DOCKERFILE}}" \
  PROV_BUILD_ARGS="${PROV_BUILD_ARGS:-}" \
  PROV_SWEEP_TP="${_TP_ARR[*]}" \
  PROV_SWEEP_ISL_OSL="${_ISLOSL_ARR[*]}" \
  PROV_SWEEP_CONC="${_CONC_ARR[*]}" \
  PROV_RECIPE_TOML="${RECIPE_TOML:-${RECIPE_DIR}/recipe.toml}" \
  python3 "${_SWEEP_DIR}/write_provenance.py" >/dev/null \
    && ok "Wrote ${RESULTS_DIR}/provenance.json" \
    || warn "Provenance write failed (continuing)."
  fi

  # ---------- Loop ----------
  local SUMMARY="${RESULTS_DIR}/summary.csv"
  echo "tp,isl,osl,conc,status,result_file" > "$SUMMARY"

  local rc_total=0
  for TP in "${_TP_ARR[@]}"; do
    for P in "${_ISLOSL_ARR[@]}"; do
      ISL="${P%,*}"; OSL="${P#*,}"
      for CONC in "${_CONC_ARR[@]}"; do
        local RESULT_FILENAME="${MODEL_NAME}_${VARIANT_NAME}_tp${TP}_isl${ISL}_osl${OSL}_c${CONC}.json"
        local CONTAINER_NAME="recipe_${MODEL_NAME//./-}_${VARIANT_NAME}_$$"
        local LOG_FILE="${RESULTS_DIR}/server_tp${TP}_isl${ISL}_osl${OSL}_c${CONC}.log"

        mapfile -t _SRV_CMD < <(_subst_placeholders "$TP" "$ISL" "$OSL" "$CONC" "${SERVER_CMD[@]}")
        # Re-quote each token so bash -lc receives a shell-safe single string.
        local _SRV_CMD_STR
        _SRV_CMD_STR=$(printf '%q ' "${_SRV_CMD[@]}")

        log "---- Run: tp=$TP isl=$ISL osl=$OSL conc=$CONC ----"

        docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
        # Guarantee cleanup even on SIGINT/SIGTERM or script exit.
        trap "docker rm -f '$CONTAINER_NAME' >/dev/null 2>&1 || true" EXIT INT TERM

        # Launch detached (no --rm: we need to capture logs after exit).
        docker run -d --name "$CONTAINER_NAME" \
          "${VENDOR_FLAGS[@]}" \
          "${HF_MOUNT[@]}" \
          "${TORCH_CACHE_MOUNT[@]}" \
          "${FILE_MOUNTS[@]}" \
          -v "${RESULTS_DIR}:/results" \
          "${HF_TOKEN_ENV[@]}" \
          ${EXTRA_DOCKER_ENV[@]+"${EXTRA_DOCKER_ENV[@]}"} \
          ${EXTRA_DOCKER_FLAGS[@]+"${EXTRA_DOCKER_FLAGS[@]}"} \
          --entrypoint=/bin/bash \
          "$IMAGE" -lc "$_SRV_CMD_STR" \
          >/dev/null

        # Wait for ready marker.
        local waited=0
        local ready=0
        while [ "$waited" -lt "$READY_TIMEOUT_S" ]; do
          if docker logs "$CONTAINER_NAME" 2>&1 | grep -q "$READY_MARKER"; then
            ready=1; break
          fi
          if ! docker ps -q --filter "name=^${CONTAINER_NAME}$" | grep -q .; then
            err "Server container exited before becoming ready."
            break
          fi
          sleep 10; waited=$((waited + 10))
        done
        if [ "$ready" -ne 1 ]; then
          err "Server failed to reach '$READY_MARKER' in ${READY_TIMEOUT_S}s."
          docker logs "$CONTAINER_NAME" > "$LOG_FILE" 2>&1 || true
          docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
          echo "$TP,$ISL,$OSL,$CONC,server_timeout," >> "$SUMMARY"
          rc_total=$((rc_total + 1))
          continue
        fi
        ok "Server is ready (after ${waited}s)."

        # Remove any stale result file before the bench run. vllm bench serve
        # appends (not overwrites) when the file exists — caused by the warmup
        # phase writing an interim JSON before the final result is written.
        # Deleting here ensures each bench run produces exactly one JSON object.
        rm -f "${RESULTS_DIR}/${RESULT_FILENAME}"

        if run_bench; then
          echo "$TP,$ISL,$OSL,$CONC,ok,$RESULT_FILENAME" >> "$SUMMARY"
        else
          err "Bench client failed."
          echo "$TP,$ISL,$OSL,$CONC,bench_failed,$RESULT_FILENAME" >> "$SUMMARY"
          rc_total=$((rc_total + 1))
        fi

        docker logs "$CONTAINER_NAME" > "$LOG_FILE" 2>&1 || true
        docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
      done
    done
  done

  # Restore ownership: the bench client runs as root inside the container and
  # writes result files as uid 0. Chown the entire results dir back to the
  # invoking user so they can read/delete results without sudo.
  local _invoke_user="${SUDO_USER:-$USER}"
  chown -R "$_invoke_user" "$RESULTS_DIR" 2>/dev/null \
    || sudo chown -R "$_invoke_user" "$RESULTS_DIR" 2>/dev/null \
    || warn "chown of $RESULTS_DIR failed — result files may be root-owned."

  echo
  ok "Sweep complete. Summary: $SUMMARY"
  if [ "$rc_total" -gt 0 ]; then
    warn "$rc_total combination(s) had failures."
    return 1
  fi
  return 0
}
