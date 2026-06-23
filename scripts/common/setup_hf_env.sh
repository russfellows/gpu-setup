#!/usr/bin/env bash
# ==============================================================================
# Hugging Face cache + env-var setup, system-wide.
#
# Writes /etc/profile.d/huggingface.sh so every interactive login shell gets
# HF_HOME pointing at the bulk-storage mount. Creates the cache directory and
# hands ownership to the invoking user.
#
# Two topologies are supported:
#   A) Dedicated data volume (typical):
#      A separate NVMe array is mounted at STORAGE_MOUNT (/mnt/data).
#      HF_HOME_PATH defaults to /mnt/data/huggingface.
#   B) Root IS the bulk storage (e.g. NVMe RAID-0 mounted at /):
#      If STORAGE_MOUNT is not present but root has >= ROOT_MIN_GB free,
#      the script auto-switches to HF_HOME_PATH=/data/huggingface on root.
#      Override HF_HOME_PATH to use a different path.
#
# Env vars:
#   HF_HOME_PATH     Cache root (default /mnt/data/huggingface)
#   STORAGE_MOUNT    Mount that must be present (default /mnt/data)
#   ROOT_MIN_GB      Minimum free GB on / to accept root fallback (default 500)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

HF_HOME_PATH="${HF_HOME_PATH:-/mnt/data/huggingface}"
STORAGE_MOUNT="${STORAGE_MOUNT:-/mnt/data}"
ROOT_MIN_GB="${ROOT_MIN_GB:-500}"

need_root

REAL_USER="$(real_user)"
REAL_GROUP="$(id -gn "$REAL_USER")"

# ---------- Verify bulk storage is available ----------
if findmnt -n "$STORAGE_MOUNT" >/dev/null 2>&1; then
  ok "$STORAGE_MOUNT is mounted."
else
  # Fall back to root filesystem if it has enough space (topology B).
  ROOT_FREE_GB=$(df -BG / | awk 'NR==2 {gsub("G","",$4); print $4}')
  if [ "$ROOT_FREE_GB" -ge "$ROOT_MIN_GB" ]; then
    warn "$STORAGE_MOUNT is not mounted, but root has ${ROOT_FREE_GB} GB free (>= ${ROOT_MIN_GB} GB threshold)."
    warn "Using root filesystem for HF cache (topology B: root IS bulk storage)."
    STORAGE_MOUNT="/"
    # Only override HF_HOME_PATH if the caller didn't already set it to
    # something other than the /mnt/data default.
    if [ "$HF_HOME_PATH" = "/mnt/data/huggingface" ]; then
      HF_HOME_PATH="/data/huggingface"
    fi
    ok "Auto-selected HF_HOME_PATH=$HF_HOME_PATH"
  else
    die "$STORAGE_MOUNT is not mounted and root has only ${ROOT_FREE_GB} GB free (< ${ROOT_MIN_GB} GB). Run scripts/common/setup_storage.sh --execute first, or override STORAGE_MOUNT/HF_HOME_PATH."
  fi
fi

# ---------- Create cache dir ----------
if [ -d "$HF_HOME_PATH" ]; then
  ok "$HF_HOME_PATH already exists."
else
  log "Creating $HF_HOME_PATH ..."
  install -d -m 0775 "$HF_HOME_PATH"
fi
log "Setting ownership of $HF_HOME_PATH to $REAL_USER:$REAL_GROUP ..."
chown -R "$REAL_USER:$REAL_GROUP" "$HF_HOME_PATH"

# ---------- Write /etc/profile.d/huggingface.sh ----------
PROFILE="/etc/profile.d/huggingface.sh"
log "Writing $PROFILE ..."
cat >"$PROFILE" <<EOF
# Managed by gpu-setup: scripts/common/setup_hf_env.sh
# Hugging Face caches and downloads land on bulk storage, not the root FS.
export HF_HOME="${HF_HOME_PATH}"
export HUGGINGFACE_HUB_CACHE="\$HF_HOME/hub"
export HF_HUB_ENABLE_HF_TRANSFER=1
# Recipes mount HF_HOME into containers at /root/.cache/huggingface
EOF
chmod 0644 "$PROFILE"

ok "HF env configured. New shells will see HF_HOME=$HF_HOME_PATH."

cat <<EOF

==============================================================================
 Next steps (run as $REAL_USER, not root):
   1. Re-source the new profile (or open a new shell):
        source $PROFILE
   2. Authenticate against the Hub:
        hf auth login
   3. Confirm a download lands under $HF_HOME_PATH:
        hf download <small-model-id>
==============================================================================
EOF
