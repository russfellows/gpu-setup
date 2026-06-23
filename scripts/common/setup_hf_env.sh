#!/usr/bin/env bash
# ==============================================================================
# Hugging Face cache + env-var setup, system-wide.
#
# Writes /etc/profile.d/huggingface.sh so every interactive login shell gets
# HF_HOME pointing at the bulk-storage mount. Creates the cache directory and
# hands ownership to the invoking user.
#
# Refuses to run unless the storage mount is actually mounted — better a
# clear error now than silently caching to the root filesystem and filling
# it up at the first model download.
#
# Env vars:
#   HF_HOME_PATH     Cache root (default /mnt/data/huggingface)
#   STORAGE_MOUNT    Mount that must be present (default /mnt/data)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

HF_HOME_PATH="${HF_HOME_PATH:-/mnt/data/huggingface}"
STORAGE_MOUNT="${STORAGE_MOUNT:-/mnt/data}"

need_root

REAL_USER="$(real_user)"
REAL_GROUP="$(id -gn "$REAL_USER")"

# ---------- Verify the bulk mount is actually mounted ----------
if ! findmnt -n "$STORAGE_MOUNT" >/dev/null 2>&1; then
  die "$STORAGE_MOUNT is not mounted. Run scripts/common/setup_storage.sh --execute first, or override STORAGE_MOUNT."
fi
ok "$STORAGE_MOUNT is mounted."

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
