#!/usr/bin/env bash
set -Eeuo pipefail

VPS_HOST="${SMC_VPS_HOST:-tradebot@45.137.153.215}"
SSH_KEY="${SMC_SSH_KEY:-$HOME/.ssh/smc_vps}"
APP_USER="${SMC_APP_USER:-tradebot}"
APP_GROUP="${SMC_APP_GROUP:-$APP_USER}"
APP_DIR="${SMC_APP_DIR:-/opt/smc-signal-engine}"
REMOTE_STAGING="${SMC_REMOTE_STAGING:-/tmp/smc-signal-engine-sync}"
SYNC_ENV="${SMC_SYNC_ENV:-0}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log() { printf '\n[vps-sync] %s\n' "$*"; }

if [ ! -f "$SSH_KEY" ]; then
  echo "SSH key not found: $SSH_KEY" >&2
  exit 1
fi
if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required on the local machine." >&2
  exit 1
fi

SSH_CMD=(ssh -o IdentitiesOnly=yes -i "$SSH_KEY")
RSYNC_RSH="ssh -o IdentitiesOnly=yes -i $SSH_KEY"

log "Checking SSH access to $VPS_HOST"
"${SSH_CMD[@]}" "$VPS_HOST" "echo SSH_OK && hostname"

log "Ensuring rsync exists on the VPS"
"${SSH_CMD[@]}" "$VPS_HOST" "command -v rsync >/dev/null 2>&1 || (sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y rsync)"

log "Preparing remote staging directory: $REMOTE_STAGING"
"${SSH_CMD[@]}" "$VPS_HOST" "rm -rf '$REMOTE_STAGING' && mkdir -p '$REMOTE_STAGING'"

EXCLUDES=(
  --exclude '.git/'
  --exclude '.venv/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '.DS_Store'
  --exclude 'logs/'
  --exclude 'reports/'
  --exclude 'data/cache/'
  --exclude 'data/live_bars/'
)
if [ "$SYNC_ENV" != "1" ]; then
  EXCLUDES+=(--exclude '.env')
fi

log "Syncing local project to remote staging"
rsync -az --delete \
  "${EXCLUDES[@]}" \
  -e "$RSYNC_RSH" \
  "$ROOT_DIR/" \
  "$VPS_HOST:$REMOTE_STAGING/"

log "Promoting staged files to $APP_DIR"
"${SSH_CMD[@]}" "$VPS_HOST" "\
  sudo mkdir -p '$APP_DIR' && \
  sudo rsync -a --delete --exclude '.env' '$REMOTE_STAGING/' '$APP_DIR/' && \
  sudo chown -R '$APP_USER:$APP_GROUP' '$APP_DIR' && \
  sudo chmod +x '$APP_DIR'/scripts/vps_*.sh\
"

log "Sync complete"
cat <<MSG

Next commands on VPS:
  cd $APP_DIR
  sudo SMC_SKIP_GIT_PULL=1 scripts/vps_deploy.sh
  sudo SMC_SKIP_GIT_PULL=1 SMC_START_SERVICES=1 scripts/vps_deploy.sh

MSG
