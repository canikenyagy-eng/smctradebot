#!/usr/bin/env bash
set -Eeuo pipefail

APP_USER="${SMC_APP_USER:-tradebot}"
APP_GROUP="${SMC_APP_GROUP:-$APP_USER}"
APP_DIR="${SMC_APP_DIR:-/opt/smc-signal-engine}"
REPO_URL="${SMC_REPO_URL:-https://github.com/canikenyagy-eng/smctradebot.git}"
BRANCH="${SMC_BRANCH:-main}"
PYTHON_BIN="${SMC_PYTHON_BIN:-python3}"

log() { printf '\n[vps-bootstrap] %s\n' "$*"; }
run_sudo() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    sudo "$@"
  fi
}
run_as_app() {
  sudo -u "$APP_USER" "$@"
}

if ! command -v apt-get >/dev/null 2>&1; then
  echo "This bootstrap script supports Ubuntu/Debian hosts with apt-get." >&2
  exit 1
fi

log "Installing OS packages"
run_sudo apt-get update
run_sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ca-certificates \
  curl \
  git \
  python3 \
  python3-pip \
  python3-venv \
  build-essential \
  rsync \
  jq

if ! id "$APP_USER" >/dev/null 2>&1; then
  log "Creating app user: $APP_USER"
  run_sudo useradd --create-home --shell /bin/bash "$APP_USER"
fi

log "Preparing app directory: $APP_DIR"
run_sudo mkdir -p "$APP_DIR"
run_sudo chown -R "$APP_USER:$APP_GROUP" "$APP_DIR"

if [ -d "$APP_DIR/.git" ]; then
  log "Repository already exists; updating $BRANCH"
  run_as_app git -C "$APP_DIR" fetch origin "$BRANCH"
  run_as_app git -C "$APP_DIR" checkout "$BRANCH"
  run_as_app git -C "$APP_DIR" pull --ff-only origin "$BRANCH"
else
  if [ "$(find "$APP_DIR" -mindepth 1 -maxdepth 1 | wc -l | tr -d ' ')" != "0" ]; then
    echo "APP_DIR is not empty and is not a git repo: $APP_DIR" >&2
    echo "Set SMC_APP_DIR to an empty directory or move existing files aside." >&2
    exit 1
  fi
  log "Cloning repository: $REPO_URL ($BRANCH)"
  if ! run_as_app git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"; then
    cat >&2 <<MSG

Git clone failed. If the GitHub repository is private, use one of these options:
  1. Configure an SSH deploy key on the VPS and set:
     SMC_REPO_URL=git@github.com:canikenyagy-eng/smctradebot.git
  2. Clone manually with your preferred GitHub auth method, then rerun this script.

MSG
    exit 1
  fi
fi

log "Creating Python virtualenv"
run_as_app "$PYTHON_BIN" -m venv "$APP_DIR/.venv"
run_as_app "$APP_DIR/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
run_as_app "$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt"

log "Preparing runtime directories"
run_as_app mkdir -p \
  "$APP_DIR/logs" \
  "$APP_DIR/reports" \
  "$APP_DIR/data/cache/ohlcv" \
  "$APP_DIR/data/live_bars/itick"

if [ ! -f "$APP_DIR/.env" ]; then
  if [ -f "$APP_DIR/.env.vps.example" ]; then
    log "Creating .env from .env.vps.example"
    run_as_app cp "$APP_DIR/.env.vps.example" "$APP_DIR/.env"
  else
    log "Creating .env from .env.example"
    run_as_app cp "$APP_DIR/.env.example" "$APP_DIR/.env"
  fi
  run_sudo chmod 600 "$APP_DIR/.env"
  run_sudo chown "$APP_USER:$APP_GROUP" "$APP_DIR/.env"
fi

log "Bootstrap complete"
cat <<MSG

Next steps on VPS:
  cd $APP_DIR
  nano .env
  scripts/vps_deploy.sh
  SMC_START_SERVICES=1 scripts/vps_deploy.sh

MSG
