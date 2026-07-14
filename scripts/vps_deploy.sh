#!/usr/bin/env bash
set -Eeuo pipefail

APP_USER="${SMC_APP_USER:-tradebot}"
APP_GROUP="${SMC_APP_GROUP:-$APP_USER}"
APP_DIR="${SMC_APP_DIR:-/opt/smc-signal-engine}"
BRANCH="${SMC_BRANCH:-main}"
SKIP_GIT_PULL="${SMC_SKIP_GIT_PULL:-0}"
SKIP_ENV_CHECK="${SMC_SKIP_ENV_CHECK:-0}"
INSTALL_SYSTEMD="${SMC_INSTALL_SYSTEMD:-1}"

log() { printf '\n[vps-deploy] %s\n' "$*"; }
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

if [ ! -d "$APP_DIR" ]; then
  echo "APP_DIR does not exist: $APP_DIR" >&2
  echo "Run scripts/vps_bootstrap.sh first or set SMC_APP_DIR." >&2
  exit 1
fi

cd "$APP_DIR"

if [ "$SKIP_GIT_PULL" != "1" ] && [ -d .git ]; then
  log "Updating git branch: $BRANCH"
  run_as_app git -C "$APP_DIR" fetch origin "$BRANCH"
  run_as_app git -C "$APP_DIR" checkout "$BRANCH"
  run_as_app git -C "$APP_DIR" pull --ff-only origin "$BRANCH"
fi

log "Ensuring runtime directories"
run_as_app mkdir -p logs reports data/cache/ohlcv data/live_bars/itick
run_sudo chown -R "$APP_USER:$APP_GROUP" logs reports data

if [ ! -x .venv/bin/python ]; then
  log "Creating missing virtualenv"
  run_as_app python3 -m venv .venv
fi

log "Installing Python dependencies"
run_as_app .venv/bin/python -m pip install --upgrade pip setuptools wheel
run_as_app .venv/bin/pip install -r requirements.txt

log "Running compile checks"
run_as_app .venv/bin/python -m py_compile \
  config.py \
  main.py \
  research/live_health_check.py \
  research/daily_live_forward_report.py \
  data/market_data.py \
  data/live_bar_provider.py \
  services/telegram.py

if [ "$SKIP_ENV_CHECK" != "1" ]; then
  log "Validating .env via Settings.from_env()"
  run_as_app .venv/bin/python - <<'PY'
from config import Settings
settings = Settings.from_env()
print("Settings OK")
print(f"pairs={','.join(settings.pairs)} data_source={settings.data_source} live_mode={settings.live_mode} scan_interval={settings.scan_interval_minutes}m")
print(f"forward_journal={settings.enable_forward_journal} feed_safe_mode={settings.enable_feed_safe_mode} health_alerts={settings.enable_health_alerts}")
PY
fi

if [ "$INSTALL_SYSTEMD" = "1" ]; then
  log "Installing systemd units"
  "$APP_DIR/scripts/vps_install_systemd.sh"
fi

log "Deploy complete"
cat <<MSG

Useful commands:
  scripts/vps_status.sh
  sudo systemctl restart smc-signal-engine
  sudo journalctl -u smc-signal-engine -f
  sudo systemctl list-timers 'smc-*'

MSG
