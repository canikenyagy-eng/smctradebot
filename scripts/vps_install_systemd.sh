#!/usr/bin/env bash
set -Eeuo pipefail

APP_USER="${SMC_APP_USER:-tradebot}"
APP_GROUP="${SMC_APP_GROUP:-$APP_USER}"
APP_DIR="${SMC_APP_DIR:-/opt/smc-signal-engine}"
HEALTHCHECK_INTERVAL="${SMC_HEALTHCHECK_INTERVAL:-5min}"
FORWARD_REPORT_ONCALENDAR="${SMC_FORWARD_REPORT_ONCALENDAR:-Mon..Fri 21:30:00}"
START_SERVICES="${SMC_START_SERVICES:-0}"
ENABLE_SERVICES="${SMC_ENABLE_SERVICES:-1}"
UNIT_DIR="${SMC_SYSTEMD_UNIT_DIR:-/etc/systemd/system}"

log() { printf '\n[vps-systemd] %s\n' "$*"; }
run_sudo() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    sudo "$@"
  fi
}
render_template() {
  local src="$1"
  local dst="$2"
  sed \
    -e "s#{{APP_USER}}#$APP_USER#g" \
    -e "s#{{APP_GROUP}}#$APP_GROUP#g" \
    -e "s#{{APP_DIR}}#$APP_DIR#g" \
    -e "s#{{HEALTHCHECK_INTERVAL}}#$HEALTHCHECK_INTERVAL#g" \
    -e "s#{{FORWARD_REPORT_ONCALENDAR}}#$FORWARD_REPORT_ONCALENDAR#g" \
    "$src" | run_sudo tee "$dst" >/dev/null
}

if ! command -v systemctl >/dev/null 2>&1; then
  echo "systemctl not found; this script requires systemd." >&2
  exit 1
fi

log "Rendering systemd units into $UNIT_DIR"
render_template "$APP_DIR/systemd/smc-signal-engine.service.template" "$UNIT_DIR/smc-signal-engine.service"
render_template "$APP_DIR/systemd/smc-healthcheck.service.template" "$UNIT_DIR/smc-healthcheck.service"
render_template "$APP_DIR/systemd/smc-healthcheck.timer.template" "$UNIT_DIR/smc-healthcheck.timer"
render_template "$APP_DIR/systemd/smc-forward-report.service.template" "$UNIT_DIR/smc-forward-report.service"
render_template "$APP_DIR/systemd/smc-forward-report.timer.template" "$UNIT_DIR/smc-forward-report.timer"

run_sudo systemctl daemon-reload

if [ "$ENABLE_SERVICES" = "1" ]; then
  log "Enabling systemd units"
  run_sudo systemctl enable smc-signal-engine.service
  run_sudo systemctl enable smc-healthcheck.timer
  run_sudo systemctl enable smc-forward-report.timer
fi

if [ "$START_SERVICES" = "1" ]; then
  log "Starting/restarting services"
  run_sudo systemctl restart smc-signal-engine.service
  run_sudo systemctl restart smc-healthcheck.timer
  run_sudo systemctl restart smc-forward-report.timer
else
  log "Services installed but not started. Set SMC_START_SERVICES=1 to start them."
fi

log "Systemd installation complete"
run_sudo systemctl --no-pager --full status smc-signal-engine.service smc-healthcheck.timer smc-forward-report.timer || true
