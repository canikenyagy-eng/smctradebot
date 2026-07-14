#!/usr/bin/env bash
set -Eeuo pipefail

APP_DIR="${SMC_APP_DIR:-/opt/smc-signal-engine}"
LINES="${SMC_STATUS_LINES:-80}"

printf '\n== systemd status ==\n'
sudo systemctl --no-pager --full status smc-signal-engine.service smc-healthcheck.timer smc-forward-report.timer || true

printf '\n== timers ==\n'
sudo systemctl list-timers 'smc-*' --no-pager || true

printf '\n== latest signal engine logs ==\n'
sudo journalctl -u smc-signal-engine.service -n "$LINES" --no-pager || true

printf '\n== latest healthcheck logs ==\n'
sudo journalctl -u smc-healthcheck.service -n 40 --no-pager || true

printf '\n== latest forward report logs ==\n'
sudo journalctl -u smc-forward-report.service -n 40 --no-pager || true

printf '\n== runtime files ==\n'
if [ -d "$APP_DIR" ]; then
  ls -lh "$APP_DIR"/logs "$APP_DIR"/reports 2>/dev/null || true
  test -f "$APP_DIR/logs/live_heartbeat.json" && cat "$APP_DIR/logs/live_heartbeat.json" || true
else
  echo "APP_DIR not found: $APP_DIR"
fi
