#!/bin/zsh
set -euo pipefail

PROJECT_DIR="/Users/kanannagiev/Documents/New project/project"
PLIST_SRC="$PROJECT_DIR/launchd/com.smc.healthcheck.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.smc.healthcheck.plist"
LOG_DIR="$HOME/Library/Logs/SMCSignalEngine"

mkdir -p "$HOME/Library/LaunchAgents" "$LOG_DIR"
cp "$PLIST_SRC" "$PLIST_DST"

launchctl bootout "gui/$(id -u)" "$PLIST_DST" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST_DST"
launchctl enable "gui/$(id -u)/com.smc.healthcheck"
launchctl kickstart -k "gui/$(id -u)/com.smc.healthcheck"

echo "Installed launch agent: $PLIST_DST"
echo "Schedule: every 300 seconds"
echo "Logs: $LOG_DIR/health-check.out.log and $LOG_DIR/health-check.err.log"
