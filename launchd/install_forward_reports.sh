#!/bin/zsh
set -euo pipefail

PROJECT_DIR="/Users/kanannagiev/Documents/New project/project"
PLIST_SRC="$PROJECT_DIR/launchd/com.smc.forwardreports.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.smc.forwardreports.plist"
LOG_DIR="$HOME/Library/Logs/SMCSignalEngine"

mkdir -p "$HOME/Library/LaunchAgents" "$LOG_DIR"
cp "$PLIST_SRC" "$PLIST_DST"

launchctl bootout "gui/$(id -u)" "$PLIST_DST" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST_DST"
launchctl enable "gui/$(id -u)/com.smc.forwardreports"

echo "Installed launch agent: $PLIST_DST"
echo "Schedule: daily at 21:30 local time"
echo "Logs: $LOG_DIR/forward-reports.out.log and $LOG_DIR/forward-reports.err.log"
