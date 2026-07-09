#!/bin/zsh
set -euo pipefail

PLIST_DST="$HOME/Library/LaunchAgents/com.smc.forwardreports.plist"

launchctl bootout "gui/$(id -u)" "$PLIST_DST" 2>/dev/null || true
rm -f "$PLIST_DST"

echo "Removed launch agent: $PLIST_DST"
