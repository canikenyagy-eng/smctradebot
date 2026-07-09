#!/bin/zsh
set -euo pipefail

PROJECT_DIR="/Users/kanannagiev/Documents/New project/project"
LOG_DIR="$PROJECT_DIR/logs"
HEALTH_LOG="$LOG_DIR/live_health_check.out.log"

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

{
  echo "=== Live health check started at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  "$PROJECT_DIR/.venv/bin/python" -m research.live_health_check --alert --output logs/live_health_status.json
  echo "=== Live health check finished at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo
} >> "$HEALTH_LOG" 2>&1

echo "Live health check completed. Log: $HEALTH_LOG"
