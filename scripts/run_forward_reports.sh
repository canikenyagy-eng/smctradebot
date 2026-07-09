#!/bin/zsh
set -euo pipefail

PROJECT_DIR="/Users/kanannagiev/Documents/New project/project"
LOG_DIR="$PROJECT_DIR/logs"
REPORT_LOG="$LOG_DIR/forward_reports_daily.out.log"

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

{
  echo "=== Forward reports started at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "Project: $PROJECT_DIR"
  echo "Journal: logs/forward_journal.jsonl"
  echo "Outcomes: logs/forward_outcomes.jsonl"
  echo "Report: reports/daily_live_forward_report.json"
  echo
  "$PROJECT_DIR/.venv/bin/python" -m research.daily_live_forward_report
  echo "=== Forward reports finished at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo
} >> "$REPORT_LOG" 2>&1

echo "Forward reports completed. Log: $REPORT_LOG"
