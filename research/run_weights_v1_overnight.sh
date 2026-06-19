#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs reports

LOG_FILE="${WEIGHTS_V1_OVERNIGHT_LOG:-logs/weights_v1_overnight.out.log}"
if [[ "${WEIGHTS_V1_OVERNIGHT_SELF_LOG:-1}" == "1" ]]; then
  : > "$LOG_FILE"
  exec > "$LOG_FILE" 2>&1
fi

PID_FILE="logs/weights_v1_overnight.pid"
if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "weights_v1 overnight job is already running with PID ${existing_pid}"
    exit 0
  fi
fi

echo "$$" > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT

export PYTHONUNBUFFERED=1
export ENABLE_BACKTEST_SNAPSHOT_CACHE=1
export BACKTEST_SNAPSHOT_CACHE_MAX_ENTRIES="${BACKTEST_SNAPSHOT_CACHE_MAX_ENTRIES:-250000}"
export MARKET_DATA_CACHE_ENABLED=1
export MARKET_DATA_CACHE_MODE=cache_only

COMMON_ARGS=(
  --pairs EURUSD,GBPUSD,USDJPY
  --history-limits 1200,3000
  --scenarios weights_v1_sizing
  --max-hold-bars 24
  --warmup-bars 120
  --cache-only
  --mc-iterations 1000
  --mc-seed 42
  --mc-ruin-dd 10
  --bar-walk-forward
  --wf-train-bars 600
  --wf-test-bars 240
  --wf-step-bars 240
  --snapshot-cache-size "${BACKTEST_SNAPSHOT_CACHE_MAX_ENTRIES}"
)

if [[ "${MAX_RUNS:-0}" != "0" ]]; then
  COMMON_ARGS+=(--max-runs "${MAX_RUNS}")
fi

started_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "Started weights_v1_sizing overnight calibration at ${started_at}"
echo "Root: ${ROOT_DIR}"
echo "Log: ${LOG_FILE}"
echo "Snapshot cache max entries: ${BACKTEST_SNAPSHOT_CACHE_MAX_ENTRIES}"

for step in 2 3; do
  output="reports/weights_v1_sizing_overnight_step${step}_1200_3000.json"
  echo
  echo "=== Running weights_v1_sizing | evaluation_step=${step} | output=${output} ==="
  .venv/bin/python -m research.tier_calibration \
    "${COMMON_ARGS[@]}" \
    --evaluation-steps "${step}" \
    --output "${output}"
done

finished_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo
echo "Finished weights_v1_sizing overnight calibration at ${finished_at}"
