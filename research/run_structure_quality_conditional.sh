#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs reports

LOG_FILE="${STRUCTURE_QUALITY_CONDITIONAL_LOG:-logs/structure_quality_conditional.out.log}"
if [[ "${STRUCTURE_QUALITY_CONDITIONAL_SELF_LOG:-1}" == "1" ]]; then
  : > "$LOG_FILE"
  exec > "$LOG_FILE" 2>&1
fi

PID_FILE="logs/structure_quality_conditional.pid"
if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "structure quality conditional calibration job is already running with PID ${existing_pid}"
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

OUTPUT="${STRUCTURE_QUALITY_CONDITIONAL_OUTPUT:-reports/structure_quality_conditional_1200_3000_step2.json}"
PAIRS="${STRUCTURE_QUALITY_PAIRS:-EURUSD,GBPUSD,USDJPY}"
HISTORY_LIMITS="${STRUCTURE_QUALITY_HISTORY_LIMITS:-1200,3000}"
EVALUATION_STEPS="${STRUCTURE_QUALITY_EVALUATION_STEPS:-2}"
BONUS_VALUES="${STRUCTURE_QUALITY_BONUS_VALUES:-4,6,8}"
MIN_SCORE="${STRUCTURE_QUALITY_MIN_SCORE_FOR_BONUS:-60}"
MAX_HOLD_BARS="${STRUCTURE_QUALITY_MAX_HOLD_BARS:-24}"
WARMUP_BARS="${STRUCTURE_QUALITY_WARMUP_BARS:-120}"

started_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "Started structure quality conditional calibration at ${started_at}"
echo "Root: ${ROOT_DIR}"
echo "Log: ${LOG_FILE}"
echo "Output: ${OUTPUT}"
echo "Pairs: ${PAIRS}"
echo "History limits: ${HISTORY_LIMITS}"
echo "Evaluation steps: ${EVALUATION_STEPS}"
echo "Scenario mode: conditional"
echo "Min score for bonus: ${MIN_SCORE}"
echo "Snapshot cache max entries: ${BACKTEST_SNAPSHOT_CACHE_MAX_ENTRIES}"

echo
echo "=== Running conditional structure quality calibration ==="
.venv/bin/python -m research.structure_quality_calibration \
  --pairs "${PAIRS}" \
  --history-limits "${HISTORY_LIMITS}" \
  --evaluation-steps "${EVALUATION_STEPS}" \
  --bonus-values "${BONUS_VALUES}" \
  --scenario-mode conditional \
  --min-score-for-bonus "${MIN_SCORE}" \
  --max-hold-bars "${MAX_HOLD_BARS}" \
  --warmup-bars "${WARMUP_BARS}" \
  --cache-only \
  --snapshot-cache-size "${BACKTEST_SNAPSHOT_CACHE_MAX_ENTRIES}" \
  --output "${OUTPUT}"

finished_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo
echo "Finished structure quality conditional calibration at ${finished_at}"
