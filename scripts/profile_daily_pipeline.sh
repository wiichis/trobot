#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

OUT_DIR="${OUT_DIR:-archivos/backtesting/daily}"
mkdir -p "$OUT_DIR"

STAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_LOG="${OUT_DIR}/profile_run_${STAMP}.log"
TIME_LOG="${OUT_DIR}/profile_time_${STAMP}.log"
SYSTEM_LOG="${OUT_DIR}/profile_system_${STAMP}.log"
SUMMARY_LOG="${OUT_DIR}/profile_summary_${STAMP}.txt"

echo "[PROFILE] Capturing baseline host metrics..."
{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "--- uptime ---"
  uptime
  echo "--- memory ---"
  if command -v free >/dev/null 2>&1; then
    free -h
  elif command -v vm_stat >/dev/null 2>&1; then
    vm_stat
  else
    echo "memory command unavailable"
  fi
  echo "--- df -h ---"
  df -h
  echo "--- cpu info ---"
  if command -v nproc >/dev/null 2>&1; then
    nproc
  elif command -v sysctl >/dev/null 2>&1; then
    sysctl -n hw.ncpu || true
  else
    echo "cpu command unavailable"
  fi
} > "$SYSTEM_LOG"

echo "[PROFILE] Running daily pipeline with profiler (PROMOTE=0)..."
if /usr/bin/time -v bash -lc "true" >/dev/null 2>/dev/null; then
  PROMOTE=0 /usr/bin/time -v -o "$TIME_LOG" bash scripts/daily_backtest_pipeline.sh > "$RUN_LOG" 2>&1
elif /usr/bin/time -l bash -lc "true" >/dev/null 2>/dev/null; then
  PROMOTE=0 /usr/bin/time -l bash scripts/daily_backtest_pipeline.sh > "$RUN_LOG" 2> "$TIME_LOG"
else
  PROMOTE=0 bash scripts/daily_backtest_pipeline.sh > "$RUN_LOG" 2>&1
  echo "verbose time profiler not available on this host" > "$TIME_LOG"
fi

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "run_log=$RUN_LOG"
  echo "time_log=$TIME_LOG"
  echo "system_log=$SYSTEM_LOG"
  echo
  echo "Key metrics:"
  if ! grep -E "Elapsed|Percent of CPU|User time|System time|Maximum resident set size|maximum resident set size|major page faults|Major .* page faults|File system inputs|File system outputs|real|user|sys" "$TIME_LOG"; then
    cat "$TIME_LOG"
  fi
} > "$SUMMARY_LOG"

echo "[PROFILE] Completed."
echo "[PROFILE] Summary: $SUMMARY_LOG"
