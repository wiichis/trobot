#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${OUT_DIR:-archivos/backtesting/daily}"
mkdir -p "$OUT_DIR"

LOSS_TRIGGER="${LOSS_TRIGGER:--1.5}"
LOOKBACK_HOURS="${LOOKBACK_HOURS:-24}"
COOLDOWN_HOURS="${COOLDOWN_HOURS:-72}"
MAX_SYMBOLS_PER_RUN="${MAX_SYMBOLS_PER_RUN:-3}"
N_TRIALS="${N_TRIALS:-45}"
LOOKBACK_DAYS_OPT="${LOOKBACK_DAYS_OPT:-60}"
LOOKBACK_DAYS_LONG="${LOOKBACK_DAYS_LONG:-120}"
SEND_ALERT="${SEND_ALERT:-1}"
APPLY="${APPLY:-0}"
REFRESH_INDICATORS="${REFRESH_INDICATORS:-0}"

STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_FILE="${OUT_DIR}/loss_recovery_${STAMP}.log"
touch "$LOG_FILE"
exec >>"$LOG_FILE" 2>&1

echo "[LOSS_RECOVERY] start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[LOSS_RECOVERY] log=${LOG_FILE}"

ARGS=(
  --pnl_csv archivos/PnL.csv
  --best_consistent pkg/best_prod_consistent.json
  --best_fallback pkg/best_prod.json
  --data_template archivos/cripto_price_5m_long.csv
  --sweep_cfg archivos/backtesting/simple_sweep.json
  --out_dir "$OUT_DIR"
  --lookback_hours "$LOOKBACK_HOURS"
  --loss_trigger "$LOSS_TRIGGER"
  --max_symbols_per_run "$MAX_SYMBOLS_PER_RUN"
  --cooldown_hours "$COOLDOWN_HOURS"
  --lookback_days_opt "$LOOKBACK_DAYS_OPT"
  --lookback_days_long "$LOOKBACK_DAYS_LONG"
  --n_trials "$N_TRIALS"
  --report "${OUT_DIR}/loss_recovery_report.json"
)

if [[ "$SEND_ALERT" == "1" ]]; then
  ARGS+=(--send_alert)
fi
if [[ "$APPLY" == "1" ]]; then
  ARGS+=(--apply)
fi
if [[ "$REFRESH_INDICATORS" == "1" ]]; then
  ARGS+=(--refresh_indicators)
fi

"$PYTHON_BIN" scripts/loss_recovery_tuner.py "${ARGS[@]}"

echo "[LOSS_RECOVERY] done_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
