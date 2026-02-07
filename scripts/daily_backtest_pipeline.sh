#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${OUT_DIR:-archivos/backtesting/daily}"
LONG_DATA="${LONG_DATA:-archivos/cripto_price_5m_long.csv}"
EVAL_DATA="${EVAL_DATA:-archivos/cripto_price_5m_eval.csv}"
SWEEP_CFG="${SWEEP_CFG:-archivos/backtesting/simple_sweep.json}"

EVAL_DAYS="${EVAL_DAYS:-90}"
LONG_CHECK_DAYS="${LONG_CHECK_DAYS:-180}"
N_TRIALS="${N_TRIALS:-180}"
RANK_BY="${RANK_BY:-pnl_net_per_trade}"
MIN_TRADES="${MIN_TRADES:-6}"
MAX_DAILY_SL_STREAK="${MAX_DAILY_SL_STREAK:-3}"
MAX_COST_RATIO="${MAX_COST_RATIO:-0.90}"
MAX_DD="${MAX_DD:-0.10}"
SECOND_PASS="${SECOND_PASS:-0}"
SKIP_PRICE_REFRESH="${SKIP_PRICE_REFRESH:-0}"

PROMOTE="${PROMOTE:-1}"
PROMOTION_DROP_PCT="${PROMOTION_DROP_PCT:-0.10}"
PROMOTION_MAX_COST_RATIO="${PROMOTION_MAX_COST_RATIO:-1.00}"
PROMOTION_MAX_DD="${PROMOTION_MAX_DD:-0.12}"
PROMOTION_MIN_TRADES_SHORT="${PROMOTION_MIN_TRADES_SHORT:-20}"
PROMOTION_MIN_TRADES_LONG="${PROMOTION_MIN_TRADES_LONG:-35}"

mkdir -p "$OUT_DIR"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_FILE="${OUT_DIR}/daily_pipeline_${STAMP}.log"
echo "[DAILY] writing log to ${LOG_FILE}"
touch "$LOG_FILE"
exec >>"$LOG_FILE" 2>&1

ACTIVE_CONSISTENT="pkg/best_prod_consistent.json"
ACTIVE_FALLBACK="pkg/best_prod.json"
SNAPSHOT_DIR="${OUT_DIR}/runtime_snapshot_${STAMP}"
mkdir -p "$SNAPSHOT_DIR"

CONSISTENT_EXISTED=0
FALLBACK_EXISTED=0
if [[ -f "$ACTIVE_CONSISTENT" ]]; then
  cp "$ACTIVE_CONSISTENT" "${SNAPSHOT_DIR}/best_prod_consistent.json"
  CONSISTENT_EXISTED=1
fi
if [[ -f "$ACTIVE_FALLBACK" ]]; then
  cp "$ACTIVE_FALLBACK" "${SNAPSHOT_DIR}/best_prod.json"
  FALLBACK_EXISTED=1
fi

restore_active_files() {
  if [[ "$CONSISTENT_EXISTED" == "1" ]]; then
    cp "${SNAPSHOT_DIR}/best_prod_consistent.json" "$ACTIVE_CONSISTENT"
  else
    rm -f "$ACTIVE_CONSISTENT"
  fi
  if [[ "$FALLBACK_EXISTED" == "1" ]]; then
    cp "${SNAPSHOT_DIR}/best_prod.json" "$ACTIVE_FALLBACK"
  else
    rm -f "$ACTIVE_FALLBACK"
  fi
}

echo "[DAILY] start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[DAILY] log=${LOG_FILE}"

echo "[DAILY] Step 1/5: refresh price files"
if [[ "$SKIP_PRICE_REFRESH" == "1" ]]; then
  echo "[DAILY] SKIP_PRICE_REFRESH=1 -> skipping price refresh"
else
  "$PYTHON_BIN" -c "import pkg.price_bingx_5m as p; p.price_bingx_5m(); p.completar_huecos_5m(); p.actualizar_long_ultimas_12h()"
fi

echo "[DAILY] Step 2/5: build eval dataset"
"$PYTHON_BIN" scripts/build_eval_dataset.py \
  --source "$LONG_DATA" \
  --output "$EVAL_DATA" \
  --days "$EVAL_DAYS" \
  --report "${OUT_DIR}/eval_dataset_report.json"

echo "[DAILY] Step 3/5: run daily sweep on eval dataset"
BT_ARGS=(
  --symbols=auto
  --data_template "$EVAL_DATA"
  --lookback_days "$EVAL_DAYS"
  --train_ratio 0
  --min_trades "$MIN_TRADES"
  --max_daily_sl_streak "$MAX_DAILY_SL_STREAK"
  --max_cost_ratio "$MAX_COST_RATIO"
  --max_dd "$MAX_DD"
  --search_mode random
  --n_trials "$N_TRIALS"
  --rank_by "$RANK_BY"
  --sweep "$SWEEP_CFG"
  --out_dir "$OUT_DIR"
  --export_best best_prod_daily_candidate.json
  --export_positive_ratio
)
if [[ "$SECOND_PASS" == "1" ]]; then
  BT_ARGS+=(--second_pass --second_topk 2)
fi
"$PYTHON_BIN" pkg/backtesting.py "${BT_ARGS[@]}"

echo "[DAILY] Step 4/5: consistency gate (90d vs long window)"
"$PYTHON_BIN" pkg/backtesting.py \
  --export_consistent_best \
  --parity_best "${OUT_DIR}/best_prod_daily_candidate.json" \
  --data_template "$LONG_DATA" \
  --out_dir "$OUT_DIR" \
  --consistency_short_days "$EVAL_DAYS" \
  --consistency_long_days "$LONG_CHECK_DAYS" \
  --consistency_max_cost_ratio 1.0 \
  --consistency_require_pnl_positive \
  --export_best "${OUT_DIR}/best_prod_daily_consistent.json"

# backtesting.py tiene side effects que pueden tocar pkg/best_prod*.json.
# Restauramos estado productivo antes de decidir si promovemos o no.
restore_active_files

echo "[DAILY] Step 5/5: evaluate and promote"
if [[ "$PROMOTE" == "1" ]]; then
  "$PYTHON_BIN" scripts/evaluate_and_promote.py \
    --candidate_best "${OUT_DIR}/best_prod_daily_consistent.json" \
    --current_best "pkg/best_prod_consistent.json" \
    --target_best "pkg/best_prod_consistent.json" \
    --target_fallback "pkg/best_prod.json" \
    --data_template "$LONG_DATA" \
    --short_days "$EVAL_DAYS" \
    --long_days "$LONG_CHECK_DAYS" \
    --allowed_pnl_drop_pct "$PROMOTION_DROP_PCT" \
    --max_cost_ratio "$PROMOTION_MAX_COST_RATIO" \
    --max_dd_abs "$PROMOTION_MAX_DD" \
    --min_trades_short "$PROMOTION_MIN_TRADES_SHORT" \
    --min_trades_long "$PROMOTION_MIN_TRADES_LONG" \
    --report "${OUT_DIR}/promotion_report.json" \
    --refresh_indicators
else
  echo "[DAILY] PROMOTE=0 -> skipping promotion step"
fi

echo "[DAILY] done_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
