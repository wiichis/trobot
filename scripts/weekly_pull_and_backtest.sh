#!/usr/bin/env bash
set -euo pipefail

# === Config ===
SSH_KEY="${SSH_KEY:-$HOME/Proyectos/ls_keys/trobot4.pem}"
SERVER_IP="${SERVER_IP:-IP_PUBLICA}"
REMOTE_USER="${REMOTE_USER:-ubuntu}"
REMOTE_BASE="${REMOTE_BASE:-/home/ubuntu/TRobot}"
LOCAL_BASE="${LOCAL_BASE:-/Users/will/Documents/proyectos/TRobot}"

REMOTE_PNL="${REMOTE_BASE}/archivos/PnL.csv"
REMOTE_LONG="${REMOTE_BASE}/archivos/cripto_price_5m_long.csv"
LOCAL_ARCHIVOS="${LOCAL_BASE}/archivos"

echo "==> Descargando PnL.csv y cripto_price_5m_long.csv..."
mkdir -p "${LOCAL_ARCHIVOS}"

scp -i "${SSH_KEY}" "${REMOTE_USER}@${SERVER_IP}:${REMOTE_PNL}" "${LOCAL_ARCHIVOS}/PnL.csv"
scp -i "${SSH_KEY}" "${REMOTE_USER}@${SERVER_IP}:${REMOTE_LONG}" "${LOCAL_ARCHIVOS}/cripto_price_5m_long.csv"

echo "==> Descarga completa en ${LOCAL_ARCHIVOS}"

# === Backtesting semanal (ajusta si quieres) ===
# Activa tu entorno local antes de correr, o ajusta la ruta del python.
# Ejemplos recomendados (comenta/descomenta):
#
# python3 pkg/backtesting.py \
#   --symbols=auto \
#   --data_template archivos/cripto_price_5m_long.csv \
#   --lookback_days 90 \
#   --train_ratio 0 \
#   --search_mode random \
#   --n_trials 600 \
#   --sweep archivos/backtesting/simple_sweep.json \
#   --export_best best_prod.json \
#   --export_positive_ratio
#
# python3 pkg/backtesting.py \
#   --export_consistent_best \
#   --parity_best pkg/best_prod.json \
#   --data_template archivos/cripto_price_5m_long.csv \
#   --consistency_short_days 90 \
#   --consistency_max_cost_ratio 1.0 \
#   --consistency_require_pnl_positive
#
# python3 pkg/backtesting.py \
#   --live_parity \
#   --symbols=auto \
#   --data_template archivos/cripto_price_5m_long.csv \
#   --parity_best pkg/best_prod.json \
#   --parity_days 90 \
#   --parity_per_symbol

echo "==> Listo."
