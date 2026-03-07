#!/usr/bin/env bash
set -euo pipefail

# === Config ===
SSH_KEY="${SSH_KEY:-$HOME/Proyectos/ls_keys/trobot4.pem}"
SERVER_IP="${SERVER_IP:-IP_PUBLICA}"
REMOTE_USER="${REMOTE_USER:-ubuntu}"
REMOTE_BASE="${REMOTE_BASE:-/home/ubuntu/TRobot}"
LOCAL_BASE="${LOCAL_BASE:-/Users/will/Documents/proyectos/TRobot}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SYNC_LONG_FROM_5M="${SYNC_LONG_FROM_5M:-1}"

REMOTE_PNL="${REMOTE_BASE}/archivos/PnL.csv"
REMOTE_5M="${REMOTE_BASE}/archivos/cripto_price_5m.csv"
REMOTE_LONG="${REMOTE_BASE}/archivos/cripto_price_5m_long.csv"
LOCAL_ARCHIVOS="${LOCAL_BASE}/archivos"

if [[ "${SERVER_IP}" == "IP_PUBLICA" || -z "${SERVER_IP}" ]]; then
  echo "ERROR: define SERVER_IP con la IP publica de produccion."
  echo "Ejemplo: SERVER_IP=98.81.217.194 $0"
  exit 1
fi

download_file() {
  local remote_path="$1"
  local local_path="$2"
  local tmp_path
  tmp_path="${local_path}.tmp.$$"

  scp -i "${SSH_KEY}" "${REMOTE_USER}@${SERVER_IP}:${remote_path}" "${tmp_path}"
  mv "${tmp_path}" "${local_path}"
}

echo "==> Descargando PnL.csv, cripto_price_5m.csv y cripto_price_5m_long.csv..."
mkdir -p "${LOCAL_ARCHIVOS}"

download_file "${REMOTE_PNL}" "${LOCAL_ARCHIVOS}/PnL.csv"
download_file "${REMOTE_5M}" "${LOCAL_ARCHIVOS}/cripto_price_5m.csv"
download_file "${REMOTE_LONG}" "${LOCAL_ARCHIVOS}/cripto_price_5m_long.csv"

echo "==> Descarga completa en ${LOCAL_ARCHIVOS}"
ls -lh \
  "${LOCAL_ARCHIVOS}/PnL.csv" \
  "${LOCAL_ARCHIVOS}/cripto_price_5m.csv" \
  "${LOCAL_ARCHIVOS}/cripto_price_5m_long.csv"

if [[ "${SYNC_LONG_FROM_5M}" == "1" ]]; then
  echo "==> Sincronizando cripto_price_5m_long.csv con el 5m local..."
  if (cd "${LOCAL_BASE}" && "${PYTHON_BIN}" -c "import pkg.price_bingx_5m as p; p.actualizar_long_ultimas_12h()"); then
    echo "==> Sync local de long completado."
  else
    echo "WARN: No se pudo sincronizar long localmente (PYTHON_BIN=${PYTHON_BIN})."
  fi
fi

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
