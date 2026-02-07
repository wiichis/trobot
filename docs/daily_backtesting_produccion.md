# Flujo diario de backtesting y pase a produccion

Este flujo crea un dataset diario de evaluacion (`cripto_price_5m_eval.csv`), optimiza parametros, valida consistencia y promueve a produccion solo si pasa gates.

## 1) Archivos nuevos del flujo

- `scripts/build_eval_dataset.py`
  - Crea `archivos/cripto_price_5m_eval.csv` desde `archivos/cripto_price_5m_long.csv`.
  - Limpia duplicados y corta ventana de N dias (default 90).
- `scripts/daily_backtest_pipeline.sh`
  - Orquesta todo el flujo diario.
- `scripts/evaluate_and_promote.py`
  - Evalua candidato vs modelo actual y promueve solo si pasa gates.
- `scripts/profile_daily_pipeline.sh`
  - Ejecuta una corrida con profiling de consumo (CPU/RAM/tiempo).

## 2) Flujo operativo diario

Paso a paso que ejecuta el pipeline:

1. Actualiza precios (`cripto_price_5m.csv` y `cripto_price_5m_long.csv`).
2. Construye `archivos/cripto_price_5m_eval.csv` con ventana de 90 dias.
3. Corre sweep diario (liviano) sobre dataset de evaluacion.
4. Exporta candidato consistente (`best_prod_daily_consistent.json`) validando 90d vs 180d.
5. Evalua gates y promueve a:
   - `pkg/best_prod_consistent.json`
   - `pkg/best_prod.json` (compatibilidad)
6. Refresca `archivos/indicadores.csv` solo si hubo promocion.

## 3) Ejecucion manual (prueba)

```bash
cd ~/TRobot
source env/bin/activate
chmod +x scripts/daily_backtest_pipeline.sh scripts/profile_daily_pipeline.sh
bash scripts/daily_backtest_pipeline.sh
```

Salida principal:
- logs en `archivos/backtesting/daily/daily_pipeline_YYYYmmdd_HHMMSS.log`
- reporte de dataset en `archivos/backtesting/daily/eval_dataset_report.json`
- reporte de promocion en `archivos/backtesting/daily/promotion_report.json`

## 4) Parametros recomendados para diario

Variables que puedes ajustar sin tocar codigo:

```bash
EVAL_DAYS=90
LONG_CHECK_DAYS=180
N_TRIALS=180
MIN_TRADES=6
MAX_DAILY_SL_STREAK=3
MAX_COST_RATIO=0.90
MAX_DD=0.10
PROMOTION_DROP_PCT=0.10
PROMOTION_MAX_COST_RATIO=1.00
PROMOTION_MAX_DD=0.12
```

Ejemplo:

```bash
EVAL_DAYS=90 N_TRIALS=150 PROMOTION_DROP_PCT=0.08 bash scripts/daily_backtest_pipeline.sh
```

## 5) Test de consumo de recursos en produccion

Primero prueba sin promover (`PROMOTE=0`) para medir costo real:

```bash
cd ~/TRobot
source env/bin/activate
bash scripts/profile_daily_pipeline.sh
```

Archivos generados:
- `archivos/backtesting/daily/profile_run_*.log`
- `archivos/backtesting/daily/profile_time_*.log`
- `archivos/backtesting/daily/profile_system_*.log`
- `archivos/backtesting/daily/profile_summary_*.txt`

Si ves picos altos de RAM/CPU:
- baja `N_TRIALS` (ej. 180 -> 120)
- desactiva `SECOND_PASS` (default ya es 0)
- evita correr pipeline al mismo tiempo que procesos pesados.

## 6) Automatizacion diaria con systemd timer (server)

Crear service:

```bash
sudo tee /etc/systemd/system/trobot-backtest-daily.service >/dev/null <<'EOF'
[Unit]
Description=TRobot daily backtesting pipeline
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=ubuntu
WorkingDirectory=/home/ubuntu/TRobot
Environment=PYTHONUNBUFFERED=1
Environment=EVAL_DAYS=90
Environment=LONG_CHECK_DAYS=180
Environment=N_TRIALS=180
Environment=MIN_TRADES=6
Environment=MAX_DAILY_SL_STREAK=3
Environment=MAX_COST_RATIO=0.90
Environment=MAX_DD=0.10
Environment=PROMOTION_DROP_PCT=0.10
ExecStart=/bin/bash -lc 'source env/bin/activate && bash scripts/daily_backtest_pipeline.sh'
EOF
```

Crear timer (ejemplo 03:30 UTC diario):

```bash
sudo tee /etc/systemd/system/trobot-backtest-daily.timer >/dev/null <<'EOF'
[Unit]
Description=Run TRobot daily backtesting pipeline

[Timer]
OnCalendar=*-*-* 03:30:00 UTC
Persistent=true
Unit=trobot-backtest-daily.service

[Install]
WantedBy=timers.target
EOF
```

Activar:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now trobot-backtest-daily.timer
systemctl list-timers | grep trobot-backtest-daily
```

Ver logs:

```bash
journalctl -u trobot-backtest-daily.service -n 200 --no-pager
```

## 7) Regla de seguridad de produccion

No se reemplaza el modelo en produccion si falla algun gate:
- trades minimos
- pnl minimo
- cost ratio maximo
- drawdown maximo
- degradacion maxima vs modelo actual (`PROMOTION_DROP_PCT`)

En ese caso, se conserva `pkg/best_prod_consistent.json` anterior.
