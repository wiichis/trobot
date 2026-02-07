# Guía rápida: operación del bot y backtesting

## 1) Producción (bot)

### Arranque
```
cd ~/TRobot
source env/bin/activate
python3 -c "import pkg.indicadores as i; i.update_indicators()"
```

Para correr en segundo plano:
```
nohup python3 main.py > bot.log 2>&1 &
```

Para correr como servicio (auto‑restart con 1h de espera):
```
sudo systemctl status trobot
```

### Logs
```
tail -f ~/TRobot/bot.log
```

Rotación (máx 5 archivos, 5MB):
```
sudo cat /etc/logrotate.d/trobot
```

### Símbolos activos
- **Data**: vienen de `pkg/symbols.json`
- **Trading**: se filtra por `pkg/best_prod_consistent.json`

Verificación rápida:
```
python3 -c "from pkg import settings; print(settings.BEST_PROD_PATH)"
```

### Checklist semanal (producción)
1) Servicio activo:
```
sudo systemctl status trobot
```
2) Logs recientes:
```
tail -n 200 ~/TRobot/bot.log
```
3) Disco:
```
df -h
```
4) PnL:
```
ls -lh ~/TRobot/archivos/PnL.csv
```


## 2) Backtesting (local)

### Descarga semanal de datos desde el server
```
chmod +x scripts/weekly_pull_and_backtest.sh
SERVER_IP=TU_IP_PUBLICA scripts/weekly_pull_and_backtest.sh
```

### Sweep recomendado (90 días, estabilidad)
```
python3 pkg/backtesting.py \
  --symbols=auto \
  --data_template archivos/cripto_price_5m_long.csv \
  --lookback_days 90 \
  --train_ratio 0 \
  --search_mode random \
  --n_trials 600 \
  --sweep archivos/backtesting/simple_sweep.json \
  --export_best best_prod.json \
  --export_positive_ratio
```

### Consistencia 90d vs largo plazo (para prod)
```
python3 pkg/backtesting.py \
  --export_consistent_best \
  --parity_best pkg/best_prod.json \
  --data_template archivos/cripto_price_5m_long.csv \
  --consistency_short_days 90 \
  --consistency_max_cost_ratio 1.0 \
  --consistency_require_pnl_positive
```

Esto genera un `best_prod_consistent.json` que se usa en producción.

### Validación parity (local)
```
python3 pkg/backtesting.py \
  --live_parity \
  --symbols=auto \
  --data_template archivos/cripto_price_5m_long.csv \
  --parity_best pkg/best_prod.json \
  --parity_days 90 \
  --parity_per_symbol
```


## 3) Recomendaciones rápidas

- No cambiar símbolos en producción sin correr backtest.
- Si el PnL semanal es negativo, revisar:
  - `cost_ratio` y símbolos con peor PnL.
  - estabilidad en 90d vs 180d.
- Mantener `best_prod_consistent.json` como fuente de verdad.


## 4) Archivos clave

- `pkg/best_prod_consistent.json` → símbolos para trading
- `pkg/symbols.json` → símbolos para data
- `archivos/indicadores.csv` → señales actuales
- `archivos/PnL.csv` → resultados reales

## 5) Flujo diario automatizado

Para automatizar backtesting diario + pase controlado a produccion, usar:

- `docs/daily_backtesting_produccion.md`
