# Flujo diario activo: loss recovery

Este es el modelo que queda activo para ajustes diarios en produccion:

- detectar perdidas recientes en `archivos/PnL.csv`
- aplicar `cooldown` por simbolo
- reoptimizar solo simbolos con perdida
- promover cambios por simbolo solo si pasan gates
- enviar alerta por Telegram

## 1) Scripts usados

- `scripts/loss_recovery_tuner.py`
- `scripts/daily_loss_recovery.sh`

## 2) Ejecucion manual

```bash
cd ~/TRobot
source env/bin/activate
chmod +x scripts/loss_recovery_tuner.py scripts/daily_loss_recovery.sh
```

Prueba sin aplicar cambios:

```bash
LOSS_TRIGGER=-1.5 APPLY=0 SEND_ALERT=1 REFRESH_INDICATORS=0 bash scripts/daily_loss_recovery.sh
```

Ejecucion real (aplica si pasa gates):

```bash
LOSS_TRIGGER=-1.5 APPLY=1 SEND_ALERT=1 REFRESH_INDICATORS=1 bash scripts/daily_loss_recovery.sh
```

## 3) Parametros recomendados

```bash
LOSS_TRIGGER=-1.5
LOOKBACK_HOURS=24
COOLDOWN_HOURS=72
MAX_SYMBOLS_PER_RUN=3
N_TRIALS=45
LOOKBACK_DAYS_OPT=60
LOOKBACK_DAYS_LONG=120
```

## 4) Archivos de salida

- `archivos/backtesting/daily/loss_recovery_report.json`
- `archivos/backtesting/daily/loss_recovery_YYYYmmdd_HHMMSS.log`
- `archivos/backtesting/daily/loss_recovery_cooldown.json`
- `archivos/backtesting/daily/best_prod_consistent_loss_merged.json` (cuando hay candidatos)

## 5) Gates de promocion por simbolo

Un simbolo se promueve solo si cumple:

- trades minimos en evaluacion corta
- mejora de PnL corto vs modelo actual
- `cost_ratio` y `max_dd` dentro de limites
- no degradar de mas la ventana larga

Si no cumple, ese simbolo no se toca.

## 6) Timer diario (opcional)

Service:

```bash
sudo tee /etc/systemd/system/trobot-loss-recovery.service >/dev/null <<'EOF'
[Unit]
Description=TRobot loss recovery tuner
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=ubuntu
WorkingDirectory=/home/ubuntu/TRobot
Environment=LOSS_TRIGGER=-1.5
Environment=LOOKBACK_HOURS=24
Environment=COOLDOWN_HOURS=72
Environment=MAX_SYMBOLS_PER_RUN=3
Environment=N_TRIALS=45
Environment=LOOKBACK_DAYS_OPT=60
Environment=LOOKBACK_DAYS_LONG=120
Environment=APPLY=1
Environment=SEND_ALERT=1
Environment=REFRESH_INDICATORS=1
ExecStart=/bin/bash -lc 'source env/bin/activate && bash scripts/daily_loss_recovery.sh'
EOF
```

Timer (ejemplo 04:10 UTC):

```bash
sudo tee /etc/systemd/system/trobot-loss-recovery.timer >/dev/null <<'EOF'
[Unit]
Description=Run TRobot loss recovery daily

[Timer]
OnCalendar=*-*-* 04:10:00 UTC
Persistent=true
Unit=trobot-loss-recovery.service

[Install]
WantedBy=timers.target
EOF
```

Activar:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now trobot-loss-recovery.timer
systemctl list-timers | grep trobot-loss-recovery
```
