# Runbook Semanal IA (TRobot)

## Objetivo
Estandarizar un flujo semanal para una IA operativa, con este orden obligatorio:
1. Actualizacion de data desde produccion.
2. Analisis de resultados (PnL real vs backtesting).
3. Corrida de backtesting semanal.
4. Seleccion de posible nuevo candidato (solo si cumple reglas).
5. Actualizacion en produccion y reinicio (solo con confirmacion humana previa).

## Reglas no negociables
- No aplicar cambios en produccion sin confirmacion explicita del operador.
- No promover candidatos con perdida acumulada en los ultimos 3 meses (90 dias) en `archivos/PnL.csv`.
- Mantener `pkg/best_prod_consistent.json` como fuente de verdad para trading en produccion.
- Antes de modificar cualquier archivo de config, crear backup local con timestamp.

## Pre-requisitos
- Repo local: `/Users/will/Documents/proyectos/TRobot`
- Key SSH: `$HOME/Proyectos/ls_keys/trobot4.pem`
- Host prod: `ubuntu@98.81.217.194`
- Python activo (local y prod).

Variables sugeridas:

```bash
export REPO="/Users/will/Documents/proyectos/TRobot"
export SERVER_IP="98.81.217.194"
export SSH_KEY="$HOME/Proyectos/ls_keys/trobot4.pem"
export REMOTE_USER="ubuntu"
export REMOTE_BASE="/home/ubuntu/TRobot"
cd "$REPO"
```

## Paso 1: Actualizacion de data desde produccion
Usar script oficial de descarga:

```bash
SERVER_IP="$SERVER_IP" bash scripts/weekly_pull_and_backtest.sh
```

Validaciones minimas:

```bash
ls -lh archivos/PnL.csv archivos/cripto_price_5m.csv archivos/cripto_price_5m_long.csv
python3 - <<'PY'
import pandas as pd
for p in ["archivos/PnL.csv","archivos/cripto_price_5m.csv","archivos/cripto_price_5m_long.csv"]:
    try:
        df = pd.read_csv(p)
        print(p, "rows=", len(df), "cols=", len(df.columns))
    except Exception as e:
        print("ERROR", p, e)
PY
```

## Paso 2: Analisis de resultados (PnL real)
Analizar 7 dias y 90 dias (3 meses), por simbolo y total:

```bash
python3 - <<'PY'
import pandas as pd

df = pd.read_csv("archivos/PnL.csv")
df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
df["income"] = pd.to_numeric(df["income"], errors="coerce")
df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
df = df.dropna(subset=["symbol","income","time"])

end_ts = df["time"].max()
w_start = end_ts - pd.Timedelta(days=7)
m3_start = end_ts - pd.Timedelta(days=90)

def summarize(since, label):
    d = df[df["time"] >= since].copy()
    g = d.groupby("symbol", as_index=False).agg(
        pnl=("income","sum"),
        eventos=("income","size")
    ).sort_values("pnl", ascending=False)
    print(f"\n=== {label} ===")
    print("total_pnl:", round(d["income"].sum(), 6), "rows:", len(d))
    print("top +:")
    print(g.head(8).to_string(index=False))
    print("top -:")
    print(g.sort_values("pnl", ascending=True).head(8).to_string(index=False))
    return g

g7 = summarize(w_start, "ULTIMOS 7 DIAS")
g90 = summarize(m3_start, "ULTIMOS 90 DIAS")
g90.to_csv("archivos/backtesting/pnl_90d_por_symbol.csv", index=False)
print("\nGuardado:", "archivos/backtesting/pnl_90d_por_symbol.csv")
PY
```

## Paso 3: Corrida de backtesting semanal base
### 3.1 Sweep semanal (90 dias)

```bash
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

### 3.2 Export consistente (90d vs largo plazo)

```bash
python3 pkg/backtesting.py \
  --export_consistent_best \
  --parity_best pkg/best_prod.json \
  --data_template archivos/cripto_price_5m_long.csv \
  --consistency_short_days 90 \
  --consistency_max_cost_ratio 1.0 \
  --consistency_require_pnl_positive
```

### 3.3 Parity del set consistente (comparacion con comportamiento esperado)

```bash
python3 pkg/backtesting.py \
  --live_parity \
  --symbols=auto \
  --data_template archivos/cripto_price_5m_long.csv \
  --parity_best pkg/best_prod_consistent.json \
  --parity_days 90 \
  --parity_per_symbol
```

Interpretacion recomendada:
- Mantener set actual si parity 90d sigue con `pnl_net` positivo y `cost_ratio` controlado.
- Ajustar si aparecen simbolos activos con degradacion clara (pnl negativo y/o costo excesivo).

## Paso 4: Seleccion de nuevo candidato (solo si existe)
Condicion obligatoria:
- El simbolo NO debe tener perdida en los ultimos 90 dias en PnL real (`pnl_90d >= 0`).

### 4.1 Construir pool elegible (fuera de activos actuales)

```bash
python3 - <<'PY'
import json
import pandas as pd

best = json.load(open("pkg/best_prod_consistent.json"))
active = {str(x.get("symbol","")).upper() for x in best if isinstance(x,dict)}

pnl = pd.read_csv("archivos/PnL.csv")
pnl["symbol"] = pnl["symbol"].astype(str).str.upper().str.strip()
pnl["income"] = pd.to_numeric(pnl["income"], errors="coerce")
pnl["time"] = pd.to_datetime(pnl["time"], utc=True, errors="coerce")
pnl = pnl.dropna(subset=["symbol","income","time"])

end_ts = pnl["time"].max()
start_ts = end_ts - pd.Timedelta(days=90)
p90 = pnl[pnl["time"] >= start_ts].copy()

g = p90.groupby("symbol", as_index=False).agg(
    pnl_90d=("income","sum"),
    eventos_90d=("income","size")
)

pool = g[(~g["symbol"].isin(active)) & (g["pnl_90d"] >= 0)].copy()
pool = pool.sort_values(["pnl_90d","eventos_90d"], ascending=[False, False])
pool.to_csv("archivos/backtesting/candidate_pool_90d.csv", index=False)

print("Activos actuales:", sorted(active))
print("Candidatos elegibles:", len(pool))
print(pool.head(10).to_string(index=False))
print("Guardado: archivos/backtesting/candidate_pool_90d.csv")
PY
```

Si `candidate_pool_90d.csv` queda vacio: no hay candidato nuevo esta semana.

### 4.2 Optimizar candidato top (si existe)
Tomar el primer simbolo de `candidate_pool_90d.csv` y correr:

```bash
export CANDIDATE="$(python3 - <<'PY'
import pandas as pd
df = pd.read_csv("archivos/backtesting/candidate_pool_90d.csv")
print("" if df.empty else df.iloc[0]["symbol"])
PY
)"
echo "CANDIDATE=$CANDIDATE"
```

```bash
python3 pkg/backtesting.py \
  --symbols "$CANDIDATE" \
  --data_template archivos/cripto_price_5m_long.csv \
  --lookback_days 90 \
  --train_ratio 0 \
  --search_mode random \
  --n_trials 500 \
  --sweep archivos/backtesting/simple_sweep.json \
  --export_best "best_prod_${CANDIDATE}.json" \
  --export_positive_ratio
```

Validar parity corto/largo del candidato:

```bash
python3 pkg/backtesting.py \
  --live_parity \
  --symbols "$CANDIDATE" \
  --data_template archivos/cripto_price_5m_long.csv \
  --parity_best "archivos/backtesting/best_prod_${CANDIDATE}.json" \
  --parity_days 90 \
  --parity_per_symbol
```

Gates automaticos obligatorios para cambiar produccion:
- `pnl_90d_real >= 0` (obligatorio).
- `parity_30d.pnl_net > 0` y `parity_90d.pnl_net > 0`.
- `parity_90d.trades >= 20`.
- `parity_90d.cost_ratio < 1.0`.
- `abs(parity_90d.max_dd) <= 0.02` (drawdown maximo 2%).
- `parity_90d.winrate >= 45`.

Regla adicional para reemplazar un simbolo activo:
- `parity_30d.pnl_new >= parity_30d.pnl_old`.
- `parity_90d.pnl_new >= parity_90d.pnl_old`.
- `abs(parity_90d.max_dd_new) <= abs(parity_90d.max_dd_old)`.
- `parity_90d.winrate_new >= parity_90d.winrate_old`.

Si falla cualquier gate, la IA debe recomendar **mantener configuracion** y no proponer despliegue.

## Paso 5: Preparar aplicacion a produccion (sin ejecutar aun)
Antes de aplicar, la IA debe entregar resumen y pedir confirmacion:
- PnL semanal real.
- PnL 90d real.
- Resultado parity de simbolos activos.
- Candidato propuesto y metricas de gates.
- Diff exacto a aplicar en `pkg/best_prod_consistent.json`.
- Decision automatica (`CAMBIAR` o `MANTENER`) con el motivo exacto segun gates.

Solo con confirmacion del operador, continuar.

## Paso 6: Aplicar en produccion y reiniciar (solo tras confirmacion)
### 6.1 Backup local

```bash
cp pkg/best_prod_consistent.json "pkg/best_prod_consistent.json.bak.$(date +%Y%m%d_%H%M%S)"
```

### 6.2 Subir config a produccion

```bash
scp -i "$SSH_KEY" pkg/best_prod_consistent.json \
  "$REMOTE_USER@$SERVER_IP:$REMOTE_BASE/pkg/best_prod_consistent.json"
```

### 6.3 Actualizar indicadores + reiniciar servicio

```bash
ssh -i "$SSH_KEY" "$REMOTE_USER@$SERVER_IP" '
  cd /home/ubuntu/TRobot &&
  source env/bin/activate &&
  python3 -c "import pkg.indicadores as i; i.update_indicators()" &&
  sudo systemctl restart trobot &&
  sleep 3 &&
  sudo systemctl status trobot --no-pager -l | sed -n "1,40p"
'
```

### 6.4 Verificar simbolos activos en runtime

```bash
ssh -i "$SSH_KEY" "$REMOTE_USER@$SERVER_IP" '
  cd /home/ubuntu/TRobot &&
  source env/bin/activate &&
  python3 -c "import pkg.indicadores as i; print(\"TRADE\", len(i.TRADE_SYMBOLS), sorted(i.TRADE_SYMBOLS))"
'
```

## Paso 7: Rollback rapido (si algo falla)
Restaurar ultimo backup local y resubir:

```bash
ls -1t pkg/best_prod_consistent.json.bak.* | head -n 1
# luego:
cp <BACKUP_FILE> pkg/best_prod_consistent.json
scp -i "$SSH_KEY" pkg/best_prod_consistent.json \
  "$REMOTE_USER@$SERVER_IP:$REMOTE_BASE/pkg/best_prod_consistent.json"
ssh -i "$SSH_KEY" "$REMOTE_USER@$SERVER_IP" 'sudo systemctl restart trobot'
```

## Formato de salida esperado para la IA
La IA debe cerrar cada corrida semanal con:
1. Estado de data (archivos y fecha de corte).
2. Resumen PnL 7d y 90d (total + top winners/losers).
3. Resultado parity de activos (conclusion: mantener o ajustar).
4. Nuevo candidato (si existe) con validacion de regla `sin perdida 90d`.
5. Accion propuesta y confirmacion requerida antes de produccion.
