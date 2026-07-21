# TRobot — Guía rápida

Bot de trading automatizado de futuros perpetuos en BingX. Opera 12 pares en USDT con una estrategia trend-following sobre velas de 5 minutos, optimizada por sweep semanal y desplegada a un servidor EC2 que corre 24/7 vía systemd.

---

## Qué hace el proyecto

1. **Recolecta datos**: cada minuto sincroniza velas 5m de BingX (`pkg/price_bingx_5m.py`).
2. **Calcula indicadores**: EMA fast/slow, RSI, ADX, ATR, volumen relativo (`pkg/indicadores.py`).
3. **Genera señales y abre/cierra posiciones**: trend-following con confirmación multi-indicador. Lógica live en `pkg/monkey_bx.py`.
4. **Gestiona TP/SL**: TP escalonado (TP1, TP2, TP3) opcional, BE-trigger después de TP1, SL fijo en `%` o ATR-trailing (`pkg/tp_stage_state.py`).
5. **Optimiza parámetros**: cada lunes corre [scripts/evaluate_pairs.py](scripts/evaluate_pairs.py) que clasifica pares por PnL real, re-optimiza los perdedores via sweep aleatorio sobre `pkg/backtesting.py`, rota el peor por uno nuevo de top-volumen.
6. **Reporta**: resúmenes horarios + diarios + alertas TP/SL/BE por Telegram (`pkg/telegram_alerts.py`).

## Lógica del scheduler (main.py)

| Frecuencia | Tarea |
|---|---|
| Cada 5 min (minuto :01,:06,…) | Pull velas 5m de BingX |
| Cada 5 min (minuto :03,:08,…) | `update_indicators` + `colocando_ordenes` (entradas) |
| Cada 50 s | `colocando_TK_SL` (gestión TP/SL en vivo) |
| Cada 5 min | `unrealized_profit_positions` (posiciones antiguas) |
| Cada 6 h | `resultado_PnL` (refresca PnL.csv desde exchange) |
| Cada hora :59 | Reporte horario por Telegram |
| Diario 23:00 UTC | Resumen diario |
| Cada 12h | Backfill velas + huecos |

## Composición actual del portfolio

10 pares (al 2026-07-20, MD5 `57daca5b`):
APT, AVAX, BCH, BNB, CFX, **DYDX**, ETH, LINK, ONDO, XMR.

🔄 **DYDX NEW** (20/07, rotó a DOT): cross-val 3/3 (30/60/80d: +0.05/+1.82/+2.31), 15 trades/80d, WR 70%, baja frecuencia = perfil que menos sufre el gap sim-real. Entra con params del sweep tal cual (tp=0.012, NO TP×2 — el sweep ya optimizó su TP). Regla de vigilancia AAVE su 1ª semana. Reglas de contrato: qty_step 0.1, price_tick 0.00001 (~$0.12).
🎯 **TP×2 aplicado** (03/07) a 8 pares (todos menos AVAX congelado y LINK NEW): el A/B de estructura de salidas mostró TPs demasiado cercanos (payoff real 0.42, perdedores 11h vs ganadores 3h). Cross-val 5/5 ventanas. Esperar: winrate más bajo, ganancias más grandes, ~20% menos trades/fees. Ver memoria `exit-structure-experiments`.
✅ **BCH NEW validó** (22/06): +2.02 real, 10/12 wins. El acierto del mes.
🔄 **LINK paramset NEW** (30/06, cross-val 5/5): validar que se realice. ⚠️ 0 trades desde 03/07 — mudo.
⏪ **CFX revertido** a `min_vol_ratio` 1.15 (30/06): volvió a plano/positivo (+0.11 la semana 13/07).
🧊 **AVAX congelado**: NO re-optimizar sus params. Revivió: +1.25 la semana 06/07. Sale de la fila de rotación.
⚠️ **Mudez de portfolio**: APT/BCH/ETH/LINK/ONDO 0 trades desde 03/07 (5/10 pares). TP×2 bajó frecuencia + fresh-cross muy selectivo. Tema #1 para revisión estructural 03/08 (re-optimizar params está congelado hasta entonces).

Pares removidos (no reincorporar a ciegas):
- DOGE-USDT (02/05) — peor 90d
- TRX-USDT (08/05) — sweep robusto sin paramset que pase filtros
- NEAR-USDT (18/05) — peor 90d (-1.91)
- HBAR-USDT (01/06) — 3 sweeps sin paramset robusto, 0 trades
- SOL-USDT (01/06) — 3 sweeps sin paramset robusto, 0 trades
- AAVE-USDT (10/06) — regla de vigilancia: perdió −1.19 su 1ª semana; cross-val regresivo 60/90d
- BTC-USDT (10/06) — 4 sweeps sin paramset robusto + 0 trades reales en 5 semanas
- DOT-USDT (20/07) — mudo crónico (2 trades desde 03/07, −0.31) + semanas rojas; rotado a DYDX

Candidatos evaluados y rechazados (10/06): HYPE, XRP, ZEC — overfit en 60/90d.
Sintéticos NC* (oro/Nasdaq/petróleo/FX): descartados — gaps de finde rompen velas 5m.

## Archivos clave

| Path | Función |
|---|---|
| `main.py` | Scheduler principal, levanta todos los jobs |
| `pkg/best_prod.json` | **Fuente de verdad de pares + parámetros activos**. Lo lee el bot al arrancar |
| `pkg/monkey_bx.py` | Lógica de entradas, salidas, TP/SL, riesgo |
| `pkg/indicadores.py` | Cálculo de indicadores y filtros de señal. Lee whitelist de `best_prod.json` |
| `pkg/backtesting.py` | Motor de backtest + sweep. (El atexit que sobrescribía `best_prod.json` fue parcheado en `27cc863`; verificar md5 tras sweeps sigue siendo buena práctica) |
| `scripts/evaluate_pairs.py` | Flujo semanal: clasificar, re-optimizar, rotar, deploy a prod |
| `scripts/weekly_pull_and_backtest.sh` | Cron wrapper de evaluate_pairs |
| `archivos/cripto_price_5m_long.csv` | Histórico de velas para backtesting |
| `archivos/PnL.csv` | PnL realizado descargado de BingX |
| `archivos/trade_closed_log.csv` | Log de cierres con razón (stop_loss, tp1/2/3) |
| `archivos/ganancias.csv` | Snapshot de balance cada minuto |
| `dashboard/` | Streamlit local para monitorear (no afecta a prod) |

## Operación — comandos comunes

### Conexión a producción

```bash
ssh -i ~/Documents/proyectos/ls_keys/trobot4.pem ubuntu@98.81.217.194
# Servicio: sudo systemctl {status|restart|stop|start} trobot
# Logs: journalctl -u trobot -f
# Working dir: ~/TRobot/  (branch main, mergea desde origin/codex/nuevo-bot-estrategia)
```

### Verificar consistencia local ↔ prod

```bash
# MD5 del archivo de pares en prod
ssh prod "cd TRobot && md5sum pkg/best_prod.json"
# vs local
md5 pkg/best_prod.json
# vs HEAD del repo
git show HEAD:pkg/best_prod.json | md5
# Los tres deben coincidir. Si no: hay drift (ver feedback_atexit_bug en memoria)
```

### Sweep semanal (ya con metodología robusta)

```bash
# Local, dry-run (no toca prod):
python3 scripts/evaluate_pairs.py --skip_sync --skip_rotation
# Default n_trials=500, train_ratio=0.66, rank_by=calmar, min_trades=15, max_dd=0.12

# Aplicar a local + deploy a prod (requiere confirmación):
python3 scripts/evaluate_pairs.py --apply --deploy --send_alert
```

### Antes de cualquier dry-run de backtesting

El bug atexit que sobrescribía `pkg/best_prod.json` fue parcheado (`27cc863`). Buena práctica que se mantiene: snapshot `md5 -q pkg/best_prod.json` antes del sweep y verificar igualdad después.

Para cross-validar paramsets de **símbolos fuera de la whitelist** (candidatos nuevos): `indicadores.py` fuerza señales en False si el símbolo no está en `best_prod.json`. Usar `TROBOT_BEST_PROD_PATH=<ruta_candidates.json>` como env var al invocar `pkg/backtesting.py --live_parity`.

## Branches

- **`main`** — branch que corre prod. Mergea desde `codex/nuevo-bot-estrategia`. Suele estar muchos commits "ahead de origin/main" porque los merges/auto-commits no se pushean a github desde prod (eso es OK).
- **`codex/nuevo-bot-estrategia`** — branch de desarrollo. Es donde se hacen commits desde local y se pushea a `origin/codex/nuevo-bot-estrategia`. Prod la mergea a su `main` local.

## Estrategia (resumen)

Trend-following con confirmación multi-indicador en velas 5m:

- **Entrada LONG**: EMA fast > EMA slow + RSI > rsi_buy + ADX > adx_min + filtros (ATR%, volumen relativo, distancia a EMA slow, opcional fresh-cross o fresh-breakout).
- **Entrada SHORT**: simétrica con RSI < rsi_sell.
- **TP**: fijo (`tp_mode=fixed`, ej 1.2-2.2%) o adaptativo por ATR (`tp_mode=atrx`).
- **SL**: porcentual fijo (`sl_mode=percent`) o ATR + trailing tras BE (`sl_mode=atr_then_trailing`).
- **Cooldown**: bars entre entradas consecutivas en el mismo símbolo.
- **Time exit**: cierre forzado tras N bars sin alcanzar TP.

Cada par tiene su propio paramset en `pkg/best_prod.json`, optimizado individualmente.

---

## Pendientes — para retomar la próxima semana

### P0 — ✅ COMPLETADO (2026-05-18)

~~Parchear `_post_run_normalize_best` en `pkg/backtesting.py`~~ → **Hecho en commit `27cc863`**. El handler ahora solo escribe a `pkg/best_prod.json` si el usuario lo pidió explícitamente vía `--export_best pkg/best_prod.json`. Verificado en el sweep semanal: archivo intacto antes/después.

### ~~P2.1 Filtro de régimen~~ — ❌ CERRADO 11/06: A y B probadas y RECHAZADAS

**Ambas direcciones falsificadas el mismo día** (A/B parity, 10 pares, 30/60/90d, logs en `archivos/backtesting/regime_validation_20260611/`):

- **Opción A (bloquear si ADX_1h<umbral)**: −51/−54/−53% del PnL. Los trades en lateral eran netos POSITIVOS (+21 USD/90d). Sensibilidad monótona: a más umbral, peor (ADX≥22 → portfolio negativo).
- **Opción B (relajar `adx_min`/`min_vol_ratio` ×0.8/×0.9 en lateral)**: Δ −5/−12/−23 USD en 30/60/90d. Monótono también (aggr ×0.7/×0.85 → −77). Los trades marginales desbloqueados pierden (XMR +0.7→−15.4). **Y los pares silenciosos NO despiertan** (CFX 7→7 trades, ETH 7→7 en 30d): su blocker no es ADX/volumen.
- Conclusión: los paramsets actuales están en un óptimo local respecto a estos knobs condicionados a régimen. No insistir con variantes (ver memoria `regime_filter_experiments`).

**Infra que queda (commiteada, inerte por default, reutilizable)**:
- `htf_mode: 'adx_only'` + knobs `regime_relax_*` en `pkg/indicadores.py` (features HTF hoisted, se computan una vez).
- Env override `TROBOT_RUNTIME_CONFIG_PATH` en `pkg/live_runtime_config.py`.
- Configs: `archivos/backtesting/configs/regime_{A_adx*,B_*}.json`.
- Harness: `TROBOT_RUNTIME_CONFIG_PATH=<cfg> python3 pkg/backtesting.py --live_parity --parity_days N --parity_per_symbol`.

### ✅ P2.1b Diagnóstico de blockers — HECHO 11/06 (herramienta: `scripts/diagnose_blockers.py`)

Descompone la señal en sus 11 condiciones y mide fail% + **near-miss** (barras donde solo falla esa condición). Valida la descomposición contra Long/Short_Signal reales. Resultados 30d en `archivos/backtesting/blockers_20260611_30d.csv`:

- **`ema_cross_recent` domina el portfolio** (blocker #1 en 13/20 par-lados, fail 86-97%): `fresh_cross_max_bars` 3-7 = el bot solo entra en los primeros 15-35 min tras un cruce EMA. La estrategia es "entrar en el cruce", no trend-following amplio.
- **CFX**: blocker es `vol_ratio` (fail 92%) — `min_vol_ratio=1.15` × media 40 barras casi nunca pasa.
- **ONDO**: `momentum_trigger` (fail 97%) — `logic=strict` exige RSI-cross Y breakout simultáneos.

**A/B quirúrgico (relajar solo el blocker #1 de cada par silencioso)**:
- ✗ AVAX fresh-cross 3→6: −19/−35/−46 (más trades = más pérdida; problema de edge, no de frecuencia)
- ✗ ETH fresh-cross 7→10: −4.6/−10.3/−12.9
- ✗ XMR fresh-cross 3→6: +5/+29 en 30/60d pero −18.8 en 90d (viola regla ≤$2)
- ✅ **CFX `min_vol_ratio` 1.15→1.0: mejora en LAS 5 ventANAS** (7/14/30/60/90d), 0→12 trades en 7d, cost_ratio 0.04-0.18. Sim con compounding exagera magnitud absoluta (esperar realización ~5%); aplicado 11/06 → vigilar.

Lección consolidada (3ª vez hoy): desbloquear near-misses solo paga cuando el edge subyacente del par es bueno (CFX era el mejor backtest del portfolio). En pares con edge débil, más trades = más pérdida.

### ✅ P1 Asimetría W/L — HECHO 03/07: time-stop RECHAZADO, TP×2 APLICADO

Diagnóstico con PnL real 90d: winrate 66% pero payoff 0.42 (ganador +0.16, perdedor −0.37) → PF bruto 0.83, negativo antes de fees. Perdedores viven 11h (SL), ganadores 3h (TP).

- ✗ **Time-stop de perdedores** (`loss_time_stop_bars` 24/36/48): empeora las 5 ventanas, monótono. **4ª confirmación** de "cortar/bloquear trades quita los netos-positivos". Infra inerte commiteada (knobs en `live_runtime_config`, check en parity-sim, configs `loss_ts_{24,36,48}.json`). NO insistir.
- ✗ `be_trigger=0`: 4/5 pero −9.57 en 60d. El BE aporta en ventanas largas.
- ✅ **TP×2 en 8 pares** (sin AVAX congelado ni LINK NEW): cross-val **5/5** (Δ +1.7/+1.1/+3.2/+6.7/+14.8), mejora repartida 8/10 pares (no la explica CFX), −20% trades, Sharpe 3.61→4.25. Aplicado 03/07. Esperar winrate ~57% con ganancias más grandes; veredicto real en 2-4 semanas.

### ✅ P4 Costos de ejecución — HECHO 03/07: TPs LIMIT maker activados

- Live ya entraba maker (PostOnly); las salidas eran todas taker. La infra `partial_limit_tp` de `monkey_bx.py` (TPs escalonados LIMIT reduce-only, one-at-a-time, fallback a market) existía completa y testeada (32 tests) pero apagada.
- 🐛 De paso: `should_fill_tp_limit`/`LimitFillPolicy` estaban hardcodeados a `None` en `pkg/backtesting.py` (resto de un módulo borrado) → `conservative_limit_fills` era código muerto en TODOS los sweeps históricos. Restaurados: fill solo si trade-through ≥buffer_bps o close confirma.
- A/B (fills conservadores + fee maker + slippage 0 en TPs vs status quo): **+0.45/+0.80/+1.15/+1.30 en 30/60/90/120d, 4/4**, peor celda por par −0.09. Fills perdidos ~2-3%.
- Aplicado: `tp_mode: partial_limit_tp` + `break_even_after_tp1: false` (mantiene BE por price-trigger idéntico a hoy) en `live_benchmark_runtime.json`. Reversible con flip de config + restart.
- 🐛 **Fix 07/07**: la primera semana los 9/9 TP LIMIT fueron rechazados por BingX (`109400: Hedge mode no acepta reduceOnly`) y cayeron al fallback market (sin pérdidas, pero sin ahorro). Fix: `tp_reduce_only: false` en config (en Hedge, side+positionSide ya define el cierre), kwargs omite el campo, y el retry sin reduceOnly cubre también LIMIT. Verificar fills LIMIT reales el 13/07.

### ❌ P2 Timeframe 15m — CERRADO 03/07: sweep completo, RECHAZADO

Hipótesis: mismo motor en 15m = menos señales pero movimientos más grandes vs costos. Falsificada:

- El estilo fresh-cross genera **~10× menos señales en 15m** (~10 trades/90d por par con la config más laxa vs ~100 en 5m). Parte era `max_dist_emaslow` calibrado a geometría 5m (en 15m la EMA slow es media de 7.5h, el precio casi siempre está a >1%), pero aun corregido el techo es estructural.
- Sweep 500 trials × 10 pares, 120d, espacio calibrado a 15m (`tf15m_validation_20260703/sweep_15m_v2.json`): 8 candidatos pasaron filtros, pero en cross-val 30/60/90/120d **pierden en las 4 ventanas** (−6.1/−13.6/−13.6/−5.0) vs baseline 5m (−1.6/+6.5/+4.2/+13.5). Overfit sobre señal escasa.
- Única señal débil: APT y ETH 15m positivos en 3-4 ventanas sin regresión (mudos crónicos en 5m), pero con 13-18 trades/120d y requeriría wiring de `entry_tf` en `indicadores.py` (live es 5m-only). NO pagar ese costo por 2 candidatos marginales; revisitar solo si APT/ETH siguen mudos ≥4 semanas más.
- Artefactos: `archivos/backtesting/tf15m_validation_20260703/`.

### Flujo semanal estándar (referencia, ya consolidado)

1. Pre-check consistencia local ↔ prod (`md5sum`); restaurar con `git checkout` si hay drift.
2. Sync PnL/ganancias/trade_closed de prod vía SCP. Para velas: **NO bajar long.csv de prod** (es la versión thin ~10-40d) — hacer **top-up directo del API BingX** (4 batches × 1000 velas por par cubren ~2 semanas de gap; dedupe por symbol+date). Backfill profundo (~35 batches ≈ 105d, el techo del API) solo para pares nuevos.
3. Análisis PnL real de la semana.
4. Dry-run `evaluate_pairs.py --skip_sync` (+ `--skip_rotation` si la rotación se maneja manual).
5. Cross-val anti-overfit (criterio refinado: 3/5 vale solo si regresión ≤$2 en ventanas perdidas).
6. Presentar propuesta → confirmar con usuario → aplicar + deploy + verificar + actualizar memoria.

### 📅 Plan con fechas (fijado 03/07 tras los 4 cambios del día)

Contexto: el 03/07 se aplicaron 4 cambios (TP×2 en 8 pares, TPs LIMIT maker, fix cooldown ×5, telemetría PnL). Necesitan ventana de realización limpia — **congelamiento de parámetros hasta el 03/08** salvo emergencia (par con pérdida real >2 USD/semana).

**Lunes 06/07 — flujo semanal en modo OBSERVACIÓN (no tocar params)**
1. Pre-check md5 local↔prod + sync PnL/ganancias/trade_closed vía SCP (velas: top-up API, no bajar long.csv).
2. **NO re-optimizar ni aplicar sweeps.** Dry-run solo como diagnóstico si se quiere.
3. Verificar ejecución de los cambios del 03/07:
   - TPs colocándose como **LIMIT** reduce-only (execution_ledger/order_register); fallback a market no dispara seguido.
   - `trade_closed_log`: cierres nuevos con `pnl_source=api`; aparecen `be_stop`/`trail_stop` (ya no todo es "stop_loss").
   - Cooldowns: tras un SL no hay re-entradas antes de 30-75 min.
4. Watch pares: **CFX** (¿volvió a plano tras revert 30/06?), **AVAX** (congelado; si pierde otra semana → candidato rotación), **LINK** (NEW 30/06, ¿realiza?), **DOT** (6ª semana muda).

**Lunes 13/07 — observación, semana 1.5 de TP×2**
- Mismas verificaciones. Primera lectura de TP×2: winrate ~57% esperado con ganadores más grandes — **NO revertir por winrate bajo**.
- Si CFX o AVAX acumulan 2 semanas malas: preparar rotación (backfill ~105d de LTC/DYDX/INJ + sweep + cross-val), **sin aplicar todavía**.

**Lunes 20/07 — evaluación intermedia (2.5 semanas de datos)**
- PnL realizado desde 03/07 vs baseline (−2 a −3 USD/mes), usando solo filas `pnl_source=api`.
- Métricas P4: % de TPs llenados como LIMIT vs fallback market. Distribución be_stop vs stop_loss real.
- Si hay candidato de rotación validado (cross-val 5/5 o 3/5 con regresión ≤$2) y CFX/AVAX siguen mal → aplicar rotación (es cambio de par, no de params congelados).

**Lunes 03/08 — veredicto TP×2 (4 semanas) + decisiones estructurales**
- Veredicto TP×2 + LIMIT TPs con un mes de realización real. Revertir solo si el PnL realizado es peor que la baseline.
- **P-proceso (punto 3)**: arranca la cadencia MENSUAL de re-optimización — primer sweep aplicable, con presupuesto de cambios (máx. 1-2 pares/mes; cada cambio debe ganarle a "no tocar nada" en cross-val). Los lunes intermedios quedan como observación/rotación.
- **P-pesos (punto 6)**: decidir con el mes de datos si concentrar capital (menos pares o pesos escalonados con cap por par) en vez del equal-weight 20% actual.

### P2 — Mejoras estratégicas adicionales

1. ~~**Filtro de régimen**~~ → ❌ cerrado (A y B rechazadas, ver arriba).
2. ~~**Anti-re-entry post-SL**~~ → ❌ **CERRADO 16/06: implementado, validado, RECHAZADO**. Knob global `post_sl_cooldown_bars` en `live_runtime_config` (default 0=off, infra inerte commiteada). A/B 10 pares: post-SL 36 barras (3h) Δ +3/−34/−41 en 30/60/90d; post-24 peor (−74/90d). Bloquea re-entradas rentables (CFX +206→+165 en 90d). Misma lección que P2.1: bloquear trades quita los netos-positivos. Configs `post_sl_{24,36}.json`, logs `post_sl_validation_20260616/`.
3. **Kill-switch DD diario**: si PnL día < −1% del balance, cortar nuevas entradas hasta el siguiente día UTC. *(siguiente candidato P2 sin probar)*
4. **Filtro de volatilidad relativa**: además de `min_atr_pct/max_atr_pct` absolutos, percentile-based para adaptarse al régimen actual de cada par.

### ✅ Bug 16/06 — desajuste de unidad en `cooldown` — CORREGIDO 03/07

`cooldown` per-símbolo era **barras en backtest** pero **minutos en live** → cooldown vivo 5× más débil que lo optimizado. Fix: live convierte barras→minutos (`_cooldown_minutes_for_symbol` retorna `cooldown*5`; alias legacy `cooldown_min` sigue en minutos) y el parity-sim usa barras directas. Los 3 motores (SimBacktester/parity/live) hablan barras. A/B parity: +2.56/+2.78/+0.49 en 30/60/90d, −0.47 en 14d, empate 7d. Efecto práctico: tras un SL se espera 30-75 min (según par) en vez de 6-15.

### P3 — Telemetría y observabilidad

0. ~~**PnL del trade_closed_log**~~ → ✅ **CORREGIDO 03/07**. El `pnl` venía de PnL.csv (refresco cada 6h) → 50% de filas en 0.0 y el resto contaminado con 48h del símbolo. Ahora: income API filtrado desde la entrada de la posición (fallbacks `price_est`/`csv_stale`, columna `pnl_source`). Cierres por stop reclasificados: `stop_loss`/`be_stop`/`trail_stop` (columna `stop_price`). ⚠️ El `pnl` de filas anteriores al 03/07 NO es confiable — filtrar por `pnl_source` en análisis.
1. **Alerta Telegram semanal**: PnL/balance, # trades, distribución razones de cierre, top losers. Si TP-rate < 25% en 7d → recomendar pausar.
2. **Dashboard mejorado**: agregar página de "salud del portfolio" con métricas de cada par (PnL 7/30/90d, winrate, pf, max_dd) y alertas visuales.
3. **CI ligero**: hook que verifique `md5sum pkg/best_prod.json` local == HEAD == prod después de cualquier deploy.

### P4 — Limpieza

1. ~~**Borrar `pkg/best_prod.json.bak.*`**~~ → ✅ HECHO 22/06: se conservó el más reciente (`.20260502_234317`), borrados 6. Son untracked (no requieren commit).
2. **Branch `main` en prod 35 commits ahead de `origin/main`**. No es problema funcional pero podría hacerse un `git push prod-merge` ocasional para mantener historia visible. No urgente.
3. **`dashboard/jobs.py` + `5_🧪_Backtesting.py` sin commitear** (sesión del dashboard). Decidir si commitear o descartar.

### Memoria persistente

Las gotchas detectadas y validadas están en `~/.claude/projects/-Users-will-Documents-proyectos-TRobot/memory/`:
- `backtest_history.md` — qué se rotó cada semana, monedas removidas/probadas
- `feedback_deploy_gotchas.md` — qué hacer cuando el deploy falla
- `feedback_atexit_bug.md` — el bug del atexit y cómo aislarse de él
