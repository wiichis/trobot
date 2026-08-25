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
- ✅ **Verificado 27/07**: el guard de notional del 07/07 funcionó — los 276 rechazos `101485` + 273 `110422` están todos concentrados en el 08/07 y desaparecen desde el 10/07. Residual: `101215` (~3 reintentos cada 3-5 días) = entradas LIMIT PostOnly que habrían cruzado el book; existe desde abril con la misma cadencia, es el comportamiento correcto de PostOnly, no una regresión.

### 🐛 Bug 27/07 — cantidad de los TPs (DYDX) — CORREGIDO Y DESPLEGADO

DYDX abrió 314.7 el 26/07 y colocó tramos **103.8 / 103.8 / 69.5**; el 4º submit salió 35.3 (4.47 USDT), bajo el mínimo de cierre de BingX, y no se envió. Quedaron **37.6 unidades (~4.8 USDT) sin cobertura de TP**, colgando sólo del SL (cerró en verde a las 06:03, +0.25). Tres defectos de cantidad, los tres reproducidos con los números reales antes de tocar código (commit `753ebd0`):

1. **Base del reparto degradada**: `set_tp_submitted` reescribía `tpN_submit_position_qty` con la posición viva en cada recolocación del mismo tramo → el 33% se calculaba sobre 314.7 → 210.9 → 107.1 y cada tramo salía más chico. Ahora la base se conserva (es el tamaño de la posición cuando se armó el plan).
2. **Remanente huérfano**: el guard del 07/07 miraba que *el tramo* llegara al mínimo, no que *lo que queda después* fuera cerrable. Un resto bajo el mínimo ya no se puede cerrar por TP en ningún ciclo posterior. Ahora `_compute_partial_limit_stage_qty` colapsa al total restante (`ok_collapsed_min_leftover`).
3. **`_round_step` perdía un step entero por binario**: `107.1/0.1 == 1070.9999…` → floor daba `107.0`; también `0.3 → 0.2`, `314.7 → 314.6`. Dejaba polvo sin gestionar en toda orden que redondeara. Pasa a `Decimal` igual que `_split_position_qtys`. **Sólo lo usa el camino de TP; `backtesting.py` no lo referencia → los sweeps históricos no cambian.**

`upsert_tp_state` con `tp_stage="none"` (apertura) ahora limpia los tramos previos, para que anclar la base no haga heredar la posición anterior. Esto además drena solo los residuos viejos de `tp_stage_state.csv` (filas `tp3_live`/`tp1_live` de posiciones ya cerradas, porque `clear_tp_state` sólo se llama en cierre por SL).

Replay del caso real post-fix: **103.8 / 103.8 / 107.1 y la posición cierra íntegra por TP**. 13 tests nuevos, 45 en total. Desplegado 28/07 01:48 UTC.

### ✅ Bug 28/07 — el TP escalonado no escalonaba — CORREGIDO Y DESPLEGADO

**Era la causa raíz aguas arriba del bug de cantidad del 27/07.** Cadena: el job de colocación (cada 50 s) recolocaba el mismo stage y `set_tp_submitted(tp_idx=1)` **sobrescribía `tp1_order_id` antes de que `_log_pending_order_transitions` pudiera atribuir el fill de la orden anterior**. Sin ese match, `_infer_tp_idx_from_state_order_id` devuelve `None`, el fill nunca se confirmaba → `tp_stage` se quedaba en `tp1_live` para siempre → `_next_tp_idx_from_stage` devolvía `1` indefinidamente → `price_src_idx = 0` siempre → **los 3 tramos salían al precio de TP1**.

Evidencia: **0 fills `tp1/tp2/tp3` en 133 cierres desde el 03/07** (`trail_stop` 116, `stop_loss` 15, `be_stop` 2). El TP escalonado llevaba ~1 mes sin escalonar; quien capturaba la ganancia era el trailing stop.

Fix (commit `077951b`, desplegado 28/07 04:17 UTC): `_reconcile_stage_before_submit()` confirma el fill del tramo vivo cuya orden ya no está pendiente, **justo antes de decidir el próximo stage**. Reutiliza `_infer_tp_fill_from_position` (misma regla de reducción ≥60% del tramo), así que una orden cancelada sin fill no avanza el stage. Sin doble conteo por los dos órdenes posibles: si el job de transiciones confirma primero, el stage ya es `tpN_filled` y la reconciliación es no-op; si reconcilia primero, suelta `tpN_order_id` y el otro job deja de matchear. **El stage 3 se deja al job de transiciones a propósito** (es el cierre de la posición; reconciliarlo aquí duplicaría `_record_trade_closed`). Replay del caso DYDX: TP1 → TP2 → TP3 y la posición cierra íntegra. 50 tests en verde.

⚠️ **Consecuencia para el veredicto de TP×2 del 03/08**: el A/B asumía TPs escalonados que se llenan, pero lo que corrió en real desde el 03/07 fue "un TP al nivel de TP1 + trailing". **El mes de datos NO valida ni invalida TP×2.** El reloj de observación de la estructura de salidas arranca de cero el 28/07.

### 🔴 Mudez del portfolio — diagnóstico 28/07: DOS causas, una corregida

> ⚠️ **Corrección 10/08 — había una TERCERA causa, y era de ejecución.** Este diagnóstico concluyó que "la mudez es de señales, no de ejecución". Eso era cierto para la mayoría de los pares, pero **no para ONDO y APT**: sus entradas LIMIT PostOnly se colocaban a −2,41% y −0,62% del mercado por el tick roto, así que **no podían llenar** aunque la señal existiera. Ambos figuran en la lista de mudos del 03/07. Ver "Bug 10/08 — tick size ausente". La lección se repite: *cuando un par no opera, descartar ejecución con evidencia, no por argumento.*

Punto de partida: 7/10 pares con **0 señales** en 10,5 días (indicadores.csv de prod sólo retiene ~10d); todo el portfolio produjo 11 señales en 28.998 barras (0,04%). La semana 20-27/07 sólo operó DYDX.

**Primero se descartó lo obvio**: la mudez es de **señales**, no de ejecución — las señales simplemente no se generan. Y NO es por falta de relajar filtros: eso ya se falsificó 4 veces (P2.1, P2.1b, P1, P2-2). El diagnóstico de blockers confirma el mismo cuadro de junio (`ema_cross_recent` blocker #1 en 9/10 pares) y la tasa de señales del **sim** apenas bajó (105 → 87 normalizado a 30d). El problema no es que los filtros se hayan endurecido: es que **live y sim divergen**.

**Causa 1 — velas en formación (CORREGIDA, commit `723e5f7`, desplegada 28/07 04:53 UTC).**
El pull corre en :01,:06,… cuando la vela de 5m aún no cerró, así que se guardaba parcial. El filtro `df_new['date'] > last_date` descartaba justamente esa vela ya cerrada en el ciclo siguiente, y el `drop_duplicates(keep='last')` nunca llegaba a reemplazarla. Firma medida contra el API (2881 velas/par, 10 pares): `open` idéntico 100%, `high_live ≤ high_api` 100%, `low_live ≥ low_api` 100%, **`close` idéntico 0,4%**. O sea: EMA/RSI/ADX/ATR del live corrían sobre closes que no eran los closes reales — y el blocker dominante (`ema_cross_recent`) se calcula sobre closes. Explica el gap de señales sim 28 vs live 10 en ventana idéntica de 10d. Fix: `>=` en vez de `>`, y `fetch_limit` 2→3. **Verificado post-deploy: 7/7 velas cerradas coinciden al 100% en OHLC** (muestra chica, ~11 min).
⚠️ **Sigue abierto**: la *decisión* de entrada se toma sobre la vela en formación (`update_indicators` + `colocando_ordenes` corren en :03,:08,… con la vela a mitad), mientras el backtest decide sobre velas cerradas.

**Causa 2 — gate de sesión heredado (NO aplicada, requiere decisión).**
`colocando_ordenes()` retorna sin hacer nada si la hora UTC no está en `session.entry_hours_utc` = `[6,7,8,9,10,11,12,14,15,16,17,19,21,22]` (14 de 24 h; bloquea 0,1,2,3,4,5,13,18,20,23). Medido sobre las señales reales: **8 de 11 (73%) cayeron en hora bloqueada**.
- El perfil viene del bloque `reference` de `live_benchmark_runtime.json`: `session_profile: liquid_utc_wo_13_18_20` con `entry_style: rsi_reversal` y `timeframe_combo: 30m_5m` — **otra estrategia**, no la actual (trend-following fresh-cross 5m).
- **El backtest nunca lo simula**: `run_live_parity_portfolio` no aplica gate horario, y el `SimBacktester` lee `entry_hours_utc` de los params por símbolo (vacío en los 10). Los params se optimizaron asumiendo 24/7 y se ejecutan con 14/24. `--entry_hours_utc` no tiene efecto en `--live_parity`.
- A/B (parity-sim, trades particionados por hora de entrada): las horas bloqueadas son **40-44% de los trades** y su PnL es −4,21 / −2,29 / **+7,62** / **+3,28** en 30/60/90/120d, con winrate igual o mejor que las permitidas (49/52/59/56% vs 49/57/58/59%). Ninguna hora bloqueada es tóxica por sí sola (peor: 05h −0,83; la mejor es 13h con +3,60).
- **Conclusión: el gate no filtra horas malas, sólo recorta caudal.** Quitarlo restauraría la coherencia sim↔live y ~40% de los trades.
- ⚠️ **Pero arregla la mudez, no la rentabilidad**: el sim da PnL negativo en las 4 ventanas, así que más trades de edge similar da *muestra*, no ganancia. El valor real es poder medir: con 5/10 pares mudos no hay muestra para ningún veredicto.

### 🐛 Bug 31/07 — el bot decidía sobre la vela EN FORMACIÓN — CORREGIDO Y DESPLEGADO

**3 días (28-31/07) sin una sola orden, y sin un solo submit al exchange.** No fue el mercado ni el exchange: era el bot.

**Lo primero que quedó claro: el fix de velas del 28/07 SÍ funcionó.** Señales antes 9 en 7,1d (38 normalizado a 30d) → después 13 en 2,9d (**135, ×3,6**), y volvieron a producir ETH, XMR, BNB y LINK, mudos desde el 03/07. 7/10 pares generando señal vs 3/10 antes. El portfolio despertó.

**Pero 0 entradas.** Causa: `ema_alert` leía `df_symbol.iloc[-1]` = **la vela en formación** (el comentario del código ya decía "solo la última vela cerrada"; la implementación no lo hacía). Esa vela lleva 1-3 min de datos, así que su volumen es ~1/5 del real → `Rel_Volume` cae a 0,04-0,49 → **el filtro de volumen la veta siempre**. Medido en prod sobre los 10 pares: `VOL_OK` **False en 10/10** en la vela en formación vs **True en 3/10** en la misma vela ya cerrada. Históricamente `VOL_OK` pasa 19,1% de las barras cerradas y ~0% de las que están en formación.

**Por qué apareció justo ahora** (importante, es el patrón a recordar): antes del 28/07 la vela parcial no se corregía nunca, así que `indicadores.csv` guardaba señales calculadas sobre velas parciales — **las mismas** que el job evaluaba. Estaban mal, pero eran consistentes entre sí. Al corregir el histórico, el CSV pasó a tener señales de velas cerradas mientras el job seguía leyendo la parcial: dos poblaciones distintas, y la parcial no dispara nunca. **Arreglar los datos destapó un bug de lectura que llevaba latente desde siempre.**

Fix (commit `15b1e1b`, desplegado 31/07 02:31 UTC):
- `pkg/indicadores.py`: nuevo `last_closed_bar()` — decide por **tiempo de cierre**, no por posición, así que no asume que la última fila siempre está en formación. `ema_alert` lo usa.
- `pkg/monkey_bx.py`: la señal viene de la vela cerrada, pero **el sizing y el precio de la orden LIMIT pasan a usar el precio vivo** (`_last_traded_price`). Con el close de una vela de hace 3-8 min la PostOnly quedaría descolocada respecto al book, o rechazada por cruzarlo (`101215`).

Verificado en vivo tras el deploy: evalúa la barra 02:30 (`vol_rel=0,78`) en lugar de la 02:35 en formación (`vol_rel=0,12`). 5 tests nuevos, 57 en total.

⚠️ **Queda como deuda**: los niveles TP/SL (`get_last_take_profit_stop_loss`, `latest_values` en `colocando_TK_SL`) siguen leyendo la última fila. Es menos grave (son niveles de precio que se recalculan cada 50 s, no un filtro binario), pero es la misma incoherencia. Ver P5.

### 🐛 Bug 10/08 — tick size ausente: envenenaba el backtest y descolocaba las órdenes en prod — CORREGIDO Y DESPLEGADO

`SYMBOL_TRADING_RULES` existe **duplicada** en `pkg/backtesting.py` y `pkg/monkey_bx.py`, y las dos derivaron. Cualquier símbolo sin entrada cae a `price_tick=0.01`. En el backtest faltaban 8 de los 10 pares (la tabla tenía 6 símbolos, **4 de ellos ya removidos** del portfolio: DOT, HBAR, TRX, DOGE); en prod faltaban 7.

**En el backtest (commit `d895c7b`) — pérdidas falsas.** Los TP se redondeaban con `ROUND_DOWN` incondicional. Para CFX (~$0.042) el tick 0.01 es el **24% del precio**, así que el TP caía **por debajo del precio de entrada** y el trade cerraba en la vela siguiente como `TP` con −13%. Firma inconfundible: **todas las salidas al mismo precio redondo** (`0.040000`), duración de exactamente 1 barra, `exit_reason=TP` con `ret_pct` fuertemente negativo.
- CFX marcaba −60 a −98 USD según la ventana y explicaba **entre el 55% y el 97% de la pérdida del portfolio**. Con el tick real (1e-05): −1,5/−2,7/**+4,4**/**+8,4**.
- ⚠️ **El sesgo no es unidireccional**: APT (1,68% del precio) y ONDO (2,87%) estaban inflados **en positivo** y al corregir el tick pasan a negativos en las 4 ventanas. Sus parámetros se optimizaron contra una función objetivo corrupta.

**En prod (commit `c0e453e`) — órdenes descolocadas.** Live redondea en dirección conservadora (`_trigger_rounding`), así que no invertía los precios, pero los descolocaba, y **el síntoma es silencioso: el par simplemente deja de operar**.

| Par | Entrada LONG (diseñada −0,02%) | TP efectivo (diseñado 1,20%) |
|---|---|---|
| ONDO | **−2,41%** → −0,03% | **3,33%** → 1,21% |
| APT | −0,62% → −0,03% | **2,75%** → 1,21% |
| CFX | −0,05% → −0,02% | 1,38% → 1,21% |

Las entradas son **PostOnly**: una orden a −2,41% del mercado casi nunca llena. Confirmado en vivo el mismo 10/08 a las 12:20, antes del deploy: `entry_order_canceled_or_expired` / `protection_timeout` en ONDO.

**Guards añadidos** (45 tests nuevos, 118 en total):
- `_assert_tick_size_sane()` en `backtesting.py`, llamado desde `load_candles()` — **rompe** con `ValueError`. Preferimos un sweep caído a un número envenenado.
- `_warn_if_tick_implausible()` en `monkey_bx.py` — **avisa** por lifecycle event (`tick_size_implausible`), una vez por símbolo por proceso. En vivo no rompe: matar el bot es peor que un tick grueso.
- Umbral 0,1% del precio; los ticks legítimos rondan 0,001-0,03% (3× de margen).
- Ticks y `qty_step` traídos de `/openApi/swap/v2/quote/contracts`, no estimados. BNB pasa de 0.1 a 0.01 (el valor real).

**Al incorporar un par nuevo**: traer `pricePrecision`/`quantityPrecision` del endpoint de contratos (tick = 10^-pricePrecision) y agregarlo a **las dos** tablas. Ver memoria `backtest_tick_size_poisoning`.

### 🐛 Bug 25/08 — el parity-sim llenaba los TP con un simple toque — CORREGIDO (`cc36b90`)

`run_live_parity_portfolio` decidía el fill con `high >= price` y podía llenar TP1, TP2 y TP3 en la **misma vela**. Ninguna de las dos cosas ocurre en vivo:

- El live deja órdenes **LIMIT maker** (`tp_mode: partial_limit_tp`). Una orden que espera en el book no se ejecuta porque el precio la toque y rebote: está en una cola, y el toque ejecuta a los de adelante. Hace falta que el mercado **atraviese** el nivel (trade-through ≥ `limit_fill_buffer_bps`, 2 bps) o que el cierre lo confirme.
- El live manda **un tramo por vez** (`tp_one_at_a_time: true`) y espera el fill antes de someter el siguiente. Medido en prod: TP1 de XMR 02:17, TP2 02:31 — 14 min de diferencia.

La lógica conservadora (`should_fill_tp_limit` / `LimitFillPolicy`) **ya existía** y se usaba en `class Backtester`, el camino del **sweep**. El parity —el que decide todos los A/B de estructura de salidas— nunca la invocaba. Dos meses de A/B de salidas se apoyaron en fills optimistas.

**El modelo se ata a la config del LIVE** (`get_tp_mode`, `get_tp_one_at_a_time`), no al flag `conservative_limit_fills` de los params: ese flag existe para que un sweep elija su supuesto de ejecución y está en `False` en los 10 pares, así que colgarlo de ahí habría dejado el parity optimista igual. Si el live vuelve a TPs market, el parity lo sigue solo.

⚠️ **El fix corrige el modelo pero NO cierra el gap**: la tasa de posiciones que tocan TP pasó de 57% a 55% (90d), contra un **38% real**. Se probó además armar el BE intrabarra (como el live, que revisa cada 50 s, en vez de con el cierre de la vela): sólo lleva 55% → 50%. **La causa dominante del resto sigue sin identificar** — queda en P5.

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

### 📅 Plan con fechas — REFIJADO 28/07

**Julio (cerrado).** Los lunes 06, 13 y 20/07 corrieron en modo observación con params congelados. El 27-28/07 se descubrió que **buena parte de lo que se creía estar midiendo no estaba ocurriendo**: los TPs nunca escalonaron y los indicadores corrían sobre velas parciales. Los datos de julio sirven para diagnóstico, **no para veredictos**.

**Estado a 28/07 05:00 UTC — 3 fixes desplegados hoy, ninguno verificado en operación real todavía**

| Fix | Commit | Desplegado | Qué falta |
|---|---|---|---|
| Cantidad de TPs (base, remanente, `_round_step`) | `753ebd0` | 28/07 01:48 | ver una posición real cerrar íntegra por TPs |
| El TP escalonado no escalonaba | `077951b` | 28/07 04:17 | ver `tp1/tp2/tp3_filled` en `execution_ledger` |
| Velas en formación | `723e5f7` | 28/07 04:53 | 7/7 velas OK post-deploy (muestra de ~11 min); confirmar con días |

**Próximos días — sólo observar, no tocar**
1. **¿Despertó el portfolio?** Contar señales en `indicadores.csv` por par. Con velas correctas debería subir de ~1 señal/día en todo el portfolio. Si NO sube, el fix de velas no era la causa principal y hay que volver al diagnóstico.
2. **¿Escalonan los TPs?** En `execution_ledger`: `tp1_filled` → `tp2_filled` → `tp3_filled` con **precios distintos**. Si todos salen al mismo precio, el fix del escalonamiento no está actuando.
3. **¿Cierran íntegras?** Ninguna posición debe dejar remanente colgando sólo del SL (era el caso DYDX del 26/07).
4. **Velas**: bajar `cripto_price_5m.csv` de prod y comparar OHLC contra el API — las cerradas deben coincidir al 100% (excluir siempre la última, está en formación por diseño).

**✅ Gate de sesión — QUITADO el 18/08 (`e172f72`) y VALIDADO el 25/08**
- Medición previa: 6/17 señales bloqueadas (35%) del 31/07-03/08 y 5/15 (33%) del 10-18/08. Dos semanas independientes, el mismo tercio.
- Implementación: flip de config, `session.entry_hours_utc: []` en `live_benchmark_runtime.json` (`is_entry_hour_allowed_utc` devuelve True si no hay horas). Revertir = restaurar `[6,7,8,9,10,11,12,14,15,16,17,19,21,22]` + restart.
- **Resultado a 7,1 días**: entradas 7 → **17**, pares operando 4 → **9/10**, PnL +0,62 → **+2,87**. Y el test directo por hora de entrada: las horas **antes bloqueadas** dieron **+1,77 con 67% de winrate** contra −0,39 y 43% de las ya permitidas.
- Se esperaba comprar *muestra* pagando pérdida a corto plazo; salió muestra **y** ganancia. No reponer el gate.

**✅ Análisis de edge — HECHO el 10/08** (el check del 06/08 no se corrió; el PnL del 10/08 lo respondió igual: −4,87 USDT en 10 días, sólo ONDO en verde). Resultado y matices en "¿hay edge?" abajo. De paso destapó el bug de tick size.
- Receta que funcionó, para repetirla: top-up de velas desde el API (los datos locales estaban 41 días viejos; **NO bajar `long.csv` de prod**), `--live_parity` con **`--symbols` explícito** (sin él usa BTC-USDT y reporta 0 trades sin avisar — P5.4), y descomponer los trades en bruto vs costos, no mirar sólo el neto.
- **Regla que se ganó con sangre**: si un solo par explica >50% del PnL del portfolio, **es una alarma de datos, no una conclusión**. CFX explicaba el 55-97% y resultó ser el tick roto.

**✅ ~17/08 — análisis de edge rehecho con datos limpios — HECHO el 18/08.** Resultado en "¿hay edge?" abajo. Sigue negativo en 4/4 y el diagnóstico se afina: 7 de 10 pares no cubren su propio costo.

**✅ ~24/08 — veredicto de estructura de salidas — EMITIDO el 25/08: NO revertir TP×2.**
Pierde en 3 de 4 ventanas con el parity ya corregido. Detalle en "Revisión semanal 25/08". Lo que queda abierto de la estructura es **TP3, que nunca llena** (0 de 39).

**🔓 Congelamiento de parámetros: VENCIDO el 24/08.** Rigió desde el 03/07. Durante su vigencia se corrigieron 7 bugs de ejecución/datos (07, 08, 13, 14, 27, 28/07, 10/08) y ninguno era un parámetro. Ahora se pueden tocar params, pero con la cadencia de abajo y **con cross-val obligatoria**.

**Cola de trabajo — en este orden**
1. **P-TP3** 🆕 *(el más maduro; evidencia directa, no depende del gap de paridad)*: TP3 está a 3,84%-7,04% y nunca llena. Opciones a medir: acercarlo, redistribuir el ladder (`TP_LADDER_FACTORS` (0.6, 1.0, 1.6)) o eliminar el tercer tramo y repartir en dos. **Ojo**: el tercer tramo hoy es el que corre con el trailing, así que quitarlo no es gratis — hay que medirlo, no asumirlo.
2. **P-paridad** *(P5.2c)*: cerrar el gap de realización de TP (sim 50-55% vs vivo 38%). Mientras siga abierto, todo A/B de salidas —incluido P-TP3— conserva un sesgo optimista de ~12-15 puntos.
3. **P-costos**: el lever que señala la medición del edge (bruto/trade ≈ costo/trade, slippage = 50% de los costos). Bajar frecuencia en los sobre-operadores (AVAX) y subir el bruto por trade.
4. **P-proceso**: cadencia MENSUAL de re-optimización (máx. 1-2 pares/mes; cada cambio debe ganarle a "no tocar nada" en cross-val).
5. **P-pesos**: concentrar capital (menos pares o pesos escalonados con cap) en vez del equal-weight actual. ⚠️ Ver la trampa de selección in-sample en "¿hay edge?".

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
4. 🆕 **Check de paridad de datos (alto valor, barato)**: comparar semanalmente el OHLC de `cripto_price_5m.csv` de prod contra el API para las velas cerradas — deben coincidir al 100%. El bug de velas parciales del 28/07 vivió meses sin detectarse y contaminó todos los indicadores. Excluir siempre la última vela.
5. 🆕 **Check de ejecución diseñada**: contar fills `tp1/tp2/tp3` en `execution_ledger`. Si en N cierres hay 0 fills de TP, algo está roto aguas arriba — fue la señal que gritó el bug del escalonamiento durante un mes sin que nadie la leyera. **Medir siempre por EVENTOS** (`tp1_filled` / `entry_order_filled`), nunca agrupando cierres de `PnL.csv` por proximidad temporal: eso dio 12% cuando el valor real era 38%.
7. 🆕 **Tests anclados a fechas fijas caducan en silencio** (25/08): `test_price_pull_partial_candles` usaba el 23/07 y empezó a fallar al pasar los 30 días de `SIGNAL_HISTORY_DAYS` — la purga borraba las velas del fixture. Falló días sin que hubiera regresión, justo en los tests que cubren el bug de velas parciales. Cualquier test que dependa de una ventana de retención debe anclarse al presente.
6. 🆕 **Entradas que expiran sin llenar** (10/08, barato y de alto valor): contar `entry_order_canceled_or_expired` con `reason=protection_timeout` por par. Un par que acumula timeouts **no está mudo por señal, está mudo por precio** — su orden PostOnly se coloca donde no puede llenar. Fue el síntoma del bug de tick size y se confundió con selectividad de filtros durante un mes.

### 🆕 P5 — Deuda de paridad live ↔ sim (abierta desde 28/07)

Todos surgieron al diagnosticar la mudez. Ninguno es un parámetro: son diferencias entre lo que prod ejecuta y lo que el backtest simula, y **hacen que los A/B midan algo distinto de lo que se cree**.

1. ~~**La decisión de entrada se toma sobre la vela en formación**~~ → ✅ **CORREGIDO 31/07** (`15b1e1b`), era la causa de 3 días sin órdenes. Ver "Bug 31/07" arriba. **Queda el residuo**: los niveles TP/SL (`get_last_take_profit_stop_loss` y `latest_values` en `colocando_TK_SL`) siguen leyendo la última fila (vela en formación). Menos grave que el filtro de volumen, pero es la misma incoherencia.
1b. **Latencia de decisión de 3-8 min.** Con el fix, a las :03 la última vela cerrada es la de :55 (cerró a :00): se decide 3 min después del cierre. El backtest decide en el cierre exacto. Se podría reducir moviendo el pull a :00,:05,… y el job de entradas a :01,:06,… Medir antes de tocar: puede no valer el riesgo de leer velas aún no publicadas por el API.
2. ~~**El gate horario no existe en el sim**~~ → ✅ **RESUELTO 18/08 quitando el gate del live**: ahora live y sim son ambos 24/7, que era la condición de paridad. Si alguna vez se repone un gate, hay que simularlo.
2b. ~~**El parity llenaba los TP con un simple toque**~~ → ✅ **CORREGIDO 25/08** (`cc36b90`). Ver "Bug 25/08".
2c. 🔴 **Gap de realización de TP, sin causa identificada (abierto).** El sim da 50-55% de posiciones que tocan TP; en vivo es **38%** (15 `tp1_filled` sobre 39 entradas desde el 28/07). Ya descartados por medición: el modelo de fill (57→55%) y el BE armado con el cierre en vez de intrabarra (55→50%). **Mientras siga abierto, todo A/B de salidas conserva un sesgo optimista de ~12-15 puntos.** Próximos sospechosos a medir: (a) la latencia de 50 s entre que se confirma un tramo y se somete el siguiente, que el sim no modela; (b) el trailing stop del live, que puede cerrar antes de TP y en el sim se actualiza sólo una vez por vela; (c) que el sim abre más posiciones que el live y el mix es distinto.
3. 🐛 **`--entry_hours_utc` no tiene efecto en `--live_parity`** — se acepta el flag y se ignora en silencio. Para medir el gate hubo que particionar los trades por hora de entrada a mano. Arreglar o al menos hacer que falle ruidosamente.
4. **`--live_parity` sin `--symbols` usa BTC-USDT por defecto** (que ni está en el portfolio) y reporta 0 trades sin avisar. Fácil de malinterpretar como "no hay señales".

### P4 — Limpieza

1. ~~**Borrar `pkg/best_prod.json.bak.*`**~~ → ✅ HECHO 22/06: se conservó el más reciente (`.20260502_234317`), borrados 6. Desde 03/08 está en `.gitignore` (sigue en disco, ya no ensucia `git status`).
2. **Branch `main` en prod 35 commits ahead de `origin/main`**. No es problema funcional pero podría hacerse un `git push prod-merge` ocasional para mantener historia visible. No urgente.
3. ~~**`dashboard/jobs.py` + `5_🧪_Backtesting.py` sin commitear**~~ → ✅ **HECHO 03/08** (`3afb438`). Verificado sin credenciales antes de commitear. `.gitignore` cubre ahora `.venv-*/`. **`git status` queda limpio.**

### Memoria persistente

Las gotchas detectadas y validadas están en `~/.claude/projects/-Users-will-Documents-proyectos-TRobot/memory/`:
- `backtest_history.md` — qué se rotó cada semana, monedas removidas/probadas
- `feedback_deploy_gotchas.md` — qué hacer cuando el deploy falla
- `feedback_atexit_bug.md` — el bug del atexit y cómo aislarse de él
- `live_sim_data_parity.md` — velas en formación (corregido 28/07) + gate horario que el sim no simula
- `backtest_tick_size_poisoning.md` — el tick por defecto (0.01) que envenenaba el backtest y descolocaba las órdenes en prod
- `execution_cost_notes.md` — TPs LIMIT maker, y el escalonamiento que no escalonaba
- `parity_sim_realization_gap.md` — cuándo el sim sobrestima y por qué mandar el real

### ✅ Verificación 03/08 — los 4 fixes de la semana pasada FUNCIONAN

Primera semana con la cadena completa operativa (señal → gate → orden → TP escalonado) desde el 03/07.

**1. Volvieron las órdenes** (valida `15b1e1b`). 20 submits desde el 31/07 02:31, **todos aceptados** (`code=0`, sin rechazos). Operaron **7 pares** — BNB, ONDO, BCH, LINK, AVAX, APT, DYDX — incluidos APT/BCH/LINK/ONDO, mudos desde el 03/07. Ritmo de cierres: **9/semana vs 3/semana** antes de los fixes.

**2. Los TPs escalonan** (valida `077951b` + `753ebd0`). Caso BCH-SHORT del 01/08, el primer `tpN_filled` tras 133 cierres sin ninguno:
- `tp1_submitted` 0.059 @ **205.99** → `tp1_filled` (+0.1675)
- confirmado con `notes=stage_advanced_before_resubmit` → **es la reconciliación del fix actuando**
- `tp2_submitted` 0.059 @ **198.65** — precio distinto: escalonó de verdad
- BE se activó tras TP1 y el resto cerró en verde. Total BCH **+0.1475**.

**3. Señales estables**: 17 en 3,9 d (**130/30d**, vs 38 pre-fix), 8/10 pares produciendo. El caudal no fue un pico.

**4. Sin remanentes huérfanos** ni rechazos `101485`/`110422`/`109400` en toda la semana.

⚠️ **Pero el PnL de la semana es −1,21 USDT** (5 cierres: 3 `stop_loss`, BCH +0,15 por TP). Acumulado agosto −1,24. **Esto era lo esperado y está advertido desde el 28/07**: arreglar la ejecución da *muestra*, no *ganancia*. Con 5 cierres no hay veredicto posible — pero ver "Edge" abajo.

### ✅ Revisión semanal 18/08 — primera semana en verde; el fix de ticks NO despertó a ONDO/APT

**Sistema**: 7,3 días estable desde el deploy del 10/08. `NRestarts=0`, 0 errores, md5 consistente, **0 avisos `tick_size_implausible`**. 9 fills de TP en la semana.

**PnL: +0,62 USDT** (vs −3,62 la semana previa) — LINK +0,80, DYDX +0,26, BCH +0,14, XMR −0,59. Balance 193,74. Sólo **2 cierres, ambos `be_stop`**: el BE protegió capital. Con 2 cierres no hay veredicto, pero rompe la racha.

⚠️ **La predicción del 10/08 falló y hay que registrarlo.** Se esperaba que el fix de ticks hiciera despertar a ONDO y APT. **Tuvieron cero eventos en 7,3 días** — ni un submit. Los `protection_timeout` pasaron de 2 a 0, pero con n=2 eso no prueba nada.
- La causa real de su mudez **no era (sólo) el precio**: es que **casi no generan señal** — 1 cada uno en 2.112 barras.
- El fix seguía siendo necesario (los precios estaban objetivamente mal), pero **no era la causa principal**. Tercera vez que se atribuye a una causa y aparece otra encima: no cerrar el diagnóstico de un par mudo con una sola explicación.

**El gate es ahora el cuello de botella medido**: de **15 señales, 5 en hora bloqueada (33%)** — APT 20h, BNB 03h, LINK 13h, AVAX 23h, LINK 01h. Las 10 que pasaron dieron 7 submits (las 3 perdidas son cooldown de DYDX y 1 de ONDO). **La única señal de APT en toda la semana murió en el gate.**

**Caudal de señales**: 15 en 7,3d = ~62/30d, la mitad de las ~130/30d medidas el 03/08. Vigilar si sigue cayendo.

### ✅ Revisión semanal 25/08 — la mejor semana; gate validado; veredicto de salidas emitido

**PnL +2,87 USDT en 7,1 días** — la mejor semana registrada. Balance **196,61** (era 193,74). Sistema estable, `NRestarts=0`, 0 errores.

**El gate quedó validado por el resultado.** Entradas que llenaron: **7 → 17**. Pares operando: **4 → 9 de 10**. APT (+0,84) y ONDO (+1,00) volvieron a operar. Y el test directo, particionando por la hora de entrada:

| Origen de la entrada | n | PnL | Winrate |
|---|---|---|---|
| Hora que ya estaba permitida | 7 | −0,39 | 43% |
| **Hora antes bloqueada** | 6 | **+1,77** | **67%** |

Las horas que el gate bloqueaba fueron las buenas. Muestra chica (n=6), pero contradice el miedo de que abrir el caudal sumaría perdedores. **No reponer el gate.**

**✅ Veredicto de estructura de salidas (el hito del ~24/08): NO revertir TP×2.** Con el parity ya corregido (ver "Bug 25/08"), revertir pierde en **3 de 4** ventanas: 30d +3,35 / 60d −3,26 / 90d −0,60 / 120d −5,63. *Antes* del fix daba 2/4 y era no concluyente — o sea que el fix del instrumento fue el que permitió emitir el veredicto.

🔴 **Lo que queda abierto y es el hallazgo más sólido: TP3 nunca llena.** **0 fills en 39 entradas** desde el 28/07 (tp1 15, tp2 8, tp3 **0**). Está a 3,84%-7,04% según el par (ETH y ONDO en 7,04%) en una estrategia de velas de 5m. El tercer tramo es el **34% de cada posición** y jamás captura: siempre termina colgado del stop. No depende del modelo de fill ni del gap de paridad — es geometría.

**La geometría del problema, para tenerla escrita** (ejemplo APT): `be_trigger` **+0,4%**, TP1 **+1,44%**, `sl_pct` **−1,5%**. Entre +0,4% y +1,44% hay **un punto entero de zona muerta** donde el BE ya está armado pero no hay TP: cualquier retroceso cierra plano. La firma en los datos: de 52 posiciones que nunca tocaron TP1, 15 perdieron −10,00 mientras 26 ganaron apenas +6,84.

⚠️ **Gotcha de medición, no repetirlo**: la tasa de TP en vivo se mide contando **eventos** (`tp1_filled` vs `entry_order_filled`), no agrupando cierres de `PnL.csv` por proximidad temporal. Agrupar con una ventana de 90 min dio 12% —cifra errónea que casi motiva un cambio de parámetros— porque parte en dos las posiciones cuyos tramos se separan horas (XMR: TP1 15:21, TP2 22:58). El número correcto es **38%**.

### 🔴 La pregunta que queda abierta: ¿hay edge?

Con la ejecución ya correcta, el diagnóstico se desplaza de "el bot no hace lo que debería" a "lo que debería hacer, ¿gana?".

**Medición del 18/08** (parity-sim, top-up de 160k velas del API — serie continua, 0 huecos —, `--symbols` explícito, ticks corregidos). Ojo con las magnitudes: el sim corre con capital 1000 y compounding mientras el balance real ronda los 194 USDT, así que están infladas ~5×. **Leer signos y proporciones, no USDT.**

| Ventana | Trades | Bruto | Costos | **Neto** | Winrate | Payoff |
|---|---|---|---|---|---|---|
| 30d | 144 | −4,88 | 32,38 | −37,26 | 56,2% | 0,44 |
| 60d | 282 | −5,32 | 66,44 | −71,76 | 54,3% | 0,48 |
| 90d | 430 | +65,29 | 100,68 | −35,40 | 58,6% | 0,59 |
| 120d | 552 | +121,43 | 130,70 | **−9,26** | 60,9% | 0,62 |

Negativo en 4/4, consistente con la medición del 10/08 (que daba −47,5/−62,5/−23,9/−18,7). **El bruto es negativo en 30 y 60d**: en el período reciente no hay ni siquiera edge bruto.

**El hallazgo accionable: 7 de 10 pares no generan bruto suficiente para pagar su propio costo.**

| ¿bruto/trade > costo/trade? | ventanas que sí |
|---|---|
| **BCH** | **4/4** ✅ |
| AVAX, BNB, CFX | 1/4 |
| APT, DYDX, ETH, LINK, ONDO, XMR | **0/4** |

- **LINK y XMR tienen bruto NEGATIVO** en 120d (−0,33 y −4,17): perderían aunque los costos fueran cero. No es un problema de costos, es que no tienen edge.
- **El slippage es el 50% de los costos** (65,2 vs 61,5 de comisiones y 4,0 de funding). Es el componente más grande y el menos atacado.
- Esto reencuadra por qué **5 de 6 A/B fallaron desde junio**: todos movían *qué* trades tomar, ninguno tocó la relación bruto/costo por trade.

⚠️ **Trampa de selección detectada, no caer en ella**: el subconjunto "solo BCH/BNB/CFX" da −0,17/+4,73/+19,48/+40,78 en 30/60/90/120d. Se ve bien, pero **mejora monótonamente con la ventana** — la selección la manda el período más largo, o sea que es selección in-sample. Sólo BCH tiene perfil robusto (4/4). No usar esto como base de una rotación sin cross-val fuera de muestra.

- El patrón de asimetría del 03/07 **persiste**: payoff 0,44-0,62 según ventana; los perdedores viven menos que los ganadores en el sim (277-321 vs 294-359 min), al revés que en el real.
- ⚠️ **Estas cifras son del 18/08 y traen el sesgo optimista del fill de TP** (corregido el 25/08). Rehacer la descomposición bruto/costos con el parity ya arreglado antes de usarla para decidir. El signo —negativo en las ventanas cortas— no debería cambiar, pero las magnitudes sí.

---

## Estado al cierre del 25/08

**Prod**: activo desde **18/08 01:16 UTC**, 7,1 días sin reinicios ni errores. HEAD `c0fc7c2`. `pkg/best_prod.json` md5 `57daca5b` coincidiendo local = HEAD = prod. Balance **196,61 USDT**.

**Portfolio**: 10 pares. **Ningún parámetro tocado desde el 03/07** — el congelamiento venció el 24/08 y hasta hoy no se tocó nada.

**PnL semana 18→25/08: +2,87 USDT** — la mejor semana registrada. Previa +0,62, anterior −3,62. Tres semanas de mejora consecutiva.

**El cambio de la semana fue el gate**: entradas 7 → 17, pares operando 4 → 9/10. Ver "Revisión semanal 25/08".

**Los 6 bugs corregidos**, todos de ejecución, datos o medición — **ninguno de parámetros**:

| Bug | Commit | Efecto real |
|---|---|---|
| Cantidad de los TPs | `753ebd0` | dejaba remanente sin cobertura de TP |
| El TP escalonado no escalonaba | `077951b` | los 3 tramos al precio de TP1, ~1 mes |
| Velas en formación en el histórico | `723e5f7` | indicadores sobre closes falsos, meses |
| Decisión sobre la vela en formación | `15b1e1b` | **0 órdenes en 3 días** |
| Tick size ausente (10/08) | `d895c7b` + `c0e453e` | backtest envenenado (CFX 55-97% de la pérdida) + ONDO/APT colocando entradas donde no llenaban |
| **Fill de TP por simple toque** (25/08) | `cc36b90` | el parity infló la tasa de TP; **2 meses de A/B de salidas** se decidieron con fills fantasma |

**Lo que se aprendió, y ya van cuatro veces**: cada uno de estos bugs se leía como una *conclusión sobre la estrategia* — "CFX no tiene edge", "ONDO y APT son mudos", "los filtros son muy selectivos", "TP×2 gana 5/5" — cuando era un defecto de ejecución, de datos o **del instrumento de medición**. La telemetría que los delataba existía y no se miraba. **Antes de concluir algo sobre la estrategia, verificar que el bot hizo lo que se cree que hizo — y que el sim mide lo que se cree que mide.**

**Al retomar, en este orden**:
1. **P-TP3**: nunca llena, 0 de 39. Es el trabajo más maduro y no depende del gap de paridad. Ver la cola en "Plan con fechas".
2. **P-paridad (P5.2c)**: sim 50-55% vs vivo 38% de realización de TP, causa sin identificar. Sesga ~12-15 puntos todo A/B de salidas.
3. **Rehacer la descomposición bruto/costos** con el parity corregido: las cifras de "¿hay edge?" son del 18/08 y traen el sesgo del fill.
4. **Vigilar el caudal de señales**: venía cayendo (~62/30d el 18/08 vs ~130/30d el 03/08) y el gate off lo compensó. Si vuelve a caer con el gate ya quitado, no queda palanca de caudal.
