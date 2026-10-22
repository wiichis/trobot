# TRobot — Guía rápida

**Desde el 22/10/2026 el foco es un bot de BTC por rangos en BingX spot.** Las reglas están en [docs/reglas_bot_btc.md](docs/reglas_bot_btc.md) (v0.1). El plan tiene cuatro fases: archivar, datos, backtest como puerta de decisión, y live sólo si el backtest pasa.

## La versión anterior (archivada, no borrada)

El bot de 5 min multi-par sobre perpetuos se archivó porque no tiene edge medido.
- **Dónde quedó**: el tag `v1-bot-5m-final` y la rama `archivo/bot-5m`.
- **Su guía completa**: [docs/legado/CLAUDE_bot_5m.md](docs/legado/CLAUDE_bot_5m.md), con dos años de bugs, experimentos y lecciones.
- **El código sigue en el repo** (`pkg/monkey_bx.py`, `pkg/indicadores.py`, `pkg/backtesting.py`…), pero sus jobs no se registran.
- **Reactivarlo**: `"bot_5m": true` en `pkg/jobs.json` + restart.

## Estado de las fases

| Fase | Estado |
|---|---|
| 0. Archivar el bot de 5 min y apagar los experimentos | ✅ 22/10 |
| 1. Velas spot 1 h de BTC + reglas del exchange (mínimo, step, tick, retiro) | ✅ 22/10 — `scripts/btc_rangos_datos.py`, paridad Binance↔BingX ~0 bps |
| 2. Backtest contra DCA puro, con selección 2021-2023 y validación 2024-2026 intacta | ❌ 22/10 — **no viable**: 0 de 5 finalistas pasan, no le gana al placebo. Ver [docs/btc_rangos_fase2_informe.md](docs/btc_rangos_fase2_informe.md) |
| 3. Bot live spot: papel → real, sólo si la Fase 2 pasa | ⛔ no se construye con la v0.1 |

Herramientas de la Fase 2: `pkg/btc_rangos_reglas.py` (señales), `pkg/btc_rangos_motor.py` (motor; el mismo que usaría el live), `scripts/btc_rangos_backtest.py` (grilla + validación + placebo, ~70 s) y `scripts/btc_acumulacion_backtest.py` (§13).

## Qué corre en el server

Los jobs se encienden en `pkg/jobs.json`. Cambiar un flag = editar + restart. Si el archivo falta, valen los defaults de `main.py`, con el bot de 5 min apagado.

| Job | Estado | Módulo |
|---|---|---|
| Resumen diario (BTC + S&P + Dow) | activo | `pkg/daily_digest.py` |
| Alerta de ventana de compra/venta de BTC | activo | `pkg/btc_alertas.py` (su gate `alerta_ventana` está en `pkg/btc_bots.json`) |
| Bot de 5 min multi-par | apagado | `pkg/monkey_bx.py` y compañía |
| Cuatro bots BTC en papel | apagado | `pkg/btc_bots.py` |
| Momentum entre pares en papel | apagado | `pkg/paper_momentum.py` |
| Monitor de funding | apagado | `pkg/funding_monitor.py` |

```bash
ssh -i ~/Documents/proyectos/ls_keys/trobot4.pem ubuntu@98.81.217.194
# sudo systemctl {status|restart|stop|start} trobot · journalctl -u trobot -f · ~/TRobot (branch main)
```

- **Deploy**: hacer push a `origin/codex/nuevo-bot-estrategia`; en prod, `git fetch && git merge origin/codex/nuevo-bot-estrategia --no-edit` + `sudo systemctl restart trobot`.
- **Consultar el exchange desde prod**: las claves vienen del `EnvironmentFile`, así que hay que cargarlas: `sudo bash -c 'set -a; . /etc/default/trobot; set +a; cd /home/ubuntu/TRobot && PYTHONPATH=. env/bin/python3 ...'`. Exigir siempre `code == 0`: sin claves el API devuelve un error que se confunde con "no hay nada".
- **Servidor**: t3.micro con 914 MB. Los backtests y los barridos se corren **siempre en el Mac**.

## Qué se recicla del bot anterior

- **Firma de requests de BingX**: `get_sign` y `send_request` en `pkg/bingx.py`. Sirven igual para los endpoints spot, que todavía no están implementados.
- **Credenciales**: `pkg/credentials.py`. Ojo: importar `pkg` arrastra las credenciales y `monkey_bx`. Los scripts de laboratorio cargan sus módulos por ruta.
- **Avisos**:
  - `pkg/lifecycle_events.emit_lifecycle_event(categoria, severidad, mensaje_md=...)`, usando `escapar_md` para los valores.
  - Cada categoría nueva va a los mapas de `pkg/telegram_alerts.py`; si no, recibe el cooldown por defecto de 900 s.
- **Indicadores**: `pkg/ta_shared.py` (ema, rsi, atr, adx).
- **Reglas de BTC**: `pkg/btc_reglas.py`.
- **Lector de zips de Binance**: `_leer_velas` en `scripts/btc_lab_temporalidades.py`.
- **Estado atómico**: `_guardar_estado` y `velas_cerradas` en `pkg/btc_bots.py`.
- **Datos**:
  - `archivos/btc/binance_1d/`: spot diario, 2017-2026.
  - `archivos/btc/binance_1h/`: **perpetuo** 1 h, 2020-2026.
  - El spot 1 h todavía hay que bajarlo.

## Lecciones que se heredan (detalle en el legado y en la memoria)

1. **Protocolo antes de mirar**: las reglas y las ventanas se fijan antes de correr. Se elige en un tramo y se valida en otro que no participó. Lo contrario terminó siempre en sobreajuste: el #1 del entrenamiento pasó 0 de 25 veces.
2. **Mes a mes, no ventanas acumuladas**: si un solo mes o un solo factor explica más del 50 % de la ganancia, es una alarma, no una conclusión.
3. **Placebo**: una regla que nunca vende con pérdida da un winrate alto por construcción. La señal de entrada tiene que ganarle a entrar al azar.
4. **La fuente de verdad es el exchange**: los fills se confirman con el estado de la orden, no con lo que el bot infiere.
5. **Los bugs viven en los bordes con el mundo real**:
   - los ids que vuelven float de un CSV;
   - el `NaN` que viene de un campo vacío;
   - un error de lectura que se toma como "vacío";
   - el reloj del server.
   Los tests tienen que pasar por el ciclo guardar → leer y por respuestas reales del exchange.
6. **Nada de valores por defecto en las reglas de contrato** (tick, step, mínimo): el tick por defecto envenenó meses de backtest.
7. **Contrastar contra un cálculo independiente**, no contra lo que el bot escribió.
