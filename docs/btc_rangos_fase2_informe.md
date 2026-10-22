# Bot BTC por rangos — informe de la Fase 2 (backtest), 22/10/2026

**Veredicto: NO viable. Según el plan, no se construye el bot live.** Ninguna combinación pasa los criterios en validación.

## Qué se midió

- **Reglas**: las de [reglas_bot_btc.md](reglas_bot_btc.md) v0.1, con los parámetros tal cual vienen en la especificación, sin optimizar.
- **Código**:
  - `pkg/btc_rangos_reglas.py`: señales.
  - `pkg/btc_rangos_motor.py`: slots, salidas y cierre mensual. Es el mismo motor que usaría el live.
  - `scripts/btc_rangos_backtest.py`: la grilla. Su protocolo quedó escrito en el encabezado antes de correrla.
- **Datos**: BTC/USDT spot 1 h de Binance, 2021-01 → 2026-10.
  - Paridad contra BingX spot en 4.776 velas: close con mediana de 0 bps y p99 de 2,6 bps.
  - Script: `scripts/btc_rangos_datos.py`.
- **Costos**: 0,1 % de comisión + 0,05 % de slippage por lado.
- **Ejecución**: la señal se toma al cierre y se compra en la apertura siguiente.
- **Salidas**: intrabarra conservador, con un piso que impide vender con pérdida.
- **Grilla**: 200 combinaciones = slots {A, B 1,5/2/3 %} × entradas {E1 2/3 %, E2, E3 7/14 d} × filtro F {no, sí} × salidas {S1 1,5/2/3 %, S2, S3}. Todas con G4.
- **DCA**: 100 fijo, 400 fijo y aleatorio de 100 a 900 (5 semillas).
- **Selección**: 2021-01 → 2023-12. **Validación**: 2024-01 → 2026-09, sin usarla para elegir.
- **Métrica**: patrimonio final en BTC del bot contra **A**, el DCA puro con los 200 USDT guardados como USDT.
  - También se mide contra **B**: DCA más los 200 USDT comprados en BTC el primer día.

**Controles hechos**:
- Sin señales, el bot da exactamente el benchmark A (verificado en 2 escenarios).
- Dos operaciones de la mejor combinación, recalculadas a mano contra las velas (señal, fill, TP, ganancia), coinciden con el motor al centavo.
- 12 tests del motor en `tests/test_btc_rangos_motor.py`.

## Resultados

**Selección (2021-2023)**: sólo **2 de 200** combinaciones le ganan al DCA puro (A), y **1 de 200** al B. La mediana es −1,1 % de BTC.

Lo que separa ganar de perder es el **costo de las atrapadas**. Con DCA 400, sobre las 200 combinaciones, en medianas:

| | ganancia realizada | costo de atrapadas | comisiones |
|---|---|---|---|
| 2021-2023 | +381 USDT | −481 USDT | 64 USDT |

Neto anual mediano: 2021 +15 · **2022 −201** · 2023 +87. Un solo año bajista se come todo lo ganado en los demás. Pasa lo que la regla anticipa: se compra la caída y la caída sigue.

Ninguna dimensión salva el resultado (puntaje mediano por dimensión, en % de BTC):

| dimensión | variantes (mediana) |
|---|---|
| slots | A −0,76 · B 1,5 % −1,53 · B 2 % −1,49 · B 3 % −1,22 |
| entradas | E1 2 % −0,85 · E1 3 % −1,00 · E2 −1,89 · E3 14 d −1,53 · E3 7 d −1,81 |
| filtro F | sin −1,34 · con −1,02 |
| salidas | S1 1,5 % −1,40 · S1 2 % −1,08 · S1 3 % −0,79 · S2 −1,05 · S3 −1,23 |

**Validación (2024-2026) de las 5 mejores**:

| combinación | sel. | val. A | val. B | placebo p90 | neto anual (DCA 400) | viable |
|---|---|---|---|---|---|---|
| A · E2 · F · S1 3 % | +0,99 | **+0,28** | −1,94 | 0,43 (real 0,02) | 2024 +28,7 · 2025 −7,0 · 2026 −23,4 | ❌ |
| A · E2 · F · S1 2 % | +0,36 | +0,00 | −2,21 | 0,14 | −0,8 · +2,4 · −21,8 | ❌ |
| A · E3 7 d · S2 | −0,09 | −0,11 | −2,32 | −0,62 | +22,6 · −1,5 · −52,9 | ❌ |
| A · E3 7 d · S3 | −0,12 | −0,11 | −2,32 | −0,60 | +19,1 · −1,3 · −51,1 | ❌ |
| A · E3 14 d · F · S3 | −0,12 | −0,10 | −2,31 | 0,06 | −2,8 · +13,4 · −21,9 | ❌ |

- La mejor queda **+0,28 % de BTC** sobre guardar los 200 en USDT: con DCA 400 son **+2,3 USDT en 33 meses**. Contra haber comprado esos 200 en BTC el primer día queda en **−173 USDT**.
- **No le gana al azar**: entrar en momentos aleatorios, con las mismas salidas, da lo mismo o más. La señal de entrada no aporta.
- **El neto anual con atrapadas es negativo en 2025 y 2026** en las 5 finalistas.
- **Sensibilidad** (todo sólo con closes) y **destino de la ganancia** (G1-G4): no cambian el veredicto. G1 (todo a holding) es apenas mejor que G4.
- **Meta de 1-2 % mensual**: se alcanza en ~50 % de los meses, pero los meses con atrapadas pierden entre −13 % y −16 % del capital del bot.

## Modo Acumulación (§13)

`scripts/btc_acumulacion_backtest.py` compara 27 variantes (tramos 3/4/6 × E1/E2/escalonado × plazo 7/14/21 días) contra comprar todo de una vez, con un inicio cada día.
- La acumulación gana en el **52-78 %** de los inicios, pero por poco: cuando pierde, pierde más, porque el precio se escapa.
- **Mejora media del precio**: entre −0,4 % y +0,5 % en selección, y **negativa en las 27 variantes en validación** (entre −0,04 % y −0,39 %).
- **Conclusión**: comprar de una vez es igual o mejor. Confirma lo que ya mostró `scripts/btc_compras_semanales.py`: elegir el momento de compra no mejora la compra fija.

## Por qué no funciona (el mecanismo)

1. **La asimetría está en el diseño**: cada ciclo ganador deja +0,7 a +2,7 % sobre un slot de ~66 USDT, menos 0,3 % de costos. Cada atrapada cuesta la caída entera del mes.
2. **"Nunca vender con pérdida" no evita la pérdida, la reclasifica**: el slot atrapado pasa a holding a un precio peor que el del DCA de ese mes. El winrate de ~88 % es un efecto de construcción, no un edge, y por eso el placebo rinde igual.
3. **Los 200 USDT casi siempre están en USDT**, y en un período en que BTC subió. Por eso el benchmark B le gana a todo por ~2 % de BTC.

## Qué queda en el repo

- **Datos**: `archivos/btc_rangos/velas_1h.csv`, `paridad_1h.csv` y `reglas_spot.json` (minQty 0,0001826 BTC ≈ 15 USDT, step 1e-6, tick 0,01).
- **Comisión de retiro de BTC en BingX**: 0,00004 BTC por la red BTC.
- **Subcuentas**: el API las soporta (hoy hay 0).
- **Resultados completos**: `archivos/btc_rangos/backtest/` (no versionado).
- **El motor y las reglas**: sirven para probar una v0.2 en minutos.

---

## v0.2 — sólo abrir slots en la ventana COMPRA (22/10/2026): tampoco es viable

**Regla, fijada antes de correr**:
- No se abre ningún slot si el último cierre diario completo está bajo su media de 100 días. Es la ventana COMPRA de `pkg/btc_alertas.py`, con la misma condición.
- Los slots ya abiertos se gestionan igual.
- Se repiten las mismas 200 combinaciones, períodos, criterios y placebo. El placebo también respeta el régimen.
- Comando: `python3 scripts/btc_rangos_backtest.py --regimen`; salidas en `archivos/btc_rangos/backtest_v02/`.

**Control**: el régimen calculado se activa en la vela 1 h que cierra el 19/08/2026, el día con el que arranca la tendencia de la alerta en vivo (`ventana_log.csv`). Hay además un test que verifica que nunca usa un día sin cerrar. El régimen permite comprar el 50 % del tiempo en la selección y el 56 % en la validación.

**Validación de las 5 mejores**: **0 de 5 viables**. Ninguna cumple los cuatro criterios a la vez.

| combinación | sel. | val. A | val. B | gana placebo | años + | viable |
|---|---|---|---|---|---|---|
| B 3 % · E1 3 % · S3 | +0,49 | +0,05 | −2,17 | ✅ | 1 de 3 | ❌ |
| A · E2 · F · S1 3 % | +0,38 | −0,79 | −2,97 | ❌ | 2 de 3 | ❌ |
| A · E1 2 % · S1 3 % | +0,29 | +0,75 | −1,50 | ❌ | 1 de 3, y un mes explica 2,6× el total | ❌ |
| B 1,5 % · E1 3 % · F · S1 2 % | +0,17 | −0,12 | −2,33 | ✅ | 2 de 3 | ❌ |
| B 1,5 % · E2 · F · S1 3 % | +0,14 | +0,17 | −2,05 | ❌ | 1 de 3 | ❌ |

En las 5, 2026 es negativo (entre −13,5 y −31 USDT). Ninguna le gana al benchmark B.

**Las 200 combinaciones en las dos versiones** (informativo, no se usó para elegir):

| | selección: mejora en | validación: mejora en | validación: mediana v0.1 → v0.2 |
|---|---|---|---|
| filtro de régimen | 189 de 200 | **39 de 200** | +0,12 → −0,23 |

- El filtro **ayuda donde se eligió** (2022, con la bajada de 2021-2022) y **empeora en 2024-2026**: bloquea compras de caídas en un mercado alcista, que eran las que rendían. Y la caída de 2026 empezó con el precio todavía sobre la media, así que igual dejó slots atrapados.
- La correlación de rankings entre selección y validación es **−0,17** con filtro y **0,02** sin él. Lo que gana en un período no anticipa lo que gana en el otro: es la misma lección de `optimizar-params-no-sirve`.
- Sin filtro, 123 de 200 combinaciones fueron positivas contra A en validación, pero por ~0,1 % de BTC (unos pocos USDT), no había forma de saber cuáles de antemano, y todas pierden contra B.

**Conclusión**: el problema no es cuándo se compra sino la estructura. La ganancia por ciclo es chica y la atrapada se lleva la caída entera. Ningún filtro de entrada probado lo cambia.

---

## v0.3 — sólo el capital del bot, sin cierre forzado (22/10/2026): pasa justo, pero no le gana a comprar BTC directamente

**Planteo del usuario**: el holding no se toca; el bot opera sólo con sus 200 USDT y su ganancia ayuda a comprar BTC nuevo.

**Regla, fijada antes de correr**:
- El capital es fijo y no hay reposición desde el DCA.
- Un slot atrapado espera, sin plazo, a poder venderse con ganancia.
- Toda la ganancia compra BTC en cada cierre mensual (G1).
- El DCA queda fuera de la cuenta, porque es igual con o sin bot.
- Las mismas 200 combinaciones, períodos, placebo y criterios, sin filtro de régimen.
- Comando: `python3 scripts/btc_rangos_backtest.py --sin_cierre`; salidas en `archivos/btc_rangos/backtest_v03/`.

**Resultado formal**: **1 de 5 finalistas pasa los 4 criterios**: B 1,5 % · E1 2 % · S1 3 %, quinta en la selección.
- Pasa el placebo por poco: +60,8 % contra un p90 de 58,4 %.
- Es la tercera variante probada en el día, así que un aprobado ajustado tiene mucha chance de ser azar.

**Contra la alternativa obvia, pierde en los dos períodos**. Valor final de los 200 USDT:

| período | precio BTC | bot v0.3 | guardar 200 USDT (A) | comprar 200 USDT en BTC el día 1 (B) |
|---|---|---|---|---|
| 2021-2023 | 32.176 → 44.947 | 261 | 200 | **279** |
| 2024-2026 | 44.947 → 84.518 | 322 | 200 | **376** |

Las otras 4 finalistas también pierden contra B en la validación (entre −21 % y −39 %).

**Como fuente de ingreso es muy irregular**: el ingreso llega cuando BTC sube, que es cuando el holding ya está ganando, y se corta cuando baja.

| | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|
| ganancia realizada (USDT) | 130 | **0** | **0** | 115 | 50 | **0** |

- Racha más larga sin ingreso: **25 meses** (dic-21 → dic-23). La racha actual es de 11 meses (nov-25 → sep-26).
- Pérdida latente máxima: 149 de 200 USDT. Hubo slots atrapados hasta 237 días, y algunos de 2021 seguían atrapados a fines de 2023.
- El ingreso medio es de 3,6-5 USDT por mes, pero concentrado en las fases alcistas.

**Por qué gana contra A y pierde contra B**: le gana a guardar USDT porque pasa parte del tiempo comprado en BTC (slots atrapados y BTC comprado con ganancias) en un período alcista. El placebo, entrando al azar, saca +35 % de mediana por la misma razón. Pero está menos tiempo expuesto que quien simplemente compra BTC, y vender cada subida de +3 % le hace perder los tramos grandes.
