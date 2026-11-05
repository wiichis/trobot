# Bot BTC por rangos en real — hitos y criterios

Dos versiones, cada una con sus reglas fijadas antes de operar:
- **v0.4** (24/10/2026 → la migración): A · E4 0,5 · S4 0,75 · G4 con reposición parcial y cierre forzado el día 3. Lo que sigue en la sección "v0.4" quedó como estaba.
- **Lotes de 3 ciclos** (desde la migración, decidida el 02/11/2026): lo que rige ahora.

## Lotes de 3 ciclos (fijado el 02/11/2026, antes de desplegar)

### Qué se prueba

- **Reglas** (`pkg/btc_rangos_lotes.py`, las mismas de la simulación de lotes, variante B):
  - **Depósitos**: cada depósito de USDT en spot es un lote, repartido en tercios entre los 3 slots. Cada tramo hace 3 ciclos (compra → venta al TP) y después su capital compra BTC de holding.
  - **Ganancia**: 90 % queda en el bot y 10 % compra BTC de holding.
  - **Atrapados**: un slot queda atrapado si cae −10 % desde su entrada o lleva 30 días abierto. Espera con su venta al TP. Lo reemplaza el primer depósito que cubra su costo, y entonces su BTC pasa a holding. Si el depósito no lo cubre, el USDT opera como un 4.º slot.
  - **El día 3**: no hay cierre forzado. Es sólo un corte contable.
  - **Entrada y salida**: las mismas de v0.4.
- **Capital**: los 186,65 USDT de v0.4. Los 3 slots abiertos pasan a ser tramos del lote inicial, con 0 ciclos contados desde la migración.
- **Decisión del usuario (02/11)**: no deposita hasta mediados de mes. Con v0.4, el cierre del 03/11 sin depósito dejaba al bot quieto un mes.
- **Lo que dice la simulación** (200 calendarios de depósitos al azar, 2021-2026; [btc_rangos_lotes_informe.md](btc_rangos_lotes_informe.md)):
  - **BTC contra el DCA puro**: −1,20 % (gana en el 6 % de los calendarios). Con 2021-2023 da −0,88 % y con 2024-2026 −1,99 %.
  - **Contra la regla v0.4**: con los mismos depósitos, v0.4 da −1,30 % en todo el período y +0,23 % en 2024-2026.
  - **Neto del trading**: +232 USDT en 6 años. Es positivo, pero lo come el USDT quieto (25 % del tiempo).
  - **Criterios**: pasa 1 de 4 (sólo el d).

### Cómo se mide (definiciones fijas)

- **Neto del ciclo** (corte del día 3, `mensual.csv`) = ganancia realizada − costo de las atrapadas reemplazadas en el ciclo − el aumento de la pérdida latente de lo abierto.
  - El % es sobre el capital del bot al empezar el ciclo.
  - **La ganancia realizada sola no cuenta.**
  - El primer corte va desde el arranque del 25/10 hasta el primer día 3 posterior a la migración, e incluye lo realizado con v0.4.
- **BTC contra el DCA puro**: holding + BTC en slots + USDT del bot ÷ precio, contra el DCA puro.
  - El DCA puro compra BTC con cada depósito en la hora en que llega, con los mismos costos.
  - Al capital inicial lo compra al precio del arranque (80.877,24, el 25/10), igual que la simulación.
- **Costo de ejecución**: igual que en v0.4, más la ganancia real contra la de la sombra en papel, que recibe los mismos depósitos (`depositos.csv` del real).
- **Tablero**: `python3 scripts/btc_rangos_hitos.py`.

### Lo que la simulación espera, para no confundir suerte con ventaja

El neto del ciclo sigue al precio de BTC (correlación 0,77). Medianas de 3.350 ciclos, de 50 calendarios de 2021-2026 (`scripts/btc_rangos_lotes_sim.py --mensual 3`):

| BTC en el ciclo | neto mediano | ciclos con neto ≥ 0 |
|---|---|---|
| cae más de 5 % | −9,8 % | 5 % |
| entre −5 % y +5 % | +1,1 % | 61 % |
| sube más de 5 % | +9,6 % | 97 % |

El neto medio es de +0,68 % por ciclo. Cada corte compara el neto real con la fila que le corresponde, y el aviso de Telegram lo muestra. Si BTC sube, el bot va a ganar: eso no prueba nada.

### Hitos

| Hito | Fecha | Qué se mira | Qué decide |
|---|---|---|---|
| M0: migración | la primera corrida después del deploy | Respaldo en `real_v04_<fecha>/`. Las 3 órdenes de venta siguen vivas. Se contrastan contra BingX directo: BTC = slots + holding, y USDT = bot + `usdt_ajeno`. | Si no coinciden, se apaga (`enabled: false`) y se corrige |
| M1: primer depósito | ~15/11/2026 | El depósito detectado en `depositos.csv`, y si reemplaza a un atrapado o va en tercios, todo contra los saldos de BingX. La sombra lo aplica en la misma hora. | Lo mismo |
| M2: primer corte | 03/12/2026 | La fila de `mensual.csv` contra un cálculo independiente con `operaciones.csv` y los precios | Lo mismo |
| H2: lectura intermedia | 03/02/2027 | El tablero completo | Nada, salvo fallas técnicas |
| **H3: veredicto** | **03/06/2027** (6 ciclos completos desde el 03/12) | Los cuatro criterios de abajo | Seguir, ajustar el monto o apagar |
| Extensión | 03/12/2027 | Si al 03/06 no hubo al menos 2 ciclos con BTC −5 % o peor, el criterio c no se puede evaluar y la prueba se alarga | |

### Criterios del veredicto (H3): funciona si cumple todos

Los mismos de v0.4:
- **a.** Neto medio de al menos **+1 % al mes** sobre los ciclos completos. La simulación da +0,68 %.
- **b.** BTC total **igual o mayor** que con el DCA puro. La simulación da −1,20 %.
- **c.** En los ciclos en que BTC cae más de 5 %, el neto medio **no es negativo**. En la simulación, sólo el 5 % de esos ciclos tienen neto ≥ 0.
- **d.** Ningún ciclo explica más del 50 % del neto.

### Cortes automáticos (el bot pausa las compras, las ventas siguen y avisa)

- El **neto acumulado** de los cortes baja de **−45 USDT** (`pausa_si_neto_menor_usdt`).
- **Cada hora**:
  - la cuenta spot tiene menos BTC que los slots abiertos, descontando lo ya vendido de una venta llena en parte;
  - o tiene menos USDT que lo que el bot cree tener, más `usdt_ajeno`.
- Una compra queda sin confirmar, o una comisión aparece cobrada en un activo desconocido.

La pausa también frena la compra de holding. La levanta una persona, después de revisar.

### Lo que no se toca durante la prueba

- **La combinación**: el bot rechaza correr si la config cambia contra el estado guardado.
- **Los depósitos**:
  - el usuario deposita lo que quiera y cuando quiera;
  - **todo USDT que aparezca en spot es del bot**, desde 15 USDT (lo menor se acumula hasta llegar);
  - el tope es de 500 USDT por depósito, y el exceso queda aparte en `usdt_ajeno`.

### Experimentos en papel junto al real (fijado el 02/11/2026, antes de desplegar)

Corren desde el deploy, sin Telegram (`pkg/btc_rangos_experimentos.py`). Arrancan del mismo punto que el real (los 3 slots abiertos del 02/11) y usan los mismos depósitos (`real/depositos.csv`).

| | Regla | En la simulación (BTC contra el DCA) |
|---|---|---|
| `exp_b1` | Lotes de **1 ciclo**: cada depósito vende una vez al TP y pasa a holding | −0,65 % en 2021-2026, la mejor del total |
| `exp_v04` | La **v0.4**: cierre forzado el día 3; los depósitos entran en el cierre y hasta entonces quedan en USDT | +0,23 % en 2024-2026, la mejor de ese tramo |

- **Medida**: patrimonio en BTC (BTC + USDT ÷ precio), todos al mismo precio, contra el real y contra un DCA puro calculado aparte (el capital a 80.877,24 y cada depósito en su hora).
- **Dónde se ve**: la sección 4 de `scripts/btc_rangos_hitos.py`; con `--json` da la tabla y la serie para el tablero web.
- **Criterio del usuario**: el **30/11** se revisa cómo van, **sin cambiar el real** por el resultado de un mes. Con la misma señal y el mismo TP, lo que cambia es cuánto tiene cada uno en BTC, así que el mes lo decide sobre todo BTC: si sube fuerte, se espera que gane la v0.4; si va de lado, los lotes de 3; si cae, la v0.4 (con el depósito en USDT).

## v0.4 (fijado el 24/10/2026, antes de operar; vigente hasta la migración)

### Qué se prueba

- **Combinación**: A · E4 0,5 · S4 0,75 · G4, con reposición parcial. Es la misma del papel.
- **Capital**: 186,65 USDT en BingX spot (todo el saldo; el usuario pidió usarlo entero en vez de 180).
- **Decisión del usuario**: pasar a real sin esperar meses de papel, porque el monto es chico, e ir midiendo.
- **Lo que dice el backtest**: no pasa los criterios. En 69 meses de 2021-2026, la ganancia realizada fue de ~5 %/mes y el neto, con el costo de las atrapadas, de −0,03 %/mes. Ver [btc_rangos_fase2_informe.md](btc_rangos_fase2_informe.md).
- **Para qué sirve la prueba en real**: confirmar o refutar eso con plata de verdad, y medir lo que el backtest no puede: deslizamiento, comisiones, rechazos.

### Cómo se mide (definiciones fijas)

- **Neto** = ganancia realizada − costo de las atrapadas en cada cierre − pérdida latente de lo que siga abierto el día de la evaluación. **La ganancia realizada sola no cuenta**: en el backtest superó el 1 % mensual en el 100 % de las ventanas de 3, 6 y 12 meses, porque un bot que nunca vende con pérdida siempre muestra ganancia.
- **BTC contra el DCA puro**: (holding + BTC en slots + USDT del bot ÷ precio) contra el DCA comprado el mismo día de cierre con el capital inicial guardado en USDT.
- **Costo de ejecución**:
  - deslizamiento de cada compra contra el mejor precio de venta del libro al mandarla (`ordenes.csv`);
  - ganancia real contra la del papel en las mismas operaciones;
  - comisiones cobradas.
- **Tablero**: `python3 scripts/btc_rangos_hitos.py`.

### Lo que el backtest espera, para no confundir suerte con ventaja

En unos pocos meses, el neto se mueve casi igual que el precio de BTC (correlación 0,92). Neto mensual mediano en 2021-2026:

| BTC en el ciclo | neto mediano | meses con neto ≥ 0 |
|---|---|---|
| cae más de 5 % | −10,1 % | 0 de 20 |
| entre −5 % y +5 % | −0,8 % | 45 % |
| sube más de 5 % | +9,1 % | 93 % |

Cada cierre mensual compara el neto real con la fila que le corresponde. Si BTC sube, el bot va a ganar: eso no prueba nada.

### Hitos

| Hito | Fecha | Qué se mira | Qué decide |
|---|---|---|---|
| H0: primera compra y primera venta | apenas pasen | El fill del exchange coincide con lo que registró el bot. Deslizamiento contra el supuesto de 5 bps. Comisión cobrada contra el 0,1 %. | Si no coinciden, se pausa y se corrige el código |
| H1: primer cierre (parcial, ~10 días) | 03/11/2026 | DCA detectado, reposición de slots y compra de holding verificados contra los saldos de BingX | Lo mismo que H0 |
| H2: lectura intermedia | 03/01/2027 | El tablero completo | Nada, salvo fallas técnicas (cortar en una caída es reaccionar al ruido) |
| **H3: veredicto** | **03/05/2027** (6 ciclos completos) | Los cuatro criterios de abajo | Seguir, ajustar el monto o apagar |
| Extensión | 03/11/2027 | Si al 03/05 no hubo al menos 2 ciclos con BTC −5 % o peor, el criterio c no se puede evaluar y la prueba se alarga | |

### Criterios del veredicto (H3): funciona si cumple todos

- **a.** Neto medio de al menos **+1 % al mes** sobre los ciclos completos. Es la meta del usuario.
- **b.** BTC total **igual o mayor** que con el DCA puro.
- **c.** En los ciclos en que BTC cae más de 5 %, el neto medio **no es negativo**. Es el criterio que distingue una fuente de ingreso aparte de una apuesta a que BTC sube; en el backtest no se cumplió nunca.
- **d.** Ningún ciclo explica más del 50 % del neto.

### Cortes automáticos (el bot pausa las compras, las ventas siguen y avisa)

- El **neto acumulado** baja de **−45 USDT** (~25 % del capital), medido en cada cierre (`pausa_si_neto_menor_usdt`).
- La cuenta spot tiene menos BTC que los slots abiertos, o menos USDT que lo que el bot cree tener en un cierre.
- Una compra queda sin confirmar.
- Una comisión aparece cobrada en un activo desconocido.

La pausa la levanta una persona, después de revisar.

### Lo que no se toca durante la prueba

- **La combinación**: el bot rechaza correr si la config cambia contra el estado guardado.
- **El monto del DCA**: el usuario deposita lo que quiera antes de cada día 3, con un tope que toma el bot de 500 USDT. Queda registrado en `mensual.csv` como `deposito_detectado`.

## Bitácora

| Fecha (UTC) | Qué pasó |
|---|---|
| 24/10 | Clave spot verificada: permisos de lectura y trading spot, sin futuros, sin retiro, IP 98.81.217.194. El server sale por IPv6, así que `bingx_spot` fuerza IPv4. Prueba de humo OK: orden límite creada, hallada por `clientOrderID` y cancelada. |
| 25/10 00:01 | Arranque con 186,65 USDT (deja aparte 0,009 USDT como `usdt_ajeno`). La sombra en papel arranca en la misma vela. El aviso dice "187" por el redondeo al mostrarlo. |
| 26/10 02:01 | **H0, primera compra ✅.** Slot 1: 0,000781 BTC por 62,21 USDT, precio medio 79.650,02, contra un ask de 79.660,47 al mandarla, o sea −1,3 bps: mejor que los +5 bps que supone el backtest. Comisión 0,000000781 BTC (0,1 %). Quedan 0,000780 BTC tras redondear al step; el polvo de 0,00000022 BTC pasa a holding. Venta límite al TP en 81.222,08. Los saldos de BingX coinciden exactos con el bot: 124,4524144 USDT libres, 0,00078 BTC bloqueados en la orden de venta y 0,00000022 BTC libres. El papel tomó la misma decisión, en el slot 1 a 79.796, y la registra a las 03:01. |
| 27-28/10 | Se simularon los lotes de ciclos y dos mejoras (menos ciclos y filtro de tendencia a los depósitos): 0 de 8 le ganan al DCA. **Decisión del usuario: el real sigue como está**, sin cambiar reglas durante la prueba. Informe: [btc_rangos_lotes_informe.md](btc_rangos_lotes_informe.md). |
| 27/10 22:01 | **H0, primera venta ✅.** La venta límite del slot 1 se llenó a 81.148,99 y cobró la comisión en USDT (0,1 %). Al 02/11 suman 4 ventas al TP: +3,62 USDT realizados, +0,20 por encima del papel en las mismas operaciones. 7 compras con deslizamiento mediano de 0 bps contra el ask. Saldos y órdenes vivas comparados directo contra BingX: coinciden exactos. |
| 02/11 | **Decisión del usuario: no deposita hasta mediados de mes y pasa a "reponer sólo cuando hay dinero"** (lotes de 3 ciclos, sin corte el día 3, como en [btc_rangos_lotes_informe.md](btc_rangos_lotes_informe.md)). El neto al 02/11 era de ≈ −0,5 USDT con BTC −4,6 % en el ciclo (el backtest esperaba −0,8 %). Con la regla actual, el cierre del 03/11 sin depósito habría atrapado los 3 slots (~4 USDT) y dejado al bot quieto. **Se posterga el cierre**: `dia_cierre` 3 → 28 en la config y en el estado del real y del papel; no hay cierre hasta el 28/11 y los slots siguen con sus ventas al TP. H1 queda sin efecto; los hitos se reinician con las reglas nuevas. |
| 02/11 | **Experimentos en papel** (lotes de 1 ciclo y v0.4), decididos con el usuario antes del deploy: corren junto al real desde el mismo punto y con los mismos depósitos; se revisan el 30/11 sin cambiar el real. Ver la sección de experimentos arriba. |
| 02/11 23:01 | **M0, migración a lotes de 3 ciclos ✅.** Deploy a las 22:03 UTC; la primera corrida migró el real, la sombra y `exp_b1` (respaldos `real_v04_202611022301`, `papel_v04_…`, `exp_b1_v04_…`); `exp_v04` arrancó del estado del papel de las 17:00 y cierra el 03/11 00:00. Contra BingX directo: BTC 0,002396444 = slots + holding del bot; USDT 1,22217461 = USDT del bot 1,21309459 + `usdt_ajeno` 0,00908002; las 3 ventas límite siguen PENDING con la misma qty y el mismo TP que calcula el motor. Sin pausa. Al precio de 76.933, el real tiene +4,68 % de BTC contra el DCA puro (que compra los 186,65 a 80.877,24 el 25/10). |
| 03/11 00:01 | **Corrida caída al reemplazar un TP.** El R de 14 días subió de ~2,06 % a 2,32 % y tocaba reponer las 3 ventas. BingX canceló la del slot 1, pero el query siguiente contestó 100404 ("order not exist") y la corrida se cortó: el slot 1 quedó una hora sin venta y los slots 2 y 3 al precio viejo. No se perdió nada: el corte contable del día 3 quedó guardado en el estado y el saldo no cambió. Arreglo: `esperar_terminada` reintenta también el 100404 (48999a5, desplegado a las 00:14), con un test que reproduce la respuesta. |
| 03/11 01:01 | La corrida siguiente colocó las 3 ventas al TP nuevo (82.172,29 / 79.171,00 / 79.375,61), con el aviso falso de "cancelada por fuera" del slot 1. Escribió el corte del 2026-10 en `mensual.csv` (neto −1,45 USDT, con el latente). Contra BingX directo: 3 ventas PENDING, BTC bloqueado 0,002395 = los 3 slots, libre 0,00000144 = holding, USDT 1,22217461 sin cambio. `exp_v04` cerró su ciclo: 3 slots atrapados a holding. |
| 04/11 15:01-16:01 | Se llenaron las ventas de los slots 2 y 3 (79.089,24 y 79.293,64; +0,89 USDT cada una) y a las 16:01 se recompró el slot 2: 0,000793 BTC a 80.456,85. Como BTC siguió subiendo después de vender, la ventaja contra el DCA bajó de +4,68 % a +3,34 %: es el costo de la regla cuando BTC sube, no una falla. El usuario no vio el aviso de la compra y sí el del papel, que llega una hora después. El log dice que el aviso del real salió a las 16:01:03 con HTTP 200 al mismo chat. Desde el 05/11 el log guarda el `message_id` de Telegram (`ok:msg=<n>`), así que la entrega se puede probar desde el server. |
| 05/11 02:24 | **Revisión contra BingX directo: coincide.** 2 ventas PENDING (slot 1 a 82.087,43 y slot 2 a 81.773,67); BTC bloqueado 0,001568 = los 2 slots; libre 0,00000165 = holding; USDT 65,50272507 = USDT del bot 65,49364505 + `usdt_ajeno`. Al 05/11: 8 compras con deslizamiento mediano de 0 bps, 6 ventas por TP, +5,40 USDT realizados, sin atrapados ni pausa. Desde la migración falta una sola corrida horaria, la del 03/11 00:01. |
