# Bot BTC por rangos en real — hitos y criterios (fijados el 24/10/2026, antes de operar)

## Qué se prueba

- **Combinación**: A · E4 0,5 · S4 0,75 · G4, con reposición parcial. Es la misma del papel.
- **Capital**: 186,65 USDT en BingX spot (todo el saldo; el usuario pidió usarlo entero en vez de 180).
- **Decisión del usuario**: pasar a real sin esperar meses de papel, porque el monto es chico, e ir midiendo.
- **Lo que dice el backtest**: no pasa los criterios. En 69 meses de 2021-2026, la ganancia realizada fue de ~5 %/mes y el neto, con el costo de las atrapadas, de −0,03 %/mes. Ver [btc_rangos_fase2_informe.md](btc_rangos_fase2_informe.md).
- **Para qué sirve la prueba en real**: confirmar o refutar eso con plata de verdad, y medir lo que el backtest no puede: deslizamiento, comisiones, rechazos.

## Cómo se mide (definiciones fijas)

- **Neto** = ganancia realizada − costo de las atrapadas en cada cierre − pérdida latente de lo que siga abierto el día de la evaluación. **La ganancia realizada sola no cuenta**: en el backtest superó el 1 % mensual en el 100 % de las ventanas de 3, 6 y 12 meses, porque un bot que nunca vende con pérdida siempre muestra ganancia.
- **BTC contra el DCA puro**: (holding + BTC en slots + USDT del bot ÷ precio) contra el DCA comprado el mismo día de cierre con el capital inicial guardado en USDT.
- **Costo de ejecución**:
  - deslizamiento de cada compra contra el mejor precio de venta del libro al mandarla (`ordenes.csv`);
  - ganancia real contra la del papel en las mismas operaciones;
  - comisiones cobradas.
- **Tablero**: `python3 scripts/btc_rangos_hitos.py`.

## Lo que el backtest espera, para no confundir suerte con ventaja

En unos pocos meses, el neto se mueve casi igual que el precio de BTC (correlación 0,92). Neto mensual mediano en 2021-2026:

| BTC en el ciclo | neto mediano | meses con neto ≥ 0 |
|---|---|---|
| cae más de 5 % | −10,1 % | 0 de 20 |
| entre −5 % y +5 % | −0,8 % | 45 % |
| sube más de 5 % | +9,1 % | 93 % |

Cada cierre mensual compara el neto real con la fila que le corresponde. Si BTC sube, el bot va a ganar: eso no prueba nada.

## Hitos

| Hito | Fecha | Qué se mira | Qué decide |
|---|---|---|---|
| H0: primera compra y primera venta | apenas pasen | El fill del exchange coincide con lo que registró el bot. Deslizamiento contra el supuesto de 5 bps. Comisión cobrada contra el 0,1 %. | Si no coinciden, se pausa y se corrige el código |
| H1: primer cierre (parcial, ~10 días) | 03/11/2026 | DCA detectado, reposición de slots y compra de holding verificados contra los saldos de BingX | Lo mismo que H0 |
| H2: lectura intermedia | 03/01/2027 | El tablero completo | Nada, salvo fallas técnicas (cortar en una caída es reaccionar al ruido) |
| **H3: veredicto** | **03/05/2027** (6 ciclos completos) | Los cuatro criterios de abajo | Seguir, ajustar el monto o apagar |
| Extensión | 03/11/2027 | Si al 03/05 no hubo al menos 2 ciclos con BTC −5 % o peor, el criterio c no se puede evaluar y la prueba se alarga | |

## Criterios del veredicto (H3): funciona si cumple todos

- **a.** Neto medio de al menos **+1 % al mes** sobre los ciclos completos. Es la meta del usuario.
- **b.** BTC total **igual o mayor** que con el DCA puro.
- **c.** En los ciclos en que BTC cae más de 5 %, el neto medio **no es negativo**. Es el criterio que distingue una fuente de ingreso aparte de una apuesta a que BTC sube; en el backtest no se cumplió nunca.
- **d.** Ningún ciclo explica más del 50 % del neto.

## Cortes automáticos (el bot pausa las compras, las ventas siguen y avisa)

- El **neto acumulado** baja de **−45 USDT** (~25 % del capital), medido en cada cierre (`pausa_si_neto_menor_usdt`).
- La cuenta spot tiene menos BTC que los slots abiertos, o menos USDT que lo que el bot cree tener en un cierre.
- Una compra queda sin confirmar.
- Una comisión aparece cobrada en un activo desconocido.

La pausa la levanta una persona, después de revisar.

## Lo que no se toca durante la prueba

- **La combinación**: el bot rechaza correr si la config cambia contra el estado guardado.
- **El monto del DCA**: el usuario deposita lo que quiera antes de cada día 3, con un tope que toma el bot de 500 USDT. Queda registrado en `mensual.csv` como `deposito_detectado`.
