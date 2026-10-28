# Bot BTC por rangos — lotes de 4 ciclos y mejoras contra el DCA (27-28/10/2026)

**Decisión del usuario (28/10):** el bot real sigue en prod como está, con A · E4_0.5 · S4_0.75 · G4, reposición parcial y cierre el día 3. Se observa cómo evoluciona según los hitos de [btc_rangos_real_hitos.md](btc_rangos_real_hitos.md). Nada de este informe se implementó en el real.

- **Script:** `scripts/btc_rangos_lotes_sim.py`. `--prueba` corre 5 pruebas: caso a mano, atrapado y 4.º slot, filtro, contraste exacto con `pkg/btc_rangos_motor.py` y filtro siempre en COMPRA = sin filtro. La corrida completa tarda ~5 min en el Mac.
- **Datos:** `archivos/btc_rangos/lotes_sim/`, con `corridas.csv`, `mejoras.csv`, `b1_mensual.csv` y `b1_mensual_por_calendario.csv`.

## Qué se probó

La regla que propuso el usuario:
- **Lotes**: cada depósito es un lote que se reparte en tercios entre 3 slots compartidos. Cada tramo hace hasta 4 ciclos (compra → venta en el TP) y después su capital compra BTC de holding (subcuenta → billetera fría).
- **Ganancia**: es del bot, con G4 (90 % se reinvierte y 10 % va a holding).
- **Atrapados**: no hay corte el día 3. Un slot queda atrapado a −10 % de su entrada o a los 30 días. Lo reemplaza el USDT nuevo que cubre su costo; si no alcanza, ese USDT opera como un 4.º slot temporal.

Las condiciones de la simulación:
- **Depósitos**: 200 calendarios al azar de 100-900 USDT por mes, en 1-3 compras.
- **Velas**: spot de 1 h, de 2021-01-03 a 2026-10-03.
- **Costos**: comisión 0,1 % y slippage 0,05 % por lado.
- **Referencia**: el DCA puro, que compra BTC con cada depósito en su misma vela.

## Resultados

### 1. Lotes de 4 ciclos: pierden contra el DCA

| BTC final contra el DCA puro (mediana) | 2021-2026 | 2021-2023 | 2024-2026 |
|---|---|---|---|
| Regla actual | −1,30 % | −1,59 % | +0,23 % |
| Lotes, variante A (literal) | −15,0 % | −13,5 % | −5,5 % |
| Lotes, variante B | −1,73 % | −1,15 % | −2,84 % |

- **Variante A:** la regla no decía qué hacer con los depósitos si el 4.º slot quedaba abierto. Esperando a que vendiera, un calendario acumuló 10.561 USDT ociosos durante 2 años.
- **Variante B:** ese USDT cuenta para reemplazar a un atrapado aunque el 4.º esté abierto. Es la versión de referencia.

### 2. Lo que produce sólo el trading (variante B, 2021-2026)

- **Neto en USDT:** +307 (ganancia de las ventas 2.857 − atrapadas 2.511 − abierto al final 41). Es positivo en el 82 % de los calendarios.
- **Neto sobre el capital que opera el bot:** +5,5 % anual. Ese capital es de unos 800-1.300 USDT.
- **Por año:** 2021 +12,9 %, 2022 −52,0 %, 2023 +53,6 %, 2024 +41,8 %, 2025 −2,2 %, 2026 −0,4 %.
- **Por qué igual pierde en BTC:** el +5,5 % no compensa el USDT quieto (23 % del tiempo) mientras BTC sube.

### 3. Cómo se miden las atrapadas: contra el día del depósito

El usuario objetó, con razón, que comparar el precio de las atrapadas con el precio medio de todo el DCA (+16,9 %) era injusto. Medido contra el precio de mercado del día en que llegó cada depósito:

| Cuándo quedó atrapado el dinero | % del dinero atrapado | Precio contra el día del depósito |
|---|---|---|
| En su 1.ª compra | 28 % | −0,1 % (igual que el DCA) |
| Tras 1 ciclo | 28 % | +2,6 % |
| Tras 2 ciclos | 22 % | +4,9 % |
| Tras 3 ciclos | 22 % | +8,8 % |

- **Comprar en caída ahorra poco:** las entradas salen −0,48 % bajo el precio medio de las 24 h previas.
- **Lo que pesa es el tiempo:** el dinero que ya hizo ciclos vuelve a entrar 2 o 3 semanas después, más caro.
- **En total**, el dinero atrapado obtuvo −2,8 % de BTC contra el DCA.

### 4. Movimientos por mes (variante B; mediana de 30 calendarios)

| | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|
| Ciclos por mes | 8,1 | 4,2 | 10,7 | 9,5 | 7,4 | 6,6 |
| Órdenes por mes | ~18 | ~9 | ~24 | ~22 | ~17 | ~15 |

### 5. Mejoras probadas (28/10): 0 de 8 pasan

Los criterios se fijaron antes de correr:
1. Mediana de la ventaja de BTC > 0 en 2021-2023 y en 2024-2026.
2. Gana en ≥ 60 % de los calendarios en cada tramo.
3. Su mediana supera al percentil 90 de su placebo.
4. Ningún mes explica más del 50 % de los aumentos de ventaja.

| Variante | 2021-2026 | 2021-2023 | 2024-2026 | Trading neto | USDT quieto |
|---|---|---|---|---|---|
| Regla actual | −1,30 % | −1,59 % | +0,23 % | +27 | — |
| 1 ciclo por lote | **−0,65 %** | −0,62 % | −0,65 % | +146 | 32 % |
| 2 ciclos | −0,90 % | −0,75 % | −1,20 % | +204 | 27 % |
| 4 ciclos (B) | −1,73 % | −1,15 % | −2,84 % | +307 | 23 % |
| Filtro + 1 ciclo | −0,80 % | −0,12 % | −1,39 % | +272 | 68 % |
| Filtro + 2 ciclos | −1,54 % | −0,97 % | −2,10 % | +40 | 50 % |
| Filtro + 4 ciclos | −2,55 % | −1,64 % | −3,58 % | +402 | 36 % |
| Filtro solo, sin bot | −1,15 % | −1,35 % | −0,23 % | — | — |

- **Filtro de tendencia:** con la ventana T1 (cierre diario sobre la media de 100 días), el depósito en VENTA espera en USDT.
- **Filtro solo, con velas diarias 2017-2026:** −0,18 %; gana en el 45 % de los calendarios. Le gana a su placebo (−1,26 %), pero no al DCA.
- **Filtro con el bot:** empeora a todas las variantes en 2024-2026.
- **Menos ciclos es mejor:** con 1 ciclo, las atrapadas quedan iguales al día del depósito (+0,2 %). Pero el lote termina comprando BTC más o menos al precio de su venta, y lo que gana en la operación lo devuelve al recomprar.

### 6. Rendimiento crudo con 1 ciclo por lote (mediana de 50 calendarios)

| Año | Capital en el bot | Ventas al mes | % crudo al mes | % neto al mes |
|---|---|---|---|---|
| 2021 | 209 | 5,5 | +9,95 % | +3,63 % |
| 2022 | 243 | 4,3 | +5,14 % | −2,60 % |
| 2023 | 234 | 6,0 | +4,14 % | +3,30 % |
| 2024 | 218 | 7,0 | +6,16 % | +3,58 % |
| 2025 | 258 | 4,7 | +3,52 % | +0,42 % |
| 2026 (ene-sep) | 260 | 3,8 | +2,94 % | −1,02 % |
| **Todo el período** | | | **+4,94 %** | **+0,80 %** |

- **% crudo:** la ganancia de las ventas sobre el capital que opera. Es positivo todos los meses, porque el bot nunca vende con pérdida.
- **% neto:** el crudo menos las atrapadas y el cambio en lo no realizado. Es negativo en el 30 % de los meses, los de caída fuerte.
- **En dinero**, el crudo son unos 10 USDT al mes, sobre un capital de ~230. El detalle mes a mes está en `b1_mensual.csv`.

## Conclusiones

1. **Ninguna variante le gana al DCA puro en BTC.** Lo más cerca es 1 ciclo por lote (−0,65 %). La regla en prod da −1,30 %.
2. **El trading deja algo de USDT** (+150 a +400 en 6 años con depósitos de ~500/mes), pero esperar para comprar BTC cuesta lo mismo o más.
3. **Las atrapadas se miden contra el día del depósito**, no contra el promedio del DCA. En la 1.ª compra cuestan lo mismo que el DCA; lo que cuesta es el dinero que vuelve a entrar después de varios ciclos.
4. **Ni el momento de entrar ni el filtro de tendencia mueven el resultado** más allá de ±1 %. Lo mismo dieron las compras semanales y el Modo Acumulación.
5. **Si algún día se cambia la regla del real**, el candidato es 1 ciclo por lote, sin filtro. Cambiarla exige archivar `archivos/btc_rangos/real/` y reiniciar los hitos.
