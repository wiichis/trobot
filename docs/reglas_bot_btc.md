# Bot BTC por rangos — Reglas v0.1

> Estado: borrador para revisión · Fecha: 2026-10-21 · Exchange: BingX (spot BTC/USDT)

## 1. Objetivo

- Aprovechar las oscilaciones de BTC dentro de rangos para generar ganancia en USDT.
- **Meta:** 1–2% mensual sobre el capital del bot (≈ 2–4 USD/mes con 200 USD), **neta de operaciones atrapadas**.
- **Benchmark:** DCA puro. El bot sirve solo si BTC nuevo con bot > BTC nuevo sin bot.

## 2. Capital y aislamiento

| Parámetro | Valor |
|---|---|
| Capital inicial | 200 USDT |
| Slots | 3 × ~66 USDT |
| Exchange | BingX spot |
| Fondeo | USDT comprado por P2P en Binance → transferencia a BingX |

- El holding existente (wallet fría) **no se conecta ni se lee** desde el bot.
- API key de BingX **sin permiso de retiro** + whitelist de IP.

## 3. Ciclo mensual

- **Día de cierre:** día 3 de cada mes (en real: entre el 1 y el 5, coincide con la compra DCA).
- **Del cierre anterior al siguiente:** el bot opera libre.
- **En el cierre:**
  1. Los slots abiertos (atrapados) pasan a holding al precio de ese momento. Se registra el precio de entrada.
  2. El DCA del mes **repone primero los slots atrapados**, en orden 1 → 2 → 3.
  3. Si el DCA no alcanza, los slots no repuestos **quedan apagados** ese mes.
  4. El sobrante del DCA se compra como BTC para holding.
- **DCA mensual:** variable, entre 100 y 900 USD.
- **Stop:** solo el cierre mensual. **No hay stop-loss por precio.**

## 4. Manejo de slots (a probar)

| Modo | Regla |
|---|---|
| **A. Independientes** | Cada slot entra con señal. Máximo 1 entrada por día. |
| **B. Escalonados** | Slot 1 entra con señal. Slot 2 entra si el precio cae X% bajo la entrada del slot 1. Slot 3 a −2X%. X a probar: 1.5%, 2%, 3%. |

## 5. Entradas (a probar)

| Código | Señal |
|---|---|
| **E1** | Caída de −2% / −3% desde el máximo de las últimas 24h. |
| **E2** | Cierre bajo la banda inferior de Bollinger (20, 2) + RSI(14) < 35. |
| **E3** | Precio dentro del 1% del mínimo del rango de 7 / 14 días. |
| **Filtro F** | No abrir entradas si ADX(14) > 25 con tendencia bajista. Se prueba con y sin filtro. |

## 6. Salidas — regla de subida (a probar)

| Código | Regla |
|---|---|
| **S1** | Take profit fijo: 1.5% / 2% / 3%. |
| **S2 (base)** | Trailing: se activa en +1.5% y vende si retrocede 0.6% desde el máximo alcanzado. |
| **S3** | Trailing con ATR: se activa en +1.5% y el retroceso permitido es 0.5 × ATR(14, 1h). |

Ninguna salida vende con pérdida. Lo que no sale antes del cierre mensual pasa a holding (sección 3).

## 7. Destino de la ganancia (a probar)

| Código | Regla |
|---|---|
| G1 | 100% a BTC holding. |
| G2 | 100% se reinvierte en slots (interés compuesto). |
| G3 | 50% slots / 50% holding. |
| **G4 (base)** | 90% se reinvierte en slots / 10% a BTC holding. |

- La reinversión se reparte en partes iguales entre los 3 slots.
- El BTC destinado a holding **se acumula en BingX** y se retira a la wallet fría solo al superar un umbral (0.005 BTC o cada trimestre, lo que ocurra primero).

## 8. Costos simulados

| Costo | Valor |
|---|---|
| Comisión spot BingX | 0.1% por lado (maker y taker) |
| Slippage | 0.05% por lado |
| Retiro BTC a wallet fría | Comisión fija vigente en BingX (pendiente confirmar) |
| P2P + transferencia de USDT | No se imputa al bot (la paga igual el DCA) |

## 9. Backtest

- **Data:** velas de 1h de BTC/USDT, últimos 3–4 años (incluye bajista 2022 y rangos 2023–2025).
- **Escenarios de DCA:** 100 fijo · 400 fijo · aleatorio entre 100 y 900.
- **Combinaciones:** Slots (A, B) × Entradas (E1, E2, E3 ± F) × Salidas (S1, S2, S3) × Ganancia (G1–G4).
- **Salida del backtest:** ranking de combinaciones por BTC extra vs DCA puro.

## 10. KPIs

| KPI | Descripción |
|---|---|
| **BTC extra vs DCA** (principal) | BTC acumulado con bot − BTC acumulado solo con DCA. |
| Ganancia realizada | USDT ganados en operaciones cerradas. |
| Costo de atrapadas | (Precio de entrada − precio de cierre del mes) × cantidad, por slot atrapado. |
| Neto mensual % | (Ganancia realizada − costo de atrapadas) / capital del bot. |
| % meses en meta | Meses con neto ≥ 1%. |
| Win rate | Operaciones cerradas en ganancia / operaciones abiertas. |
| Ciclos por mes | Operaciones completas (compra + venta) por mes. |
| Comisiones pagadas | Total en USDT. |
| Peor mes | Neto mensual más bajo. |

## 11. Criterio de viabilidad

La estrategia se considera viable si, en el periodo completo del backtest:

1. El **BTC extra vs DCA es positivo**.
2. El **neto anual con atrapadas es positivo** en al menos la mayoría de los años probados.
3. La meta de 1–2% mensual se alcanza en un % de meses razonable (a definir con los resultados).

## 12. Pendientes

- [ ] Confirmar la comisión actual de retiro de BTC en BingX.
- [ ] Confirmar el monto mínimo de orden spot BTC/USDT en BingX.
- [ ] Definir la fuente de la data histórica de 1h.
- [ ] Confirmar si BingX permite subcuentas (para separar el Modo Acumulación).

## 13. Modo Acumulación (compras grandes de holding)

Para compras extraordinarias de BTC para holding (ej. 2000 USD), fuera del DCA mensual. Reusa las señales de entrada del bot, pero **solo compra, nunca vende**.

| Parámetro | Valor base |
|---|---|
| Monto | Variable (ej. 2000 USD) |
| Tramos | 4 × 25% del monto |
| Entrada por tramo | Señal E1 o E2, o escalonado (−2%, −4%, −6% desde el precio de inicio) |
| Fecha límite | 14 días desde el inicio |
| Al vencer | Lo no ejecutado se compra a mercado |
| Destino | 100% a holding |

**Reglas:**
- Si hay slots atrapados del bot al iniciar, el monto **repone primero esos slots** (igual que el cierre mensual).
- **Contabilidad separada del bot:** subcuenta de BingX si está disponible. Si no, prefijo en `clientOrderId` (`BOT-` / `ACU-`).
- El capital de acumulación **no amplía los slots del bot**, salvo que el backtest lo valide.
- **Retiro:** al terminar la acumulación, se retira todo a la wallet fría (supera el umbral de la sección 7).

**Backtest:**
- Comparar contra el benchmark **lump sum**: comprar todo el monto de una vez el día de inicio.
- Variantes: tramos (3 / 4 / 6), señal (E1 / E2 / escalonado), fecha límite (7 / 14 / 21 días).
- KPI: precio promedio obtenido vs lump sum, y % de veces que la acumulación le gana.
