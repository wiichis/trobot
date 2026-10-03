#!/usr/bin/env python3
"""Cuatro bots sobre BTC, cada uno con 25% del capital (02/10/2026) — simulación histórica.

Pedido del usuario: operar sólo BTC "como si fueran 4 bots" con características distintas.
Las reglas salen de `scripts/btc_lab_temporalidades.py` (mismos costos, funding real) y
NO se optimizan:
  A  tendencia lenta: diario, precio vs media de 100 días, long/short.
  B  tendencia rápida: 4 h, Donchian 20/10, long/short.
  C  cobro de funding: BTC al contado + short del perpetuo por el mismo monto, con la
     mitad del capital en cada pata (short a 1x). Cobra (o paga) el funding sobre la mitad
     del capital. Entra una vez: 0,1% de comisión spot + 0,07% perpetuo, ida y vuelta.
     Supone que el precio del contado y del perpetuo se mueven igual (sin base).
  D  compra de caídas: diario, long si RSI 14 < 30, sale al cruzar 50; sin shorts.
Período común: feb-2020 → sep-2026 (funding real desde ene-20). Reparto fijo: cada bot
compone su 25% por separado, sin mover capital entre bots.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import btc_lab_temporalidades as lab  # noqa: E402

DESDE, HASTA = pd.Timestamp("2020-02-01"), pd.Timestamp("2026-10-01")
COSTO_CARRY = 2 * (0.001 + 0.0007)


def diario(neto):
    return (1 + neto).groupby(neto.index.floor("D")).prod() - 1


def main():
    series, funding = lab.cargar(), lab.cargar_funding()
    d1, h4 = series["1d"], series["4h"]
    bots = {}
    n, _, tr_a = lab.simular(d1, lab.posiciones(d1, "T1", False), lab.funding_por_barra(d1, 24, funding), DESDE, HASTA)
    bots["A tendencia lenta (1d)"] = (diario(n), len(tr_a))
    n, _, tr_b = lab.simular(h4, lab.posiciones(h4, "T3", False), lab.funding_por_barra(h4, 4, funding), DESDE, HASTA)
    bots["B tendencia rápida (4h)"] = (diario(n), len(tr_b))
    f = funding.loc[DESDE:HASTA - pd.Timedelta(seconds=1)]
    c = (0.5 * f).groupby(f.index.floor("D")).sum()
    c.iloc[0] -= COSTO_CARRY / 2
    c.iloc[-1] -= COSTO_CARRY / 2
    bots["C cobro de funding"] = (c, 1)
    n, _, tr_d = lab.simular(d1, lab.posiciones(d1, "R1", True), lab.funding_por_barra(d1, 24, funding), DESDE, HASTA)
    bots["D compra de caídas (1d)"] = (diario(n), len(tr_d))

    dias = pd.date_range(DESDE, HASTA - pd.Timedelta(days=1), freq="D")
    r = pd.DataFrame({k: v.reindex(dias).fillna(0.0) for k, (v, _) in bots.items()})
    eq = (1 + r).cumprod()
    cartera = eq.mean(axis=1)                       # 25% fijo en cada uno, sin rebalancear
    eq["CARTERA (25% c/u)"] = cartera
    btc = d1["close"].reindex(dias).ffill()
    eq["holding BTC"] = btc / btc.iloc[0]

    anios = len(dias) / 365.25
    filas = []
    for k in eq.columns:
        e = eq[k]
        por_anio = e.resample("YE").last().pct_change().fillna(e.resample("YE").last().iloc[0] - 1)
        filas.append(dict(bot=k, total=e.iloc[-1] - 1, anual=e.iloc[-1] ** (1 / anios) - 1,
                          caida_max=(e / e.cummax() - 1).min(),
                          operaciones=bots[k][1] if k in bots else None,
                          **{str(y.year): v for y, v in por_anio.items()}))
    t = pd.DataFrame(filas).set_index("bot")
    pct = lambda x: f"{x*100:+.0f}%" if pd.notna(x) else ""
    print(f"Período {DESDE:%d/%m/%y} → {HASTA:%d/%m/%y} ({anios:.1f} años)\n")
    print(t.drop(columns="operaciones").apply(lambda col: col.map(pct)).to_string())
    print("\nOperaciones en el período:", {k: v[1] for k, v in bots.items()})
    print("\nCorrelación de los retornos diarios:")
    print(r.corr().round(2).to_string())
    out = lab.REPO / "archivos/analisis/btc_cuatro_bots.csv"
    t.to_csv(out)


if __name__ == "__main__":
    main()
