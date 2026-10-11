#!/usr/bin/env python3
"""Tasas base de BTC para el resumen diario (11/10/2026).

Qué pasó históricamente en los 7 y 30 días siguientes según el estado del día:
tendencia (cierre sobre o bajo la media de 100 días, la regla T1 de pkg/btc_reglas.py)
y RSI 14 diario en cuatro tramos. Velas diarias spot de Binance, ago-2017 → hoy.

Sirve para decir "qué se puede esperar" sin inventar una predicción: la conclusión de fondo
es que la dirección a 7 días es casi una moneda al aire (49-57% en casi todos los estados),
y que lo que sí es estable es el RANGO típico (p10-p90).

Ojo al leerlo: los días se solapan (un mismo movimiento cuenta muchas veces), así que la
muestra efectiva es bastante menor que `n`. Tramos con n < 60 se reemplazan por el total de
su tendencia en pkg/daily_digest.py.

Uso: python3 scripts/btc_tasas_base.py  → escribe pkg/btc_tasas_base.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "scripts"))
sys.path.insert(0, str(RAIZ))
import btc_lab_temporalidades as lab  # noqa: E402
from pkg.btc_reglas import rsi  # noqa: E402

TRAMOS = [0, 30, 50, 70, 101]
NOMBRES = ["<30", "30-50", "50-70", ">70"]


def tabla(g: pd.DataFrame) -> dict:
    return {"n": int(len(g)),
            "sube7": round(float((g.f7 > 0).mean()), 3), "med7": round(float(g.f7.median()), 4),
            "p10_7": round(float(g.f7.quantile(0.1)), 4), "p90_7": round(float(g.f7.quantile(0.9)), 4),
            "sube30": round(float((g.f30 > 0).mean()), 3), "med30": round(float(g.f30.median()), 4),
            "p10_30": round(float(g.f30.quantile(0.1)), 4), "p90_30": round(float(g.f30.quantile(0.9)), 4)}


def main():
    c = lab.cargar()["1d"]["close"]
    ma = c.rolling(100).mean()
    df = pd.DataFrame({
        "tend": np.where(c > ma, "sobre", "bajo"),
        "rsi": pd.cut(rsi(c), TRAMOS, labels=NOMBRES, right=False),
        "f7": c.shift(-7) / c - 1, "f30": c.shift(-30) / c - 1,
    }, index=c.index).iloc[100:].dropna()
    out = {"_comentario": __doc__.strip().splitlines()[0],
           "periodo": f"{df.index[0].date()} a {df.index[-1].date()}",
           "fuente": "velas diarias spot BTCUSDT de Binance",
           "todos": tabla(df), "por_tendencia": {}, "por_estado": {}}
    for t, g in df.groupby("tend"):
        out["por_tendencia"][t] = tabla(g)
    for (t, r), g in df.groupby(["tend", "rsi"], observed=True):
        out["por_estado"][f"{t}|{r}"] = tabla(g)
    destino = RAIZ / "pkg" / "btc_tasas_base.json"
    destino.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("escrito", destino, "|", out["periodo"])


if __name__ == "__main__":
    main()
