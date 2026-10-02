#!/usr/bin/env python3
"""¿La señal rinde distinto según la tendencia del MERCADO? (02/10/2026)

Hipótesis (propuesta #3 del 02/10): el edge de la señal aparece en meses de tendencia
generalizada; si es así, una señal ALINEADA con la tendencia del mercado debería rendir más
que una a contramano. Sólo mide: no filtra nada ni toca el bot.

Mercado = índice equiponderado de los 25 pares con velas (10 del portfolio + 15
candidatos): retorno log medio por hora de los pares disponibles. Rasgos, calculados SÓLO
con velas anteriores a la señal (la hora cerrada previa):
- PRINCIPAL (fijado antes de mirar): signo del retorno del índice en las 72 h previas.
- secundario: signo del retorno en las 24 h previas.
Una señal está "alineada" si su lado coincide con ese signo (long con mercado subiendo).

Entrada: el CSV de señales de scripts/edge_horizonte.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent


def indice_mercado(archivos) -> pd.Series:
    """Índice horario equiponderado (log-retorno medio de los pares disponibles), acumulado."""
    rets = []
    for f in archivos:
        d = pd.read_csv(f, usecols=["symbol", "close", "date"])
        d["date"] = pd.to_datetime(d["date"], utc=True, format="mixed")
        for sym, g in d.groupby("symbol"):
            h = g.set_index("date")["close"].resample("1h").last().dropna()
            rets.append(np.log(h).diff().rename(sym))
    r = pd.concat(rets, axis=1)
    r = r.loc[:, ~r.columns.duplicated()]
    return r.mean(axis=1, skipna=True).fillna(0).cumsum().rename("idx")


def rasgos(idx: pd.Series, fechas: pd.Series) -> pd.DataFrame:
    """Retorno del índice en las 24 y 72 h previas a cada fecha (hora cerrada anterior)."""
    hora = fechas.dt.floor("1h") - pd.Timedelta(hours=1)
    v = idx.reindex(idx.index.union(pd.DatetimeIndex(hora.unique()))).ffill()
    def ret(h):
        return v.reindex(hora).to_numpy() - v.reindex(hora - pd.Timedelta(hours=h)).to_numpy()
    return pd.DataFrame({"mkt24": ret(24), "mkt72": ret(72)}, index=fechas.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--senales", default=str(REPO / "archivos/analisis/edge_horizonte_20261002.csv"))
    ap.add_argument("--velas", nargs="*", default=[str(REPO / "archivos/velas_limpias_5m.csv"),
                                                    str(REPO / "archivos/candidatos/velas_candidatos.csv")])
    ap.add_argument("--H", type=int, default=8)
    args = ap.parse_args()

    s = pd.read_csv(args.senales)
    s["fecha"] = pd.to_datetime(s["fecha"], utc=True)
    s = s[s["H"] == args.H].reset_index(drop=True)
    idx = indice_mercado(args.velas)
    s = pd.concat([s, rasgos(idx, s["fecha"])], axis=1)
    sg = np.where(s["lado"] == "long", 1, -1)
    for k in ("mkt72", "mkt24"):
        s[f"alin_{k}"] = np.sign(s[k]) == sg

    print(f"H={args.H} h — {len(s)} señales, índice de {idx.index.min():%d/%m/%y} a {idx.index.max():%d/%m/%y}\n")
    for k, etiqueta in (("mkt72", "PRINCIPAL: tendencia 72 h"), ("mkt24", "secundario: tendencia 24 h")):
        print(etiqueta)
        for regla in ("sin_stop", "stop", "stop_trail"):
            a, b = s[s[f"alin_{k}"]][regla], s[~s[f"alin_{k}"]][regla]
            se = np.sqrt(a.var() / len(a) + b.var() / len(b))
            print(f"  {regla:10s} alineadas {a.mean()*100:+.3f}% (n={len(a)}) | a contramano {b.mean()*100:+.3f}% "
                  f"(n={len(b)}) | diferencia {(a.mean()-b.mean())*100:+.3f}% (t={(a.mean()-b.mean())/se:+.2f})")
        m = s.assign(mes=s["fecha"].dt.strftime("%Y-%m")).groupby(["mes", f"alin_{k}"])["stop_trail"].mean().unstack()
        m.columns = ["contramano", "alineadas"]
        m["dif"] = m["alineadas"] - m["contramano"]
        print(f"  por mes (stop_trail, %): alineadas > contramano en {(m['dif'] > 0).sum()} de {m['dif'].notna().sum()} meses")
        print((m * 100).round(2).to_string().replace("\n", "\n    "))
        print()
    out = REPO / "archivos/analisis" / f"regimen_mercado_H{args.H}.csv"
    s.to_csv(out, index=False)


if __name__ == "__main__":
    main()
