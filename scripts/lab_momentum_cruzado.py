#!/usr/bin/env python3
"""Momentum entre pares (cross-sectional) — variantes fijadas de antemano (02/10/2026).

Viene de scripts/lab_estrategias.py: de 6 familias, la única positiva neta fue F5 (largo
top-3 / corto bottom-3 por retorno 72 h, 24 h): +0,28% bruto, +0,12% neto, t 0,9. A horizonte
corto los costos pesan mucho; acá se prueban horizontes más largos sobre un universo más
amplio (39 pares).

Variantes (NO se optimizan; se fijaron antes de correr):
  V1  mirada 72 h, mantener 24 h, rebalanceo diario,  3 largos / 3 cortos   (la F5 original)
  V2  mirada 7 d,  mantener 7 d,  rebalanceo semanal, 5 largos / 5 cortos
  V3  mirada 7 d,  mantener 3 d,  rebalanceo c/3 d,   5 largos / 5 cortos
  V4  mirada 7 d,  mantener 7 d,  rebalanceo semanal, 5 largos, sin cortos
      (V4 tiene exposición al mercado: se reporta además su exceso sobre el promedio del
      universo en el mismo período)

Señal con el cierre de la hora t; entrada en la apertura de t+1; salida al cierre de la
última hora del período. Costos taker: 16 bps por vuelta y por pata.

Criterio de "prometedora" (fijado antes de mirar): neto positivo en las DOS mitades
(dic-abr y may-sep), t > 2 en el total y ningún mes con más del 50% de la suma de los
meses positivos.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
COSTO = 2 * (0.0005 + 0.0003)
VARIANTES = {
    "V1_72h_24h_3x3": dict(look=72, hold=24, n_l=3, n_c=3),
    "V2_7d_7d_5x5": dict(look=168, hold=168, n_l=5, n_c=5),
    "V3_7d_3d_5x5": dict(look=168, hold=72, n_l=5, n_c=5),
    "V4_7d_7d_solo_largos": dict(look=168, hold=168, n_l=5, n_c=0),
}


def _a_1h(d):
    d = d.set_index("date").sort_index()
    return d.resample("1h").agg({"open": "first", "close": "last"}).dropna()


def cargar_universo():
    series = {}
    for f in (REPO / "archivos/velas_limpias_5m.csv", REPO / "archivos/candidatos/velas_candidatos.csv"):
        d = pd.read_csv(f, usecols=["symbol", "open", "close", "date"])
        d["date"] = pd.to_datetime(d["date"], utc=True, format="mixed")
        for sym, g in d.groupby("symbol"):
            series.setdefault(sym, _a_1h(g))
    for z in sorted((REPO / "archivos/candidatos/binance").glob("*USDT-5m-*.zip")):
        sym = z.name.split("USDT-5m-")[0] + "-USDT"
        if sym in series and not str(sym).startswith("_"):
            continue
        series.setdefault("_" + sym, []).append(z)
    for k in [k for k in series if k.startswith("_")]:
        sym, frames = k[1:], []
        for z in series.pop(k):
            with zipfile.ZipFile(z) as zf:
                raw = zf.read(zf.namelist()[0]).decode()
            df = pd.read_csv(io.StringIO(raw), header=0 if raw.startswith("open_time") else None).iloc[:, :5]
            df.columns = ["ot", "open", "high", "low", "close"]
            ot = pd.to_numeric(df["ot"])
            df["date"] = pd.to_datetime(ot, unit="us" if ot.max() > 1e14 else "ms", utc=True)
            frames.append(df[["date", "open", "close"]])
        series[sym] = _a_1h(pd.concat(frames))
    closes = pd.DataFrame({s: h["close"] for s, h in series.items()})
    opens = pd.DataFrame({s: h["open"] for s, h in series.items()})
    fin = closes.index.max() - pd.Timedelta(hours=1)
    return closes.loc[:fin], opens.loc[:fin]


def correr(closes, opens, look, hold, n_l, n_c):
    idx = closes.index
    filas = []
    t_i = look + 1
    while t_i + hold < len(idx):
        t = idx[t_i]
        ret = (closes.iloc[t_i] / closes.iloc[t_i - look] - 1)
        ie, js = t_i + 1, t_i + hold
        ent, sal = opens.iloc[ie], closes.iloc[js]
        ok = ret.notna() & ent.notna() & sal.notna()
        r = ret[ok].sort_values()
        if len(r) >= n_l + n_c + 5:
            mercado = float((sal[ok] / ent[ok] - 1).mean())
            patas = [(s, 1) for s in r.index[-n_l:]] + ([(s, -1) for s in r.index[:n_c]] if n_c else [])
            for s, lado in patas:
                bruto = lado * (sal[s] / ent[s] - 1)
                filas.append({"entrada": idx[ie], "symbol": s, "lado": lado, "bruto": bruto,
                              "neto": bruto - COSTO, "exceso": bruto - lado * mercado, "universo": int(ok.sum())})
        t_i += hold
    return pd.DataFrame(filas)


def resumir(nombre, t):
    t = t.copy()
    t["mes"] = pd.to_datetime(t["entrada"]).dt.strftime("%Y-%m")
    t["mitad"] = np.where(pd.to_datetime(t["entrada"]) < pd.Timestamp("2026-05-01", tz="UTC"), "dic-abr", "may-sep")
    tt = lambda x: x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 2 else np.nan
    meses = t.groupby("mes")["neto"].mean()
    pos = meses[meses > 0]
    conc = pos.max() / pos.sum() if len(pos) else 1.0
    mit = t.groupby("mitad")["neto"].agg(["mean", "size"])
    prometedora = bool((mit["mean"] > 0).all() and tt(t["neto"]) > 2 and conc <= 0.5)
    print(f"\n{nombre}: {len(t)} patas, universo medio {t['universo'].mean():.0f} pares")
    print(f"  bruto {t['bruto'].mean()*100:+.3f}% | neto {t['neto'].mean()*100:+.3f}% (t={tt(t['neto']):+.2f}) | "
          f"gana {(t['neto'] > 0).mean():.0%} | meses + {(meses > 0).sum()}/{len(meses)} | concentración {conc:.0%}")
    print("  por mitad:", {k: f"{v['mean']*100:+.3f}% (n={int(v['size'])})" for k, v in mit.iterrows()})
    if "solo_largos" in nombre:
        print(f"  exceso sobre el promedio del universo: {t['exceso'].mean()*100:+.3f}% (t={tt(t['exceso']):+.2f})")
    else:
        print("  por pata:", t.groupby("lado")["neto"].agg(lambda x: f"{x.mean()*100:+.3f}%").to_dict())
    print("  por mes (neto %):", (meses * 100).round(2).to_dict())
    print("  →", "PROMETEDORA" if prometedora else "no cumple el criterio")
    return prometedora


def main():
    closes, opens = cargar_universo()
    print(f"universo: {closes.shape[1]} pares, {closes.index.min():%d/%m/%y} → {closes.index.max():%d/%m/%y}")
    out = REPO / "archivos/analisis"
    out.mkdir(parents=True, exist_ok=True)
    for nombre, v in VARIANTES.items():
        t = correr(closes, opens, **v)
        t.to_csv(out / f"momentum_{nombre}.csv", index=False)
        resumir(nombre, t)


if __name__ == "__main__":
    main()
