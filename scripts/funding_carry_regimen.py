#!/usr/bin/env python3
"""Carry de funding con filtro de régimen, 09/10/2026. Reglas fijadas ANTES de correr:
 - lookback 30d, n in {10, 5}, rebalanceo semanal, costo 0,17% por entrada o salida.
 - filtro: invertir sólo si el funding anualizado medio (trailing 30d, sobre notional) de los n pares
   elegidos >= umbral; si no, todo a efectivo. Umbrales: sin filtro, 10% (principal), 5% (sensibilidad).
Fuentes: 'binance' (120 pares, 2019-2026) y 'bingx' (funding y perp de BingX, contado Binance, 2023-2026)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

R = Path(__file__).resolve().parent.parent / "archivos"
COSTO = 0.5 * (0.0010 + 0.0007)

def panel(fuente, min_vol):
    fdir = R / ("funding" if fuente == "binance" else "funding_bingx")
    fr, pr, sp, vol = {}, {}, {}, {}
    for d in sorted(fdir.glob("*USDT")):
        try:
            f = pd.read_csv(d / "funding.csv"); p = pd.read_csv(d / "perp_1d.csv"); s = pd.read_csv(R / "funding" / d.name / "spot_1d.csv")
        except Exception: continue
        if len(f) < 100: continue
        f["dia"] = pd.to_datetime(f.fundingTime, unit="ms").dt.floor("D")
        fr[d.name] = f.groupby("dia").fundingRate.sum()
        for dic, x, c in ((pr, p, "close"), (sp, s, "close"), (vol, p, "quote_vol")):
            dic[d.name] = pd.Series(x[c].to_numpy(), index=pd.to_datetime(x.t, unit="ms").dt.floor("D"))
    FR, PR, SP, VOL = [pd.DataFrame(x).sort_index() for x in (fr, pr, sp, vol)]
    idx = FR.index.intersection(PR.index).intersection(SP.index)
    FR, PR, SP, VOL = (x.loc[idx] for x in (FR, PR, SP, VOL))
    ok = (VOL.rolling(30).mean() > min_vol) & FR.notna() & PR.notna() & SP.notna()
    ok &= (SP.pct_change() - PR.pct_change()).abs().rolling(30).max() < 0.25   # descarta datos rotos
    return FR, (SP.pct_change() - PR.pct_change()), ok, idx

def correr(P, lb, n, umbral, reb=7):
    FR, base, ok, idx = P
    fl = FR.rolling(lb, min_periods=lb).mean()
    pesos = pd.DataFrame(0.0, index=idx, columns=FR.columns); w = pd.Series(0.0, index=FR.columns)
    cost = pd.Series(0.0, index=idx)
    for i, t in enumerate(idx):
        if i % reb == 0 and i > lb:
            sc = fl.loc[t].where(ok.loc[t]).dropna(); sc = sc[sc > 0]
            new = pd.Series(0.0, index=FR.columns)
            if len(sc):
                top = sc.nlargest(n)
                if umbral is None or top.mean() * 365 >= umbral: new[top.index] = 1.0 / n
            cost.iloc[i] = COSTO * (new - w).abs().sum(); w = new
        pesos.iloc[i] = w
    W = pesos.shift(1)
    r = 0.5 * (W * (FR + base)).sum(axis=1) - cost
    ini = idx[lb + 2]
    return r.loc[ini:], W.loc[ini:]

def res(r):
    eq = (1 + r).cumprod(); a = eq.iloc[-1] ** (365 / len(r)) - 1
    return a, (eq / eq.cummax() - 1).min()

if __name__ == "__main__":
    for fuente, mv in (("binance", 5e6), ("bingx", 2e6)):
        P = panel(fuente, mv)
        print(f"\n=== {fuente}: {P[0].shape[1]} pares, {P[3][0].date()} -> {P[3][-1].date()}")
        filas = []
        for n in (10, 5):
            for u in (None, 0.10, 0.05):
                r, W = correr(P, 30, n, u)
                a, dd = res(r); inv = (W.sum(axis=1) > 0).mean()
                por_anio = r.groupby(r.index.year).sum().round(3).to_dict()
                h = len(r) // 2; a1, a2 = res(r.iloc[:h])[0], res(r.iloc[h:])[0]
                a365, _ = res(r.iloc[-365:]); a180, _ = res(r.iloc[-180:])
                filas.append(dict(n=n, umbral=u, anual=a, caida=dd, invertido=inv, mitad1=a1, mitad2=a2, ult365=a365, ult180=a180, **{str(k): v for k, v in por_anio.items()}))
        pd.set_option("display.width", 250); print(pd.DataFrame(filas).round(3).to_string(index=False))
