#!/usr/bin/env python3
"""Laboratorio de carry de funding (long contado + short perpetuo), 09/10/2026.
Capital C: mitad en contado, mitad de margen del short a 1x (notional N = C/2 por pata).
Retorno diario sobre C = 0.5*(funding_dia + ret_contado - ret_perp). Costo por entrada o salida
de un par: 0.5*(0.10% contado + 0.07% perp) sobre C asignado... ver COSTO_LADO.
Variantes fijadas ANTES de mirar: lookback {14,30} x N {5,10}, rebalanceo 7 días."""
import json
from pathlib import Path
import numpy as np, pandas as pd

D = Path(__file__).resolve().parent.parent / "archivos/funding"
COSTO_LADO = 0.5 * (0.0010 + 0.0007)      # por entrada O salida, sobre el capital asignado al par
MIN_VOL = 5e6                              # USDT/día de volumen perp (media 30d) para ser elegible

def cargar():
    fr, pr, sp, vol = {}, {}, {}, {}
    for d in sorted(D.glob("*USDT")):
        try:
            f = pd.read_csv(d / "funding.csv"); p = pd.read_csv(d / "perp_1d.csv"); s = pd.read_csv(d / "spot_1d.csv")
        except Exception: continue
        f["dia"] = pd.to_datetime(f.fundingTime, unit="ms").dt.floor("D")
        fr[d.name] = f.groupby("dia").fundingRate.sum()
        for dic, x, col in ((pr, p, "close"), (sp, s, "close"), (vol, p, "quote_vol")):
            dic[d.name] = pd.Series(x[col].to_numpy(), index=pd.to_datetime(x.t, unit="ms").dt.floor("D"))
    return [pd.DataFrame(x).sort_index() for x in (fr, pr, sp, vol)]

FR, PR, SP, VOL = cargar()
idx = FR.index.intersection(PR.index).intersection(SP.index)
FR, PR, SP, VOL = (x.loc[idx] for x in (FR, PR, SP, VOL))
base = (SP.pct_change() - PR.pct_change())              # ganancia diaria de la base por unidad de notional
bruto_dia = FR + base                                    # por unidad de notional (no de capital)
elegible = (VOL.rolling(30).mean() > MIN_VOL) & FR.notna() & PR.notna() & SP.notna()

def backtest(lb, n, reb=7, etiqueta=""):
    fr_lb = FR.rolling(lb, min_periods=lb).mean()
    pesos = pd.DataFrame(0.0, index=idx, columns=FR.columns)
    w = pd.Series(0.0, index=FR.columns)
    costos = pd.Series(0.0, index=idx)
    for i, t in enumerate(idx):
        if i % reb == 0 and i > lb:
            sc = fr_lb.loc[t].where(elegible.loc[t]).dropna()
            sc = sc[sc > 0]                                   # sólo si el funding reciente es positivo
            new = pd.Series(0.0, index=FR.columns)
            if len(sc): new[sc.nlargest(n).index] = 1.0 / n   # n fijo: lo no invertido queda en efectivo
            costos.iloc[i] = COSTO_LADO * (new - w).abs().sum()
            w = new
        pesos.iloc[i] = w
    # el peso fijado al cierre de t rige el retorno de t+1 (sin lookahead)
    ret = 0.5 * (pesos.shift(1) * bruto_dia).sum(axis=1) - costos
    ret = ret.loc[idx[lb + 2]:]
    return ret, pesos.shift(1).loc[ret.index], costos.loc[ret.index]

def resumen(r, nombre):
    eq = (1 + r).cumprod(); anual = eq.iloc[-1] ** (365 / len(r)) - 1
    dd = (eq / eq.cummax() - 1).min(); sh = r.mean() / r.std() * np.sqrt(365) if r.std() else np.nan
    return dict(nombre=nombre, dias=len(r), anual=anual, caida=dd, sharpe=sh, total=eq.iloc[-1] - 1)

if __name__ == "__main__":
    print("pares:", FR.shape[1], "| días:", len(idx), idx[0].date(), "->", idx[-1].date())
    # 1) Nivel y persistencia
    anual = (FR.mean() * 365)
    print("\nFunding anualizado, mediana entre pares:", f"{anual.median():.1%}", "| BTC:", f"{anual.get('BTCUSDT', np.nan):.1%}")
    ult = FR.loc[FR.index[-90]:].mean() * 365
    print("Últimos 90d: mediana", f"{ult.median():.1%}", "| p75", f"{ult.quantile(.75):.1%}", "| top5:", ", ".join(f"{k[:-4]} {v:.0%}" for k, v in ult.nlargest(5).items()))
    for lb in (14, 30):
        a = FR.rolling(lb).mean(); b = FR.rolling(lb).mean().shift(-lb)
        cs = [a.loc[t].rank().corr(b.loc[t].rank()) for t in idx[lb::lb] if t in b.index and b.loc[t].notna().sum() > 20]
        print(f"Persistencia lookback {lb}d -> siguientes {lb}d: Spearman medio {np.nanmean(cs):+.2f} (n={len(cs)} cortes, {np.mean(np.array(cs)>0):.0%} positivos)")
    # 2) Estrategia
    res = []
    for lb in (14, 30):
        for n in (5, 10):
            r, w, c = backtest(lb, n)
            m = resumen(r, f"lb{lb}_n{n}")
            m["inv_medio"] = (w.sum(axis=1) > 0).mean(); m["costos_anual"] = c.sum() * 365 / len(c)
            m["mitad1"] = resumen(r.iloc[:len(r)//2], "")["anual"]; m["mitad2"] = resumen(r.iloc[len(r)//2:], "")["anual"]
            m["ult180"] = resumen(r.iloc[-180:], "")["anual"]; m["ult90"] = resumen(r.iloc[-90:], "")["anual"]
            res.append(m)
            r.to_csv(D / f"carry_{m['nombre']}.csv")
    df = pd.DataFrame(res).set_index("nombre")
    pd.set_option("display.width", 200); print("\n", df.round(3).to_string())
