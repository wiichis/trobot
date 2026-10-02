#!/usr/bin/env python3
"""Laboratorio de familias de estrategia (02/10/2026) — primer barrido, SIN optimizar.

Contexto: la señal fresh-cross 5m del bot no tiene edge persistente (scripts/edge_horizonte.py)
y optimizar sus params empeora el resultado fuera de muestra (scripts/optimizar_parity.py).
Hipótesis del usuario: no hay UN edge fijo; hay edges que dependen del régimen. Este script
mide familias clásicas con parámetros fijados de antemano, por mes y por régimen de
mercado, para ver si alguna tiene edge y en qué condiciones.

Reglas del barrido (fijadas antes de mirar resultados):
- Velas de 1 h (resampleadas de las 5m). Señal al cierre de la vela t, entrada en la
  APERTURA de t+1. Stops sobre high/low horarios; si en una vela se tocan stop y salida,
  manda el stop (conservador).
- Costos: taker en entrada y salida (5 bps c/u) + 3 bps de slippage c/u = 16 bps por vuelta.
- Una posición por par y familia a la vez (no se superponen).
- Régimen de mercado (índice equiponderado de los pares): "tendencia" si |retorno 72 h| del
  índice supera su mediana de los 30 días previos; si no, "rango". Calculado con datos
  previos a la entrada.

Familias:
  F1 breakout 48 h, stop 2×ATR(14, 1h) con trailing, sin TP.
  F2 momentum de serie: signo del retorno 72 h, se mantiene 24 h.
  F3 reversión RSI(14, 1h): <25 compra, >75 vende; sale al cruzar 50 o a las 12 h.
  F4 reversión Bollinger(20, 2σ): cierre fuera de banda, contra; sale en la media o a las 12 h.
  F5 momentum entre pares: cada día 00:00 UTC, largo las 3 con mayor retorno 72 h y corto
     las 3 con menor; 24 h.
  F6 reversión entre pares: cada día, largo las 3 con PEOR retorno 24 h y corto las 3 mejores; 24 h.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
COSTO_VUELTA = 2 * (0.0005 + 0.0003)


def cargar_1h(archivos):
    series = {}
    for f in archivos:
        d = pd.read_csv(f)
        d["date"] = pd.to_datetime(d["date"], utc=True, format="mixed")
        for sym, g in d.groupby("symbol"):
            if sym in series:
                continue
            g = g.set_index("date").sort_index()
            h = g.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
            series[sym] = h.iloc[:-1]   # la última hora puede estar incompleta
    return series


def atr(h, n=14):
    pc = h["close"].shift()
    tr = pd.concat([h["high"] - h["low"], (h["high"] - pc).abs(), (h["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def recorrer(h, señales, salida):
    """Simula trades no superpuestos. `señales`: Serie {+1,-1,0} al cierre de t.
    `salida(i_entrada, lado, j)` -> (salir: bool, precio o None) evaluado en la vela j."""
    o, hi, lo, c = (h[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    s = señales.to_numpy()
    idx = h.index
    trades, i, n = [], 0, len(h)
    while i < n - 1:
        if s[i] == 0 or np.isnan(s[i]):
            i += 1
            continue
        lado, ie = int(s[i]), i + 1
        entrada = o[ie]
        j, precio = ie, None
        while j < n:
            salir, precio = salida(ie, lado, j)
            if salir:
                break
            j += 1
        if j >= n:
            break
        trades.append({"entrada": idx[ie], "salida": idx[j], "lado": lado,
                       "ret": lado * (precio / entrada - 1) - COSTO_VUELTA})
        i = j + 1
    return trades


def familias_por_par(sym, h):
    out = []
    c, hi, lo = h["close"], h["high"], h["low"]
    o_np, hi_np, lo_np, c_np = (h[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    a = atr(h).to_numpy(float)

    # F1 breakout 48 h con trailing 2×ATR
    up = c > hi.shift(1).rolling(48).max()
    dn = c < lo.shift(1).rolling(48).min()
    s1 = pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=h.index)
    estado = {}
    def sal_f1(ie, lado, j):
        if j == ie:
            estado["stop"] = o_np[ie] - lado * 2 * a[ie - 1]
        stop = estado["stop"]
        if (lado == 1 and lo_np[j] <= stop) or (lado == -1 and hi_np[j] >= stop):
            return True, stop
        nuevo = c_np[j] - lado * 2 * a[j]
        estado["stop"] = max(stop, nuevo) if lado == 1 else min(stop, nuevo)
        return False, None
    out += [dict(t, familia="F1_breakout", symbol=sym) for t in recorrer(h, s1, sal_f1)]

    # F2 momentum de serie: signo del retorno 72 h, 24 h
    s2 = np.sign(c / c.shift(72) - 1).fillna(0)
    out += [dict(t, familia="F2_momentum_serie", symbol=sym)
            for t in recorrer(h, s2, lambda ie, lado, j: (j - ie >= 23, c_np[j]))]

    # F3 reversión RSI
    r = rsi(c).to_numpy(float)
    s3 = pd.Series(np.where(r < 25, 1, np.where(r > 75, -1, 0)), index=h.index)
    def sal_f3(ie, lado, j):
        cruzo = (lado == 1 and r[j] >= 50) or (lado == -1 and r[j] <= 50)
        return (cruzo or j - ie >= 11), c_np[j]
    out += [dict(t, familia="F3_reversion_rsi", symbol=sym) for t in recorrer(h, s3, sal_f3)]

    # F4 reversión Bollinger
    m = c.rolling(20).mean()
    sd = c.rolling(20).std()
    s4 = pd.Series(np.where(c < m - 2 * sd, 1, np.where(c > m + 2 * sd, -1, 0)), index=h.index)
    m_np = m.to_numpy(float)
    def sal_f4(ie, lado, j):
        llego = (lado == 1 and c_np[j] >= m_np[j]) or (lado == -1 and c_np[j] <= m_np[j])
        return (llego or j - ie >= 11), c_np[j]
    out += [dict(t, familia="F4_reversion_bollinger", symbol=sym) for t in recorrer(h, s4, sal_f4)]
    return out


def familias_cruzadas(series):
    closes = pd.DataFrame({s: h["close"] for s, h in series.items()})
    opens = pd.DataFrame({s: h["open"] for s, h in series.items()})
    out = []
    dias = closes.index[(closes.index.hour == 0)]
    for t in dias:
        ie = closes.index.searchsorted(t) + 1
        je = ie + 23
        if je >= len(closes.index):
            break
        hora_e, hora_s = closes.index[ie], closes.index[je]
        for familia, look, signo in (("F5_momentum_cruzado", 72, 1), ("F6_reversion_cruzada", 24, -1)):
            if closes.index.searchsorted(t) - look < 0:
                continue
            ret = (closes.loc[t] / closes.shift(look).loc[t] - 1).dropna()
            ent, sal = opens.loc[hora_e], closes.loc[hora_s]
            ret = ret[ent.reindex(ret.index).notna() & sal.reindex(ret.index).notna()]
            if len(ret) < 10:
                continue
            orden = ret.sort_values()
            largos = orden.index[-3:] if signo == 1 else orden.index[:3]
            cortos = orden.index[:3] if signo == 1 else orden.index[-3:]
            for sym, lado in [(x, 1) for x in largos] + [(x, -1) for x in cortos]:
                out.append({"familia": familia, "symbol": sym, "entrada": hora_e, "salida": hora_s, "lado": lado,
                            "ret": lado * (sal[sym] / ent[sym] - 1) - COSTO_VUELTA})
    return out


def regimen(series, fechas):
    rets = pd.DataFrame({s: np.log(h["close"]).diff() for s, h in series.items()})
    idx = rets.mean(axis=1, skipna=True).fillna(0).cumsum()
    r72 = (idx - idx.shift(72)).abs()
    umbral = r72.rolling(24 * 30, min_periods=24 * 10).median()
    tend = (r72 > umbral).shift(1)       # sólo con datos previos
    return tend.reindex(fechas, method="ffill").to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--velas", nargs="*", default=[str(REPO / "archivos/velas_limpias_5m.csv"),
                                                    str(REPO / "archivos/candidatos/velas_candidatos.csv")])
    ap.add_argument("--out", default=str(REPO / "archivos/analisis/lab_trades.csv"))
    args = ap.parse_args()
    series = cargar_1h(args.velas)
    trades = []
    for sym, h in sorted(series.items()):
        trades += familias_por_par(sym, h)
    trades += familias_cruzadas(series)
    t = pd.DataFrame(trades)
    t["regimen"] = np.where(regimen(series, pd.DatetimeIndex(t["entrada"])) == True, "tendencia", "rango")  # noqa: E712
    t["mes"] = pd.to_datetime(t["entrada"]).dt.strftime("%Y-%m")
    t.to_csv(args.out, index=False)

    pd.set_option("display.width", 220)
    print(f"{len(series)} pares, {len(t)} trades, {t['entrada'].min():%d/%m/%y} → {t['entrada'].max():%d/%m/%y}\n")
    g = t.groupby("familia")["ret"]
    res = pd.DataFrame({"n": g.size(), "media_%": g.mean() * 100,
                        "t": g.mean() / (g.std() / np.sqrt(g.size())), "gana_%": g.apply(lambda x: (x > 0).mean() * 100)})
    meses = t.groupby(["familia", "mes"])["ret"].mean().unstack()
    res["meses+"] = (meses > 0).sum(axis=1).astype(str) + "/" + meses.notna().sum(axis=1).astype(str)
    print("GENERAL"); print(res.round(3).to_string())
    print("\nPOR RÉGIMEN (media %, n)")
    rg = t.groupby(["familia", "regimen"])["ret"].agg(["mean", "size", "std"])
    rg["t"] = rg["mean"] / (rg["std"] / np.sqrt(rg["size"]))
    rg["mean"] *= 100
    print(rg[["mean", "size", "t"]].round(3).unstack().to_string())
    print("\nPOR MES (media %)"); print((meses * 100).round(2).to_string())


if __name__ == "__main__":
    main()
