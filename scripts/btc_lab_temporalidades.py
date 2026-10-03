#!/usr/bin/env python3
"""Trading de BTC solo, en varias temporalidades (02/10/2026) — reglas clásicas SIN optimizar.

Pregunta del usuario: que el bot opere sólo BTC (long y short) y genere un % de ganancia.
Se miden 4 reglas clásicas con parámetros fijados ANTES de mirar, iguales en barras para
todas las temporalidades, cada una en versión long/short y sólo-long.

Datos (Binance, data.binance.vision):
- 1d: spot BTCUSDT, ago-17 → oct-26 (`archivos/btc/binance_1d/`). Se evalúa 2019-01 →
  2026-10; lo anterior sólo calienta indicadores.
- 4h / 1h: perpetuo BTCUSDT a 1h, ene-20 → sep-26 (`archivos/btc/binance_1h/`). Se evalúa
  desde feb-20 (enero calienta las medias).
- Funding real del perpetuo BTCUSDT desde ene-20 (`archivos/btc/binance_funding/`); antes
  de eso (sólo 2019 en 1d) se supone 0,01% cada 8 h.

Primera corrida (misma fecha) con sólo 10 meses intradía y funding supuesto: 0 de 32;
15m perdía 24-89% en todas las reglas por costos, así que se deja fuera.

Ejecución: señal con el CIERRE de la barra t, operación a la APERTURA de t+1. 1x, sin stops
fuera de la propia regla.
Costos: taker 5 bps + 2 bps de slippage por lado (14 bps por vuelta; dar vuelta de long a
short paga los dos lados). Funding: la posición vigente en cada liquidación (cada 8 h)
paga posición × tasa (el long paga si la tasa es positiva, el short cobra).

Reglas (n en barras; código en pkg/btc_reglas.py):
  T1 precio vs media simple de 100: long arriba, short abajo.
  T2 cruce de medias exponenciales 20/50: long si EMA20 > EMA50, short si no.
  T3 Donchian 20/10: entra al cerrar sobre el máximo de 20 barras (bajo el mínimo para
     short); sale al cerrar bajo el mínimo de 10 (sobre el máximo de 10 para short).
  R1 reversión RSI 14: long si RSI < 30, short si RSI > 70; sale al cruzar 50.
En la versión sólo-long las señales de short dejan la posición en cero.

Criterio fijado de antemano para "candidata" (igual para todas las temporalidades):
retorno neto positivo en las dos mitades (2019/2020-2022 y 2023-2026), ≥ 5 años
positivos y caída máxima < 35%. Son 24 combinaciones: que una pase por azar es
esperable. Una candidata es una hipótesis, no una estrategia.
En 4h/1h se informa además 2020-2025: la primera corrida ya había mirado 2026.
"""
from __future__ import annotations

import importlib.util
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent

# Las reglas viven en pkg/btc_reglas.py (las usa también el bot en vivo). Se carga por ruta
# para no importar el paquete `pkg` entero (credenciales, monkey_bx...).
_spec = importlib.util.spec_from_file_location("btc_reglas", REPO / "pkg" / "btc_reglas.py")
_reglas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reglas)
rsi, posiciones = _reglas.rsi, _reglas.posiciones
COSTO_LADO = 0.0005 + 0.0002
FUNDING_8H_SUPUESTO = 0.0001
SALIDA = REPO / "archivos/analisis/btc_lab_temporalidades.csv"


def _leer_zip(z):
    with zipfile.ZipFile(z) as zf:
        return zf.read(zf.namelist()[0]).decode()


def _leer_velas(carpeta, patron):
    frames = []
    for z in sorted((REPO / carpeta).glob(patron)):
        raw = _leer_zip(z)
        df = pd.read_csv(io.StringIO(raw), header=0 if raw.startswith("open_time") else None).iloc[:, :5]
        df.columns = ["ot", "open", "high", "low", "close"]
        ot = pd.to_numeric(df["ot"])
        df["date"] = pd.to_datetime(ot, unit="us" if ot.max() > 1e14 else "ms")
        frames.append(df[["date", "open", "high", "low", "close"]])
    return pd.concat(frames).drop_duplicates("date").set_index("date").sort_index().astype(float)


def cargar():
    series = {"1d": _leer_velas("archivos/btc/binance_1d", "BTCUSDT-1d-*.zip")}
    h1 = _leer_velas("archivos/btc/binance_1h", "BTCUSDT-1h-*.zip")
    series["4h"] = h1.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    series["1h"] = h1
    return series


def cargar_funding():
    fr = []
    for z in sorted((REPO / "archivos/btc/binance_funding").glob("BTCUSDT-fundingRate-*.zip")):
        fr.append(pd.read_csv(io.StringIO(_leer_zip(z))))
    f = pd.concat(fr)
    idx = pd.to_datetime(f["calc_time"], unit="ms").dt.floor("min")
    return pd.Series(f["last_funding_rate"].to_numpy(float), index=idx).sort_index()


def funding_por_barra(h, horas_barra, funding):
    """Suma de las tasas liquidadas dentro de cada barra [open_i, open_i + duración)."""
    t = funding.index.to_numpy()
    acum = np.concatenate([[0.0], np.cumsum(funding.to_numpy())])
    ini = h.index.to_numpy()
    fin = (h.index + pd.Timedelta(hours=horas_barra)).to_numpy()
    real = acum[np.searchsorted(t, fin, side="left")] - acum[np.searchsorted(t, ini, side="left")]
    supuesto = FUNDING_8H_SUPUESTO * horas_barra / 8
    return pd.Series(np.where(h.index < funding.index.min(), supuesto, real), index=h.index)


def simular(h, pos, fund, desde, hasta):
    """Posición decidida al cierre de t, vigente de open[t+1] a open[t+2]. Devuelve
    (retorno neto por barra, funding pagado por barra, trades) dentro de [desde, hasta)."""
    o = h["open"]
    r = (o.shift(-1) / o - 1)                       # de la apertura de i a la de i+1
    vig = pos.shift(1).fillna(0.0)                  # vigente durante la barra i
    costo = (vig - vig.shift(1).fillna(0.0)).abs() * COSTO_LADO
    pago_funding = vig * fund
    bruto = vig * r - pago_funding
    neto = bruto - costo
    mask = (h.index >= desde) & (h.index < hasta) & r.notna()
    neto, bruto, vig, pago_funding = neto[mask], bruto[mask], vig[mask], pago_funding[mask]
    # por operación: retorno compuesto de la posición (con funding) menos entrada y salida
    trades = []
    ini, acum, lado = None, 1.0, 0.0
    for t, v, x in zip(vig.index, vig.to_numpy(), bruto.to_numpy()):
        if v != lado:
            if lado != 0:
                trades.append(dict(entrada=ini, lado=lado, ret=acum - 1 - 2 * COSTO_LADO))
            ini, acum, lado = t, 1.0, v
        if lado != 0:
            acum *= 1 + x
    if lado != 0:
        trades.append(dict(entrada=ini, lado=lado, ret=acum - 1 - 2 * COSTO_LADO, abierta=True))
    return neto, pago_funding, pd.DataFrame(trades)


def metricas(neto, trades, horas_barra):
    eq = (1 + neto).cumprod()
    total = eq.iloc[-1] - 1 if len(eq) else 0.0
    anios = len(neto) * horas_barra / 24 / 365.25
    dd = (eq / eq.cummax() - 1).min() if len(eq) else 0.0
    n = len(trades)
    w = trades.ret[trades.ret > 0] if n else pd.Series(dtype=float)
    l = trades.ret[trades.ret <= 0] if n else pd.Series(dtype=float)
    return dict(retorno=total, anual=(1 + total) ** (1 / anios) - 1 if anios > 0 and total > -1 else np.nan,
                caida_max=dd, trades=n, acierto=(trades.ret > 0).mean() if n else np.nan,
                gana_medio=w.mean() if len(w) else np.nan, pierde_medio=l.mean() if len(l) else np.nan,
                t=trades.ret.mean() / (trades.ret.std() / np.sqrt(n)) if n > 2 else np.nan)


def main():
    series = cargar()
    funding = cargar_funding()
    horas = {"1d": 24, "4h": 4, "1h": 1}
    cortes = {"1d": ("2019-01-01", "2023-01-01", "2026-10-02"),
              "4h": ("2020-02-01", "2023-01-01", "2026-10-01"),
              "1h": ("2020-02-01", "2023-01-01", "2026-10-01")}
    filas = []
    for tf, h in series.items():
        a, m, b = (pd.Timestamp(x) for x in cortes[tf])
        fund = funding_por_barra(h, horas[tf], funding)
        bh = {k: h["open"].asof(fin) / h["open"].asof(ini) - 1 for k, (ini, fin) in
              {"total": (a, b), "mitad1": (a, m), "mitad2": (m, b)}.items()}
        c = h["close"].loc[a:b]
        for regla in ("T1", "T2", "T3", "R1"):
            for solo_long in (False, True):
                pos = posiciones(h, regla, solo_long)
                fila = dict(tf=tf, regla=regla, modo="solo long" if solo_long else "long/short",
                            holding=bh["total"], holding_m1=bh["mitad1"], holding_m2=bh["mitad2"],
                            holding_caida=(c / c.cummax() - 1).min())
                neto, pf, tr = simular(h, pos, fund, a, b)
                fila.update(metricas(neto, tr, horas[tf]))
                fila["funding_pagado"] = pf.sum()
                for k, (ini, fin) in {"m1": (a, m), "m2": (m, b)}.items():
                    n2, _, t2 = simular(h, pos, fund, ini, fin)
                    fila[f"ret_{k}"] = metricas(n2, t2, horas[tf])["retorno"]
                if tf != "1d":
                    n3, _, t3 = simular(h, pos, fund, a, pd.Timestamp("2026-01-01"))
                    fila["ret_2020_2025"] = metricas(n3, t3, horas[tf])["retorno"]
                anual = neto.groupby(neto.index.year).apply(lambda x: (1 + x).prod() - 1)
                fila["anios_pos"] = f"{int((anual > 0).sum())}/{len(anual)}"
                fila["por_anio"] = " ".join(f"{y % 100:02d}:{v*100:+.0f}" for y, v in anual.items())
                fila["candidata"] = bool(fila["ret_m1"] > 0 and fila["ret_m2"] > 0
                                         and (anual > 0).sum() >= 5 and fila["caida_max"] > -0.35)
                filas.append(fila)
    t = pd.DataFrame(filas)
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(SALIDA, index=False)

    pct = lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "—"
    pd.set_option("display.width", 250)
    for tf in ("1d", "4h", "1h"):
        g = t[t.tf == tf]
        a, m, b = cortes[tf]
        print(f"\n=== {tf}  {a} → {b}  | holding BTC: {pct(g.holding.iloc[0])} "
              f"(mitades {pct(g.holding_m1.iloc[0])} / {pct(g.holding_m2.iloc[0])}, caída máx {pct(g.holding_caida.iloc[0])})")
        cols = dict(regla=g.regla, modo=g.modo, retorno=g.retorno.map(pct), anual=g.anual.map(pct),
                    mitad1=g.ret_m1.map(pct), mitad2=g.ret_m2.map(pct), caida_max=g.caida_max.map(pct),
                    trades=g.trades, acierto=g.acierto.map(lambda x: f"{x:.0%}" if pd.notna(x) else "—"),
                    gana=g.gana_medio.map(pct), pierde=g.pierde_medio.map(pct), t=g.t.round(2),
                    funding=g.funding_pagado.map(pct))
        if tf != "1d":
            cols["2020-25"] = g.ret_2020_2025.map(pct)
        cols["años+"] = g.anios_pos
        cols["candidata"] = g.candidata.map({True: "SÍ", False: ""})
        print(pd.DataFrame(cols).to_string(index=False))
        for _, f in g.iterrows():
            print(f"   {f.regla} {f.modo:10s} por año: {f.por_anio}")
    print(f"\nCandidatas: {int(t.candidata.sum())} de {len(t)}")


if __name__ == "__main__":
    main()
