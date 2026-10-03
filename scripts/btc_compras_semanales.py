#!/usr/bin/env python3
"""Compras semanales de BTC: ¿mejora el resultado comprar según el precio? (02/10/2026)

Pregunta del usuario: compra BTC para tenerlo (holding) y no sabe en qué momento comprar.
Se simula una compra semanal con el mismo depósito y se comparan reglas fijadas ANTES de
mirar resultados (no se optimizan).

Datos: velas diarias spot BTCUSDT de Binance (data.binance.vision), ago-17 → oct-26, en
`archivos/btc/binance_1d/`. La historia anterior a 2019 sólo calienta los indicadores.

Reglas comunes:
- Cada lunes entran 100 USDT a la caja. La compra se hace a la APERTURA del lunes; los
  indicadores usan sólo cierres hasta el domingo (sin mirar el futuro).
- Comisión 0,1% sobre lo comprado. Lo que queda sin gastar en la caja cuenta al final por
  su valor en USDT.
- Valor final = BTC × último cierre + caja.

Estrategias:
  A   compra fija: 100 USDT cada lunes.
  B1  según la media de 200 días (r = cierre / media 200 d): ×2 si r < 1; ×1 si r < 1,5;
      ×0,5 si no.
  B2  según la distancia al máximo histórico de cierre (dd): ×2 si dd ≤ −40%; ×1,5 si
      dd ≤ −20%; ×1 si dd ≤ −5%; ×0,5 si está a menos de 5% del máximo.
  B3  según el RSI semanal (14 semanas): ×2 si < 40; ×1 hasta 70; ×0,5 si > 70.
      En B1-B3 la compra es (multiplicador × 100) acotada a la caja disponible: comprar
      "más" sólo es posible con lo que se ahorró antes. Nunca deja de comprar del todo.
  C   esperar caída: acumula y compra TODA la caja sólo cuando el precio está ≥20% bajo
      el máximo histórico.
  D   referencia: todo el dinero de las semanas del período invertido el primer lunes
      (supone tener el total desde el principio; flujos distintos, no es comparable 1 a 1).

Criterio fijado de antemano: una regla B o C MEJORA a A si su valor final supera al de A
por ≥1% en las dos mitades (2019-2022 y 2023-2026) y en ≥5 de los 7 arranques anuales
(enero de 2019 a 2025, todos hasta el final).
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATOS = REPO / "archivos/btc/binance_1d"
SALIDA = REPO / "archivos/analisis/btc_compras_semanales.csv"
DEPOSITO = 100.0
COMISION = 0.001


def cargar():
    frames = []
    for z in sorted(DATOS.glob("BTCUSDT-1d-*.zip")):
        with zipfile.ZipFile(z) as zf:
            raw = zf.read(zf.namelist()[0]).decode()
        df = pd.read_csv(io.StringIO(raw), header=0 if raw.startswith("open_time") else None).iloc[:, :5]
        df.columns = ["ot", "open", "high", "low", "close"]
        ot = pd.to_numeric(df["ot"])
        df["date"] = pd.to_datetime(ot, unit="us" if ot.max() > 1e14 else "ms").dt.normalize()   # UTC, sin tz
        frames.append(df[["date", "open", "high", "low", "close"]])
    d = pd.concat(frames).drop_duplicates("date").set_index("date").sort_index()
    faltan = pd.date_range(d.index.min(), d.index.max(), freq="D").difference(d.index)
    if len(faltan):
        raise ValueError(f"faltan {len(faltan)} días, p. ej. {list(faltan[:5])}")
    return d


def rsi_wilder(c, n=14):
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def lunes_con_indicadores(d):
    c = d["close"]
    ind = pd.DataFrame({"sma200": c.rolling(200).mean(), "ath": c.cummax(), "cierre": c})
    semanal = c.resample("W-SUN").last()
    ind["rsi_sem"] = rsi_wilder(semanal).reindex(c.index, method="ffill")
    ind = ind.shift(1)                       # el lunes sólo se conoce hasta el domingo
    lunes = d.index[d.index.dayofweek == 0]
    out = ind.loc[lunes].copy()
    out["precio"] = d.loc[lunes, "open"]
    out["r200"] = out["cierre"] / out["sma200"]
    out["dd"] = out["cierre"] / out["ath"] - 1
    return out


def mult_b1(f):
    return 2.0 if f.r200 < 1.0 else (1.0 if f.r200 < 1.5 else 0.5)


def mult_b2(f):
    if f.dd <= -0.40:
        return 2.0
    if f.dd <= -0.20:
        return 1.5
    if f.dd <= -0.05:
        return 1.0
    return 0.5


def mult_b3(f):
    return 2.0 if f.rsi_sem < 40 else (1.0 if f.rsi_sem <= 70 else 0.5)


def simular(semanas, regla, precio_final):
    caja = btc = gastado = 0.0
    compras = 0
    for _, f in semanas.iterrows():
        caja += DEPOSITO
        if regla == "A":
            monto = DEPOSITO
        elif regla == "C":
            monto = caja if f.dd <= -0.20 else 0.0
        else:
            monto = {"B1": mult_b1, "B2": mult_b2, "B3": mult_b3}[regla](f) * DEPOSITO
        monto = min(monto, caja)
        if monto > 0:
            btc += monto * (1 - COMISION) / f.precio
            caja -= monto
            gastado += monto
            compras += 1
    depositado = DEPOSITO * len(semanas)
    return dict(depositado=depositado, btc=btc, costo_medio=gastado / btc if btc else np.nan,
                caja_final=caja, valor_final=btc * precio_final + caja, compras=compras)


def simular_todo_al_inicio(semanas, precio_final):
    depositado = DEPOSITO * len(semanas)
    btc = depositado * (1 - COMISION) / semanas["precio"].iloc[0]
    return dict(depositado=depositado, btc=btc, costo_medio=depositado / btc, caja_final=0.0,
                valor_final=btc * precio_final, compras=1)


def main():
    d = cargar()
    lunes = lunes_con_indicadores(d)
    precio_final = float(d["close"].iloc[-1])
    print(f"BTC diario {d.index.min():%d/%m/%y} → {d.index.max():%d/%m/%y}; último cierre {precio_final:,.0f}")
    fin = lunes.index.max()
    periodos = {"completo 2019-2026": ("2019-01-01", fin), "mitad 2019-2022": ("2019-01-01", "2022-12-31"),
                "mitad 2023-2026": ("2023-01-01", fin)}
    periodos.update({f"desde ene-{y}": (f"{y}-01-01", fin) for y in range(2019, 2026)})
    filas = []
    for nombre, (ini, fn) in periodos.items():
        sem = lunes.loc[ini:fn].dropna(subset=["sma200", "rsi_sem"])
        # cada período se valúa con el cierre del último día del período
        pf = float(d["close"].loc[:fn].iloc[-1]) if fn != fin else precio_final
        base = None
        for regla in ("A", "B1", "B2", "B3", "C", "D"):
            r = simular_todo_al_inicio(sem, pf) if regla == "D" else simular(sem, regla, pf)
            base = r["valor_final"] if regla == "A" else base
            filas.append(dict(periodo=nombre, regla=regla, semanas=len(sem), **r,
                              vs_A=r["valor_final"] / base - 1, multiplo=r["valor_final"] / r["depositado"]))
    t = pd.DataFrame(filas)
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(SALIDA, index=False)

    pd.set_option("display.width", 200)
    for nombre in periodos:
        g = t[t.periodo == nombre]
        print(f"\n{nombre} ({int(g.semanas.iloc[0])} semanas, depositado {g.depositado.iloc[0]:,.0f})")
        print(g[["regla", "valor_final", "multiplo", "vs_A", "costo_medio", "caja_final", "compras"]]
              .assign(valor_final=lambda x: x.valor_final.round(0), multiplo=lambda x: x.multiplo.round(2),
                      vs_A=lambda x: (x.vs_A * 100).round(1), costo_medio=lambda x: x.costo_medio.round(0),
                      caja_final=lambda x: x.caja_final.round(0))
              .to_string(index=False))

    print("\nCRITERIO (mejora a A: ≥+1% en las dos mitades y en ≥5 de 7 arranques anuales)")
    arr = [f"desde ene-{y}" for y in range(2019, 2026)]
    for regla in ("B1", "B2", "B3", "C"):
        g = t[t.regla == regla].set_index("periodo")["vs_A"]
        mitades = g["mitad 2019-2022"] >= 0.01 and g["mitad 2023-2026"] >= 0.01
        gana_arr = int((g[arr] >= 0.01).sum())
        veredicto = "MEJORA" if mitades and gana_arr >= 5 else "no mejora"
        print(f"  {regla}: mitades {g['mitad 2019-2022']*100:+.1f}% / {g['mitad 2023-2026']*100:+.1f}%, "
              f"arranques ganados {gana_arr}/7 → {veredicto}")


if __name__ == "__main__":
    main()
