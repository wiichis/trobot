#!/usr/bin/env python3
"""¿Cuánto edge tiene la señal de entrada si se la mantiene un horizonte fijo? (02/10/2026)

Re-mide la observación del 31/08 ("+0,198% neto por señal manteniendo 8 h con el stop
actual") con las herramientas de hoy: señales del mismo `_calc_symbol` que corre el bot,
params vigentes (activos de best_prod.json + banca de bench.json) y la config viva
(filtro de régimen ADX 1h).

Para cada señal (primera de cada racha: se descartan las que repiten el mismo lado del
mismo par dentro de los 30 min previos) se simula la salida con tres reglas:

- `sin_stop`: se cierra al horizonte, pase lo que pase.
- `stop`: stop inicial = SL_L/SL_S de la vela de la señal (el que coloca el live), si no
  se tocó antes, se cierra al horizonte.
- `stop_trail`: igual pero el stop sigue al SL del indicador vela a vela (monótono, con la
  vela ANTERIOR, sin lookahead) — el trailing ATR del live, sin TPs ni break-even.

Entrada: al cierre de la vela de la señal con fee maker (2 bps, lo medido en BingX).
Salida: fee taker (5 bps) + slippage `calc_slippage_rate` (el del parity).
Referencia ("baseline"): el retorno al mismo horizonte de TODAS las velas del par, con el
signo del lado — la ventaja es retorno de la señal − baseline.

Escribe archivos/analisis/edge_horizonte_<fecha>.csv (una fila por señal y horizonte).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pkg"))

HORIZONTES_H = (2, 4, 8, 12, 24)
FEE_MAKER = 0.0002
FEE_TAKER = 0.0005
RACHA_BARRAS = 6          # 30 min
BARRAS_H = 12             # velas de 5m por hora


def params_vigentes(archivos=None):
    out = {}
    for f in (archivos or (REPO / "pkg/best_prod.json", REPO / "pkg/bench.json")):
        f = Path(f)
        for e in json.loads(f.read_text()):
            if e.get("params"):
                out[e["symbol"].upper()] = dict(e["params"])
    return out


def simular_par(sym, df, p):
    from pkg import indicadores as ind
    from pkg.backtesting import calc_slippage_rate
    d = ind._calc_symbol(df.copy(), sym, params_override=p).sort_values("date").reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"], utc=True)
    c, h, l = d["close"].to_numpy(float), d["high"].to_numpy(float), d["low"].to_numpy(float)
    atr = d["ATR_pct"].fillna(0).to_numpy(float)
    sl_l, sl_s = d["SL_L"].to_numpy(float), d["SL_S"].to_numpy(float)
    n = len(d)
    filas = []
    # baseline: retorno medio a cada horizonte sobre todas las velas
    base = {H: float(np.nanmean(c[H * BARRAS_H:] / c[:-H * BARRAS_H] - 1)) for H in HORIZONTES_H}
    for lado, col, sl_arr, sg in (("long", "Long_Signal", sl_l, 1), ("short", "Short_Signal", sl_s, -1)):
        sig = d[col].astype(bool).to_numpy()
        idx = np.flatnonzero(sig)
        ult = -10 ** 9
        for i in idx:
            if i - ult <= RACHA_BARRAS:
                ult = i
                continue
            ult = i
            e = c[i]   # entra al cierre de la vela de la señal
            stop0 = sl_arr[i]
            if not np.isfinite(stop0) or (sg == 1 and stop0 >= e) or (sg == -1 and stop0 <= e):
                stop0 = np.nan
            for H in HORIZONTES_H:
                j_fin = i + H * BARRAS_H
                if j_fin >= n:
                    continue
                costo_e = FEE_MAKER
                # sin stop
                r_ns = sg * (c[j_fin] / e - 1)
                slip_fin = calc_slippage_rate(atr[j_fin])
                res = {"sin_stop": r_ns - costo_e - FEE_TAKER - slip_fin}
                # stop fijo y stop con trailing
                for regla in ("stop", "stop_trail"):
                    stop = stop0
                    salida = None
                    for j in range(i + 1, j_fin + 1):
                        if regla == "stop_trail" and np.isfinite(sl_arr[j - 1]) and np.isfinite(stop):
                            stop = max(stop, sl_arr[j - 1]) if sg == 1 else min(stop, sl_arr[j - 1])
                        if np.isfinite(stop) and ((sg == 1 and l[j] <= stop) or (sg == -1 and h[j] >= stop)):
                            salida = (stop, j)
                            break
                    if salida is None:
                        r = sg * (c[j_fin] / e - 1) - slip_fin
                    else:
                        r = sg * (salida[0] / e - 1) - calc_slippage_rate(atr[salida[1]])
                    res[regla] = r - costo_e - FEE_TAKER
                filas.append({"symbol": sym, "lado": lado, "fecha": d["date"].iloc[i], "H": H,
                              "baseline": sg * base[H], **res})
    return filas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "archivos/velas_limpias_5m.csv"))
    ap.add_argument("--out", default=str(REPO / "archivos/analisis"))
    ap.add_argument("--params", nargs="*", default=None,
                    help="json(s) de params (default: best_prod.json + bench.json vigentes)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    from pkg.bench import habilitar_en_indicadores
    params = params_vigentes(args.params)
    habilitar_en_indicadores(list(params))
    velas = pd.read_csv(args.data)
    filas = []
    for sym, p in sorted(params.items()):
        df = velas[velas["symbol"] == sym]
        if df.empty:
            continue
        filas += simular_par(sym, df, p)
        print(sym, "ok", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    r = pd.DataFrame(filas)
    f = out / f"edge_horizonte_{pd.Timestamp.utcnow():%Y%m%d}{args.tag}.csv"
    r.to_csv(f, index=False)
    print("escrito", f, len(r), "filas")


if __name__ == "__main__":
    main()
