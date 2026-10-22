"""Modo Acumulación (§13 de docs/reglas_bot_btc.md) contra comprar todo de una vez — 22/10/2026.

Una compra grande para holding se parte en tramos y cada tramo espera una señal; lo que no se
ejecutó al vencer el plazo se compra a mercado. Se compara el precio promedio obtenido (con
costos) contra el lump sum: comprar todo en la apertura del día de inicio.

Fijado antes de correr:
- Inicio: cada día a las 00:00 UTC; selección 2021-01-03 → 2023-12-31, validación 2024-01-03 →
  plazo antes del fin de los datos. Se informan las dos por separado.
- Variantes: tramos {3, 4, 6} × señal {E1 2 %, E2, escalonado} × plazo {7, 14, 21} días.
  * E1/E2: un tramo en la apertura siguiente a cada vela con señal (`pkg/btc_rangos_reglas.py`,
    sin filtro), como máximo uno por día UTC.
  * Escalonado: el primer tramo en el inicio; el tramo k (k = 1…n−1) cuando el low toca
    precio_inicio × (1 − 2 % × k), comprado a ese nivel (o a la apertura si abre debajo).
  * Al vencer: el resto a la apertura de la vela del vencimiento.
- Costos: comisión 0,1 % + slippage 0,05 % por compra, iguales para el lump sum.
- KPIs: mejora del precio promedio contra el lump sum (positivo = la acumulación compró más
  barato) y % de inicios en que gana.

Uso: python3 scripts/btc_acumulacion_backtest.py
"""
from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "archivos" / "btc_rangos" / "backtest"
COSTO = (1 + 0.0005) / (1 - 0.001)  # precio efectivo por BTC = precio × COSTO

spec = importlib.util.spec_from_file_location("btc_rangos_reglas", REPO / "pkg" / "btc_rangos_reglas.py")
rr = importlib.util.module_from_spec(spec)
sys.modules["btc_rangos_reglas"] = rr
spec.loader.exec_module(rr)

TRAMOS = [3, 4, 6]
SENALES = ["E1_2", "E2", "escalonado"]
PLAZOS = [7, 14, 21]
PERIODOS = {"seleccion": ("2021-01-03", "2024-01-01"), "validacion": ("2024-01-03", "2026-10-22")}


def precio_promedio(precios: list[float]) -> float:
    """Mismo monto en cada tramo: el precio promedio es la media armónica de los precios."""
    p = np.asarray(precios) * COSTO
    return len(p) / np.sum(1 / p)


def acumular(v, sen_idx, dia, i0, n, senal, plazo_h):
    o, l = v["open"], v["low"]
    fin = i0 + plazo_h
    precios = []
    if senal == "escalonado":
        p0 = o[i0]
        precios.append(p0)
        for k in range(1, n):
            nivel = p0 * (1 - 0.02 * k)
            idx = np.nonzero(l[i0:fin] <= nivel)[0]
            if not len(idx):
                break
            j = i0 + idx[0]
            precios.append(min(nivel, o[j]))
    else:
        ultimo_dia = None
        # señal en la vela j → compra en la apertura de j+1, dentro del plazo
        for j in sen_idx[(sen_idx >= i0 - 1) & (sen_idx < fin - 1)]:
            if len(precios) == n:
                break
            if dia[j + 1] == ultimo_dia:
                continue
            precios.append(o[j + 1])
            ultimo_dia = dia[j + 1]
    while len(precios) < n:
        precios.append(o[fin])
    return precio_promedio(precios)


def main() -> None:
    v = pd.read_csv(REPO / "archivos/btc_rangos/velas_1h.csv", parse_dates=["date"], index_col="date")
    ind = rr.indicadores(v)
    arr = {k: v[k].to_numpy() for k in ("open", "low")}
    dia = v.index.normalize().to_numpy()
    sen = {s: np.nonzero(rr.senal_entrada(v, ind, s, False).to_numpy())[0] for s in ("E1_2", "E2")}
    sen["escalonado"] = np.array([], dtype=int)
    filas = []
    for nombre, (a, b) in PERIODOS.items():
        inicios = np.nonzero((v.index >= a) & (v.index < b) & (v.index.hour == 0))[0]
        for n, s, plazo in itertools.product(TRAMOS, SENALES, PLAZOS):
            ph = plazo * 24
            mejoras = []
            for i0 in inicios:
                if i0 + ph >= len(v):
                    continue
                pa = acumular(arr, sen[s], dia, i0, n, s, ph)
                lump = arr["open"][i0] * COSTO
                mejoras.append((lump / pa - 1) * 100)
            m = np.asarray(mejoras)
            filas.append({"periodo": nombre, "tramos": n, "senal": s, "plazo_d": plazo, "inicios": len(m),
                          "mejora_media_pct": m.mean(), "mejora_mediana_pct": np.median(m),
                          "gana_pct": (m > 0).mean() * 100, "peor_pct": m.min(), "mejor_pct": m.max()})
    df = pd.DataFrame(filas)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "acumulacion.csv", index=False)
    piv = df.pivot_table(index=["tramos", "senal", "plazo_d"], columns="periodo",
                         values=["mejora_media_pct", "gana_pct"]).round(2)
    print(piv.to_string())


if __name__ == "__main__":
    main()
