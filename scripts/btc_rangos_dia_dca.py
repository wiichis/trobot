"""¿Depende el resultado del día en que cae el DCA? (22/10/2026, objeción del usuario).

El DCA compra el día de cierre al precio que haya, igual con bot y sin bot. Para descartar que
el día 3 haya sido un día "barato" que favorece al DCA puro, se repite la comparación con el
cierre y el DCA en cada día del 1 al 28. Mismo escenario que `btc_rangos_venta_anual.py`: las
5 finalistas de la v0.4, 2021-01-03 → 2026-10-03, venta anual del 10 % del holding.

Uso: python3 scripts/btc_rangos_dia_dca.py [--procesos 8] [--parcial]
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("va", REPO / "scripts" / "btc_rangos_venta_anual.py")
va = importlib.util.module_from_spec(_spec)
sys.modules["va"] = va
_spec.loader.exec_module(va)
v4, bt, mm = va.v4, va.bt, va.mm
PERIODO = (bt.SEL[0], bt.VAL[1])


def correr(args):
    combo, dia, esc = args
    (s, x), e, sa = combo
    cfg = mm.Config(slots=s, escalon_r=x, salida=sa, ganancia="G4", venta_anual=va.VENTA, dia_cierre=dia,
                    reposicion_parcial=v4.parcial())
    r = mm.simular(v4._V, v4._SEN[e], v4._IND["atr"], cfg, bt.dca_fn(esc), *PERIODO, rango=v4._IND["rango"])
    m, p = r.motor, r.precio_fin
    con = (m.holding_btc + m.btc_en_slots()) * p + m.usdt_bot() + m.usdt_para_holding + m.ventas_usdt
    a = m.bench_btc * p + m.cfg.capital + m.bench_ventas_usdt
    b = (m.bench_btc + m.bench_b_btc) * p + m.bench_ventas_usdt + m.bench_b_ventas_usdt
    nombre = bt.clave({"slots": s, "escalon": x, "entrada": e, "filtro": False, "salida": sa})
    return {"combo": nombre, "dia": dia, "dca": esc, "ingreso_bot": pd.DataFrame(m.meses)["realizado"].sum(),
            "dif_A": con - a, "dif_B": con - b}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procesos", type=int, default=8)
    ap.add_argument("--parcial", action="store_true")
    args = ap.parse_args()
    if args.parcial:
        os.environ["TROBOT_REPOSICION_PARCIAL"] = "1"
    tareas = list(itertools.product(va.FINALISTAS, range(1, 29), ["fijo100", "fijo400"]))
    with Pool(args.procesos, initializer=v4._init) as pool:
        df = pd.DataFrame(pool.map(correr, tareas, chunksize=4))
    out = REPO / "archivos" / "btc_rangos" / "backtest_v04" / f"dia_dca{'_parcial' if v4.parcial() else ''}.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    g = df.groupby(["combo", "dca"]).agg(
        ingreso_bot=("ingreso_bot", "median"),
        dias_gana_A=("dif_A", lambda x: int((x > 0).sum())), dif_A_mediana=("dif_A", "median"),
        dif_A_mejor=("dif_A", "max"), dif_A_peor=("dif_A", "min"),
        dias_gana_B=("dif_B", lambda x: int((x > 0).sum())), dif_B_mediana=("dif_B", "median"))
    print("Sobre 28 días posibles de DCA (USDT, 2021-26, venta anual 10 %):")
    print(g.round(0).to_string())
    print(f"\nResultados en {out}")


if __name__ == "__main__":
    main()
