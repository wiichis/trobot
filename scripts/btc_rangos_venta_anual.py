"""v0.4 con venta anual del 10 % del holding (22/10/2026, pedido del usuario).

El usuario sostiene que el costo de las atrapadas sólo se hace efectivo cuando vende BTC, y
vende ~10 % del holding por año. Se simulan los dos mundos con la misma regla: cada 3 de enero
se vende el 10 % del holding (con bot y sin bot; también la parte extra del benchmark B).

Al final se compara el valor total en USDT:
  con bot = BTC (holding + slots) × precio final + USDT del bot + USDT de las ventas anuales
  A (DCA puro, 200 USDT guardados) = BTC × precio + 200 + USDT de las ventas
  B (DCA puro, 200 USDT comprados en BTC el día 1) = BTC × precio + USDT de las ventas
Combinaciones: las 5 finalistas de la v0.4 (elegidas en la selección, sin cambios).

Uso: python3 scripts/btc_rangos_venta_anual.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("v4", REPO / "scripts" / "btc_rangos_v04.py")
v4 = importlib.util.module_from_spec(_spec)
sys.modules["v4"] = v4
_spec.loader.exec_module(v4)
bt, mm = v4.bt, v4.mm

FINALISTAS = [(("A", 0.0), "E4_0.5", "S5"), (("A", 0.0), "E4_0.5", "S4_0.5"), (("B", 1.0), "E4_0.5", "S4_0.75"),
              (("B", 1.0), "E4_1", "S4_0.75"), (("A", 0.0), "E4_0.5", "S4_0.75")]
PERIODOS = {"2021-23": bt.SEL, "2024-26": bt.VAL, "2021-26": (bt.SEL[0], bt.VAL[1])}
VENTA = 0.10


def uno(combo, escenario, periodo) -> dict:
    (s, x), e, sa = combo
    cfg = mm.Config(slots=s, escalon_r=x, salida=sa, ganancia="G4", venta_anual=VENTA)
    r = mm.simular(v4._V, v4._SEN[e], v4._IND["atr"], cfg, bt.dca_fn(escenario), *periodo, rango=v4._IND["rango"])
    m, p = r.motor, r.precio_fin
    con = (m.holding_btc + m.btc_en_slots()) * p + m.usdt_bot() + m.usdt_para_holding + m.ventas_usdt
    a = m.bench_btc * p + m.cfg.capital + m.bench_ventas_usdt
    b = (m.bench_btc + m.bench_b_btc) * p + m.bench_ventas_usdt + m.bench_b_ventas_usdt
    me = pd.DataFrame(m.meses)
    return {"ingreso_bot": me["realizado"].sum(), "ventas_con_bot": m.ventas_usdt, "ventas_sin_bot": m.bench_ventas_usdt,
            "btc_con_bot": m.holding_btc + m.btc_en_slots(), "btc_sin_bot": m.bench_btc,
            "total_con_bot": con, "total_A": a, "total_B": b, "dif_A": con - a, "dif_B": con - b}


def main() -> None:
    v4._init()
    filas = []
    for combo in FINALISTAS:
        nombre = bt.clave({"slots": combo[0][0], "escalon": combo[0][1], "entrada": combo[1], "filtro": False,
                           "salida": combo[2]})
        for per_n, per in PERIODOS.items():
            res = {esc: uno(combo, esc, per) for esc in bt.ESCENARIOS}
            aleat = pd.DataFrame([res[k] for k in res if k.startswith("aleat")]).mean()
            for esc_n, val in (("DCA 100", res["fijo100"]), ("DCA 400", res["fijo400"]), ("DCA aleatorio", aleat)):
                filas.append({"combo": nombre, "periodo": per_n, "dca": esc_n, **dict(val)})
    df = pd.DataFrame(filas)
    out = REPO / "archivos" / "btc_rangos" / "backtest_v04" / "venta_anual_10.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    cols = ["ingreso_bot", "ventas_con_bot", "ventas_sin_bot", "btc_con_bot", "btc_sin_bot", "total_con_bot",
            "total_A", "dif_A", "total_B", "dif_B"]
    for per_n in PERIODOS:
        print(f"\n=== {per_n} ===")
        print(df[df.periodo == per_n].set_index(["combo", "dca"])[cols].round(3).to_string())
    print(f"\nResultados en {out}")


if __name__ == "__main__":
    main()
