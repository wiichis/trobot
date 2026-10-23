"""Bot BTC por rangos v0.4 — objetivos variables según el día (22/10/2026).

Pedido del usuario: el bot no espera un 3 % fijo; la ganancia se calcula según cómo está el
día, con una meta modesta de 1-2 % al mes (15-20 % al año) como fuente adicional. Arranca con
200 USDT; cuando el usuario compra BTC (DCA mensual, §3 y §13) ese dinero repone primero los
slots atrapados, que pasan a su holding (es BTC que iba a comprar igual) y el bot sigue
operando. Ganancia: 90 % queda en el bot, 10 % a holding (G4).

FIJADO ANTES DE CORRER:
- R = mediana de 14 días cerrados de (máximo − mínimo) / cierre (`rango_diario`). Diario spot
  de Binance desde 2017 + días completos de las velas 1 h.
- Entradas E4: close <= máximo de 24 h × (1 − g × R), g ∈ {0,5; 1}. Sin filtro F.
- Salidas: S4_f TP = entrada × (1 + max(f × R, 0,6 %)), f ∈ {0,25; 0,5; 0,75}, con el R
  vigente; S5 trailing que se arma en +max(0,5 R; 0,6 %) y vende al retroceder 0,2 R.
- Slots: A (independientes, 1 entrada por día) o B escalonado cada 1 R. 2 × 2 × 4 = 16.
- Estructura de v0.1: cierre mensual el día 3 (atrapados → holding, el DCA repone en orden).
  DCA 100 fijo, 400 fijo y aleatorio 100-900 (5 semillas). Costos 0,1 % + 0,05 % por lado.
- Selección 2021-01-03 → 2024-01-03; validación 2024-01-03 → 2026-10-03.
- Se elige por INGRESO: neto mensual medio (ganancia realizada − costo de atrapadas, en % del
  capital del bot) con DCA 400. Las 5 mejores van a validación.
- Viable, en validación, si cumple todo:
  (1) neto mensual medio ≥ 1 % con DCA 400 (la meta del usuario, ≈ 12 % al año);
  (2) BTC final ≥ DCA puro (puntaje A ≥ 0, promedio de escenarios);
  (3) neto anual positivo en la mayoría de los años;
  (4) neto mensual medio mayor que el percentil 90 del placebo (entradas al azar con la misma
      probabilidad, mismas salidas, 50 semillas, DCA 400);
  (5) ningún ciclo explica más del 50 % del neto total.

`--parcial` (22/10/2026): reposición parcial de los slots (ver `Config.reposicion_parcial`);
salidas en `archivos/btc_rangos/backtest_v04_parcial/`.

Uso: python3 scripts/btc_rangos_v04.py [--procesos 8] [--parcial]
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "archivos" / "btc_rangos" / "backtest_v04"

_spec = importlib.util.spec_from_file_location("bt", REPO / "scripts" / "btc_rangos_backtest.py")
bt = importlib.util.module_from_spec(_spec)
sys.modules["bt"] = bt
_spec.loader.exec_module(bt)
rr, mm = bt.rr, bt.mm

SLOTS = [("A", 0.0), ("B", 1.0)]
ENTRADAS = ["E4_0.5", "E4_1"]
SALIDAS = ["S4_0.25", "S4_0.5", "S4_0.75", "S5"]
N_PLACEBO = 50


def parcial() -> bool:
    """Se pasa por variable de entorno para que llegue a los procesos del Pool."""
    return os.environ.get("TROBOT_REPOSICION_PARCIAL") == "1"

_V = _IND = None
_SEN: dict = {}


def diario_ohlc(v1h: pd.DataFrame) -> pd.DataFrame:
    lab_spec = importlib.util.spec_from_file_location("btc_lab", REPO / "scripts" / "btc_lab_temporalidades.py")
    lab = importlib.util.module_from_spec(lab_spec)
    lab_spec.loader.exec_module(lab)
    d = lab._leer_velas("archivos/btc/binance_1d", "BTCUSDT-1d-*.zip")
    g = v1h.resample("1D")
    extra = g.agg({"open": "first", "high": "max", "low": "min", "close": "last"})[g["close"].count() == 24]
    return pd.concat([d, extra[extra.index > d.index[-1]]]).sort_index()


def _init():
    global _V, _IND
    _V = pd.read_csv(REPO / "archivos/btc_rangos/velas_1h.csv", parse_dates=["date"], index_col="date")
    _IND = rr.indicadores(_V)
    _IND["rango"] = rr.rango_diario(diario_ohlc(_V), _V.index)
    for e in ENTRADAS:
        _SEN[e] = rr.senal_entrada(_V, _IND, e, False)


def correr(args) -> dict:
    (slots, esc_r), entrada, salida, escenario, periodo, placebo = args
    cfg = mm.Config(slots=slots, escalon_r=esc_r, salida=salida, ganancia="G4", reposicion_parcial=parcial())
    sen = _SEN[entrada]
    if placebo is not None:
        en = (sen.index >= periodo[0]) & (sen.index < periodo[1])
        p = sen[en].mean()
        rng = np.random.default_rng(10_000 + placebo)
        sen = pd.Series(rng.random(len(_V)) < p, index=_V.index)
    r = mm.simular(_V, sen, _IND["atr"], cfg, bt.dca_fn(escenario), *periodo, rango=_IND["rango"])
    out = {"slots": slots, "escalon": esc_r, "entrada": entrada, "filtro": False, "salida": salida,
           "ganancia": "G4", "escenario": escenario, "intrabar": True, "placebo": placebo,
           "periodo": "sel" if periodo == bt.SEL else "val"}
    out.update(bt.metricas(r))
    me = pd.DataFrame(r.motor.meses)
    out["realizado_mensual_pct"] = float((me["realizado"] / me["capital_inicio"]).mean() * 100)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procesos", type=int, default=8)
    ap.add_argument("--parcial", action="store_true", help="reposición parcial de los slots")
    args = ap.parse_args()
    global OUT
    if args.parcial:
        os.environ["TROBOT_REPOSICION_PARCIAL"] = "1"
        OUT = OUT.parent / "backtest_v04_parcial"
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    combos = list(itertools.product(SLOTS, ENTRADAS, SALIDAS))
    with Pool(args.procesos, initializer=_init) as pool:
        tareas = [(s, e, sa, esc, per, None) for (s, e, sa), esc, per in
                  itertools.product(combos, bt.ESCENARIOS, (bt.SEL, bt.VAL))]
        df = pd.DataFrame(pool.map(correr, tareas, chunksize=2))
        sel = bt.resumen_por_combo(df[df.periodo == "sel"].drop(columns="periodo"))
        val = bt.resumen_por_combo(df[df.periodo == "val"].drop(columns="periodo"))
        for nombre, rs, d in (("sel", sel, df[df.periodo == "sel"]), ("val", val, df[df.periodo == "val"])):
            d = d.assign(combo=d.apply(bt.clave, axis=1))
            rs["realizado_mensual_pct_base"] = d[d.escenario == "fijo400"].set_index("combo")["realizado_mensual_pct"]
            rs["neto_mensual_dca100"] = d[d.escenario == "fijo100"].set_index("combo")["neto_mensual_medio_pct"]
        df.to_csv(OUT / "corridas.csv", index=False)
        sel = sel.sort_values("neto_mensual_medio_pct_base", ascending=False)
        sel.to_csv(OUT / "seleccion_ranking.csv")
        val.to_csv(OUT / "validacion_16.csv")
        finalistas = sel.head(5).index.tolist()
        mapa = {bt.clave({"slots": s, "escalon": x, "entrada": e, "filtro": False, "salida": sa}): ((s, x), e, sa)
                for (s, x), e, sa in combos}
        tareas = [(*mapa[c], "fijo400", bt.VAL, k) for c in finalistas for k in range(N_PLACEBO)]
        plac = pd.DataFrame(pool.map(correr, tareas, chunksize=4))
    plac["combo"] = plac.apply(bt.clave, axis=1)
    plac.to_csv(OUT / "placebo.csv", index=False)

    cols = ["neto_mensual_medio_pct_base", "realizado_mensual_pct_base", "neto_mensual_dca100", "puntaje_a",
            "puntaje_b", "meses_en_meta_pct_base", "peor_mes_pct_base", "ciclos_mes_base", "atrapadas_base",
            "comisiones_base", "neto_anual_base"]
    pd.set_option("display.width", 250)
    print("SELECCIÓN 2021-23 (16 combinaciones, ordenadas por neto mensual con DCA 400):")
    print(sel[cols].round(2).to_string())
    filas = []
    for c in finalistas:
        v = val.loc[c]
        p90 = plac.loc[plac.combo == c, "neto_mensual_medio_pct"].quantile(0.9)
        anual = pd.Series(json.loads(v["neto_anual_base"]))
        pasa = {"c1_meta_1pct": v["neto_mensual_medio_pct_base"] >= 1.0,
                "c2_btc_vs_dca": v["puntaje_a"] >= 0,
                "c3_mayoria_anios": (anual > 0).sum() > len(anual) / 2,
                "c4_placebo": v["neto_mensual_medio_pct_base"] > p90,
                "c5_concentracion": pd.notna(v["concentracion_base"]) and v["concentracion_base"] <= 0.5}
        filas.append({"combo": c, "sel_neto_mes": sel.loc[c, "neto_mensual_medio_pct_base"],
                      "val_neto_mes": v["neto_mensual_medio_pct_base"], "val_realizado_mes": v["realizado_mensual_pct_base"],
                      "val_neto_mes_dca100": v["neto_mensual_dca100"], "placebo_p90": p90,
                      "val_puntaje_a": v["puntaje_a"], "val_puntaje_b": v["puntaje_b"],
                      "val_neto_anual": v["neto_anual_base"], "meses_meta": v["meses_en_meta_pct_base"],
                      "peor_mes": v["peor_mes_pct_base"], "ops_mes": v["ciclos_mes_base"], **pasa,
                      "viable": all(pasa.values())})
    fin = pd.DataFrame(filas)
    fin.to_csv(OUT / "finalistas.csv", index=False)
    print("\nVALIDACIÓN 2024-26, finalistas:")
    print(fin.round(2).T.to_string())
    print("\nVALIDACIÓN, las 16 (informativo):")
    print(val[cols].sort_values("neto_mensual_medio_pct_base", ascending=False).round(2).to_string())
    print(f"Total {time.time() - t0:.0f} s. Resultados en {OUT}")


if __name__ == "__main__":
    main()
