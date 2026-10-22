"""Backtest del bot BTC por rangos (Fase 2, 22/10/2026) — la puerta antes de construir el live.

Especificación: `docs/reglas_bot_btc.md` v0.1. Motor: `pkg/btc_rangos_motor.py` (el mismo que
usaría el live); señales: `pkg/btc_rangos_reglas.py`. Datos: `archivos/btc_rangos/velas_1h.csv`
(`scripts/btc_rangos_datos.py`), Binance spot 1 h con paridad contra BingX de ~0 bps de mediana.

PROTOCOLO, fijado antes de correr:
- Combinaciones de señal: slots {A, B 1,5 %, B 2 %, B 3 %} × entradas {E1 2 %, E1 3 %, E2,
  E3 7 d, E3 14 d} × filtro F {no, sí} × salidas {S1 1,5 %, S1 2 %, S1 3 %, S2, S3} = 200,
  todas con G4 (la base). G no cambia qué trades se toman: se evalúa sólo en las finalistas.
- Escenarios de DCA: 100 fijo, 400 fijo, aleatorio uniforme 100-900 (múltiplos de 10) con
  5 semillas fijas. El bot y el benchmark reciben el mismo DCA en el mismo momento.
- Métrica: BTC extra contra el benchmark A en % =
  (patrimonio en BTC del bot − DCA puro con los 200 USDT guardados) / DCA puro.
  Patrimonio = holding + BTC en slots + (USDT del bot + USDT reservado para holding) / precio.
  Puntaje = promedio de {100 fijo, 400 fijo, promedio de las 5 aleatorias}.
  También se informa contra el benchmark B (los 200 comprados en BTC el primer día).
- Selección: ciclos 2021-01-03 → 2024-01-03. Validación: 2024-01-03 → 2026-10-03, que no
  participa en elegir. Sólo las 5 mejores de la selección se miran en validación.
- Placebo: entradas al azar con la misma probabilidad por vela que la señal real en el período,
  mismos slots y salidas, 50 semillas, DCA 400 fijo. La señal "le gana al placebo" si su BTC
  extra supera el percentil 90 del placebo.
- Viable, en validación, si: (1) BTC extra vs A > 0 en el puntaje; (2) neto anual con
  atrapadas positivo en la mayoría de los años; (3) le gana al placebo; (4) ningún ciclo
  mensual explica más del 50 % del neto total.
- Sensibilidad: las finalistas también con `intrabar=False` (todo sólo con closes).

v0.2 (`--regimen`, 22/10/2026, fijado antes de correr): todo lo anterior idéntico, más el
régimen R de `pkg/btc_rangos_reglas.py` aplicado a todas las combinaciones (no se abren slots
si el último cierre diario está bajo la media de 100 días). El placebo también respeta el
régimen: entradas al azar sólo en las velas permitidas, con la misma probabilidad que la
señal real tiene en ellas. Cierre diario: spot diario de Binance (`archivos/btc/binance_1d/`,
desde 2017, así la media está calentada en 2021) y, después de su último día, días completos
armados con las velas 1 h. Salidas en `archivos/btc_rangos/backtest_v02/`.

Uso: python3 scripts/btc_rangos_backtest.py [--procesos 8]
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "archivos" / "btc_rangos" / "backtest"


def _cargar(nombre: str):
    spec = importlib.util.spec_from_file_location(nombre, REPO / "pkg" / f"{nombre}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


rr = _cargar("btc_rangos_reglas")
mm = _cargar("btc_rangos_motor")

SEL = (pd.Timestamp("2021-01-03"), pd.Timestamp("2024-01-03"))
VAL = (pd.Timestamp("2024-01-03"), pd.Timestamp("2026-10-03"))
SLOTS = [("A", 0.0), ("B", 0.015), ("B", 0.02), ("B", 0.03)]
ENTRADAS = list(rr.ENTRADAS)
FILTROS = [False, True]
SALIDAS = list(mm.SALIDAS)
SEMILLAS_DCA = [1, 2, 3, 4, 5]
N_PLACEBO = 50
META_MES = 0.01

_V = _IND = _REG = None
_SEN: dict = {}


def cierre_diario(v1h: pd.DataFrame) -> pd.Series:
    """Close diario spot: Binance 1d y, después de su último día, días completos de las 1 h."""
    lab_spec = importlib.util.spec_from_file_location("btc_lab", REPO / "scripts" / "btc_lab_temporalidades.py")
    lab = importlib.util.module_from_spec(lab_spec)
    lab_spec.loader.exec_module(lab)
    d = lab._leer_velas("archivos/btc/binance_1d", "BTCUSDT-1d-*.zip")["close"]
    g = v1h["close"].resample("1D")
    extra = g.last()[g.count() == 24]
    return pd.concat([d, extra[extra.index > d.index[-1]]]).sort_index()


def _init(regimen: bool = False):
    global _V, _IND, _REG
    _V = pd.read_csv(REPO / "archivos/btc_rangos/velas_1h.csv", parse_dates=["date"], index_col="date")
    _IND = rr.indicadores(_V)
    _REG = rr.regimen_compra(cierre_diario(_V), _V.index) if regimen else None
    for e, f in itertools.product(ENTRADAS, FILTROS):
        s = rr.senal_entrada(_V, _IND, e, f)
        _SEN[(e, f)] = s & _REG if regimen else s


def dca_fn(escenario: str):
    if escenario.startswith("fijo"):
        monto = float(escenario[4:])
        return lambda k: monto
    rng = np.random.default_rng(int(escenario.split("_")[1]))
    seq = rng.integers(10, 91, size=200) * 10.0
    return lambda k: float(seq[k - 1])


ESCENARIOS = ["fijo100", "fijo400"] + [f"aleat_{s}" for s in SEMILLAS_DCA]


def metricas(r) -> dict:
    m = r.motor
    pat = r.patrimonio_btc
    meses = pd.DataFrame(m.meses)
    meses["neto"] = meses["realizado"] - meses["atrapadas_costo"]
    meses["neto_pct"] = meses["neto"] / meses["capital_inicio"]
    meses["anio"] = meses["ciclo"].str[:4]
    anual = meses.groupby("anio")["neto"].sum()
    ops = pd.DataFrame(m.operaciones)
    total_neto = meses["neto"].sum()
    conc = meses["neto"].max() / total_neto if total_neto > 0 else np.nan
    n_meses = len(meses)
    entradas = int(meses["entradas"].sum())
    return {
        "extra_a_pct": (pat - r.bench_a_btc) / r.bench_a_btc * 100,
        "extra_b_pct": (pat - r.bench_b_btc) / r.bench_b_btc * 100,
        "extra_a_btc": pat - r.bench_a_btc,
        "patrimonio_btc": pat,
        "bench_a_btc": r.bench_a_btc,
        "realizado": meses["realizado"].sum(),
        "atrapadas_costo": meses["atrapadas_costo"].sum(),
        "neto_total": total_neto,
        "neto_mensual_medio_pct": meses["neto_pct"].mean() * 100,
        "meses_en_meta_pct": (meses["neto_pct"] >= META_MES).mean() * 100,
        "peor_mes_pct": meses["neto_pct"].min() * 100,
        "anios_pos": int((anual > 0).sum()),
        "anios": len(anual),
        "concentracion": conc,
        "entradas": entradas,
        "cierres": len(ops),
        "winrate_pct": len(ops) / entradas * 100 if entradas else np.nan,
        "ciclos_mes": len(ops) / n_meses if n_meses else 0.0,
        "atrapadas": int(meses["n_atrapadas"].sum()),
        "comisiones": m.comisiones,
        "neto_anual": json.dumps({k: round(v, 2) for k, v in anual.items()}),
    }


def correr(args) -> dict:
    (slots, esc), entrada, filtro, salida, ganancia, escenario, periodo, intrabar, placebo = args
    cfg = mm.Config(slots=slots, escalon=esc or 0.02, salida=salida, ganancia=ganancia, intrabar=intrabar)
    sen = _SEN[(entrada, filtro)]
    if placebo is not None:
        en_periodo = (sen.index >= periodo[0]) & (sen.index < periodo[1])
        permitido = _REG.to_numpy() if _REG is not None else np.ones(len(_V), dtype=bool)
        p = sen[en_periodo & permitido].mean()
        rng = np.random.default_rng(10_000 + placebo)
        sen = pd.Series((rng.random(len(_V)) < p) & permitido, index=_V.index)
    r = mm.simular(_V, sen, _IND["atr"], cfg, dca_fn(escenario), *periodo)
    out = {"slots": slots, "escalon": esc, "entrada": entrada, "filtro": filtro, "salida": salida,
           "ganancia": ganancia, "escenario": escenario, "intrabar": intrabar, "placebo": placebo}
    out.update(metricas(r))
    return out


def clave(row) -> str:
    esc = f"{row['escalon']:.3f}" if row["slots"] == "B" else "-"
    return f"{row['slots']}{esc}|{row['entrada']}|F{int(row['filtro'])}|{row['salida']}"


def puntaje(df: pd.DataFrame, col: str = "extra_a_pct") -> pd.Series:
    """Promedio de {fijo100, fijo400, media de las aleatorias} por combinación."""
    df = df.assign(grupo=np.where(df["escenario"].str.startswith("aleat"), "aleat", df["escenario"]))
    g = df.groupby(["combo", "grupo"])[col].mean().unstack()
    return g.mean(axis=1)


def resumen_por_combo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["combo"] = df.apply(clave, axis=1)
    base = df[df["escenario"] == "fijo400"].set_index("combo")
    out = pd.DataFrame({"puntaje_a": puntaje(df), "puntaje_b": puntaje(df, "extra_b_pct")})
    cols = ["realizado", "atrapadas_costo", "neto_total", "neto_mensual_medio_pct", "meses_en_meta_pct",
            "peor_mes_pct", "anios_pos", "anios", "concentracion", "entradas", "cierres", "ciclos_mes",
            "atrapadas", "comisiones", "neto_anual", "extra_a_pct"]
    out = out.join(base[cols].add_suffix("_dca400"))
    return out.sort_values("puntaje_a", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procesos", type=int, default=8)
    ap.add_argument("--regimen", action="store_true", help="v0.2: sólo abrir slots en ventana COMPRA")
    args = ap.parse_args()
    global OUT
    if args.regimen:
        OUT = OUT.parent / "backtest_v02"
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with Pool(args.procesos, initializer=_init, initargs=(args.regimen,)) as pool:
        # 1) Selección
        tareas = [(s, e, f, sa, "G4", esc, SEL, True, None)
                  for s, e, f, sa, esc in itertools.product(SLOTS, ENTRADAS, FILTROS, SALIDAS, ESCENARIOS)]
        print(f"Selección: {len(tareas)} corridas…", flush=True)
        sel = pd.DataFrame(pool.map(correr, tareas, chunksize=4))
        sel.to_csv(OUT / "seleccion_corridas.csv", index=False)
        rs = resumen_por_combo(sel)
        rs.to_csv(OUT / "seleccion_ranking.csv")
        print(f"  {time.time() - t0:.0f} s. Combinaciones con puntaje A > 0: {(rs['puntaje_a'] > 0).sum()} de {len(rs)}")
        print(rs.head(10)[["puntaje_a", "puntaje_b", "neto_mensual_medio_pct_dca400", "atrapadas_dca400",
                           "cierres_dca400", "anios_pos_dca400"]].round(3).to_string())

        finalistas = rs.head(5).index.tolist()
        combos = {}
        for _, row in sel.iterrows():
            combos.setdefault(clave(row), ((row["slots"], row["escalon"]), row["entrada"], row["filtro"], row["salida"]))

        # 2) Validación de las finalistas (y las mismas en selección con todas las G y sin intrabar)
        tareas = []
        for c in finalistas:
            s, e, f, sa = combos[c]
            for esc in ESCENARIOS:
                tareas.append((s, e, f, sa, "G4", esc, VAL, True, None))
                tareas.append((s, e, f, sa, "G4", esc, VAL, False, None))
                for g in ("G1", "G2", "G3"):
                    tareas.append((s, e, f, sa, g, esc, VAL, True, None))
            for k in range(N_PLACEBO):
                tareas.append((s, e, f, sa, "G4", "fijo400", VAL, True, k))
                tareas.append((s, e, f, sa, "G4", "fijo400", SEL, True, k))
        print(f"Validación + placebo + sensibilidad: {len(tareas)} corridas…", flush=True)
        val = pd.DataFrame(pool.map(correr, tareas, chunksize=4))
    val["combo"] = val.apply(clave, axis=1)
    val.to_csv(OUT / "validacion_corridas.csv", index=False)

    filas = []
    for c in finalistas:
        v = val[(val["combo"] == c)]
        real = v[v["placebo"].isna() & v["intrabar"] & (v["ganancia"] == "G4")]
        rv = resumen_por_combo(real.drop(columns="combo")).iloc[0]
        plac_val = v[v["placebo"].notna() & (v["escenario"] == "fijo400")]
        p90 = plac_val["extra_a_pct"].quantile(0.9) if len(plac_val) else np.nan
        real400 = real[real["escenario"] == "fijo400"].iloc[0]
        sens = resumen_por_combo(v[v["placebo"].isna() & ~v["intrabar"]].drop(columns="combo")).iloc[0]["puntaje_a"]
        gs = {}
        for g in ("G1", "G2", "G3"):
            vg = v[v["placebo"].isna() & v["intrabar"] & (v["ganancia"] == g)]
            gs[f"puntaje_{g}"] = resumen_por_combo(vg.drop(columns="combo")).iloc[0]["puntaje_a"]
        pasa = {
            "c1_extra_pos": rv["puntaje_a"] > 0,
            "c2_mayoria_anios": rv["anios_pos_dca400"] > rv["anios_dca400"] / 2,
            "c3_gana_placebo": real400["extra_a_pct"] > p90,
            "c4_concentracion": (rv["concentracion_dca400"] <= 0.5) if pd.notna(rv["concentracion_dca400"]) else False,
        }
        filas.append({"combo": c, "puntaje_sel": rs.loc[c, "puntaje_a"], "puntaje_val_a": rv["puntaje_a"],
                      "puntaje_val_b": rv["puntaje_b"], "val_extra_a_dca400": real400["extra_a_pct"],
                      "placebo_p50": plac_val["extra_a_pct"].median(), "placebo_p90": p90,
                      "puntaje_val_sin_intrabar": sens, **gs, "puntaje_G4": rv["puntaje_a"],
                      "neto_anual_dca400": rv["neto_anual_dca400"], "meses_en_meta_pct": rv["meses_en_meta_pct_dca400"],
                      "peor_mes_pct": rv["peor_mes_pct_dca400"], "ciclos_mes": rv["ciclos_mes_dca400"],
                      "atrapadas": rv["atrapadas_dca400"], "comisiones": rv["comisiones_dca400"],
                      "concentracion": rv["concentracion_dca400"], **pasa, "viable": all(pasa.values())})
    fin = pd.DataFrame(filas)
    fin.to_csv(OUT / "finalistas.csv", index=False)
    pd.set_option("display.width", 250)
    print(fin.round(3).T.to_string())
    print(f"Total {time.time() - t0:.0f} s. Resultados en {OUT}")


if __name__ == "__main__":
    main()
