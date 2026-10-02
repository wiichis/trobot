#!/usr/bin/env python3
"""Optimizador de parámetros sobre el PARITY (el modelo de ejecución del live) — 02/10/2026.

Por qué existe: el sweep de `pkg/backtesting.py` (clase `Backtester`) sólo opera en el
último `train_ratio` de los datos y elige ahí, así que su "test" es el período de
selección; además genera otras señales que el live (~1,6× menos posiciones por día). Los
45 candidatos del 02/10 salieron positivos en su "test" (45/45) y sólo 14/45 en el parity
sobre el mismo período. Ver CLAUDE.md, "Candidatos 02/10".

Diseño (fijado ANTES de mirar resultados):

    ... hold2 (50d al 22/02) ... hold1 (60d al 23/04) ... [ ENTRENAMIENTO 90d ][ TEST 60d ] fin

- Cada combinación se simula con `run_live_parity_portfolio` (ejecución del live, filtro
  de régimen, fills PostOnly, fees maker, peso 0,20), el par solo.
- **Se elige SÓLO con el entrenamiento.** Requisitos: ≥ MIN_TRADES_TRAIN posiciones y al
  menos 2 de los 3 tercios del entrenamiento positivos; entre los que cumplen, gana el de
  mayor PnL. El candidato final es el #1 del entrenamiento — no se elige mirando el test.
- El test (últimos 60d) y las dos ventanas de falsación **no participan de la elección**:
  sólo miden al #1 (y, como información, al top-5).
- La combinación vigente del par (si existe en `--best`) entra como trial 0, así el
  resultado se compara contra "no tocar nada" con la misma vara.

Veredicto del #1: PASA si es positivo en test, en hold1 y en hold2, con ≥ MIN_TRADES_TEST
posiciones en test, y —si hay paramset vigente— le gana en test y en las dos falsaciones.

Uso:
    python3 scripts/optimizar_parity.py --symbol BCH-USDT --data archivos/velas_limpias_5m.csv \\
        --best pkg/best_prod.json --trials 300 --seed 101 --out archivos/optimizacion
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pkg"))

TRAIN_DIAS = 90
TEST_DIAS = 60
WARMUP_DIAS = 10
HOLDS = {"hold1": ("2026-04-23", 60), "hold2": ("2026-02-22", 50)}
# En POSICIONES (no en tramos de TP, que es lo que cuenta el `min_trades` del sweep: ~2 por
# posición). Calibrado a la frecuencia de la estrategia antes de la primera corrida real: el
# paramset vigente de BCH hace 10 posiciones en 90 días y 12 en 60.
MIN_TRADES_TRAIN = 10
MIN_TRADES_TEST = 6
CAPITAL = 1000.0
PESO = 0.20
TOP_K = 5


# ---------------------------------------------------------------- piezas puras (testeadas)


def ventanas(fin: pd.Timestamp) -> Dict[str, tuple]:
    """{nombre: (inicio, fin)} de cada ventana. El entrenamiento termina donde empieza el test."""
    test_ini = fin - pd.Timedelta(days=TEST_DIAS)
    train_ini = test_ini - pd.Timedelta(days=TRAIN_DIAS)
    out = {"train": (train_ini, test_ini), "test": (test_ini, fin)}
    for nombre, (f, dias) in HOLDS.items():
        f = pd.Timestamp(f, tz="UTC")
        out[nombre] = (f - pd.Timedelta(days=dias), f)
    return out


def tercios(inicio: pd.Timestamp, fin: pd.Timestamp) -> List[tuple]:
    paso = (fin - inicio) / 3
    return [(inicio + paso * k, inicio + paso * (k + 1)) for k in range(3)]


def puntaje_train(res: Dict) -> Optional[float]:
    """PnL del entrenamiento si la combinación es elegible; None si no lo es."""
    if res.get("n", 0) < MIN_TRADES_TRAIN:
        return None
    if sum(1 for x in res.get("tercios", []) if x > 0) < 2:
        return None
    return float(res["pnl"])


def ranking(trials: Sequence[Dict]) -> List[Dict]:
    """Trials elegibles ordenados por puntaje de entrenamiento (desempate: más trades)."""
    eleg = [t for t in trials if puntaje_train(t["train"]) is not None]
    return sorted(eleg, key=lambda t: (puntaje_train(t["train"]), t["train"]["n"]), reverse=True)


def _pos(res: Dict) -> bool:
    """Positivo. Una ventana SIN DATOS (par de historia corta) no cuenta como aprobada:
    sin falsación no hay evidencia fuera de muestra."""
    return res.get("pnl") is not None and res["pnl"] > 0


def veredicto(cand: Dict, vigente: Optional[Dict]) -> tuple:
    checks = {
        "test_positivo": _pos(cand["test"]),
        "hold1_positivo": _pos(cand["hold1"]),
        "hold2_positivo": _pos(cand["hold2"]),
        f"test_trades>={MIN_TRADES_TEST}": cand["test"]["n"] >= MIN_TRADES_TEST,
    }
    if vigente is not None:
        for w in ("test", "hold1", "hold2"):
            a, b = cand[w].get("pnl"), vigente[w].get("pnl")
            checks[f"le_gana_al_vigente_en_{w}"] = a is not None and b is not None and a > b
    return all(checks.values()), checks


def muestrear(espacio: Dict[str, list], n: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    claves = sorted(espacio)
    vistos, out = set(), []
    intentos = 0
    while len(out) < n and intentos < n * 20:
        intentos += 1
        p = {k: rng.choice(espacio[k]) for k in claves}
        if p.get("ema_fast", 0) >= p.get("ema_slow", 10 ** 9):
            continue
        clave = json.dumps(p, sort_keys=True)
        if clave in vistos:
            continue
        vistos.add(clave)
        out.append(p)
    return out


# ---------------------------------------------------------------- simulación


def _simular(args):
    """Corre el parity de UN par con UN set de params en las ventanas pedidas."""
    symbol, params, pedidos = args
    from pkg import backtesting as bt
    from pkg.bench import habilitar_en_indicadores
    habilitar_en_indicadores([symbol])   # sin esto un par fuera de best_prod da 0 trades
    p = dict(params)
    p["peso"] = PESO
    out = {}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump([{"symbol": symbol, "params": p}], fh)
        best_path = fh.name
    try:
        for nombre, (ini, fin, archivo) in pedidos.items():
            dias = int(math.ceil((fin - ini).total_seconds() / 86400)) + WARMUP_DIAS
            r = bt.run_live_parity_portfolio([symbol], archivo, CAPITAL, None, best_path=best_path,
                                             lookback_days=dias, return_trades=True)
            pos = {}
            for t in r["trades_list"]:
                et = pd.Timestamp(t.entry_time)
                et = et.tz_localize("UTC") if et.tzinfo is None else et
                if et < ini:
                    continue   # trades del precalentamiento
                q = pos.setdefault(t.position_id, {"pnl": 0.0, "et": et})
                q["pnl"] += t.pnl()
            pnls = [q["pnl"] for q in pos.values()]
            res = {"pnl": round(sum(pnls), 3), "n": len(pnls)}
            if nombre == "train":
                res["tercios"] = [round(sum(q["pnl"] for q in pos.values() if a <= q["et"] < b), 3)
                                  for a, b in tercios(ini, fin)]
            out[nombre] = res
    finally:
        Path(best_path).unlink(missing_ok=True)
    return out


def _cortar(df_sym: pd.DataFrame, hasta: pd.Timestamp, carpeta: Path, nombre: str) -> str:
    p = carpeta / f"{nombre}.csv"
    d = df_sym[df_sym["dt"] <= hasta].drop(columns="dt")
    d.to_csv(p, index=False)
    return str(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--data", required=True, help="CSV de velas 5m que contenga el par")
    ap.add_argument("--best", default=None, help="best_prod/bench json con el paramset vigente (trial 0)")
    ap.add_argument("--space", default=str(REPO / "archivos/backtesting/simple_sweep.json"))
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(REPO / "archivos/optimizacion"))
    args = ap.parse_args()

    sym = args.symbol.upper()
    out_dir = Path(args.out) / f"{sym.replace('-USDT', '')}_s{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    df = df[df["symbol"] == sym].copy()
    if df.empty:
        raise SystemExit(f"{sym} no está en {args.data}")
    df["dt"] = pd.to_datetime(df["date"], utc=True, format="mixed")
    fin = df["dt"].max()
    W = ventanas(fin)

    inicio_datos = df["dt"].min()
    con_datos = {n for n, (a, b) in W.items() if inicio_datos <= a - pd.Timedelta(days=WARMUP_DIAS)}
    if "train" not in con_datos:
        raise SystemExit(f"{sym}: la historia (desde {inicio_datos:%d/%m}) no cubre el entrenamiento")
    sin_datos = [n for n in ("test", "hold1", "hold2") if n not in con_datos]
    if sin_datos:
        print(f"⚠️ {sym}: historia desde {inicio_datos:%d/%m}; sin datos para {sin_datos} (cuentan como no aprobadas)")
    archivos = {n: _cortar(df, b, out_dir, f"velas_{n}") for n, (a, b) in W.items() if n in con_datos}
    pedidos_train = {"train": (W["train"][0], W["train"][1], archivos["train"])}
    pedidos_eval = {n: (W[n][0], W[n][1], archivos[n]) for n in ("test", "hold1", "hold2") if n in con_datos}

    espacio = json.loads(Path(args.space).read_text())
    combos = muestrear(espacio, args.trials, args.seed)
    vigente_params = None
    if args.best:
        for e in json.loads(Path(args.best).read_text()):
            if str(e.get("symbol", "")).upper() == sym:
                vigente_params = dict(e.get("params", {}))
    todos = ([{"id": "vigente", "params": vigente_params}] if vigente_params else []) + \
            [{"id": f"t{i:04d}", "params": p} for i, p in enumerate(combos)]

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res_train = list(ex.map(_simular, [(sym, t["params"], pedidos_train) for t in todos], chunksize=4))
    for t, r in zip(todos, res_train):
        t["train"] = r["train"]
    rk = ranking([t for t in todos if t["id"] != "vigente"])
    a_evaluar = rk[:TOP_K]
    vigente = next((t for t in todos if t["id"] == "vigente"), None)
    if vigente:
        a_evaluar = a_evaluar + [vigente]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res_eval = list(ex.map(_simular, [(sym, t["params"], pedidos_eval) for t in a_evaluar]))
    for t, r in zip(a_evaluar, res_eval):
        t.update(r)
        for n in sin_datos:
            t[n] = {"pnl": None, "n": 0, "sin_datos": True}

    elegido = rk[0] if rk else None
    informe = {"symbol": sym, "seed": args.seed, "trials": len(combos), "fin_datos": str(fin),
               "ventanas": {n: [str(a), str(b)] for n, (a, b) in W.items()},
               "elegibles": len(rk), "segundos": round(time.time() - t0, 1),
               "vigente": {k: vigente[k] for k in ("train", "test", "hold1", "hold2")} if vigente else None,
               "top": [{k: t[k] for k in ("id", "train", "test", "hold1", "hold2", "params")} for t in rk[:TOP_K]]}
    if elegido:
        ok, checks = veredicto(elegido, vigente)
        informe["elegido"] = {"id": elegido["id"], "pasa": ok, "checks": checks, "params": elegido["params"]}
    (out_dir / "informe.json").write_text(json.dumps(informe, indent=1, default=str), encoding="utf-8")
    for f in out_dir.glob("velas_*.csv"):
        f.unlink()

    print(f"{sym} s{args.seed}: {len(combos)} trials, {len(rk)} elegibles, {informe['segundos']} s")
    fmt = lambda r: "    s/d" if r.get("pnl") is None else f"{r['pnl']:+7.2f}"
    fila = lambda t: (f"train {t['train']['pnl']:+7.2f} (n={t['train']['n']:3d}, tercios {t['train']['tercios']}) | "
                      f"test {fmt(t['test'])} (n={t['test']['n']}) | hold1 {fmt(t['hold1'])} | "
                      f"hold2 {fmt(t['hold2'])}")
    if vigente:
        print("  vigente :", fila(vigente))
    for i, t in enumerate(rk[:TOP_K], 1):
        print(f"  #{i} {t['id']}:", fila(t))
    if elegido:
        print("  ELEGIDO (#1 del entrenamiento):", "PASA" if informe["elegido"]["pasa"] else "NO PASA",
              {k: v for k, v in informe["elegido"]["checks"].items() if not v})


if __name__ == "__main__":
    main()
