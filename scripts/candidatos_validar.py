#!/usr/bin/env python3
"""Valida los paramsets de los CANDIDATOS con el protocolo completo (01/10/2026).

Para cada `archivos/candidatos/sweeps/best_<PAR>_s<SEMILLA>.json` corre el parity (con la
config viva: filtro de régimen, ejecución del live) en:
- 4 ventanas recientes (30/60/90/120d hasta el fin de los datos),
- 2 ventanas de FALSACIÓN que el sweep no vio: 60d al 23/04 y 50d al 22/02,
- 6 meses aislados de 30 días, para ver si el resultado depende de un solo mes.

Un candidato PASA si: positivo en las 4 recientes, positivo en las 2 de falsación, al
menos 15 trades en 120d, positivo en al menos 4 de 6 meses y ningún mes explica más de la
mitad de la suma de los meses positivos. A un par nuevo no hay paramset "actual" contra
el cual compararlo, así que la vara es absoluta.

El par se simula solo, con peso 0,20 (el tamaño por trade del live).
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pkg"))

DIR = REPO / "archivos" / "candidatos"
SWEEPS = DIR / "sweeps"
VAL = DIR / "validacion"
CAPITAL = 1000.0


def _ventanas(fin):
    w = {f"w{d}": (None, d) for d in (30, 60, 90, 120)}
    w["hold1"] = (pd.Timestamp("2026-04-23", tz="UTC"), 60)
    w["hold2"] = (pd.Timestamp("2026-02-22", tz="UTC"), 50)
    for k in range(1, 7):
        w[f"mes{k}"] = (fin - pd.Timedelta(days=30 * (k - 1)), 30)
    return w


def _datos_truncados(fin):
    VAL.mkdir(parents=True, exist_ok=True)
    full = pd.read_csv(DIR / "velas_candidatos.csv")
    full["dt"] = pd.to_datetime(full["date"], utc=True)
    rutas = {}
    for nombre, (corte, _) in _ventanas(fin).items():
        p = VAL / f"velas_{nombre}.csv"
        if corte is None:
            rutas[nombre] = str(DIR / "velas_candidatos.csv")
            continue
        if not p.exists():
            full[full["dt"] <= corte].drop(columns="dt").to_csv(p, index=False)
        rutas[nombre] = str(p)
    return rutas


def _correr(args):
    best_path, data, dias, nombre = args
    from pkg import backtesting as bt
    sym = json.loads(Path(best_path).read_text())[0]["symbol"]
    r = bt.run_live_parity_portfolio([sym], data, CAPITAL, None, best_path=best_path,
                                     lookback_days=dias, return_trades=True)
    pos = {}
    for t in r["trades_list"]:
        p = pos.setdefault(t.position_id, [0.0, 0.0])
        p[0] += t.pnl()
        p[1] += t.entry_price * t.qty
    n = len(pos)
    pnl = sum(v[0] for v in pos.values())
    pct = (sum(v[0] / v[1] for v in pos.values() if v[1]) / n) if n else 0.0
    return nombre, {"pnl": round(pnl, 3), "n": n, "por_trade_pct": round(pct, 5)}


def veredicto(res):
    rec = [res[f"w{d}"]["pnl"] for d in (30, 60, 90, 120)]
    hold = [res["hold1"]["pnl"], res["hold2"]["pnl"]]
    meses = [res[f"mes{k}"]["pnl"] for k in range(1, 7)]
    pos = [m for m in meses if m > 0]
    concentracion = (max(pos) / sum(pos)) if pos else 1.0
    checks = {
        "4_recientes_positivas": all(x > 0 for x in rec),
        "2_falsaciones_positivas": all(x > 0 for x in hold),
        "trades_120d>=15": res["w120"]["n"] >= 15,
        "meses_positivos>=4": len(pos) >= 4,
        "ningun_mes>50%": concentracion <= 0.5,
    }
    return all(checks.values()), checks, round(concentracion, 2)


def main():
    fin = pd.to_datetime(pd.read_csv(DIR / "velas_candidatos.csv", usecols=["date"])["date"], utc=True).max()
    rutas = _datos_truncados(fin)
    ventanas = _ventanas(fin)
    trabajos, cands = [], []
    for f in sorted(SWEEPS.glob("best_*_s*.json")):
        data = json.loads(f.read_text())
        data = data if isinstance(data, list) else data.get("pairs", [])
        if not data:
            continue
        e = data[0]
        params = dict(e.get("params", {}))
        params["peso"] = 0.20
        cand = VAL / f"cand_{f.stem[5:]}.json"
        cand.write_text(json.dumps([{"symbol": e["symbol"], "params": params}]), encoding="utf-8")
        cands.append(cand)
        for nombre, (_, dias) in ventanas.items():
            trabajos.append((str(cand), rutas[nombre], dias, f"{cand.stem}|{nombre}"))
    resultados = {}
    with ProcessPoolExecutor(max_workers=6) as ex:
        for clave, r in ex.map(_correr, trabajos):
            c, nombre = clave.split("|")
            resultados.setdefault(c, {})[nombre] = r
    filas = []
    for c, res in sorted(resultados.items()):
        ok, checks, conc = veredicto(res)
        filas.append({"candidato": c[5:], "pasa": ok, **{k: res[k]["pnl"] for k in ventanas},
                      "n120": res["w120"]["n"], "por_trade_120": res["w120"]["por_trade_pct"],
                      "concentracion": conc, "falla": ", ".join(k for k, v in checks.items() if not v)})
    df = pd.DataFrame(filas)
    df.to_csv(VAL / "resumen.csv", index=False)
    pd.set_option("display.width", 300)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
