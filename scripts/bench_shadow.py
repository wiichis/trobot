#!/usr/bin/env python3
"""Revisión semanal de la banca (01/10/2026). Ver pkg/bench.py y CLAUDE.md, "Banca de pares".

Para cada par ACTIVO evalúa si debería pasar a banca (sim en 4 ventanas + PnL real en
bloques de 14 días). Para cada par EN BANCA con params, lo simula HACIA ADELANTE desde su
fecha de ingreso (con velas de precalentamiento para que los indicadores no arranquen en
frío) y evalúa si puede volver a operar o si se venció su plazo.

NO cambia nada: propone, y la decisión la toma el usuario. Escribe un reporte en
archivos/bench/reportes/.

Uso (local, después de bajar de prod long.csv, bench.csv y PnL.csv):
    python3 scripts/bench_shadow.py --data archivos/cripto_price_5m_long.csv \\
        --bench_data archivos/cripto_price_5m_bench.csv --pnl archivos/PnL.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pkg"))

from pkg import backtesting as bt  # noqa: E402
from pkg import bench  # noqa: E402

VENTANAS = (30, 60, 90, 120)
WARMUP_DIAS = 10


def _velas(paths):
    frames = []
    for p in paths:
        if p and Path(p).exists():
            d = pd.read_csv(p)
            d["date"] = pd.to_datetime(d["date"], utc=True, errors="coerce", format="mixed")
            frames.append(d.dropna(subset=["symbol", "date"]))
    if not frames:
        raise SystemExit("No hay velas para simular")
    df = pd.concat(frames).drop_duplicates(["symbol", "date"], keep="last").sort_values(["symbol", "date"])
    # la última vela de cada serie puede estar en formación
    df = df[df["date"] < df.groupby("symbol")["date"].transform("max")]
    return df


def _escribir_csv(df, dirpath, nombre):
    p = Path(dirpath) / nombre
    out = df.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    out.to_csv(p, index=False)
    return str(p)


def _por_posicion(trades):
    rows = [dict(symbol=t.symbol, pid=t.position_id, entry_time=t.entry_time,
                 pnl=t.pnl(), notional=t.entry_price * t.qty) for t in trades]
    if not rows:
        return pd.DataFrame(columns=["symbol", "pid", "entry_time", "pnl", "notional"])
    df = pd.DataFrame(rows)
    return df.groupby(["symbol", "pid"]).agg(entry_time=("entry_time", "first"), pnl=("pnl", "sum"),
                                             notional=("notional", "sum")).reset_index()


def bloques_reales(pnl_csv, symbols, fin, n=4, dias=14):
    if not pnl_csv or not Path(pnl_csv).exists():
        return {s: [] for s in symbols}
    p = pd.read_csv(pnl_csv)
    p["ts"] = pd.to_datetime(p["time"], errors="coerce").dt.tz_localize("UTC")
    out = {}
    for s in symbols:
        q = p[p["symbol"] == s]
        vals = []
        for k in range(n, 0, -1):
            a, b = fin - pd.Timedelta(days=dias * k), fin - pd.Timedelta(days=dias * (k - 1))
            vals.append(float(q[(q["ts"] >= a) & (q["ts"] < b)]["income"].sum()))
        out[s] = vals
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "archivos/cripto_price_5m_long.csv"))
    ap.add_argument("--bench_data", default=str(REPO / "archivos/cripto_price_5m_bench.csv"))
    ap.add_argument("--pnl", default=str(REPO / "archivos/PnL.csv"))
    ap.add_argument("--best", default=str(REPO / "pkg/best_prod.json"))
    ap.add_argument("--bench", default=str(bench.BENCH_PATH))
    ap.add_argument("--capital", type=float, default=None, help="balance real (default: último de ganancias.csv)")
    ap.add_argument("--out", default=str(REPO / "archivos/bench/reportes"))
    args = ap.parse_args()

    capital = args.capital
    if capital is None:
        g = REPO / "archivos/ganancias.csv"
        try:
            capital = float(pd.read_csv(g, header=None).iloc[-1, 1])
        except Exception:
            capital = 188.0

    best = json.loads(Path(args.best).read_text(encoding="utf-8"))
    activos = sorted({e["symbol"].upper() for e in best})
    banca = bench.load_bench(Path(args.bench))
    # Los pares en banca no están en best_prod.json: sin esto su forward daría 0 trades.
    bench.habilitar_en_indicadores([e["symbol"] for e in banca])
    velas = _velas([args.data, args.bench_data])
    fin = velas["date"].max()
    hoy = fin.date()

    informe = {"generado": datetime.now(timezone.utc).isoformat(), "fin_datos": str(fin),
               "capital": capital, "activos": {}, "banca": {}}

    with tempfile.TemporaryDirectory() as tmp:
        data_path = _escribir_csv(velas, tmp, "velas.csv")

        # --- activos: ¿alguno debería ir a banca? ---
        sim = {s: {} for s in activos}
        for d in VENTANAS:
            r = bt.run_live_parity_portfolio(activos, data_path, capital, None, best_path=args.best,
                                             lookback_days=d, return_trades=True)
            pos = _por_posicion(r["trades_list"])
            for s in activos:
                sim[s][f"{d}d"] = round(float(pos[pos["symbol"] == s]["pnl"].sum()), 3)
        reales = bloques_reales(args.pnl, activos, fin)
        for s in activos:
            ok, motivo = bench.debe_ir_a_banca(sim[s], reales[s])
            informe["activos"][s] = {"sim": sim[s], "real_bloques_14d": [round(x, 3) for x in reales[s]],
                                     "propone_banca": ok, "motivo": motivo}

        # --- banca: forward desde `desde` ---
        for e in banca:
            s = e["symbol"]
            fila = {"estado": e["estado"], "desde": e.get("desde")}
            if e["estado"] != "banca" or not e.get("params"):
                fila["nota"] = "en observación: sólo se le bajan velas"
                informe["banca"][s] = fila
                continue
            if s not in set(velas["symbol"]):
                fila["nota"] = "sin velas todavía"
                informe["banca"][s] = fila
                continue
            desde = pd.Timestamp(e["desde"], tz="UTC")
            dias = max(1, (fin - desde).days + 1)
            params = dict(e["params"])
            params.setdefault("peso", 0.20)
            best_tmp = Path(tmp) / f"best_{s}.json"
            best_tmp.write_text(json.dumps([{"symbol": s, "params": params}]), encoding="utf-8")
            r = bt.run_live_parity_portfolio([s], data_path, capital, None, best_path=str(best_tmp),
                                             lookback_days=dias + WARMUP_DIAS, return_trades=True)
            pos = _por_posicion(r["trades_list"])
            pos = pos[pd.to_datetime(pos["entry_time"], utc=True) >= desde]
            n = int(len(pos))
            pct = float((pos["pnl"] / pos["notional"]).mean()) if n else 0.0
            ok, motivo = bench.puede_activarse(bool(e.get("protocolo_ok", False)), n, pct)
            fila.update({"trades_forward": n, "pnl_forward": round(float(pos["pnl"].sum()), 3),
                         "pnl_por_trade_pct": round(pct, 5), "protocolo_ok": bool(e.get("protocolo_ok", False)),
                         "puede_activarse": ok, "motivo": motivo,
                         "vencido": (not ok) and bench.vencio_en_banca(e["desde"], hoy)})
            informe["banca"][s] = fila

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = hoy.strftime("%Y%m%d")
    (out / f"banca_{stamp}.json").write_text(json.dumps(informe, indent=2, ensure_ascii=False), encoding="utf-8")

    lineas = [f"# Banca — {hoy} (datos hasta {fin:%d/%m %H:%M} UTC, capital {capital:.2f})", "",
              "## Activos", "", "| par | sim 30/60/90/120d | real bloques 14d | ¿a banca? |", "|---|---|---|---|"]
    for s, f in informe["activos"].items():
        lineas.append(f"| {s} | {' / '.join(f'{v:+.2f}' for v in f['sim'].values())} | "
                      f"{' / '.join(f'{v:+.2f}' for v in f['real_bloques_14d'])} | "
                      f"{'**SÍ**' if f['propone_banca'] else 'no'} |")
    lineas += ["", "## Banca", "", "| par | estado | desde | trades fwd | PnL fwd | por trade | ¿activar? | vencido |",
               "|---|---|---|---|---|---|---|---|"]
    for s, f in informe["banca"].items():
        if "trades_forward" not in f:
            lineas.append(f"| {s} | {f['estado']} | {f.get('desde')} | — | — | — | {f.get('nota')} | — |")
            continue
        lineas.append(f"| {s} | {f['estado']} | {f['desde']} | {f['trades_forward']} | {f['pnl_forward']:+.3f} | "
                      f"{f['pnl_por_trade_pct']:+.2%} | {'**SÍ**' if f['puede_activarse'] else 'no — ' + f['motivo']} | "
                      f"{'**SÍ**' if f['vencido'] else 'no'} |")
    (out / f"banca_{stamp}.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
