"""Tablero de hitos del bot BTC por rangos en real (24/10/2026; lotes de 3 ciclos desde el 02/11).
Reglas en `docs/btc_rangos_real_hitos.md`, fijadas antes de operar con cada versión.

Lee `archivos/btc_rangos/real/` (y `papel/` como sombra, si arrancaron juntos) y muestra:
1. Ejecución: deslizamiento de cada compra contra el libro al mandarla y contra la compra del
   papel en la misma vela; comisiones reales.
2. Depósitos y corte a corte (día 3): ganancia realizada, costo de las atrapadas reemplazadas,
   cambio en la pérdida latente, neto, cuánto se movió BTC y qué esperaba la simulación.
3. Los cuatro criterios del veredicto, con el valor de hoy.

Uso (en el server o con copias locales): python3 scripts/btc_rangos_hitos.py [--dir archivos/btc_rangos]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
# lotes de 3 ciclos, `btc_rangos_lotes_sim.py --mensual 3` (igual que `ESPERADO_SIMULACION` del real)
ESPERADO = ((-5.0, "cae >5 %", -9.8), (5.0, "±5 %", 1.1), (float("inf"), "sube >5 %", 9.6))
VEREDICTO = "03/06/2027"


def esperado(var):
    for tope, texto, neto in ESPERADO:
        if var < tope:
            return texto, neto
    return ESPERADO[-1][1:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(REPO / "archivos" / "btc_rangos"))
    base = Path(ap.parse_args().dir)
    real, sombra = base / "real", base / "papel"
    est = json.loads((real / "estado.json").read_text(encoding="utf-8"))
    leer = lambda f: pd.read_csv(f) if f.exists() else pd.DataFrame()
    ops, meses, ordenes, eq, deps = (leer(real / f) for f in ("operaciones.csv", "mensual.csv", "ordenes.csv",
                                                               "equity.csv", "depositos.csv"))
    print(f"Real desde {est['inicio'][:16]}, lotes desde {est.get('inicio_lotes', '—')[:16]} · "
          f"pausa: {est.get('pausa') or 'no'}")

    print("\n1) EJECUCIÓN")
    compras = ordenes[(ordenes.get("motivo") == "compra") & (ordenes.get("ejecutado", 0) > 0)] if len(ordenes) else ordenes
    if len(compras):
        desliz = (compras["precio_medio"] / compras["ask"] - 1) * 1e4
        print(f"   compras: {len(compras)} · deslizamiento contra el ask al mandar: mediana {desliz.median():.1f} bps, "
              f"peor {desliz.max():.1f} bps (el backtest supone 5 bps)")
        unicas = ordenes.drop_duplicates("order_id", keep="last")
        fee_usdt = unicas["fee"].fillna(0) * unicas["precio_medio"].where(unicas["fee_asset"] == "BTC", 1.0)
        print(f"   comisiones cobradas: {fee_usdt.sum():.2f} USDT")
    else:
        print("   todavía no hay compras")
    p_ops = leer(sombra / "operaciones.csv")
    if len(ops) and len(p_ops):
        a = ops.merge(p_ops, on=["slot", "entrada_t"], suffixes=("_real", "_papel"))
        if len(a):
            print(f"   ventas emparejadas con el papel: {len(a)} de {len(ops)} · ganancia real − papel: "
                  f"{(a['ganancia_real'] - a['ganancia_papel']).sum():+.2f} USDT")

    print("\n2) DEPÓSITOS Y CORTES DEL DÍA 3")
    if len(deps):
        d = deps[deps["lote"] > 0]
        print(f"   depósitos: {len(d)} por {d['monto'].sum():.2f} USDT"
              + (f" · reemplazaron atrapados: {int((d['reemplaza'].fillna('').astype(str) != '').sum())}" if len(d) else ""))
    if len(meses):
        meses["neto_pct"] = meses["neto"] / meses["capital_inicio"] * 100
        meses["var_btc"] = (meses["precio_cierre"] / meses["precio_inicio"] - 1) * 100
        for _, f in meses.iterrows():
            ref = "BTC {:+.1f} % ({}) → simulación ~{:+.1f} %".format(f["var_btc"], *esperado(f["var_btc"]))
            print(f"   {f['ciclo']}: realizado {f['realizado']:+7.2f} · atrapadas {f['atrapadas_costo']:6.2f} · "
                  f"pérdida latente {f['latente']:+6.2f} · neto {f['neto']:+7.2f} USDT ({f['neto_pct']:+5.1f} %) · "
                  f"{int(f['ventas'])} ventas · {ref}")
    else:
        print("   todavía no hubo corte")

    print(f"\n3) CRITERIOS (veredicto en el corte del {VEREDICTO})")
    if len(meses):
        completos = meses.iloc[1:] if len(meses) > 1 else meses.iloc[0:0]  # el primer ciclo es parcial
        neto_medio = completos["neto_pct"].mean() if len(completos) else float("nan")
        caidas = completos[completos["var_btc"] < -5]
        total = completos["neto"].sum()
        conc = completos["neto"].max() / total if total > 0 else float("nan")
        print(f"   a) neto medio ≥ +1 %/mes: {neto_medio:+.2f} % en {len(completos)} ciclos completos" if len(completos)
              else "   a) neto medio ≥ +1 %/mes: todavía no hay ciclos completos (el primero es parcial)")
        print(f"   c) neto en los meses con BTC −5 % o peor ≥ 0: {len(caidas)} meses así"
              + (f", neto medio {caidas['neto_pct'].mean():+.2f} %" if len(caidas) else " (hacen falta 2)"))
        if len(completos):
            print(f"   d) ningún mes > 50 % del neto: {conc:.0%}" if total > 0 else "   d) neto total ≤ 0")
    if len(eq) and "latente" in eq:
        u = eq.iloc[-1]
        con, sin = u["patrimonio_btc"], u["bench_btc"]
        print(f"   b) BTC con bot {con:.6f} vs DCA puro {sin:.6f} ({(con / sin - 1) * 100:+.2f} %) · "
              f"pérdida latente hoy {u['latente']:+.2f} USDT · atrapados {int(u['atrapados'])}")


if __name__ == "__main__":
    main()
