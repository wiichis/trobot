#!/usr/bin/env python3
"""Reporte de la prueba sin dinero del momentum entre pares (pkg/paper_momentum.py).

Uso (bajar primero los archivos de prod):
    scp -i <pem> ubuntu@<prod>:TRobot/archivos/paper/momentum_*.{csv,json} archivos/paper/
    python3 scripts/paper_momentum_reporte.py

Aplica los criterios fijados de antemano en pkg/paper_momentum.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent


def main():
    cfg = json.loads((REPO / "pkg/paper_momentum.json").read_text())
    crit = cfg["criterios"]
    f = REPO / "archivos/paper/momentum_trades.csv"
    if not f.exists():
        print("Todavía no hay patas cerradas (el primer cierre llega en el segundo rebalanceo).")
        return
    t = pd.read_csv(f).dropna(subset=["neto"])
    por_reb = t.groupby("rebalanceo")["neto"].mean()
    n_reb = len(por_reb)
    media = t["neto"].mean()
    tstat = media / (t["neto"].std() / np.sqrt(len(t))) if len(t) > 2 else float("nan")
    pos = (por_reb > 0).mean() if n_reb else float("nan")
    print(f"Rebalanceos cerrados: {n_reb} de {crit['revision_en_rebalanceos']} para la revisión | patas: {len(t)}")
    print(f"Neto medio por pata: {media*100:+.3f}% (t={tstat:+.2f}) | bruto {t['bruto'].mean()*100:+.3f}%")
    print(f"Rebalanceos positivos: {pos:.0%}")
    print("Por lado:", t.groupby("lado")["neto"].agg(lambda x: f"{x.mean()*100:+.3f}% (n={len(x)})").to_dict())
    print("Por rebalanceo (neto medio %):", (por_reb * 100).round(3).to_dict())
    print(f"Acumulado (suma de rebalanceos, cartera equiponderada): {por_reb.sum()*100:+.2f}%")
    if n_reb < crit["revision_en_rebalanceos"]:
        print("→ Sin veredicto: faltan rebalanceos.")
    elif media >= 0.002 and pos >= 0.60:
        print("→ ÉXITO según el criterio:", crit["exito"])
    elif media <= 0:
        print("→ FRACASO según el criterio:", crit["fracaso"])
    else:
        print("→ INTERMEDIO:", crit["intermedio"])


if __name__ == "__main__":
    main()
