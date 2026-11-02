"""Dos experimentos en papel junto al real (02/11/2026), sin Telegram.

El 02/11 el real pasó a lotes de 3 ciclos. Para ver cómo se comportan las otras dos reglas que
mejor salieron en la simulación, corren en papel desde el mismo punto (el estado del 02/11, con
3 slots abiertos) y con los mismos depósitos del real (`real/depositos.csv`):

- `exp_b1`: lotes de **1 ciclo** (`pkg/btc_rangos.py` con `pkg/btc_rangos_exp_b1.json`). Fue la
  mejor de la simulación en 2021-2026: −0,65 % de BTC contra el DCA.
- `exp_v04`: la regla **v0.4** con el cierre forzado del día 3 (`pkg/btc_rangos_v04.py`). Fue
  la mejor en 2024-2026: +0,23 %.

El usuario fijó el criterio antes de arrancar: el 30/11 se mira cómo van, sin cambiar el real
por el resultado de un mes. `comparacion()` arma la tabla que muestran
`scripts/btc_rangos_hitos.py` y el tablero.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from . import btc_rangos as papel
from . import btc_rangos_v04 as v04

log = logging.getLogger(__name__)

PKG = Path(__file__).resolve().parent
BASE = papel.BASE_DIR / "archivos" / "btc_rangos"
EXPERIMENTOS = {
    "exp_b1": {"nombre": "Lotes de 1 ciclo", "config": PKG / "btc_rangos_exp_b1.json"},
    "exp_v04": {"nombre": "Regla v0.4 (cierre el día 3)", "config": PKG / "btc_rangos_exp_v04.json"},
}
# las cuatro carpetas que se comparan, en el orden de la tabla
FILAS = (("real", "Real · lotes de 3 ciclos"), ("papel", "Sombra · lotes de 3 ciclos"),
         ("exp_b1", EXPERIMENTOS["exp_b1"]["nombre"]), ("exp_v04", EXPERIMENTOS["exp_v04"]["nombre"]))
COMISION, SLIPPAGE = 0.001, 0.0005


def run_btc_rangos_experimentos(ahora=None, fuentes=None, base: Optional[Path] = None) -> Dict:
    """Job horario (:01, después del real y de la sombra). Uno que falla no frena al otro."""
    base = Path(base or BASE)
    out = {}
    for clave, e in EXPERIMENTOS.items():
        try:
            if clave == "exp_v04":
                v04.CONFIG_PATH, v04.OUT_DIR = e["config"], base / clave  # el módulo es de una sola instancia
                out[clave] = v04.run_btc_rangos_v04(ahora, fuentes)
            else:
                out[clave] = papel.run_btc_rangos(ahora, fuentes, config_path=e["config"], out_dir=base / clave)
        except Exception as exc:
            log.exception("btc_rangos_experimentos: %s falló: %s", clave, exc)
            out[clave] = None
    return out


def _leer(f: Path) -> pd.DataFrame:
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


def dca_puro_btc(base: Path, precio_inicio: float, capital: float) -> float:
    """Cálculo propio, no el del motor: el capital comprado al precio de arranque del real y cada
    depósito en su hora, con la comisión y el slippage del backtest."""
    deps = _leer(base / "real" / "depositos.csv")
    compras = [(capital, precio_inicio)]
    if len(deps):
        compras += [(float(d.monto), float(d.precio)) for d in deps[deps["lote"] > 0].itertuples()]
    return sum(u * (1 - COMISION) / (p * (1 + SLIPPAGE)) for u, p in compras)


def serie(base: Optional[Path] = None, desde: Optional[str] = None) -> Dict[str, List[Dict]]:
    """Ventaja contra el DCA puro hora a hora de cada uno (cada foto a su propio precio), desde
    `desde` (por defecto, el arranque de los lotes en el real)."""
    base = Path(base or BASE)
    est_real = papel.leer_estado(base / "real") or {}
    desde = pd.Timestamp(desde or est_real.get("inicio_lotes") or "2026-11-02")
    desde = desde.tz_convert(None) if desde.tzinfo else desde
    deps = _leer(base / "real" / "depositos.csv")
    deps = deps[deps["lote"] > 0] if len(deps) else deps
    capital = float((est_real.get("combo") or {}).get("capital") or 0.0)
    p0 = float(est_real.get("precio_inicio") or float("nan"))
    out = {}
    for clave, _ in FILAS:
        eq = _leer(base / clave / "equity.csv")
        if not len(eq) or "patrimonio_btc" not in eq:
            continue
        ts = pd.to_datetime(eq["ts"], utc=True).dt.tz_convert(None)
        eq = eq.assign(ts=ts)[ts >= desde]
        filas = []
        for x in eq.itertuples():
            llegados = deps[pd.to_datetime(deps["t"]) <= x.ts] if len(deps) else deps
            dca = sum(u * (1 - COMISION) / (p * (1 + SLIPPAGE))
                      for u, p in [(capital, p0)] + [(float(d.monto), float(d.precio)) for d in llegados.itertuples()])
            filas.append({"ts": x.ts.isoformat(), "precio": float(x.precio), "patrimonio_btc": float(x.patrimonio_btc),
                          "vs_dca_pct": (float(x.patrimonio_btc) / dca - 1) * 100 if dca else float("nan")})
        out[clave] = filas
    return out


def comparacion(base: Optional[Path] = None, precio: Optional[float] = None) -> Dict:
    """Patrimonio de cada uno en BTC, todos valuados al mismo precio (el último del real, si no
    se da otro), contra el DCA puro y contra el real."""
    base = Path(base or BASE)
    est_real = papel.leer_estado(base / "real") or {}
    eq_real = _leer(base / "real" / "equity.csv")
    if precio is None and len(eq_real):
        precio = float(eq_real["precio"].iloc[-1])
    filas: List[Dict] = []
    for clave, nombre in FILAS:
        d = base / clave
        eq, ops = _leer(d / "equity.csv"), _leer(d / "operaciones.csv")
        est = papel.leer_estado(d) if (d / "estado.json").exists() else None
        if not len(eq) or est is None or "patrimonio_btc" not in eq:
            filas.append({"clave": clave, "nombre": nombre, "corre": False})
            continue
        u = eq.iloc[-1]
        btc = float(u["btc_slots"] + u["holding_btc"])
        usdt = (float(u["patrimonio_btc"]) - btc) * float(u["precio"])  # todo el USDT, también el que espera
        desde = pd.Timestamp(est.get("inicio_lotes") or est.get("inicio_exp") or est["inicio"])
        desde = desde.tz_convert(None) if desde.tzinfo else desde
        ventas = ops[pd.to_datetime(ops["salida_t"]) >= desde] if len(ops) else ops  # sólo desde el arranque
        filas.append({"clave": clave, "nombre": nombre, "corre": True, "actualizado": est.get("actualizado"),
                      "btc": btc, "usdt": usdt, "holding_btc": float(u["holding_btc"]),
                      "btc_slots": float(u["btc_slots"]),
                      "patrimonio_btc": btc + usdt / precio if precio else float("nan"),
                      "ventas": len(ventas), "realizado": float(ventas["ganancia"].sum()) if len(ventas) else 0.0,
                      "atrapados": int(u["atrapados"]) if "atrapados" in u and pd.notna(u["atrapados"]) else None})
    dca = dca_puro_btc(base, float(est_real.get("precio_inicio") or float("nan")),
                       float((est_real.get("combo") or {}).get("capital") or 0.0)) if est_real else float("nan")
    real = next((f for f in filas if f["clave"] == "real" and f["corre"]), None)
    for f in filas:
        if f["corre"]:
            f["vs_dca_pct"] = (f["patrimonio_btc"] / dca - 1) * 100 if dca == dca and dca else float("nan")
            f["vs_real_pct"] = (f["patrimonio_btc"] / real["patrimonio_btc"] - 1) * 100 if real else float("nan")
    return {"precio": precio, "dca_puro_btc": dca, "filas": filas}
