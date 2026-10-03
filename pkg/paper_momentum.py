"""Prueba hacia adelante SIN DINERO del momentum entre pares (V3) — 02/10/2026.

Configuración y criterios de evaluación en `pkg/paper_momentum.json` (fijados antes de
empezar). Resultado en `archivos/paper/`:
- `momentum_estado.json`  próximo rebalanceo y número de rebalanceos hechos.
- `momentum_libro.csv`    patas abiertas (símbolo, lado, precio y hora de entrada).
- `momentum_trades.csv`   patas cerradas con su resultado bruto y neto.

Diseño:
- Corre como job HORARIO del bot y decide con el estado en disco si toca rebalancear. Un
  job "cada 3 días" de `schedule` se reinicia con cada restart del bot (el de 12 h de
  long.csv no llegó a correr el 02/10 por eso); el estado en disco sobrevive.
- No manda órdenes ni toca nada del camino de trading. NUNCA levanta excepción: en
  `main.py` una excepción en un job corta el loop entero.
- Precios: velas de 1 h de BingX, sólo las CERRADAS. Cierre del libro y entrada del nuevo
  al close de la última hora cerrada — equivale al backtest, que entra en la apertura de la
  hora siguiente a la señal.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

log = logging.getLogger("trobot")

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "paper_momentum.json"
OUT_DIR = BASE_DIR / "archivos" / "paper"


def _cfg() -> Dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _rutas():
    return (OUT_DIR / "momentum_estado.json", OUT_DIR / "momentum_libro.csv", OUT_DIR / "momentum_trades.csv")


def _precios_1h(symbol: str, limit: int, ahora: pd.Timestamp) -> pd.Series:
    """Closes horarios CERRADOS (índice = apertura de la hora, UTC)."""
    from .price_bingx_5m import _fetch_bingx_candles
    velas = _fetch_bingx_candles(symbol, limit, interval="1h")
    if not velas:
        return pd.Series(dtype=float)
    d = pd.DataFrame(velas)
    d["date"] = pd.to_datetime(d["date"], utc=True)
    d = d[d["date"] + pd.Timedelta(hours=1) <= ahora]          # fuera la hora en formación
    return d.drop_duplicates("date").set_index("date")["close"].sort_index()


def rebalancear(precios: Dict[str, pd.Series], libro: pd.DataFrame, cfg: Dict,
                ahora: pd.Timestamp, n_reb: int):
    """Lógica pura: cierra `libro` y arma el nuevo. Devuelve (cerradas, libro_nuevo, resumen)."""
    look, n_l, n_c = int(cfg["lookback_h"]), int(cfg["n_largos"]), int(cfg["n_cortos"])
    costo = float(cfg["costo_por_vuelta"])
    ultimo, ret = {}, {}
    for sym, s in precios.items():
        if s is None or s.empty:
            continue
        h_ult = s.index.max()
        ultimo[sym] = (h_ult, float(s.iloc[-1]))
        h_ini = h_ult - pd.Timedelta(hours=look)
        if h_ini in s.index and s.loc[h_ini] > 0:
            ret[sym] = float(s.iloc[-1] / s.loc[h_ini] - 1)

    cerradas = []
    for _, pata in libro.iterrows():
        sym, lado, pe = pata["symbol"], int(pata["lado"]), float(pata["precio_entrada"])
        if sym not in ultimo:
            cerradas.append({**pata.to_dict(), "salida": str(ahora), "precio_salida": None,
                             "bruto": None, "neto": None, "nota": "sin precio de salida"})
            continue
        ps = ultimo[sym][1]
        bruto = lado * (ps / pe - 1)
        cerradas.append({**pata.to_dict(), "salida": str(ultimo[sym][0] + pd.Timedelta(hours=1)),
                         "precio_salida": ps, "bruto": bruto, "neto": bruto - costo, "nota": ""})

    nuevo = []
    if len(ret) >= n_l + n_c + 5:
        orden = sorted(ret, key=ret.get)
        patas = [(s, 1) for s in orden[-n_l:]] + ([(s, -1) for s in orden[:n_c]] if n_c else [])
        for sym, lado in patas:
            nuevo.append({"rebalanceo": n_reb + 1, "symbol": sym, "lado": lado,
                          "entrada": str(ultimo[sym][0] + pd.Timedelta(hours=1)),
                          "precio_entrada": ultimo[sym][1], "retorno_lookback": round(ret[sym], 5)})
    netos = [c["neto"] for c in cerradas if c["neto"] is not None]
    resumen = {"rebalanceo": n_reb + 1, "universo_con_precio": len(ret), "patas_cerradas": len(cerradas),
               "neto_medio_cerradas": (sum(netos) / len(netos)) if netos else None,
               "largos": [s for s, l in [(p["symbol"], p["lado"]) for p in nuevo] if l == 1],
               "cortos": [s for s, l in [(p["symbol"], p["lado"]) for p in nuevo] if l == -1]}
    return cerradas, pd.DataFrame(nuevo), resumen


def run_paper_momentum(ahora: Optional[datetime] = None, fetch=None) -> Optional[Dict]:
    """Job horario. Devuelve el resumen si rebalanceó, None si no tocaba o si falló."""
    try:
        cfg = _cfg()
        if not cfg.get("enabled", False):
            return None
        ahora = pd.Timestamp(ahora or datetime.now(timezone.utc))
        ahora = ahora.tz_localize("UTC") if ahora.tzinfo is None else ahora.tz_convert("UTC")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        f_estado, f_libro, f_trades = _rutas()
        estado = json.loads(f_estado.read_text()) if f_estado.exists() else {"rebalanceos": 0, "proximo": None}
        if estado.get("proximo") and ahora < pd.Timestamp(estado["proximo"]):
            return None

        fetch = fetch or _precios_1h
        limite = int(cfg["lookback_h"]) + 12
        precios = {}
        for sym in cfg["universo"]:
            try:
                precios[sym] = fetch(sym, limite, ahora)
            except Exception as exc:
                log.warning("paper_momentum: sin precios de %s: %s", sym, exc)
            time.sleep(0.1)

        libro = pd.read_csv(f_libro) if f_libro.exists() else pd.DataFrame(columns=["symbol", "lado", "precio_entrada"])
        cerradas, nuevo, resumen = rebalancear(precios, libro, cfg, ahora, int(estado["rebalanceos"]))
        if nuevo.empty:
            log.warning("paper_momentum: universo insuficiente (%s pares con precio); se reintenta en 1 h",
                        resumen["universo_con_precio"])
            return None
        if cerradas:
            t = pd.DataFrame(cerradas)
            t.to_csv(f_trades, mode="a", header=not f_trades.exists(), index=False)
        nuevo.to_csv(f_libro, index=False)
        hora = ahora.floor("1h")
        estado = {"rebalanceos": resumen["rebalanceo"], "ultimo": str(hora),
                  "proximo": str(hora + pd.Timedelta(hours=int(cfg["hold_h"])))}
        f_estado.write_text(json.dumps(estado, indent=1), encoding="utf-8")
        resumen["proximo"] = estado["proximo"]
        _avisar(resumen, f_trades)
        return resumen
    except Exception as exc:   # nunca tirar el bot por la prueba sin dinero
        log.warning("paper_momentum falló: %s", exc)
        return None


def _pct(x: float) -> str:
    return (f"{x:+.2f}%").replace(".", ",")


def mensaje_md(resumen: Dict, neto_cerradas: Optional[float], acumulado_pct: Optional[float]) -> str:
    """Mensaje de Telegram (Markdown) de un rebalanceo."""
    from .lifecycle_events import escapar_md
    corto = lambda syms: escapar_md(", ".join(str(x).replace("-USDT", "") for x in syms)) or "—"
    lineas = [f"_Prueba SIN DINERO · rebalanceo {resumen['rebalanceo']}_",
              f"🟢 Largos: {corto(resumen['largos'])}",
              f"🔴 Cortos: {corto(resumen['cortos'])}"]
    if neto_cerradas is not None:
        lineas.append(f"Patas que cerraron: `{_pct(neto_cerradas * 100)}` neto medio")
    if acumulado_pct is not None:
        lineas.append(f"Desde el inicio: `{_pct(acumulado_pct)}` neto medio por pata")
    if resumen.get("proximo"):
        lineas.append(f"Próximo rebalanceo: {pd.Timestamp(resumen['proximo']):%d/%m %H:%M} UTC")
    return "\n".join(lineas)


def _avisar(resumen: Dict, f_trades: Path) -> None:
    try:
        from .lifecycle_events import emit_lifecycle_event
        acumulado = None
        if f_trades.exists():
            t = pd.read_csv(f_trades)
            if "neto" in t and t["neto"].notna().any():
                acumulado = round(float(t["neto"].mean()) * 100, 3)
        nm = resumen["neto_medio_cerradas"]
        emit_lifecycle_event(
            "paper_momentum_rebalanceo", "INFO",
            rebalanceo=resumen["rebalanceo"],
            neto_medio_patas_cerradas_pct=None if nm is None else round(nm * 100, 3),
            neto_medio_acumulado_pct=acumulado,
            largos=",".join(resumen["largos"]), cortos=",".join(resumen["cortos"]),
            detalle="prueba SIN DINERO del momentum entre pares (V3); no se mandan órdenes",
            mensaje_md=mensaje_md(resumen, nm, acumulado),
        )
    except Exception:
        pass
