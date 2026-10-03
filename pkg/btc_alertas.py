"""Alerta de ventana de compra / venta de BTC (03/10/2026).

Pedido del usuario: tener a mano cuándo comprar y cuándo vender su BTC (holding), con un
precio de referencia. Son señales de dos reglas MEDIDAS (las de los bots A y D de
`pkg/btc_bots.py`, código en `pkg/btc_reglas.py`), no una recomendación:

- Tendencia (regla T1, vela diaria): cierre sobre la media de 100 días = ventana de
  COMPRA; debajo = ventana de VENTA. El precio de referencia es la propia media: si el
  cierre diario la cruza, la ventana cambia. Seguida con un holding al contado
  2019-2026: +2.845% contra +2.193% del holding, caída máxima −39% contra −77%,
  invertido el 57% del tiempo, ~6 compras por año y sólo el 24% salen bien
  (`scripts/btc_lab_temporalidades.py`, costos de contado).
- Caída fuerte (regla R1, vela diaria): RSI 14 bajo 30 = ventana de COMPRA por caída;
  se cierra cuando el RSI vuelve a 50. 28 veces en 2019-2026. Ojo: el RSI ALTO no es
  señal de venta (vender/shortear con RSI > 70 perdió −90% en el laboratorio).

Corre cada hora (:02) y decide una vez por vela diaria cerrada. Avisa sólo cuando una
ventana cambia (un mensaje por ciclo; Telegram silencia una categoría 15 min) y una vez
al arrancar con el estado actual. El estado vive en `archivos/btc_bots/ventana_estado.json`
y el historial en `ventana_log.csv`. NUNCA levanta excepción.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd

from .btc_reglas import posiciones, rsi

log = logging.getLogger("trobot")

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "btc_bots.json"
OUT_DIR = BASE_DIR / "archivos" / "btc_bots"
VENTANA_VELAS = 500
MIN_VELAS = 120
NOTA = "Señal de una regla medida (2019-2026), no una recomendación."


def _n(x: float, dec: int = 0) -> str:
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _rutas():
    return OUT_DIR / "ventana_estado.json", OUT_DIR / "ventana_log.csv"


def _desde(serie: pd.Series) -> pd.Timestamp:
    """Fecha de la primera vela del tramo actual (desde cuándo rige el estado)."""
    cambio = serie.ne(serie.shift(1))
    return serie.index[cambio].max()


def evaluar(v: pd.DataFrame) -> Dict:
    """Estado de las dos ventanas con la última vela diaria CERRADA de `v`."""
    tendencia = posiciones(v, "T1", True)          # 1 sobre la media de 100, 0 debajo
    caida = posiciones(v, "R1", True)              # 1 desde RSI < 30 hasta volver a 50
    c = v["close"]
    return {"barra": str(v.index[-1]), "cierre": float(c.iat[-1]),
            "media100": float(c.rolling(100).mean().iat[-1]), "rsi": float(rsi(c).iat[-1]),
            "tendencia": int(tendencia.iat[-1]), "tendencia_desde": str(_desde(tendencia).date()),
            "caida": int(caida.iat[-1]), "caida_desde": str(_desde(caida).date())}


def texto_estado(e: Dict) -> str:
    """Una línea para el resumen diario."""
    dist = (e["cierre"] / e["media100"] - 1) * 100
    ventana = "COMPRA (tendencia alcista)" if e["tendencia"] else "VENTA (tendencia bajista)"
    extra = " · además ventana de compra por caída fuerte" if e["caida"] else ""
    return (f"{ventana} desde {e['tendencia_desde']} · cierre {_n(e['cierre'])} · "
            f"referencia media 100 d {_n(e['media100'])} ({_n(dist, 1)}%) · RSI {_n(e['rsi'], 1)}{extra}")


def _mensajes(prev: Optional[Dict], e: Dict) -> Dict[str, str]:
    ref = f"{_n(e['media100'])} (media de 100 días)"
    campos = {}
    if prev is None or prev["tendencia"] != e["tendencia"]:
        prefijo = "" if prev is not None else "Estado actual: "
        if e["tendencia"]:
            campos["ventana"] = f"{prefijo}🟢 COMPRA — tendencia alcista desde {e['tendencia_desde']}"
            campos["como_leerla"] = f"Sigue abierta mientras el cierre diario esté sobre {ref}. Si cierra debajo, pasa a venta."
        else:
            campos["ventana"] = f"{prefijo}🔴 VENTA — tendencia bajista desde {e['tendencia_desde']}"
            campos["como_leerla"] = f"Sigue abierta mientras el cierre diario esté bajo {ref}. Si cierra encima, pasa a compra."
    if prev is not None and prev["caida"] != e["caida"]:
        campos["caida_fuerte"] = ("🟢 COMPRA por caída fuerte: RSI diario bajo 30. Se cierra cuando el RSI vuelva a 50."
                                  if e["caida"] else "Se cerró la ventana de compra por caída (el RSI volvió a 50).")
    elif prev is None and e["caida"]:
        campos["caida_fuerte"] = f"🟢 además ventana de COMPRA por caída fuerte desde {e['caida_desde']} (RSI bajo 30, se cierra en 50)."
    if campos:
        campos.update(precio_cierre=_n(e["cierre"]), precio_referencia=ref, rsi_diario=_n(e["rsi"], 1), nota=NOTA)
    return campos


def _fecha(iso: str) -> str:
    return pd.Timestamp(iso).strftime("%d/%m/%Y")


def _pct(x: float) -> str:
    if round(abs(x), 1) == 0:
        return "0,0%"
    return ("+" if x > 0 else "-") + _n(abs(x), 1) + "%"


def mensaje_md(prev: Optional[Dict], e: Dict) -> str:
    """Mensaje de Telegram (Markdown) para un cambio de ventana o el estado al arrancar."""
    lineas = []
    media, cierre = _n(e["media100"]), _n(e["cierre"])
    if prev is None or prev["tendencia"] != e["tendencia"]:
        if prev is None:
            lineas.append("*Estado actual*")
        if e["tendencia"]:
            lineas.append(f"🟢 *COMPRA* · tendencia alcista desde el {_fecha(e['tendencia_desde'])}")
        else:
            lineas.append(f"🔴 *VENTA* · tendencia bajista desde el {_fecha(e['tendencia_desde'])}")
        lineas += ["",
                   f"Cierre BTC     `{cierre}`",
                   f"Referencia     `{media}` (media de 100 días)",
                   f"Distancia      `{_pct((e['cierre'] / e['media100'] - 1) * 100)}`",
                   f"RSI diario     `{_n(e['rsi'], 1)}`",
                   ""]
        if e["tendencia"]:
            lineas.append(f"Sigue abierta mientras el cierre diario esté sobre `{media}`. "
                          f"Si cierra debajo, pasa a 🔴 *VENTA*.")
        else:
            lineas.append(f"Sigue abierta mientras el cierre diario esté bajo `{media}`. "
                          f"Si cierra encima, pasa a 🟢 *COMPRA*.")
    caida_cambio = prev is not None and prev["caida"] != e["caida"]
    if caida_cambio or (prev is None and e["caida"]):
        if lineas:
            lineas.append("")
        if e["caida"]:
            lineas.append(f"🟢 *COMPRA por caída fuerte* · RSI diario `{_n(e['rsi'], 1)}` (bajo 30)")
            lineas.append("Se cierra cuando el RSI vuelva a 50.")
        else:
            lineas.append(f"⚪ Se cerró la ventana de compra por caída: el RSI volvió a `{_n(e['rsi'], 1)}`.")
        if not (prev is None or prev["tendencia"] != e["tendencia"]):
            lineas.append(f"Cierre BTC `{cierre}`")
    lineas += ["", "_Señal de una regla medida en 2019-2026; no es una recomendación._"]
    return "\n".join(lineas)


def linea_md(e: Dict) -> str:
    """Una línea (Markdown) con el estado del día, para el resumen diario de los bots."""
    ventana = "🟢 COMPRA" if e["tendencia"] else "🔴 VENTA"
    extra = " · 🟢 caída fuerte" if e["caida"] else ""
    return (f"{ventana} desde el {_fecha(e['tendencia_desde'])} · referencia `{_n(e['media100'])}` · "
            f"RSI `{_n(e['rsi'], 1)}`{extra}")


def _emitir(**campos) -> None:
    try:
        from .lifecycle_events import emit_lifecycle_event
        emit_lifecycle_event("btc_ventana", "INFO", **campos)
    except Exception:
        pass


def leer_estado() -> Optional[Dict]:
    try:
        f, _ = _rutas()
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
    except Exception:
        return None


def run_btc_alertas(ahora: Optional[datetime] = None, fuentes: Optional[Dict[str, Callable]] = None) -> Optional[Dict]:
    """Job horario. Devuelve los campos del aviso si avisó, {} si no hubo cambio, None si no corrió."""
    try:
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if not cfg.get("alerta_ventana", True):
            return None
        symbol = cfg.get("symbol", "BTC-USDT")
        ahora = pd.Timestamp(ahora or datetime.now(timezone.utc))
        ahora = ahora.tz_localize("UTC") if ahora.tzinfo is None else ahora.tz_convert("UTC")
        fuentes = fuentes or {}
        from .btc_bots import _velas_bingx, velas_cerradas
        velas_fn = fuentes.get("velas") or (lambda interval, limit: _velas_bingx(symbol, interval, limit))

        prev = leer_estado()
        v = velas_cerradas(velas_fn("1d", VENTANA_VELAS), 24, ahora)
        if len(v) < MIN_VELAS:
            log.warning("btc_alertas: %s velas diarias cerradas; se reintenta", len(v))
            return None
        if prev is not None and prev.get("barra") == str(v.index[-1]):
            return {}
        e = evaluar(v)
        campos = _mensajes(prev, e)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        f_estado, f_log = _rutas()
        tmp = f_estado.with_suffix(".tmp")
        tmp.write_text(json.dumps(e, indent=1), encoding="utf-8")
        os.replace(tmp, f_estado)
        pd.DataFrame([{**e, "aviso": " | ".join(campos.get(k, "") for k in ("ventana", "caida_fuerte")).strip(" |")}]) \
            .to_csv(f_log, mode="a", header=not f_log.exists(), index=False)
        if campos:
            _emitir(**campos, mensaje_md=mensaje_md(prev, e))
        return campos
    except Exception as exc:   # nunca tirar el bot por la alerta
        log.warning("btc_alertas falló: %s", exc)
        return None
