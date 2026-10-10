"""Modo ligero (10/10/2026): saltar `update_indicators()` cuando el bot está ocioso.

Condiciones (TODAS): el modo está habilitado, `entries.enabled` es false, la última foto de
órdenes pendientes (`order_pending_prev.csv`, que `colocando_ordenes` actualiza cada 5 min) está
vacía y `tp_stage_state.csv` no tiene ninguna fila con etapa distinta de `none` (o, si la tiene,
el exchange confirma que ese par está plano). Ante cualquier
duda (archivo ilegible, columna ausente) NO se salta: el costo de saltar mal es un SL calculado
con indicadores viejos, el de no saltar es sólo CPU.

Lo único que se omite es el recálculo de indicadores. Se sigue corriendo `colocando_ordenes` entero
(registro de órdenes y transiciones, confirmación de fills) y `colocando_TK_SL` cada 50 s.
Dos redes de seguridad: el primer ciclo tras arrancar siempre calcula (así sigue saliendo el aviso
`entries_paused`), y cada `max_saltos_seguidos`+1 ciclos se calcula igual (indicadores con ≤1 h).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

log = logging.getLogger("trobot")

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "light_mode.json"
PENDING_PREV = BASE_DIR / "archivos" / "order_pending_prev.csv"
TP_STATE = BASE_DIR / "archivos" / "tp_stage_state.csv"

_saltos_seguidos = 0
_primer_ciclo = True
_ultimo_estado: Optional[bool] = None


def _config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"enabled": False}


def _sin_ordenes() -> bool:
    if not PENDING_PREV.exists():
        return False
    try:
        d = pd.read_csv(PENDING_PREV, dtype=str)
        return "orderId" in d.columns and len(d) == 0
    except pd.errors.EmptyDataError:
        return True                     # archivo vacío = sin órdenes
    except Exception:
        return False


def _posicion_abierta(symbol: str) -> Optional[bool]:
    """Lectura ESTRICTA de la posición en el exchange. None si no se pudo leer con certeza:
    `total_positions` devuelve lo mismo para 'no hay' que para 'no pude leer', y aquí un
    falso 'plana' sería justo el error caro."""
    try:
        import pkg.bingx
        r = json.loads(pkg.bingx.perpetual_swap_positions(symbol))
        if r.get("code") != 0 or not isinstance(r.get("data"), list):
            return None
        return any(abs(float(x.get("positionAmt", 0) or 0)) > 0 for x in r["data"])
    except Exception:
        return None


def _sin_posiciones_en_estado(posicion_abierta: Optional[Callable[[str], Optional[bool]]] = None) -> bool:
    """Sin filas de TP activas. Una fila activa pero huérfana (la posición ya cerró y, con las
    entradas pausadas, nada la limpia: `clear_tp_state` sólo corre en cierres por SL) se descarta
    si el exchange confirma que el par está plano."""
    if not TP_STATE.exists():
        return True
    try:
        d = pd.read_csv(TP_STATE, dtype=str)
        if "tp_stage" not in d.columns or "symbol" not in d.columns:
            return False
        activas = d[d["tp_stage"].fillna("none").str.strip().str.lower() != "none"]
        if activas.empty:
            return True
        consulta = posicion_abierta or _posicion_abierta
        return all(consulta(str(sym).strip()) is False for sym in activas["symbol"])
    except pd.errors.EmptyDataError:
        return True
    except Exception:
        return False


def skip_indicators(entradas_habilitadas: Optional[Callable[[], bool]] = None,
                    posicion_abierta: Optional[Callable[[str], Optional[bool]]] = None) -> bool:
    """True si este ciclo puede omitir el cálculo de indicadores. Nunca levanta excepción."""
    global _saltos_seguidos, _primer_ciclo, _ultimo_estado
    try:
        cfg = _config()
        if not cfg.get("enabled", False):
            return False
        if entradas_habilitadas is None:
            from .live_runtime_config import are_entries_enabled as entradas_habilitadas
        ocioso = (not entradas_habilitadas()) and _sin_ordenes() and _sin_posiciones_en_estado(posicion_abierta)
        if ocioso != _ultimo_estado:
            log.info("light_mode: %s", "ocioso, se omite el recálculo de indicadores" if ocioso else "activo, se calculan indicadores")
            _ultimo_estado = ocioso
        if not ocioso:
            _saltos_seguidos = 0
            return False
        if _primer_ciclo:
            _primer_ciclo = False
            return False
        if _saltos_seguidos >= int(cfg.get("max_saltos_seguidos", 11)):
            _saltos_seguidos = 0
            return False
        _saltos_seguidos += 1
        return True
    except Exception as exc:
        log.warning("light_mode falló (%s); se calcula normal", exc)
        return False
