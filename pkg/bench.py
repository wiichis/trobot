"""Banca de pares: los que NO operan pero se siguen simulando hacia adelante (01/10/2026).

Diseño acordado el 01/10 (ver CLAUDE.md, "Banca de pares"):

- `pkg/best_prod.json` sigue siendo la única fuente de verdad de lo que OPERA.
- `pkg/bench.json` lista los pares en banca. Nada del camino de órdenes la lee: los pares
  en banca no entran a `currencies_list()`, ni a `cripto_price_5m.csv`, ni a
  `indicadores.csv`. Sólo se les bajan velas a un archivo propio, y se simulan con el
  parity una vez por semana (`scripts/bench_shadow.py`).
- Las transiciones (activo → banca, banca → activo, banca → fuera) NO son automáticas:
  el script las propone y la decisión la toma el usuario.

Estados de una entrada de `bench.json`:
- "banca": tiene params y se simula hacia adelante desde `desde`.
- "observacion": candidato sin params validados todavía; sólo se le bajan velas.

Por qué los criterios son lentos y exigen muestra (medido el 01/10): el resultado de un
par NO persiste de un mes al siguiente (Spearman medio −0,07 en el sim, ~0,07 en real).
Un interruptor "activo si el mes pasado fue positivo" habría hecho +3,09 contra +18,63
de dejar todos, y los apagados habrían hecho +15,54. Por eso entrar o salir exige
evidencia acumulada, no un mes.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PKG_DIR = Path(__file__).resolve().parent

_env_bench = os.getenv("TROBOT_BENCH_PATH", "").strip()
BENCH_PATH = Path(_env_bench).expanduser() if _env_bench else PKG_DIR / "bench.json"

ESTADOS = ("banca", "observacion")

# --- Criterio activo → banca -------------------------------------------------------
# Negativo en el sim en al menos 3 de las 4 ventanas (30/60/90/120d) Y negativo en real
# en al menos 3 de los últimos 4 bloques de 14 días. Las dos cosas: el sim solo no
# alcanza (sesgo y ruido) y el real solo tiene ~2 cierres por par y por bloque.
MIN_VENTANAS_SIM_NEGATIVAS = 3
MIN_BLOQUES_REALES_NEGATIVOS = 3

# --- Criterio banca → activo --------------------------------------------------------
# Haber pasado el protocolo de backtest (4 ventanas + falsación) y además acumular
# trades simulados HACIA ADELANTE (con velas que no existían al elegir los params).
MIN_TRADES_FORWARD = 15
# El parity es optimista: sobre las mismas posiciones da ~0,13 USDT más de bruto por
# posición de ~37 USDT que el real (cruce del 23/09). Para que el real esperado sea
# positivo, el forward simulado tiene que superar ese sesgo.
SESGO_SIM_POR_TRADE_PCT = 0.0035

# --- Criterio banca → fuera ---------------------------------------------------------
# Dos ciclos mensuales de re-optimización sin llegar a activarse.
DIAS_MAX_EN_BANCA = 60


def load_bench(path: Optional[Path] = None) -> List[Dict]:
    """Lee la banca. Si el archivo no existe, la banca está vacía.

    Un archivo corrupto levanta excepción a propósito: callar ese error haría que la banca
    "desaparezca" sin aviso. Los llamadores del camino vivo (el job de velas) ya la
    atrapan para no tirar el bot.
    """
    p = Path(path) if path else BENCH_PATH
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{p}: se esperaba una lista de pares")
    out = []
    for row in data:
        if not isinstance(row, dict):
            continue
        sym = str(row.get("symbol", "")).upper().strip()
        if not sym:
            continue
        estado = str(row.get("estado", "observacion")).strip().lower()
        if estado not in ESTADOS:
            raise ValueError(f"{p}: estado desconocido '{estado}' para {sym}")
        row = dict(row)
        row["symbol"] = sym
        row["estado"] = estado
        out.append(row)
    return out


def bench_symbols(path: Optional[Path] = None, excluir: Iterable[str] = ()) -> List[str]:
    """Símbolos de la banca (cualquier estado), sin los que ya operan."""
    excl = {str(s).upper().strip() for s in excluir}
    return sorted({r["symbol"] for r in load_bench(path)} - excl)


def _n_negativos(valores: Iterable[Optional[float]]) -> Tuple[int, int]:
    vals = [float(v) for v in valores if v is not None]
    return sum(1 for v in vals if v < 0), len(vals)


def debe_ir_a_banca(sim_ventanas: Dict[str, float],
                    bloques_reales: Sequence[float]) -> Tuple[bool, str]:
    """¿Un par ACTIVO debería pasar a la banca?

    `sim_ventanas`: PnL simulado por ventana, p.ej. {'30d': -9.0, '60d': -10.3, ...}.
    `bloques_reales`: PnL real de los últimos bloques de 14 días, del más viejo al más
    nuevo. Se usan los últimos 4. Un bloque en 0 (sin operar) NO cuenta como negativo.
    """
    neg_sim, n_sim = _n_negativos(sim_ventanas.values())
    ultimos = list(bloques_reales)[-4:]
    neg_real, n_real = _n_negativos(ultimos)
    ok_sim = n_sim >= 4 and neg_sim >= MIN_VENTANAS_SIM_NEGATIVAS
    ok_real = n_real >= 4 and neg_real >= MIN_BLOQUES_REALES_NEGATIVOS
    motivo = (f"sim negativo en {neg_sim}/{n_sim} ventanas, "
              f"real negativo en {neg_real}/{n_real} bloques de 14 días")
    return ok_sim and ok_real, motivo


def puede_activarse(paso_protocolo: bool, trades_forward: int,
                    pnl_forward_por_trade_pct: float) -> Tuple[bool, str]:
    """¿Un par en BANCA puede volver a operar?

    `pnl_forward_por_trade_pct`: PnL simulado hacia adelante por trade, como fracción del
    notional de la posición (0,004 = +0,4%).
    """
    if not paso_protocolo:
        return False, "no pasó el protocolo de backtest (4 ventanas + falsación)"
    if trades_forward < MIN_TRADES_FORWARD:
        return False, f"{trades_forward} trades forward, faltan {MIN_TRADES_FORWARD - trades_forward}"
    margen = pnl_forward_por_trade_pct - SESGO_SIM_POR_TRADE_PCT
    if margen <= 0:
        return False, (f"forward {pnl_forward_por_trade_pct:+.2%}/trade no supera el sesgo "
                       f"del sim ({SESGO_SIM_POR_TRADE_PCT:.2%})")
    return True, f"forward {pnl_forward_por_trade_pct:+.2%}/trade en {trades_forward} trades"


def _a_fecha(x) -> date:
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    return datetime.fromisoformat(str(x)[:10]).date()


def vencio_en_banca(desde, hoy=None) -> bool:
    """¿Lleva más de DIAS_MAX_EN_BANCA días en banca? (sólo aplica si no puede activarse)."""
    hoy = _a_fecha(hoy) if hoy is not None else date.today()
    return (hoy - _a_fecha(desde)).days > DIAS_MAX_EN_BANCA
