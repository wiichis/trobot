# pkg/cfg_loader.py
#
# Carga la whitelist de símbolos desde la fuente única: best_prod.json
#
import json
import logging
from pathlib import Path
from typing import Any, List
from .settings import BEST_PROD_PATH, DEFAULT_SYMBOLS

log = logging.getLogger("trobot")


def _extract_symbols(obj: Any) -> List[str]:
    """Extrae lista de símbolos desde estructura JSON (lista o dict)."""
    if isinstance(obj, list):
        symbols = []
        for item in obj:
            if isinstance(item, str):
                symbols.append(item)
            elif isinstance(item, dict):
                sym = item.get("symbol")
                if isinstance(sym, str):
                    symbols.append(sym)
        return symbols

    if isinstance(obj, dict):
        for key in ("winners", "symbols", "whitelist"):
            val = obj.get(key)
            if isinstance(val, list):
                return _extract_symbols(val)
            if isinstance(val, dict):
                return [str(k) for k in val.keys()]

    return []


def load_best_symbols() -> List[str]:
    """Carga la whitelist desde best_prod.json (fuente única)."""
    try:
        with BEST_PROD_PATH.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        syms = _extract_symbols(cfg)
        if syms:
            log.info("✅ Whitelist desde best_prod.json (%d)", len(syms))
            return syms
        log.warning("best_prod.json sin símbolos válidos; usando defaults.")
    except FileNotFoundError:
        log.warning("best_prod.json no encontrado. Usando DEFAULT_SYMBOLS.")
    except Exception as e:
        log.error("Error leyendo best_prod.json: %s", e)

    return DEFAULT_SYMBOLS
