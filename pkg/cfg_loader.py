# pkg/cfg_loader.py
import json
import logging
from pathlib import Path
from typing import Any, List
from .settings import BEST_PROD_PATH, DEFAULT_SYMBOLS

log = logging.getLogger("trobot")
SYMBOLS_PATH = Path(__file__).resolve().parent / "symbols.json"

def _extract_symbols(obj: Any) -> List[str]:
    """
    Devuelve una lista de símbolos desde distintas estructuras válidas:
    - Lista raíz: ["XRP-USDT", "AVAX-USDT", ...]
    - Dict con claves: winners | symbols | whitelist
      * Si es lista -> usarla
      * Si es dict  -> usar keys()
    """
    # Caso 1: el JSON completo es una lista (puede ser de strings o dicts con 'symbol')
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

    # Caso 2: el JSON es un dict con alguna de las claves conocidas
    if isinstance(obj, dict):
        for key in ("winners", "symbols", "whitelist"):
            val = obj.get(key)
            if isinstance(val, list):
                out = []
                for item in val:
                    if isinstance(item, str):
                        out.append(item)
                    elif isinstance(item, dict):
                        sym = item.get("symbol")
                        if isinstance(sym, str):
                            out.append(sym)
                return out
            if isinstance(val, dict):
                return [str(k) for k in val.keys()]

    # Si no se pudo extraer nada, devolver lista vacía
    return []

def load_best_symbols() -> List[str]:
    """
    Carga la whitelist de símbolos desde pkg/symbols.json (si existe) o pkg/best_prod.json.
    Si no existe o es inválido, regresa DEFAULT_SYMBOLS.
    """
    try:
        with SYMBOLS_PATH.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        syms = _extract_symbols(cfg)
        if syms:
            log.info("✅ Whitelist derivada de symbols.json (%d)", len(syms))
            return syms
        log.warning("Config symbols.json sin lista de símbolos usable; se intenta best_prod.json.")
    except FileNotFoundError:
        pass
    except Exception as e:
        log.error("Config inválida en symbols.json: %s", e)

    try:
        with BEST_PROD_PATH.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        syms = _extract_symbols(cfg)
        if syms:
            log.info("✅ Whitelist derivada de best_prod.json (%d)", len(syms))
            return syms
        log.warning("Config best_prod.json sin lista de símbolos usable; se usan defaults.")
    except FileNotFoundError:
        log.warning("No se encontró best_prod.json. Usando DEFAULT_SYMBOLS (%d).", len(DEFAULT_SYMBOLS))
    except Exception as e:
        log.error("Config inválida en best_prod.json: %s", e)

    return DEFAULT_SYMBOLS
