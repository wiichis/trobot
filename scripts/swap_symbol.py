#!/usr/bin/env python3
"""
Herramienta CLI para reemplazar una moneda de la whitelist por otra.

Acciones principales:
1. Actualiza pkg/best_prod.json (y opcionalmente los params del nuevo símbolo).
2. Sincroniza DEFAULT_SYMBOLS en pkg/settings.py (fallback).
3. Purga el histórico del símbolo removido en los CSV relevantes de ./archivos.
4. (Opcional) dispara la descarga inicial de velas para que el símbolo nuevo
   comience a generar histórico inmediatamente.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = REPO_ROOT / "pkg"
ARCHIVOS_DIR = REPO_ROOT / "archivos"

BEST_PROD_PATH = PKG_DIR / "best_prod.json"
SETTINGS_PATH = PKG_DIR / "settings.py"
SYMBOLS_PATH = PKG_DIR / "symbols.json"

DEFAULT_PURGE_TARGETS = [
    ARCHIVOS_DIR / "cripto_price_5m.csv",
    ARCHIVOS_DIR / "cripto_price_5m_long.csv",
    ARCHIVOS_DIR / "cripto_price_30m.csv",
    ARCHIVOS_DIR / "indicadores.csv",
]


def _normalize_symbol(symbol: str) -> str:
    """Normaliza el símbolo a MAYÚSCULAS y agrega sufijo -USDT si falta un quote."""
    sym = str(symbol or "").strip().upper()
    if not sym:
        raise ValueError("Símbolo vacío.")
    if "-" not in sym:
        sym = f"{sym}-USDT"
    return sym


def _load_params(args: argparse.Namespace) -> Dict[str, Any]:
    if args.params_file and args.params_json:
        raise ValueError("Usa solo uno de --params-file o --params-json.")
    payload: Dict[str, Any] = {}
    if args.params_file:
        data = json.loads(Path(args.params_file).read_text(encoding="utf-8"))
    elif args.params_json:
        data = json.loads(args.params_json)
    else:
        return payload

    if isinstance(data, dict):
        if "params" in data and "symbol" in data:
            candidate = data.get("params") or {}
        else:
            candidate = data
    else:
        raise ValueError("Los parámetros deben ser un objeto JSON.")

    if not isinstance(candidate, dict):
        raise ValueError("Los parámetros deben ser un dict plano.")
    return candidate


def _symbol_from_entry(entry: Any) -> str:
    if isinstance(entry, str):
        return entry.upper()
    if isinstance(entry, dict):
        sym = entry.get("symbol") or entry.get("Symbol")
        if isinstance(sym, str):
            return sym.upper()
    return ""

def _extract_symbols_payload(data: Any) -> List[str]:
    if isinstance(data, list):
        return [str(item).upper() for item in data if isinstance(item, str)]
    if isinstance(data, dict):
        for key in ("symbols", "whitelist", "winners"):
            val = data.get(key)
            if isinstance(val, list):
                return [str(item).upper() for item in val if isinstance(item, str)]
            if isinstance(val, dict):
                return [str(k).upper() for k in val.keys()]
    return []


def update_best_prod(remove_sym: str, add_sym: str, params: Dict[str, Any], dry_run: bool) -> Tuple[int, int]:
    if not BEST_PROD_PATH.exists():
        raise FileNotFoundError(f"No existe {BEST_PROD_PATH}")

    data = json.loads(BEST_PROD_PATH.read_text(encoding="utf-8"))
    removed = 0

    if isinstance(data, list):
        list_kind = "dict" if any(isinstance(item, dict) for item in data) else "str"
        new_entries = []
        insert_idx = None
        for item in data:
            sym = _symbol_from_entry(item)
            if sym == remove_sym or sym == add_sym:
                removed += int(sym == remove_sym)
                if insert_idx is None and sym == remove_sym:
                    insert_idx = len(new_entries)
                continue
            new_entries.append(item)

        if insert_idx is None:
            insert_idx = len(new_entries)

        if list_kind == "str":
            new_entry: Any = add_sym
        else:
            new_entry = {"symbol": add_sym}
            if params:
                new_entry["params"] = params

        new_entries.insert(insert_idx, new_entry)
        final_data = new_entries

    elif isinstance(data, dict):
        new_map = {}
        for key, value in data.items():
            sym = str(key).upper()
            if sym == remove_sym or sym == add_sym:
                removed += int(sym == remove_sym)
                continue
            new_map[key] = value

        new_map[add_sym] = params or {}
        final_data = new_map
    else:
        raise TypeError("best_prod.json debe ser lista o dict.")

    if not dry_run:
        BEST_PROD_PATH.write_text(
            json.dumps(final_data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return removed, len(final_data)


def _extract_default_symbols(source: str) -> List[str]:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "DEFAULT_SYMBOLS":
                    values = ast.literal_eval(node.value)
                    return [str(v).upper() for v in values]
    raise ValueError("No se encontró DEFAULT_SYMBOLS en settings.py")


def update_default_symbols(remove_sym: str, add_sym: str, dry_run: bool) -> List[str]:
    text = SETTINGS_PATH.read_text(encoding="utf-8")
    symbols = _extract_default_symbols(text)

    idx = None
    for i, sym in enumerate(symbols):
        if sym == remove_sym:
            idx = i
            break
    if idx is not None:
        symbols.pop(idx)
    # Evita duplicados
    symbols = [s for s in symbols if s != add_sym]
    insert_pos = idx if idx is not None else len(symbols)
    symbols.insert(insert_pos, add_sym)

    if not dry_run:
        new_block = ",\n    ".join(f'"{sym}"' for sym in symbols)
        replacement = f"DEFAULT_SYMBOLS = [\n    {new_block}\n]"
        pattern = re.compile(r"DEFAULT_SYMBOLS\s*=\s*\[[^\]]*\]", re.DOTALL)
        updated_text = pattern.sub(replacement, text, count=1)
        SETTINGS_PATH.write_text(updated_text, encoding="utf-8")
    return symbols

def update_symbols_file(remove_sym: str, add_sym: str, dry_run: bool) -> List[str]:
    symbols: List[str] = []
    if SYMBOLS_PATH.exists():
        try:
            data = json.loads(SYMBOLS_PATH.read_text(encoding="utf-8"))
            symbols = _extract_symbols_payload(data)
        except Exception as exc:
            print(f"⚠️  No se pudo leer symbols.json: {exc}", file=sys.stderr)
            symbols = []
    if not symbols:
        symbols = _extract_default_symbols(SETTINGS_PATH.read_text(encoding="utf-8"))

    idx = None
    for i, sym in enumerate(symbols):
        if sym == remove_sym:
            idx = i
            break
    if idx is not None:
        symbols.pop(idx)
    symbols = [s for s in symbols if s != add_sym]
    insert_pos = idx if idx is not None else len(symbols)
    symbols.insert(insert_pos, add_sym)

    if not dry_run:
        SYMBOLS_PATH.write_text(json.dumps(symbols, indent=2) + "\n", encoding="utf-8")
    return symbols


def purge_history(remove_sym: str, dry_run: bool) -> Dict[str, int]:
    stats: Dict[str, int] = {}
    for csv_path in DEFAULT_PURGE_TARGETS:
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            stats[csv_path.name] = -1
            print(f"⚠️  No se pudo leer {csv_path.name}: {exc}", file=sys.stderr)
            continue

        if "symbol" not in df.columns:
            continue
        mask = df["symbol"].astype(str).str.upper() == remove_sym
        removed_rows = int(mask.sum())
        if not removed_rows:
            continue
        stats[csv_path.name] = removed_rows
        if not dry_run:
            df_filtered = df.loc[~mask].copy()
            df_filtered.to_csv(csv_path, index=False)
    return stats


def fetch_new_data(skip_download: bool) -> str:
    if skip_download:
        return "omitido (--skip-download)"
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from pkg import price_bingx_5m

        price_bingx_5m.price_bingx_5m()
        price_bingx_5m.actualizar_long_ultimas_12h()
        return "ejecutado"
    except Exception as exc:
        return f"falló: {exc}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reemplaza un símbolo existente por uno nuevo y limpia históricos."
    )
    parser.add_argument("--remove", required=True, help="Símbolo a quitar (ej. XRP-USDT)")
    parser.add_argument("--add", required=True, help="Símbolo a agregar (ej. SOL-USDT)")
    parser.add_argument(
        "--params-file",
        help="Archivo JSON con los parámetros del símbolo nuevo (solo el dict de params).",
    )
    parser.add_argument(
        "--params-json",
        help='Cadena JSON inline con los parámetros del nuevo símbolo. Ej: \'{"tp": 0.01}\'',
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="No disparar la descarga inicial de velas al final del proceso.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Muestra los cambios que se harían pero sin escribir archivos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    remove_sym = _normalize_symbol(args.remove)
    add_sym = _normalize_symbol(args.add)
    if remove_sym == add_sym:
        raise ValueError("El símbolo nuevo debe ser diferente al que se remueve.")

    params = _load_params(args)
    dry = args.dry_run

    removed_count, total_entries = update_best_prod(remove_sym, add_sym, params, dry)
    defaults = update_default_symbols(remove_sym, add_sym, dry)
    symbols_list = update_symbols_file(remove_sym, add_sym, dry)
    purge_stats = purge_history(remove_sym, dry)
    fetch_status = fetch_new_data(args.skip_download or dry)

    print("=== Resumen del swap ===")
    print(f"best_prod.json: -{removed_count} entradas, total ahora {total_entries}.")
    print(f"DEFAULT_SYMBOLS: {len(defaults)} símbolos -> {', '.join(defaults)}")
    print(f"symbols.json: {len(symbols_list)} símbolos -> {', '.join(symbols_list)}")
    if purge_stats:
        for fname, count in purge_stats.items():
            if count >= 0:
                print(f"{fname}: {count} filas purgadas.")
            else:
                print(f"{fname}: no se pudo purgar (ver logs).")
    else:
        print("Históricos: sin filas que purgar.")
    print(f"Captura inicial: {fetch_status}")


if __name__ == "__main__":
    main()
