# -*- coding: utf-8 -*-
"""
backtesting.py — TRobot (versión simple-realista)

Objetivo: motor de backtesting compacto y realista pero simple, alineado con
`archivos/instrucciones_backtesting.md`.

Características principales implementadas:
- Base 5m (sin marco 30m).
- ATR(14) absoluto y ATR%_5m = ATR/close para slippage y filtros.
- Filtro horario: no abrir nuevas posiciones ±30 min de 00:00, 08:00, 16:00 UTC.
- Slippage dinámico simple: max(0.02%), 0.25 × ATR%_5m, tope 0.06%.
- Comisiones taker 0.05% por lado. Funding 0.01%/8h prorrateado por barra.
- Position sizing: min(10% equity, (0.5% equity) / ATR(14)).
- Entradas: EMA cruz + RSI + ADX + cierre rompe high/low reciente.
- Salidas: TP fijo (0.5–3%) o k*ATR o sin TP; SL por ATR (fijo→trailing) / trailing desde el inicio / % fijo.
- Salvaguardas: una ejecución por vela; filtro por ATR%.
- Validación: split 70/30 con embargo de 1 hora.
- Métricas de salida + cost_ratio.

Dependencias: pandas, numpy.
No requiere TA‑Lib.
"""
from __future__ import annotations
import os
import json
import math
import argparse
import re
import itertools
import sys
import atexit
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

#
# ==========================
# Rutas por defecto
# ==========================
DEFAULT_DATA_TEMPLATE = 'archivos/cripto_price_5m_long.csv'  # CSV único con múltiples símbolos
DEFAULT_OUT_DIR = 'archivos/backtesting'
SYMBOLS_FILE = os.path.join(os.path.dirname(__file__), 'symbols.json')
TP_LADDER_FACTORS = (0.6, 1.0, 1.6)
TP_SPLITS_DEFAULT = (0.40, 0.40, 0.20)
TP1_DEFAULT_FACTOR = 0.70
FUNDING_EVENTS_CACHE: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}

# ==========================
# Configuración SIMPLE-RUN (ejecuta con un solo comando, sin flags)
# ==========================
# Modo de uso:
# 1) Ajusta SIMPLE_RUN['MODE'] y demás campos.
# 2) Ejecuta: `p3 pkg/backtesting.py`  (sin argumentos)
#    El script inyectará los flags apropiados automáticamente.
# MODE disponibles:
#   - 'SWEEP_PRESET'  : usa un preset interno (ej. 'core_winners')
#   - 'SWEEP_RANGES'  : usa rangos definidos en SIMPLE_RUN['RANGES'] (se guarda un JSON temporal)
#   - 'SINGLE'        : corre con parámetros fijos en SIMPLE_RUN['SINGLE_PARAMS']
#   - 'BEST_FROM_FILE': usa --load_best (y opcionalmente winners_from_best)
SIMPLE_RUN: Dict[str, object] = {
    'ENABLE': True,
    'MODE': 'SWEEP_RANGES',
    'SYMBOLS': 'auto',
    'DATA_TEMPLATE': DEFAULT_DATA_TEMPLATE,
    'CAPITAL': 300.0,
    'OUT_DIR': DEFAULT_OUT_DIR,
    'RANK_BY': 'pnl_net',
    'SEARCH_MODE': 'random',        # 'grid' o 'random'
    'N_TRIALS': 320,                 # combos aleatorios por símbolo (si SEARCH_MODE='random')
    'RANDOM_SEED': 123,
    'SECOND_PASS': True,
    'SECOND_TOPK': 3,
    'FILTERS': {'min_trades': 3, 'min_winrate': 0.0, 'max_cost_ratio': 0.92, 'max_dd': 0.20},
    'EXPORT_BEST': 'best_prod.json',
    'RANGES': {
        'tp': [0.008, 0.010, 0.012, 0.013, 0.015, 0.017, 0.019, 0.022, 0.025],
        'tp_mode': ['fixed', 'atrx', 'none'],
        'tp_atr_mult': [0.0, 0.4, 0.6, 0.8, 1.0],
        'ema_fast': [10, 13, 20, 21, 34],
        'ema_slow': [55, 72, 89],
        'rsi_buy': [54, 56, 58, 60, 62],
        'rsi_sell': [40, 42, 44, 46, 48],
        'adx_min': [12, 15, 18, 22],
        'min_atr_pct': [0.0010, 0.0012, 0.0015, 0.0020],
        'max_atr_pct': [0.010, 0.012, 0.015, 0.020],
        'atr_mult': [1.4, 1.6, 1.8, 2.0, 2.2],
        'sl_mode': ['atr_then_trailing', 'atr_trailing_only', 'percent'],
        'sl_pct': [0.010, 0.015, 0.020],
        'be_trigger': [0.0035, 0.0045, 0.0055],
        'cooldown': [5, 10, 15],
        'logic': ['any', 'strict'],
        'hhll_lookback': [8, 10, 12, 14, 18],
        'time_exit_bars': [24, 30, 36, 48, 60],
        'max_dist_emaslow': [0.008, 0.010, 0.012],
        'fresh_cross_max_bars': [4, 6, 8],
        'require_rsi_cross': [True, False],
        # Nuevos endurecedores:
        'min_ema_spread': [0.0008, 0.0010, 0.0012],   # separación mínima EMA_f/EMA_s
        'require_close_vs_emas': [True, False],        # permite escenarios más flexibles
        'min_vol_ratio': [1.00, 1.05, 1.10, 1.15],     # volumen relativo al MA
        'vol_ma_len': [20, 30, 40],                   # lookback MA de volumen
        'adx_slope_len': [3, 4],                       # barras para pendiente ADX
        'adx_slope_min': [0.2, 0.3, 0.5, 0.7],        # pendiente mínima ADX
        'fresh_breakout_only': [False, True],          # controla rupturas frescas
    },
    'LOAD_BEST_FILE': None,
    'WINNERS_FROM_BEST': False,
    'BEST_THRESHOLD': 0.0,
    'PORTFOLIO_EVAL': True,
}

def _build_simple_argv(cfg: Dict[str, object]) -> list:
    """Construye sys.argv para ejecutar el script sin flags según SIMPLE_RUN."""
    argv = [sys.argv[0]]

    def add(flag: str, val):
        if val is None:
            return
        argv.append(flag)
        argv.append(str(val))

    # Base
    add('--symbols', cfg.get('SYMBOLS', 'auto'))
    add('--data_template', cfg.get('DATA_TEMPLATE', DEFAULT_DATA_TEMPLATE))
    add('--capital', cfg.get('CAPITAL', 1000.0))
    add('--out_dir', cfg.get('OUT_DIR', DEFAULT_OUT_DIR))
    add('--rank_by', cfg.get('RANK_BY', 'pnl_net'))
    filters = cfg.get('FILTERS') or {}
    if filters.get('min_trades'):
        add('--min_trades', filters['min_trades'])
    if (filters.get('min_winrate') or 0) > 0:
        add('--min_winrate', filters['min_winrate'])
    if filters.get('max_cost_ratio') is not None:
        add('--max_cost_ratio', filters['max_cost_ratio'])
    if filters.get('max_dd') is not None:
        add('--max_dd', filters['max_dd'])
    if cfg.get('EXPORT_BEST'):
        add('--export_best', cfg.get('EXPORT_BEST'))

    # Añadir flags de búsqueda random/grid si están presentes
    if cfg.get('SEARCH_MODE'):
        add('--search_mode', cfg.get('SEARCH_MODE'))
    if cfg.get('N_TRIALS') is not None:
        add('--n_trials', cfg.get('N_TRIALS'))
    if cfg.get('RANDOM_SEED') is not None:
        add('--random_seed', cfg.get('RANDOM_SEED'))
    # Propagar flags de segunda pasada si están presentes
    if cfg.get('SECOND_PASS'):
        argv.append('--second_pass')
    if cfg.get('SECOND_TOPK') is not None:
        add('--second_topk', cfg.get('SECOND_TOPK'))
    if cfg.get('PORTFOLIO_EVAL'):
        argv.append('--portfolio_eval_best')

    mode = cfg.get('MODE', 'SWEEP_RANGES')
    if mode == 'SWEEP_RANGES':
        ranges = cfg.get('RANGES') or {}
        out_dir = cfg.get('OUT_DIR', DEFAULT_OUT_DIR)
        os.makedirs(out_dir, exist_ok=True)
        sweep_path = os.path.join(out_dir, 'simple_sweep.json')
        with open(sweep_path, 'w') as f:
            json.dump(ranges, f, indent=2)
        add('--sweep', sweep_path)
    elif mode == 'SINGLE':
        sp = cfg.get('SINGLE_PARAMS') or {}
        symbols = sp.get('symbol_list')
        if symbols:
            try:
                idx = argv.index('--symbols')
                argv[idx + 1] = symbols
            except ValueError:
                add('--symbols', symbols)
        for k in ['tp','tp_mode','tp_atr_mult','atr_mult','sl_mode','sl_pct',
                  'ema_fast','ema_slow','rsi_buy','rsi_sell','adx_min',
                  'min_atr_pct','max_atr_pct','be_trigger','cooldown','logic']:
            if k in sp and sp[k] is not None:
                add('--' + k, sp[k])
    # 'BEST_FROM_FILE' branch omitted as not needed for minimal config

    return argv

# ==========================
# Normalización de archivo 'best' exportado
# ==========================
def _normalize_best_file(path_in: str, save_to: str = os.path.join('pkg', 'best_prod.json')) -> None:
    """
    Garantiza que el JSON de 'best' tenga el esquema:
        [ {"symbol": "AVAX-USDT", "params": {...}}, ... ]
    Soporta entradas como lista de strings, dict por símbolo o lista de dicts heterogénea.
    """
    try:
        if not os.path.exists(path_in):
            return
        with open(path_in, 'r') as f:
            data = json.load(f)
    except Exception:
        return

    out = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                out.append({'symbol': item.upper(), 'params': {}})
            elif isinstance(item, dict):
                # caso preferido: {"symbol": "X", "params": {...}}
                if 'symbol' in item and 'params' in item:
                    sym = str(item.get('symbol', '')).upper()
                    params = item.get('params') or {}
                    out.append({'symbol': sym, 'params': params if isinstance(params, dict) else {}})
                else:
                    # fallback: dict con claves de params + 'symbol'
                    sym = str(item.get('symbol', '')).upper()
                    if sym:
                        params = {k: v for k, v in item.items() if k.lower() != 'symbol'}
                        out.append({'symbol': sym, 'params': params})
    elif isinstance(data, dict):
        # {"AVAX-USDT": {...}, "BNB-USDT": {...}}
        for sym, params in data.items():
            out.append({'symbol': str(sym).upper(), 'params': params if isinstance(params, dict) else {}})
    else:
        return

    try:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        with open(save_to, 'w') as f:
            json.dump(out, f, indent=2)
    except Exception:
        # en caso de fallo de escritura, no bloquear ejecución
        pass


def _post_run_normalize_best():
    """
    Al terminar la ejecución, intenta normalizar el archivo de 'best' sin asumir
    exactamente dónde lo escribió el runner.
    """
    candidates = set()
    # lo que el usuario configuró
    cfg_path = SIMPLE_RUN.get('EXPORT_BEST') or 'best_prod.json'
    candidates.add(cfg_path)
    # posible ubicación en out_dir
    out_dir = SIMPLE_RUN.get('OUT_DIR', DEFAULT_OUT_DIR) or DEFAULT_OUT_DIR
    candidates.add(os.path.join(out_dir, os.path.basename(cfg_path)))
    # fallback por defecto
    candidates.add(os.path.join(out_dir, 'best_prod.json'))
    for p in list(candidates):
        try:
            _normalize_best_file(p, os.path.join('pkg', 'best_prod.json'))
        except Exception:
            continue

atexit.register(_post_run_normalize_best)

# ==========================
# Funding dinámico (opcional)
# ==========================

def _load_funding_events_csv(path: Optional[str]) -> Dict[str, List[Tuple[int, float]]]:
    """Carga un CSV con columnas [symbol, time/date, rate] y retorna eventos por símbolo.

    El CSV debe contener al menos:
        - symbol (o pair)
        - time|timestamp|date (UTC). Si es numérico, se asume ms.
        - rate|funding_rate|funding

    Devuelve dict {SYMBOL: [(ts_ms, rate), ...]} ordenado temporalmente.
    """
    if not path:
        return {}
    abspath = os.path.abspath(path)
    cached = FUNDING_EVENTS_CACHE.get(abspath)
    if cached is not None:
        return cached

    result: Dict[str, List[Tuple[int, float]]] = {}
    if not os.path.exists(abspath):
        print(f"[FUNDING] Archivo {abspath} no encontrado; se usará tasa fija.")
        FUNDING_EVENTS_CACHE[abspath] = result
        return result

    try:
        df = pd.read_csv(abspath)
    except Exception as exc:
        print(f"[FUNDING][WARN] No se pudo leer {abspath}: {exc}")
        FUNDING_EVENTS_CACHE[abspath] = result
        return result

    if df.empty:
        FUNDING_EVENTS_CACHE[abspath] = result
        return result

    cols = {c.lower(): c for c in df.columns}
    sym_col = cols.get('symbol') or cols.get('pair')
    time_col = cols.get('time') or cols.get('timestamp') or cols.get('date')
    rate_col = cols.get('rate') or cols.get('funding_rate') or cols.get('funding')

    if not sym_col or not time_col or not rate_col:
        print(f"[FUNDING][WARN] CSV {abspath} no tiene columnas estándar (symbol,time,rate). Campos={list(df.columns)}")
        FUNDING_EVENTS_CACHE[abspath] = result
        return result

    work = df[[sym_col, time_col, rate_col]].copy()
    work[sym_col] = work[sym_col].astype(str).str.upper().str.strip()
    if np.issubdtype(work[time_col].dtype, np.number):
        ts = pd.to_datetime(work[time_col], unit='ms', utc=True, errors='coerce')
    else:
        ts = pd.to_datetime(work[time_col], utc=True, errors='coerce')
    work['_ts'] = ts
    work['_rate'] = pd.to_numeric(work[rate_col], errors='coerce')
    work = work.dropna(subset=[sym_col, '_ts', '_rate'])
    if work.empty:
        FUNDING_EVENTS_CACHE[abspath] = result
        return result
    work['_ts_ms'] = (work['_ts'].astype('int64') // 10**6)

    for sym, sdf in work.groupby(sym_col):
        tuples = []
        for _, row in sdf.sort_values('_ts')[[ '_ts_ms', '_rate']].iterrows():
            try:
                ts_ms = int(row['_ts_ms'])
                rate = float(row['_rate'])
            except Exception:
                continue
            tuples.append((ts_ms, rate))
        if tuples:
            result[sym] = tuples

    FUNDING_EVENTS_CACHE[abspath] = result
    if result:
        print(f"[FUNDING] Eventos cargados para {len(result)} símbolos desde {abspath}")
    return result

def _normalize_funding_events_for(symbol: str, funding_map: Dict[str, List[Tuple[int, float]]]) -> List[Tuple[pd.Timestamp, float]]:
    events: List[Tuple[pd.Timestamp, float]] = []
    if not funding_map:
        return events
    raw = funding_map.get(symbol.upper()) or funding_map.get(symbol)
    if not raw:
        return events
    for ts_raw, rate in raw:
        try:
            if isinstance(ts_raw, (int, float)) and not isinstance(ts_raw, bool):
                ts = pd.to_datetime(int(ts_raw), unit='ms', utc=True)
            else:
                ts = pd.to_datetime(ts_raw, utc=True, errors='coerce')
            if ts is None or pd.isna(ts):
                continue
            if ts.tzinfo is None:
                ts = ts.tz_localize('UTC')
            else:
                ts = ts.tz_convert('UTC')
            events.append((ts, float(rate)))
        except Exception:
            continue
    return events


def _as_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
    try:
        return bool(value)
    except Exception:
        return default


def _as_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _load_best_params(path: str) -> List[Dict[str, object]]:
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[PORTFOLIO][WARN] No se pudo leer {path}: {exc}")
        return []
    best = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                sym = str(item.get('symbol', '')).upper()
                if sym:
                    params = item.get('params') or {}
                    best.append({'symbol': sym, 'params': params})
    elif isinstance(data, dict):
        for sym, params in data.items():
            best.append({'symbol': str(sym).upper(), 'params': params if isinstance(params, dict) else {}})
    return best


def _run_portfolio_from_best(best_path: str, args, funding_map: Dict[str, List[Tuple[int, float]]]):
    entries = _load_best_params(best_path)
    if not entries:
        print(f"[PORTFOLIO][WARN] No hay parámetros válidos en {best_path}")
        return

    trades_map: Dict[str, List[Trade]] = {}
    print(f"[PORTFOLIO] Evaluando set de {len(entries)} símbolos desde {best_path}")
    for entry in entries:
        sym = entry['symbol']
        params = entry.get('params') or {}
        try:
            df_sym = load_candles(args.data_template, sym)
        except Exception as exc:
            print(f"[PORTFOLIO][WARN] No se pudo cargar {sym}: {exc}")
            continue
        time_exit_param = params.get('time_exit_bars', args.time_exit_bars)
        if time_exit_param in ('', None):
            time_exit_param = None
        else:
            time_exit_param = _as_int(time_exit_param, time_exit_param)

        bt = Backtester(
            symbol=sym,
            df5m=df_sym,
            initial_equity=args.capital,
            tp_pct=_as_float(params.get('tp', args.tp), args.tp),
            tp_mode=params.get('tp_mode', args.tp_mode),
            tp_atr_mult=_as_float(params.get('tp_atr_mult', args.tp_atr_mult), args.tp_atr_mult),
            atr_mult_sl=_as_float(params.get('atr_mult', args.atr_mult), args.atr_mult),
            sl_mode=params.get('sl_mode', args.sl_mode),
            sl_pct=_as_float(params.get('sl_pct', args.sl_pct), args.sl_pct),
            ema_fast=_as_int(params.get('ema_fast', args.ema_fast), args.ema_fast),
            ema_slow=_as_int(params.get('ema_slow', args.ema_slow), args.ema_slow),
            rsi_buy=_as_int(params.get('rsi_buy', args.rsi_buy), args.rsi_buy),
            rsi_sell=_as_int(params.get('rsi_sell', args.rsi_sell), args.rsi_sell),
            adx_min=_as_int(params.get('adx_min', args.adx_min), args.adx_min),
            min_atr_pct=_as_float(params.get('min_atr_pct', args.min_atr_pct), args.min_atr_pct),
            max_atr_pct=_as_float(params.get('max_atr_pct', args.max_atr_pct), args.max_atr_pct),
            be_trigger=_as_float(params.get('be_trigger', args.be_trigger), args.be_trigger),
            cooldown_bars=_as_int(params.get('cooldown', args.cooldown), args.cooldown),
            logic=str(params.get('logic', args.logic)),
            hhll_lookback=_as_int(params.get('hhll_lookback', args.hhll_lookback), args.hhll_lookback),
            time_exit_bars=time_exit_param,
            max_dist_emaslow=_as_float(params.get('max_dist_emaslow', args.max_dist_emaslow), args.max_dist_emaslow),
            fresh_cross_max_bars=_as_int(params.get('fresh_cross_max_bars', args.fresh_cross_max_bars), args.fresh_cross_max_bars),
            require_rsi_cross=_as_bool(params.get('require_rsi_cross', args.require_rsi_cross), args.require_rsi_cross),
            min_ema_spread=float(params.get('min_ema_spread', args.min_ema_spread)),
            require_close_vs_emas=_as_bool(params.get('require_close_vs_emas', args.require_close_vs_emas), args.require_close_vs_emas),
            min_vol_ratio=_as_float(params.get('min_vol_ratio', args.min_vol_ratio), args.min_vol_ratio),
            vol_ma_len=_as_int(params.get('vol_ma_len', args.vol_ma_len), args.vol_ma_len),
            adx_slope_len=_as_int(params.get('adx_slope_len', args.adx_slope_len), args.adx_slope_len),
            adx_slope_min=_as_float(params.get('adx_slope_min', args.adx_slope_min), args.adx_slope_min),
            fresh_breakout_only=_as_bool(params.get('fresh_breakout_only', args.fresh_breakout_only), args.fresh_breakout_only),
            tp_splits=params.get('tp_splits'),
            tp_ladder_factors=params.get('tp_factors') or params.get('tp_ladder_factors'),
            tp1_factor=params.get('tp1_factor'),
            tp1_pct_override=params.get('tp1_pct_override'),
            funding_events=_normalize_funding_events_for(sym, funding_map),
        )
        res = bt.run(train_ratio=0.7)
        trades_map[sym] = bt.trades
        print(f"  {sym}: trades={res['trades']} pnl={res['pnl_net']} winrate={res['winrate_pct']}%")

    if not trades_map:
        print("[PORTFOLIO][WARN] No se generaron trades para evaluar el portafolio.")
        return

    portfolio = simulate_portfolio(
        trades_map,
        initial_equity=args.capital,
        max_positions=args.portfolio_max_positions,
        max_alloc_pct=args.portfolio_alloc_pct,
    )
    print("\n[PORTFOLIO] Resultado agregado desde best:")
    print(f"  Trades ejecutados: {portfolio['trades']} (omitidos: {portfolio['skipped_trades']})")
    print(f"  PnL neto: {portfolio['pnl_net']} | Winrate: {portfolio['winrate_pct']}%")
    print(f"  Max DD: {portfolio['max_dd_pct']} | Sharpe anualizado: {portfolio['sharpe_annual']}")
    print(f"  Cost ratio: {portfolio['cost_ratio']} (comisiones={portfolio['commissions']}, slip={portfolio['slippage_cost']}, funding={portfolio['funding_cost']})\n")

# ==========================
# Detección de símbolos disponibles en CSV único
# ==========================

def _list_symbols_from_csv(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path, usecols=None)
    cols_lower = {c.lower(): c for c in df.columns}
    col_sym = 'symbol' if 'symbol' in df.columns else cols_lower.get('symbol')
    if col_sym is None:
        return []
    return [str(s) for s in df[col_sym].dropna().unique().tolist()]


def _norm_symbol(s: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', str(s).upper())


def _extract_symbols_from_obj(obj: object) -> List[str]:
    if isinstance(obj, list):
        out = []
        for item in obj:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                sym = item.get('symbol')
                if isinstance(sym, str):
                    out.append(sym)
        return out
    if isinstance(obj, dict):
        for key in ('symbols', 'whitelist', 'winners'):
            val = obj.get(key)
            if isinstance(val, list):
                out = []
                for item in val:
                    if isinstance(item, str):
                        out.append(item)
                    elif isinstance(item, dict):
                        sym = item.get('symbol')
                        if isinstance(sym, str):
                            out.append(sym)
                return out
            if isinstance(val, dict):
                return [str(k) for k in val.keys()]
    return []


def _load_symbols_file(path: str) -> List[str]:
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return []
    syms = _extract_symbols_from_obj(data)
    return [str(s).upper().strip() for s in syms if str(s).strip()]

# ==========================
# Progreso
# ==========================

def _print_progress(done: int, total: int, prefix: str = "[SWEEP] Progreso"):
    if total <= 0:
        return
    width = 28  # ancho de la barra
    ratio = max(0.0, min(1.0, done / float(total)))
    filled = int(width * ratio)
    bar = '█' * filled + '·' * (width - filled)
    pct = int(ratio * 100)
    msg = f"\r{prefix}: |{bar}| {done}/{total} ({pct}%)"
    print(msg, end='', flush=True)
    if done >= total:
        print()  # salto de línea final

# ==========================
# Utilidades
# ==========================

def ensure_dt_utc(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col], utc=True)
    return df.sort_values(col).reset_index(drop=True)


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df['high'], df['low'], df['close']
    prev_c = c.shift(1)
    tr = pd.concat([
        (h - l),
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    # Implementación clásica de ADX sin TA‑Lib
    h, l, c = df['high'], df['low'], df['close']
    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = atr(df, period=1)  # true range sin suavizar para DI
    atr_n = tr.ewm(alpha=1/period, adjust=False).mean()

    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr_n
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr_n

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    adx_val = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx_val.fillna(0)



def nearest_funding_minutes(ts: pd.Timestamp) -> int:
    """Minutos hasta el funding más cercano (00:00, 08:00, 16:00 UTC)."""
    ts_utc = ts.tz_convert('UTC') if ts.tzinfo else ts.tz_localize('UTC')
    base = ts_utc.normalize()
    anchors = [base + pd.Timedelta(hours=h) for h in (0, 8, 16)]
    # Considera también el ancla del día anterior y siguiente por seguridad
    anchors += [base - pd.Timedelta(hours=8), base + pd.Timedelta(hours=24)]
    deltas = [abs((ts_utc - a).total_seconds())/60 for a in anchors]
    return int(min(deltas))


def in_funding_window(ts: pd.Timestamp, window_min: int = 30) -> bool:
    return nearest_funding_minutes(ts) <= window_min


def calc_slippage_rate(atr_pct: float) -> float:
    # 0.02% mínimo, 0.25 * ATR%_5m con tope 0.06%
    s = max(0.0002, 0.25 * float(atr_pct))
    return min(s, 0.0006)


def round_qty(qty: float, step: float = 0.0001) -> float:
    if step <= 0:
        return qty
    return math.floor(qty / step) * step


# ==========================
# Helper: barras desde último evento True
# ==========================
def bars_since(event: pd.Series) -> pd.Series:
    """Cuenta barras desde el último True (0 en la barra del evento). NaN si nunca ocurrió."""
    vals = event.astype(bool).values
    out = np.full(len(vals), np.nan, dtype=float)
    last = -1
    for i, flag in enumerate(vals):
        if flag:
            last = i
            out[i] = 0.0
        elif last >= 0:
            out[i] = float(i - last)
    return pd.Series(out, index=event.index)


# ==========================
# Data classes
# ==========================
@dataclass
class Trade:
    symbol: str
    side: str  # 'long' o 'short'
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    qty: float = 0.0
    commission_in: float = 0.0
    commission_out: float = 0.0
    slippage_in: float = 0.0
    slippage_out: float = 0.0
    funding_cost: float = 0.0

    def pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        direction = 1 if self.side == 'long' else -1
        gross = direction * (self.exit_price - self.entry_price) * self.qty
        costs = self.commission_in + self.commission_out + self.slippage_in + self.slippage_out + self.funding_cost
        return gross - costs


# ==========================
# Núcleo del backtest
# ==========================
class Backtester:
    def __init__(self,
                 symbol: str,
                 df5m: pd.DataFrame,
                 initial_equity: float = 1000.0,
                 tp_pct: float = 0.01,            # 1%
                 tp_mode: str = 'fixed',          # 'fixed' | 'atrx' | 'none'
                 tp_atr_mult: float = 0.0,        # usado si tp_mode='atrx'
                 atr_mult_sl: float = 2.0,
                 sl_mode: str = 'atr_then_trailing',  # 'atr_then_trailing' | 'atr_trailing_only' | 'percent'
                 sl_pct: float = 0.0,             # usado si sl_mode='percent'
                 ema_fast: int = 20,
                 ema_slow: int = 50,
                 rsi_buy: int = 55,
                 rsi_sell: int = 45,
                 adx_min: int = 20,
                 taker_fee: float = 0.0005,       # 0.05%
                 funding_8h: float = 0.0001,      # 0.01% por 8h
                 min_atr_pct: float = 0.0015,     # 0.15%
                 max_atr_pct: float = 0.02,       # 2.0%
                 lot_step: float = 0.001,
                 be_trigger: float = 0.0045,
                 cooldown_bars: int = 10,
                 logic: str = 'any',
                 hhll_lookback: int = 10,
                 time_exit_bars: Optional[int] = 30,
                 max_dist_emaslow: float = 0.015,
                 fresh_cross_max_bars: int = 6,
                 require_rsi_cross: bool = True,
                 min_ema_spread: float = 0.0,
                 require_close_vs_emas: bool = False,
                 min_vol_ratio: float = 0.0,
                 vol_ma_len: int = 50,
                 adx_slope_len: int = 3,
                 adx_slope_min: float = 0.0,
                 fresh_breakout_only: bool = False,
                 tp_splits: Optional[List[float]] = None,
                 tp_ladder_factors: Optional[List[float]] = None,
                 tp1_factor: Optional[float] = None,
                 tp1_pct_override: Optional[float] = None,
                 funding_events: Optional[List[Tuple[pd.Timestamp, float]]] = None
                 ):
        self.symbol = symbol
        self.df5m = df5m.copy()
        self.equity = initial_equity
        self.initial_equity = initial_equity
        self.tp_pct = tp_pct
        self.tp_mode = tp_mode
        self.tp_atr_mult = tp_atr_mult
        self.atr_mult_sl = atr_mult_sl
        self.sl_mode = sl_mode
        self.sl_pct = sl_pct
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_buy = rsi_buy
        self.rsi_sell = rsi_sell
        self.adx_min = adx_min
        self.taker_fee = taker_fee
        self.funding_8h = funding_8h
        self.min_atr_pct = min_atr_pct
        self.max_atr_pct = max_atr_pct
        self.lot_step = lot_step

        self.trades: List[Trade] = []
        self.open_trade: Optional[Trade] = None
        self.last_exec_bar: Optional[pd.Timestamp] = None

        self.be_trigger = be_trigger    # 0.45% para activar BE
        self.cooldown_bars = cooldown_bars
        self.cooldown = 0               # barras de enfriamiento tras SL
        self.be_active = False          # bandera de breakeven del trade actual
        self.logic = logic
        self.hhll_lookback = hhll_lookback
        self.time_exit_bars = time_exit_bars
        self.max_dist_emaslow = float(max_dist_emaslow)
        self.fresh_cross_max_bars = int(fresh_cross_max_bars)
        self.require_rsi_cross = bool(require_rsi_cross)

        self.min_ema_spread = float(min_ema_spread)
        self.require_close_vs_emas = bool(require_close_vs_emas)
        self.min_vol_ratio = float(min_vol_ratio)
        self.vol_ma_len = int(vol_ma_len)
        self.adx_slope_len = int(adx_slope_len)
        self.adx_slope_min = float(adx_slope_min)
        self.fresh_breakout_only = bool(fresh_breakout_only)
        self.tp_splits = tuple(tp_splits) if tp_splits else TP_SPLITS_DEFAULT
        self.tp_ladder_factors = tuple(tp_ladder_factors) if tp_ladder_factors else TP_LADDER_FACTORS
        self.tp1_factor = tp1_factor if tp1_factor is not None else TP1_DEFAULT_FACTOR
        self.tp1_pct_override = tp1_pct_override
        self.funding_events = sorted(funding_events, key=lambda x: x[0]) if funding_events else []
        self.funding_idx = 0

        self._prepare_features()

        # Estado interno para gestionar posiciones con parciales
        self.position_open_qty = 0.0
        self.position_entry_fee_remaining = 0.0
        self.position_slippage_in_remaining = 0.0
        self.position_funding_remaining = 0.0
        self.tp_plan: List[Dict[str, float]] = []

    def _prepare_features(self):
        df = self.df5m
        df = ensure_dt_utc(df)
        # Indicadores 5m
        df['ema_f'] = ema(df['close'], self.ema_fast)
        df['ema_s'] = ema(df['close'], self.ema_slow)
        df['rsi'] = rsi(df['close'], 14)
        df['atr'] = atr(df, 14)
        df['atr_pct'] = (df['atr'] / df['close']).clip(lower=0)
        df['adx'] = adx(df, 14)
        # Señales de volumen y tendencia ADX (pendiente)
        try:
            vlen = max(2, int(self.vol_ma_len))
        except Exception:
            vlen = 50
        df['vol_ma'] = df['volume'].rolling(vlen, min_periods=vlen).mean()
        try:
            alen = max(1, int(self.adx_slope_len))
        except Exception:
            alen = 3
        df['adx_slope'] = df['adx'] - df['adx'].shift(alen)
        # High/Low recientes (price action) con lookback configurable
        lookback = int(self.hhll_lookback) if self.hhll_lookback is not None else 0
        if lookback > 0:
            df['hh'] = df['high'].rolling(lookback, min_periods=lookback).max().shift(1)
            df['ll'] = df['low'].rolling(lookback, min_periods=lookback).min().shift(1)
            # Breakouts frescos (primera ruptura del rango hh/ll)
            df['fresh_long_break'] = (df['close'] > df['hh']) & (df['close'].shift(1) <= df['hh'].shift(1))
            df['fresh_short_break'] = (df['close'] < df['ll']) & (df['close'].shift(1) >= df['ll'].shift(1))
        else:
            df['hh'] = np.nan
            df['ll'] = np.nan
            df['fresh_long_break'] = False
            df['fresh_short_break'] = False

        # Cruces EMA/RSI recientes (alineado a indicadores.py)
        ema_up_evt = (df['ema_f'] > df['ema_s']) & (df['ema_f'].shift(1) <= df['ema_s'].shift(1))
        ema_dn_evt = (df['ema_f'] < df['ema_s']) & (df['ema_f'].shift(1) >= df['ema_s'].shift(1))
        df['ema_up_bars'] = bars_since(ema_up_evt)
        df['ema_dn_bars'] = bars_since(ema_dn_evt)

        rsi_up_evt = (df['rsi'] >= self.rsi_buy) & (df['rsi'].shift(1) < self.rsi_buy)
        rsi_dn_evt = (df['rsi'] <= self.rsi_sell) & (df['rsi'].shift(1) > self.rsi_sell)
        df['rsi_up_bars'] = bars_since(rsi_up_evt)
        df['rsi_dn_bars'] = bars_since(rsi_dn_evt)

        look = int(self.fresh_cross_max_bars) if self.fresh_cross_max_bars is not None else 0
        if look > 0:
            df['ema_cross_up_recent'] = ema_up_evt.rolling(look, min_periods=1).max().astype(bool)
            df['ema_cross_dn_recent'] = ema_dn_evt.rolling(look, min_periods=1).max().astype(bool)
            df['rsi_cross_up_recent'] = rsi_up_evt.rolling(look, min_periods=1).max().astype(bool)
            df['rsi_cross_dn_recent'] = rsi_dn_evt.rolling(look, min_periods=1).max().astype(bool)
        else:
            df['ema_cross_up_recent'] = df['ema_f'] > df['ema_s']
            df['ema_cross_dn_recent'] = df['ema_f'] < df['ema_s']
            df['rsi_cross_up_recent'] = df['rsi'] >= self.rsi_buy
            df['rsi_cross_dn_recent'] = df['rsi'] <= self.rsi_sell

        self.df5m = df

    def _process_funding_events(self, ts: pd.Timestamp, price: float) -> float:
        if not self.funding_events:
            return 0.0
        applied = 0.0
        while self.funding_idx < len(self.funding_events):
            evt_ts, rate = self.funding_events[self.funding_idx]
            if ts < evt_ts:
                break
            try:
                rate_f = float(rate)
            except Exception:
                rate_f = 0.0
            if self.open_trade is not None and self.position_open_qty > 0 and rate_f != 0.0:
                direction = 1.0 if self.open_trade.side == 'long' else -1.0
                charge = self.position_open_qty * price * rate_f * direction
                self.position_funding_remaining += charge
                applied += charge
            self.funding_idx += 1
        return applied

    def _compute_tp_prices(self, entry_price: float, atr_abs: float, side: str) -> List[float]:
        prices: List[float] = []
        direction = 1.0 if side == 'long' else -1.0
        if self.tp_mode == 'fixed' and self.tp_pct > 0:
            for mult in self.tp_ladder_factors:
                prices.append(entry_price * (1 + direction * self.tp_pct * float(mult)))
        elif self.tp_mode == 'atrx' and atr_abs > 0 and self.tp_atr_mult > 0:
            target = entry_price + direction * (self.tp_atr_mult * atr_abs)
            prices = [target for _ in self.tp_ladder_factors]
        return [float(px) for px in prices if px and px > 0]

    def _apply_tp1_adjustment(self, prices: List[float], entry_price: float, side: str) -> List[float]:
        if not prices:
            return prices
        direction = 1.0 if side == 'long' else -1.0
        if self.tp1_pct_override is not None and self.tp1_pct_override > 0:
            prices[0] = entry_price * (1 + direction * float(self.tp1_pct_override))
            return prices
        factor = self.tp1_factor if (self.tp1_factor is not None and 0 < self.tp1_factor < 1) else TP1_DEFAULT_FACTOR
        base = prices[0]
        prices[0] = entry_price + (base - entry_price) * float(factor)
        return prices

    def _prepare_tp_plan(self, qty: float, entry_price: float, atr_abs: float, side: str) -> List[Dict[str, float]]:
        if qty <= 0:
            return []
        prices = self._compute_tp_prices(entry_price, atr_abs, side)
        if not prices:
            return []
        prices = self._apply_tp1_adjustment(prices, entry_price, side)
        count = len(prices)
        splits = self.tp_splits[:count]
        if not splits or sum(splits) <= 0:
            splits = tuple(1.0 for _ in range(count))
        total = float(sum(splits))
        fractions = [float(s) / total for s in splits]
        plan: List[Dict[str, float]] = []
        remaining = qty
        for idx in range(count):
            split_qty = round_qty(qty * fractions[idx], self.lot_step)
            if idx == count - 1:
                split_qty = max(0.0, remaining)
            remaining -= split_qty
            filled = split_qty <= 0
            plan.append({
                'price': prices[idx],
                'qty': max(split_qty, 0.0),
                'label': f"TP{idx + 1}",
                'filled': filled,
            })
        if plan and remaining > 1e-9:
            plan[-1]['qty'] = max(0.0, plan[-1]['qty'] + remaining)
        return plan

    def _tp_targets_hit(self, row: pd.Series) -> List[Dict[str, float]]:
        hits: List[Dict[str, float]] = []
        if not self.tp_plan or self.open_trade is None:
            return hits
        for target in self.tp_plan:
            if target.get('filled') or target.get('qty', 0.0) <= 0:
                continue
            price = float(target['price'])
            if self.open_trade.side == 'long':
                if row['high'] >= price:
                    hits.append(target)
            else:
                if row['low'] <= price:
                    hits.append(target)
        return hits

    def _allocate_entry_cost(self, qty_close: float) -> Tuple[float, float, float]:
        if qty_close <= 0 or self.position_open_qty <= 0:
            return 0.0, 0.0, 0.0
        ratio = qty_close / self.position_open_qty
        fee_share = self.position_entry_fee_remaining * ratio
        slip_in_share = self.position_slippage_in_remaining * ratio
        funding_share = self.position_funding_remaining * ratio
        self.position_entry_fee_remaining -= fee_share
        self.position_slippage_in_remaining -= slip_in_share
        self.position_funding_remaining -= funding_share
        return fee_share, slip_in_share, funding_share

    def _close_position_portion(self, qty_close: float, exit_price: float, ts: pd.Timestamp, atr_pct: float) -> Tuple[float, float, float]:
        if self.open_trade is None or qty_close <= 0:
            return 0.0, 0.0, 0.0
        qty_close = min(qty_close, self.position_open_qty)
        if qty_close <= 0:
            return 0.0, 0.0, 0.0
        entry_fee_share, slip_in_share, funding_share = self._allocate_entry_cost(qty_close)
        slip_rate = calc_slippage_rate(atr_pct)
        if self.open_trade.side == 'long':
            actual_exit_notional = exit_price * (1 - slip_rate) * qty_close
        else:
            actual_exit_notional = exit_price * (1 + slip_rate) * qty_close
        exit_fee = self._apply_commission(actual_exit_notional)
        sl_out = exit_price * slip_rate * qty_close
        trade = Trade(
            symbol=self.symbol,
            side=self.open_trade.side,
            entry_time=self.open_trade.entry_time,
            entry_price=self.open_trade.entry_price,
            exit_time=ts,
            exit_price=exit_price,
            qty=qty_close,
            commission_in=entry_fee_share,
            commission_out=exit_fee,
            slippage_in=slip_in_share,
            slippage_out=sl_out,
            funding_cost=funding_share,
        )
        pnl = trade.pnl()
        self.trades.append(trade)
        self.position_open_qty -= qty_close
        return pnl, entry_fee_share + exit_fee, slip_in_share + sl_out

    def _flatten_position(self):
        self.open_trade = None
        self.position_open_qty = 0.0
        self.position_entry_fee_remaining = 0.0
        self.position_slippage_in_remaining = 0.0
        self.position_funding_remaining = 0.0
        self.tp_plan = []
        self.be_active = False

    # --------------------------
    # Reglas
    # --------------------------
    def _allow_new_trade(self, row: pd.Series) -> bool:
        ts = row['date']
        atr_pct = float(row.get('atr_pct', np.nan))
        if in_funding_window(ts, 30):
            return False
        if not (self.min_atr_pct <= atr_pct <= self.max_atr_pct):
            return False
        # una ejecución por vela
        if self.last_exec_bar is not None and ts == self.last_exec_bar:
            return False
        # cooldown activo
        if self.cooldown > 0:
            return False
        # Anti-chase: distancia a EMA_slow
        try:
            ema_s = float(row.get('ema_s', np.nan))
            close = float(row.get('close', np.nan))
            if not (math.isnan(ema_s) or ema_s == 0 or math.isnan(close)):
                dist = abs(close / ema_s - 1.0)
                if dist > float(self.max_dist_emaslow):
                    return False
        except Exception:
            pass
        return True

    def _entry_signal(self, row: pd.Series) -> Optional[str]:
        # Tendencia 5m
        trend_up = row['ema_f'] > row['ema_s']
        trend_dn = row['ema_f'] < row['ema_s']
        # Fuerza
        strong = row['adx'] >= self.adx_min
        # Price action: cierre fuera de hh/ll
        long_break = (row['close'] > row['hh']) if not math.isnan(row['hh']) else False
        short_break = (row['close'] < row['ll']) if not math.isnan(row['ll']) else False

        # Si se exige breakout fresco, sustituye las condiciones de ruptura
        if getattr(self, 'fresh_breakout_only', False):
            long_break = bool(row.get('fresh_long_break', False))
            short_break = bool(row.get('fresh_short_break', False))

        # Disparadores de RSI (reciente vs. nivel)
        if self.require_rsi_cross:
            rsi_trig_long = bool(row.get('rsi_cross_up_recent', False))
            rsi_trig_short = bool(row.get('rsi_cross_dn_recent', False))
        else:
            rsi_trig_long = row['rsi'] >= self.rsi_buy
            rsi_trig_short = row['rsi'] <= self.rsi_sell

        if self.logic == 'strict':
            long_ok = trend_up and strong and rsi_trig_long and long_break
            short_ok = trend_dn and strong and rsi_trig_short and short_break
        else:
            long_ok = trend_up and strong and (rsi_trig_long or long_break)
            short_ok = trend_dn and strong and (rsi_trig_short or short_break)

        # --- Gate por recencia de cruces ---
        ema_long_recent = bool(row.get('ema_cross_up_recent', False))
        ema_short_recent = bool(row.get('ema_cross_dn_recent', False))
        if long_ok and not ema_long_recent:
            long_ok = False
        if short_ok and not ema_short_recent:
            short_ok = False

        # --- Endurecedores adicionales ---
        # 1) Separación mínima entre EMAs (evitar cruces planos)
        ema_spread_ok = True
        try:
            if self.min_ema_spread > 0:
                ema_s = float(row.get('ema_s', np.nan))
                ema_f = float(row.get('ema_f', np.nan))
                if not (math.isnan(ema_s) or math.isnan(ema_f) or ema_s == 0):
                    ema_spread_ok = (abs(ema_f - ema_s) / abs(ema_s)) >= self.min_ema_spread
                else:
                    ema_spread_ok = False
        except Exception:
            ema_spread_ok = False

        # 2) Precio relativo a EMAs
        price_long_ok = True
        price_short_ok = True
        if self.require_close_vs_emas:
            try:
                price_long_ok = (row['close'] > row['ema_f']) and (row['ema_f'] > row['ema_s'])
                price_short_ok = (row['close'] < row['ema_f']) and (row['ema_f'] < row['ema_s'])
            except Exception:
                price_long_ok = price_short_ok = False

        # 3) Pendiente mínima de ADX (tendencia ganando fuerza)
        adx_slope_ok = True
        if self.adx_slope_min > 0:
            try:
                adx_slope_ok = float(row.get('adx_slope', 0.0)) >= self.adx_slope_min
            except Exception:
                adx_slope_ok = False

        # 4) Filtro de volumen: volumen actual por encima de su media
        vol_ok = True
        if self.min_vol_ratio > 0:
            try:
                vma = float(row.get('vol_ma', np.nan))
                vol_ok = (not math.isnan(vma) and vma > 0 and float(row.get('volume', 0.0)) >= self.min_vol_ratio * vma)
            except Exception:
                vol_ok = False

        if long_ok:
            if not ema_spread_ok: long_ok = False
            if not adx_slope_ok: long_ok = False
            if not vol_ok: long_ok = False
            if not price_long_ok: long_ok = False
        if short_ok:
            if not ema_spread_ok: short_ok = False
            if not adx_slope_ok: short_ok = False
            if not vol_ok: short_ok = False
            if not price_short_ok: short_ok = False

        if long_ok and not short_ok:
            return 'long'
        if short_ok and not long_ok:
            return 'short'
        return None

    def _risk_position_size(self, price: float, atr_abs: float) -> float:
        if atr_abs <= 0:
            return 0.0
        risk_amt = 0.005 * self.equity            # 0.5% del equity
        qty_risk = risk_amt / atr_abs
        qty_cap = (0.10 * self.equity) / price    # 10% del equity
        qty = min(qty_risk, qty_cap)
        return round_qty(max(qty, 0.0), self.lot_step)

    def _apply_commission(self, notional: float) -> float:
        return abs(notional) * self.taker_fee

    def _update_trailing_sl(self, row: pd.Series, direction: str, entry_price: float, atr_abs: float, anchor: float) -> float:
        # trailing basado en ATR
        if direction == 'long':
            new_anchor = max(anchor, row['close'])
            sl = new_anchor - self.atr_mult_sl * atr_abs
        else:
            new_anchor = min(anchor, row['close'])
            sl = new_anchor + self.atr_mult_sl * atr_abs
        return sl

    def run(self, train_ratio: float = 0.7) -> Dict[str, float]:
        df = self.df5m.copy()
        if len(df) < 100:
            raise ValueError("Datos insuficientes")

        # Split + embargo 1h
        cut_idx = int(len(df) * train_ratio)
        cut_time = df.loc[cut_idx, 'date']
        oos_start = cut_time + pd.Timedelta(hours=1)

        commissions = slippages = funding_costs = 0.0
        realized_pnl = 0.0
        anchor_price = None  # para trailing
        open_bar_idx = None  # índice de barra donde se abrió la posición (para time-exit)

        # === Track equity curve ===
        equity_curve = []
        equity_time = []

        for i, row in df.iterrows():
            ts = row['date']
            if self.cooldown > 0:
                self.cooldown -= 1
            price = float(row['close'])
            atr_abs = float(row['atr']) if not math.isnan(row['atr']) else 0.0
            atr_pct = float(row['atr_pct']) if not math.isnan(row['atr_pct']) else 0.0

            if self.funding_events:
                funding_costs += self._process_funding_events(ts, price)
            elif self.open_trade is not None:
                # funding prorrateado constante (legacy)
                notional = self.position_open_qty * price
                per_bar_rate = self.funding_8h * (5.0 / 480.0)
                fcost = abs(notional) * per_bar_rate
                self.position_funding_remaining += fcost
                funding_costs += fcost

            # Cierre (TP / SL)
            if self.open_trade is not None:
                t = self.open_trade
                # Activación de breakeven cuando el precio avanza a favor
                if not self.be_active:
                    if t.side == 'long' and row['high'] >= t.entry_price * (1 + self.be_trigger):
                        self.be_active = True
                    elif t.side == 'short' and row['low'] <= t.entry_price * (1 - self.be_trigger):
                        self.be_active = True

                # SL: según modo
                if anchor_price is None:
                    anchor_price = t.entry_price

                if self.sl_mode == 'percent':
                    # SL fijo por porcentaje, sin trailing (salvo BE para no perder)
                    if t.side == 'long':
                        sl_price = t.entry_price * (1 - max(self.sl_pct, 0.0))
                        if self.be_active:
                            sl_price = max(sl_price, t.entry_price)
                    else:
                        sl_price = t.entry_price * (1 + max(self.sl_pct, 0.0))
                        if self.be_active:
                            sl_price = min(sl_price, t.entry_price)
                else:
                    # Modos basados en ATR
                    if (self.sl_mode == 'atr_trailing_only') or (self.be_active and self.sl_mode == 'atr_then_trailing'):
                        # Trailing por ATR
                        sl_price = self._update_trailing_sl(row, t.side, t.entry_price, atr_abs, anchor_price)
                        anchor_price = max(anchor_price, price) if t.side == 'long' else min(anchor_price, price)
                        # Nunca peor que la entrada una vez activo
                        if t.side == 'long':
                            sl_price = max(sl_price, t.entry_price)
                        else:
                            sl_price = min(sl_price, t.entry_price)
                    else:
                        # SL fijo inicial por ATR (antes de BE)
                        if t.side == 'long':
                            sl_price = t.entry_price - self.atr_mult_sl * atr_abs
                        else:
                            sl_price = t.entry_price + self.atr_mult_sl * atr_abs

                # Procesar TP escalonados (puede vaciar toda la posición)
                tp_hits = self._tp_targets_hit(row)
                if tp_hits:
                    for target in tp_hits:
                        if self.open_trade is None or self.position_open_qty <= 0:
                            break
                        qty_tp = min(float(target.get('qty', 0.0)), self.position_open_qty)
                        if qty_tp <= 0:
                            target['filled'] = True
                            continue
                        pnl_tp, comm_tp, slip_tp = self._close_position_portion(qty_tp, float(target['price']), ts, atr_pct)
                        realized_pnl += pnl_tp
                        commissions += comm_tp
                        slippages += slip_tp
                        self.equity += pnl_tp
                        target['filled'] = True
                        target['qty'] = 0.0
                        if self.position_open_qty <= 0:
                            self.last_exec_bar = ts
                            anchor_price = None
                            open_bar_idx = None
                            self._flatten_position()
                            break
                    if self.open_trade is None:
                        continue

                # ¿Se golpea SL?
                if t.side == 'long':
                    hit_sl = row['low'] <= sl_price
                else:
                    hit_sl = row['high'] >= sl_price

                exit_reason = None
                exit_price = None
                if hit_sl:
                    exit_reason = 'SL'
                    exit_price = sl_price

                # Salida por tiempo (si está activada y no golpeó TP/SL)
                if exit_reason is None and self.time_exit_bars and open_bar_idx is not None:
                    if (i - open_bar_idx) >= int(self.time_exit_bars):
                        exit_reason = 'TIME'
                        exit_price = price  # cierra a precio de cierre de la barra

                if exit_reason is not None:
                    qty_to_close = self.position_open_qty
                    pnl_exit, comm_exit, slip_exit = self._close_position_portion(qty_to_close, float(exit_price), ts, atr_pct)
                    realized_pnl += pnl_exit
                    commissions += comm_exit
                    slippages += slip_exit
                    self.equity += pnl_exit
                    self.last_exec_bar = ts  # una ejecución por vela
                    anchor_price = None
                    open_bar_idx = None
                    if exit_reason == 'SL':
                        self.cooldown = self.cooldown_bars
                    self._flatten_position()
                    # Track equity stepwise per barra
                    equity_curve.append(self.equity)
                    equity_time.append(ts)
                    continue  # no abrir en la misma vela

            # Entrada (solo OOS y si no hay posición)
            if ts < oos_start:
                # Track equity stepwise per barra
                equity_curve.append(self.equity)
                equity_time.append(ts)
                continue  # no operamos en train; se podría usar para optimizar

            if self.open_trade is None:
                if not self._allow_new_trade(row):
                    # Track equity stepwise per barra
                    equity_curve.append(self.equity)
                    equity_time.append(ts)
                    continue
                signal = self._entry_signal(row)
                if signal is None:
                    # Track equity stepwise per barra
                    equity_curve.append(self.equity)
                    equity_time.append(ts)
                    continue

                qty = self._risk_position_size(price, atr_abs)
                if qty <= 0:
                    # Track equity stepwise per barra
                    equity_curve.append(self.equity)
                    equity_time.append(ts)
                    continue

                slip_rate = calc_slippage_rate(atr_pct)
                entry_price = price
                sl_in = price * slip_rate * qty
                if signal == 'long':
                    actual_entry_notional = price * (1 + slip_rate) * qty
                else:
                    actual_entry_notional = price * (1 - slip_rate) * qty
                fee = self._apply_commission(actual_entry_notional)

                self.open_trade = Trade(
                    symbol=self.symbol,
                    side=signal,
                    entry_time=ts,
                    entry_price=entry_price,
                    qty=qty,
                    commission_in=fee,
                    slippage_in=sl_in,
                )
                self.position_open_qty = qty
                self.position_entry_fee_remaining = fee
                self.position_slippage_in_remaining = sl_in
                self.position_funding_remaining = 0.0
                self.tp_plan = self._prepare_tp_plan(qty, entry_price, atr_abs, signal)
                self.last_exec_bar = ts
                open_bar_idx = i
                # Si el SL es 'atr_trailing_only', activar trailing desde el inicio
                if self.sl_mode == 'atr_trailing_only':
                    self.be_active = True
                    anchor_price = entry_price
            # Track equity stepwise per barra
            equity_curve.append(self.equity)
            equity_time.append(ts)

        # Si quedó abierta la posición, la cerramos al último precio
        if self.open_trade is not None and self.position_open_qty > 0:
            row = df.iloc[-1]
            ts = row['date']
            price = float(row['close'])
            atr_pct = float(row['atr_pct'])
            pnl_exit, comm_exit, slip_exit = self._close_position_portion(self.position_open_qty, price, ts, atr_pct)
            realized_pnl += pnl_exit
            commissions += comm_exit
            slippages += slip_exit
            self.equity += pnl_exit
            self._flatten_position()
            equity_curve.append(self.equity)
            equity_time.append(ts)

        gross_pnl = sum([(1 if t.side=='long' else -1) * (t.exit_price - t.entry_price) * t.qty for t in self.trades if t.exit_price is not None])
        costs_total = commissions + slippages + funding_costs
        cost_ratio = (costs_total / abs(gross_pnl)) if gross_pnl != 0 else np.nan

        wins = [t for t in self.trades if t.pnl() > 0]
        winrate = (len(wins) / len(self.trades)) * 100 if self.trades else 0.0

        # ==== Métricas adicionales ====
        # Profit Factor
        pos = sum([t.pnl() for t in self.trades if t.pnl() > 0])
        neg = -sum([t.pnl() for t in self.trades if t.pnl() < 0])
        profit_factor = (pos / neg) if neg > 0 else None

        # Max Drawdown (sobre equity_curve)
        max_dd_pct = None
        if len(equity_curve) >= 2:
            eq = np.array(equity_curve, dtype=float)
            peak = np.maximum.accumulate(eq)
            dd = (eq - peak) / peak
            max_dd_pct = float(dd.min()) if len(dd) else None

        # Sharpe anualizado usando retornos por barra de equity
        sharpe_annual = None
        if len(equity_curve) >= 3:
            eq = np.array(equity_curve, dtype=float)
            rets = np.diff(eq) / eq[:-1]
            if rets.std() > 1e-12:
                bars_per_year = 288 * 365  # 5m bars ≈ 105,120/año
                sharpe_annual = float((rets.mean() / rets.std()) * np.sqrt(bars_per_year))

        result = {
            'symbol': self.symbol,
            'trades': len(self.trades),
            'equity_final': round(self.equity, 2),
            'pnl_net': round(self.equity - self.initial_equity, 2),
            'winrate_pct': round(winrate, 2),
            'profit_factor': round(float(profit_factor), 3) if profit_factor is not None else None,
            'commissions': round(commissions, 4),
            'slippage_cost': round(slippages, 4),
            'funding_cost': round(funding_costs, 4),
            'cost_ratio': round(float(cost_ratio), 4) if not np.isnan(cost_ratio) else None,
            'max_dd_pct': round(float(max_dd_pct), 4) if max_dd_pct is not None else None,
            'sharpe_annual': round(float(sharpe_annual), 3) if sharpe_annual is not None else None,
        }
        return result

    def export_trades(self, out_path: str):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        rows = []
        for t in self.trades:
            rows.append({
                'symbol': t.symbol,
                'side': t.side,
                'entry_time': t.entry_time,
                'entry_price': t.entry_price,
                'exit_time': t.exit_time,
                'exit_price': t.exit_price,
                'qty': t.qty,
                'pnl': t.pnl(),
                'commission_in': t.commission_in,
                'commission_out': t.commission_out,
                'slippage_in': t.slippage_in,
                'slippage_out': t.slippage_out,
                'funding_cost': t.funding_cost,
            })
        pd.DataFrame(rows).to_csv(out_path, index=False)


def simulate_portfolio(trades_by_symbol: Dict[str, List[Trade]],
                       initial_equity: float,
                       max_positions: int = 3,
                       max_alloc_pct: float = 0.30) -> Dict[str, object]:
    events: List[Tuple[str, pd.Timestamp, str, int, Trade]] = []
    for sym, trades in trades_by_symbol.items():
        if not trades:
            continue
        upper_sym = str(sym).upper()
        for idx, trade in enumerate(trades):
            if trade.entry_time is None or trade.exit_time is None:
                continue
            entry_ts = pd.Timestamp(trade.entry_time)
            exit_ts = pd.Timestamp(trade.exit_time)
            if entry_ts.tzinfo is None:
                entry_ts = entry_ts.tz_localize('UTC')
            else:
                entry_ts = entry_ts.tz_convert('UTC')
            if exit_ts.tzinfo is None:
                exit_ts = exit_ts.tz_localize('UTC')
            else:
                exit_ts = exit_ts.tz_convert('UTC')
            events.append(('exit', exit_ts, upper_sym, idx, trade))
            events.append(('entry', entry_ts, upper_sym, idx, trade))

    events.sort(key=lambda e: (e[1], 0 if e[0] == 'exit' else 1, e[2]))

    active: Dict[Tuple[str, int], Dict[str, float]] = {}
    cash = float(initial_equity)
    equity = float(initial_equity)
    portfolio_trades: List[Trade] = []
    equity_curve: List[float] = [equity]
    equity_time: List[pd.Timestamp] = []
    skipped = 0

    for etype, ts, sym, idx, trade in events:
        key = (sym, idx)
        if etype == 'exit':
            pos = active.pop(key, None)
            if not pos:
                continue
            scale = pos['scale']
            notional = pos['notional']
            cash += notional
            pnl = trade.pnl() * scale
            cash += pnl
            equity += pnl
            scaled_trade = Trade(
                symbol=trade.symbol,
                side=trade.side,
                entry_time=trade.entry_time,
                entry_price=trade.entry_price,
                exit_time=trade.exit_time,
                exit_price=trade.exit_price,
                qty=trade.qty * scale,
                commission_in=trade.commission_in * scale,
                commission_out=trade.commission_out * scale,
                slippage_in=trade.slippage_in * scale,
                slippage_out=trade.slippage_out * scale,
                funding_cost=trade.funding_cost * scale,
            )
            portfolio_trades.append(scaled_trade)
            equity_curve.append(equity)
            equity_time.append(ts)
        else:
            if max_positions and max_positions > 0 and len(active) >= max_positions:
                skipped += 1
                continue
            notional_full = abs(trade.entry_price * trade.qty)
            if notional_full <= 0:
                continue
            cap_limit = notional_full
            if max_alloc_pct and max_alloc_pct > 0:
                cap_limit = min(cap_limit, equity * max_alloc_pct)
            alloc = min(cap_limit, cash)
            if alloc <= 0:
                skipped += 1
                continue
            scale = alloc / notional_full
            if scale <= 1e-9:
                skipped += 1
                continue
            cash -= alloc
            active[key] = {'scale': scale, 'notional': alloc}

    gross_pnl = sum([(1 if t.side == 'long' else -1) * (t.exit_price - t.entry_price) * t.qty for t in portfolio_trades if t.exit_price is not None])
    commissions = sum([(t.commission_in + t.commission_out) for t in portfolio_trades])
    slippages = sum([(t.slippage_in + t.slippage_out) for t in portfolio_trades])
    funding_costs = sum([t.funding_cost for t in portfolio_trades])
    costs_total = commissions + slippages + funding_costs
    cost_ratio = (costs_total / abs(gross_pnl)) if gross_pnl else None

    wins = [t for t in portfolio_trades if t.pnl() > 0]
    winrate = (len(wins) / len(portfolio_trades)) * 100 if portfolio_trades else 0.0

    pos = sum([t.pnl() for t in portfolio_trades if t.pnl() > 0])
    neg = -sum([t.pnl() for t in portfolio_trades if t.pnl() < 0])
    profit_factor = (pos / neg) if neg > 0 else None

    max_dd_pct = None
    if len(equity_curve) >= 2:
        eq = np.array(equity_curve, dtype=float)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        max_dd_pct = float(dd.min()) if len(dd) else None

    sharpe_annual = None
    if len(equity_curve) >= 3:
        eq = np.array(equity_curve, dtype=float)
        rets = np.diff(eq) / eq[:-1]
        if rets.std() > 1e-12:
            bars_per_year = 288 * 365
            sharpe_annual = float((rets.mean() / rets.std()) * np.sqrt(bars_per_year))

    return {
        'trades': len(portfolio_trades),
        'pnl_net': round(equity - initial_equity, 2),
        'winrate_pct': round(winrate, 2),
        'profit_factor': round(float(profit_factor), 3) if profit_factor is not None else None,
        'commissions': round(commissions, 4),
        'slippage_cost': round(slippages, 4),
        'funding_cost': round(funding_costs, 4),
        'cost_ratio': round(float(cost_ratio), 4) if cost_ratio is not None else None,
        'max_dd_pct': round(float(max_dd_pct), 4) if max_dd_pct is not None else None,
        'sharpe_annual': round(float(sharpe_annual), 3) if sharpe_annual is not None else None,
        'equity_final': round(equity, 2),
        'equity_curve': equity_curve,
        'equity_time': equity_time,
        'skipped_trades': skipped,
        'executed_trades': portfolio_trades,
    }


# ==========================
# CLI
# ==========================

def load_candles(template_or_path: str, symbol: str) -> pd.DataFrame:
    """Carga velas 5m desde:
    - Una ruta *templated* con `{symbol}` (ej. 'archivos/{symbol}_5m.csv'), o
    - Un CSV único con múltiples símbolos (ej. 'archivos/cripto_price_5m_long.csv').
    """
    # Caso 1: plantilla por símbolo
    if '{symbol}' in template_or_path:
        path = template_or_path.replace('{symbol}', symbol)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No existe el CSV para {symbol}: {path}")
        df = pd.read_csv(path)
    else:
        # Caso 2: CSV único con columna 'symbol'
        path = template_or_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"No existe el archivo de velas: {path}")
        df_all = pd.read_csv(path)
        cols_lower = {c.lower(): c for c in df_all.columns}
        # Normaliza nombre de columna 'symbol' si viene en mayúsculas u otro casing
        if 'symbol' not in df_all.columns:
            if 'symbol' in cols_lower:
                df_all = df_all.rename(columns={cols_lower['symbol']: 'symbol'})
            else:
                raise ValueError(f"El archivo {path} no contiene columna 'symbol'. Columnas: {list(df_all.columns)}")
        # Normaliza símbolos para permitir 'BTCUSDT' o 'BTC-USDT'
        df_all['symbol_norm'] = (
            df_all['symbol'].astype(str)
            .str.upper()
            .str.replace(r'[^A-Z0-9]', '', regex=True)
        )
        target_norm = re.sub(r'[^A-Z0-9]', '', symbol.upper())
        df = df_all[df_all['symbol_norm'] == target_norm].copy()
        if df.empty:
            # Intento alterno con guion por si viene en otro formato
            sym_alt = f"{symbol[:-4]}-{symbol[-4:]}" if len(symbol) > 4 else symbol
            target_norm_alt = re.sub(r'[^A-Z0-9]', '', sym_alt.upper())
            df = df_all[df_all['symbol_norm'] == target_norm_alt].copy()
        if df.empty:
            ej = df_all['symbol'].iloc[0] if len(df_all) else 'N/A'
            raise ValueError(f"No hay datos para el símbolo {symbol} en {path}. Ejemplo en CSV: '{ej}'")

    # Normaliza nombres básicos y valida columnas mínimas
    cols_map_lower = {c.lower(): c for c in df.columns}
    rename = {}
    for c in ['symbol','open','high','low','close','volume','date']:
        if c not in df.columns and c in cols_map_lower:
            rename[cols_map_lower[c]] = c
    if rename:
        df = df.rename(columns=rename)

    missing = [c for c in ['open','high','low','close','volume','date'] if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas ({missing}) en {path}")

    # Garantiza columna 'symbol'
    if 'symbol' not in df.columns:
        df['symbol'] = symbol

    # Ordena por fecha y devuelve columnas esperadas en orden
    df = df[['symbol','open','high','low','close','volume','date']]
    df = ensure_dt_utc(df, 'date')
    if 'symbol_norm' in df.columns: df = df.drop(columns=['symbol_norm'])
    return df


#
# ==========================
# Worker global para multiprocessing (evita pickling de closures)
# ==========================

def run_job(job):
    sym, p, data_template, capital, out_dir = job[:5]
    funding_map = job[5] if len(job) >= 6 else {}
    try:
        dfj = load_candles(data_template, sym)
        funding_events = _normalize_funding_events_for(sym, funding_map) if isinstance(funding_map, dict) else []
        bt = Backtester(
            symbol=sym,
            df5m=dfj,
            initial_equity=capital,
            tp_pct=p.get('tp', 0.01),
            tp_mode=p.get('tp_mode', 'fixed'),
            tp_atr_mult=p.get('tp_atr_mult', 0.0),
            atr_mult_sl=p.get('atr_mult', 2.0),
            sl_mode=p.get('sl_mode', 'atr_then_trailing'),
            sl_pct=p.get('sl_pct', 0.0),
            ema_fast=p['ema_fast'],
            ema_slow=p['ema_slow'],
            rsi_buy=p['rsi_buy'],
            rsi_sell=p['rsi_sell'],
            adx_min=p['adx_min'],
            min_atr_pct=p['min_atr_pct'],
            max_atr_pct=p['max_atr_pct'],
            be_trigger=p['be_trigger'],
            cooldown_bars=p['cooldown'],
            logic=p['logic'],
            hhll_lookback=p.get('hhll_lookback', 10),
            time_exit_bars=p.get('time_exit_bars', 30),
            max_dist_emaslow=p.get('max_dist_emaslow', 0.015),
            fresh_cross_max_bars=p.get('fresh_cross_max_bars', 6),
            require_rsi_cross=p.get('require_rsi_cross', True),
            min_ema_spread=p.get('min_ema_spread', 0.0),
            require_close_vs_emas=p.get('require_close_vs_emas', False),
            min_vol_ratio=p.get('min_vol_ratio', 0.0),
            vol_ma_len=p.get('vol_ma_len', 50),
            adx_slope_len=p.get('adx_slope_len', 3),
            adx_slope_min=p.get('adx_slope_min', 0.0),
            fresh_breakout_only=p.get('fresh_breakout_only', False),
            tp_splits=p.get('tp_splits'),
            tp_ladder_factors=p.get('tp_factors') or p.get('tp_ladder_factors'),
            tp1_factor=p.get('tp1_factor'),
            tp1_pct_override=p.get('tp1_pct_override'),
            funding_events=funding_events,
        )
        res = bt.run(train_ratio=0.7)
        res['params'] = p
        res['symbol'] = sym
        # Ya no guardamos trades en cada combinación del sweep aquí.
        return res
    except Exception as e:
        return {'symbol': sym, 'error': str(e), 'params': p}

def main():
    # --- SIMPLE RUN: si no hay argumentos en CLI y está habilitado, inyecta flags ---
    try:
        if len(sys.argv) == 1 and isinstance(SIMPLE_RUN, dict) and SIMPLE_RUN.get('ENABLE', False):
            sys.argv = _build_simple_argv(SIMPLE_RUN)
            print(f"[SIMPLE] Ejecutando en modo simple → MODE={SIMPLE_RUN.get('MODE')} | symbols={SIMPLE_RUN.get('SYMBOLS')}")
    except Exception as _e:
        print(f"[SIMPLE][WARN] No se pudo activar modo simple: {_e}")
    parser = argparse.ArgumentParser(description='TRobot Backtesting simple-realista')
    parser.add_argument('--symbols', type=str, default='BTCUSDT', help='Lista separada por comas')
    parser.add_argument('--data_template', type=str, default=DEFAULT_DATA_TEMPLATE, help='Ruta con {symbol} o CSV único (default: DEFAULT_DATA_TEMPLATE)')
    parser.add_argument('--capital', type=float, default=1000.0)
    parser.add_argument('--tp', type=float, default=0.01, help='Take Profit en fracción (0.01 = 1%)')
    parser.add_argument('--atr_mult', type=float, default=2.0)
    parser.add_argument('--ema_fast', type=int, default=20)
    parser.add_argument('--ema_slow', type=int, default=50)
    parser.add_argument('--rsi_buy', type=int, default=55)
    parser.add_argument('--rsi_sell', type=int, default=45)
    parser.add_argument('--adx_min', type=int, default=15)
    parser.add_argument('--logic', type=str, default='any', choices=['strict','any'], help="'strict' = tendencia & fuerza & momentum & cierre fuera; 'any' = tendencia & fuerza & (momentum OR cierre fuera)")
    parser.add_argument('--min_atr_pct', type=float, default=0.0012, help='Límite inferior de ATR% (0.0012 = 0.12%)')
    parser.add_argument('--max_atr_pct', type=float, default=0.012, help='Límite superior de ATR% (0.012 = 1.2%)')
    parser.add_argument('--be_trigger', type=float, default=0.0045, help='Activación de breakeven (fracción, 0.0045 = 0.45%)')
    parser.add_argument('--cooldown', type=int, default=10, help='Barras de enfriamiento tras SL')
    parser.add_argument('--hhll_lookback', type=int, default=10, help='Lookback para HH/LL en velas 5m (rupturas)')
    parser.add_argument('--time_exit_bars', type=int, default=30, help='Barras máximas a mantener una posición OOS antes de cerrarla (0 desactiva)')
    parser.add_argument('--max_dist_emaslow', type=float, default=0.010, help='Distancia máxima relativa a EMA_slow para abrir (0.010 = 1.0%)')
    parser.add_argument('--fresh_cross_max_bars', type=int, default=3, help='Máximo de barras para considerar cruce EMA/RSI como reciente')
    parser.add_argument('--require_rsi_cross', dest='require_rsi_cross', action='store_true', help='Exigir cruce reciente de RSI (ON)')
    parser.add_argument('--no_require_rsi_cross', dest='require_rsi_cross', action='store_false', help='No exigir cruce reciente de RSI')
    parser.set_defaults(require_rsi_cross=True)
    parser.add_argument('--min_ema_spread', type=float, default=0.001, help='Mínima separación relativa entre EMA_f y EMA_s (0.001 = 0.1%)')
    parser.add_argument('--require_close_vs_emas', dest='require_close_vs_emas', action='store_true', help='Exigir close>EMA_f>EMA_s (long) o close<EMA_f<EMA_s (short)')
    parser.add_argument('--no_require_close_vs_emas', dest='require_close_vs_emas', action='store_false', help='No exigir relación close/EMAs')
    parser.set_defaults(require_close_vs_emas=True)
    parser.add_argument('--min_vol_ratio', type=float, default=1.1, help='Volumen mínimo como múltiplo de su media (1.2 = 120%)')
    parser.add_argument('--vol_ma_len', type=int, default=30, help='Ventana para media de volumen')
    parser.add_argument('--adx_slope_len', type=int, default=3, help='Barras para calcular pendiente de ADX')
    parser.add_argument('--adx_slope_min', type=float, default=0.4, help='Pendiente mínima de ADX requerida')
    parser.add_argument('--fresh_breakout_only', dest='fresh_breakout_only', action='store_true', help='Exigir que el breakout HH/LL sea el primero (fresco)')
    parser.add_argument('--no_fresh_breakout_only', dest='fresh_breakout_only', action='store_false', help='Permitir rupturas no frescas')
    parser.set_defaults(fresh_breakout_only=False)
    parser.add_argument('--portfolio_mode', action='store_true', help='Simular todos los símbolos compartiendo el mismo capital')
    parser.add_argument('--portfolio_max_positions', type=int, default=3, help='Máximo de posiciones abiertas simultáneas (0 = sin límite)')
    parser.add_argument('--portfolio_alloc_pct', type=float, default=0.30, help='Fracción máxima del equity que puede asignar cada trade (0.30 = 30%)')
    parser.add_argument('--portfolio_eval_best', action='store_true', help='Después de un sweep, evalúa el portafolio usando el archivo exportado en --export_best')
    parser.add_argument('--funding_csv', type=str, default=None, help='CSV opcional con historial de funding (symbol,time,rate)')
    parser.add_argument('--sweep', type=str, default=None, help='Ruta a JSON con listas de parámetros para barrido (grid)')
    parser.add_argument('--tp_mode', type=str, default='fixed', choices=['fixed','atrx','none'], help="Modo de TP: 'fixed'=% , 'atrx'=k*ATR, 'none'=sin TP")
    parser.add_argument('--tp_atr_mult', type=float, default=0.0, help='Multiplicador de ATR para TP cuando tp_mode=atrx')
    parser.add_argument('--sl_mode', type=str, default='atr_then_trailing', choices=['atr_then_trailing','atr_trailing_only','percent'], help="Modo de SL: ATR fijo y luego trailing, solo trailing por ATR, o porcentaje fijo")
    parser.add_argument('--sl_pct', type=float, default=0.0, help='SL en porcentaje (0.01 = 1%) cuando sl_mode=percent')
    parser.add_argument('--out_dir', type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument('--rank_by', type=str, default='pnl_net', choices=['pnl_net','sharpe_annual','profit_factor','winrate_pct','cost_ratio','max_dd_pct','calmar'], help="Métrica para ranking y selección por símbolo. 'calmar' = pnl_net / |max_dd_pct|")
    parser.add_argument('--min_trades', type=int, default=0, help='Filtra combinaciones con menos de este número de trades')
    parser.add_argument('--min_winrate', type=float, default=0.0, help='Filtra combinaciones con winrate &lt; a este porcentaje')
    parser.add_argument('--max_cost_ratio', type=float, default=None, help='Filtra combinaciones con cost_ratio &gt; a este valor')
    parser.add_argument('--max_dd', type=float, default=None, help='Filtra combinaciones con drawdown máximo absoluto mayor a este valor (ej. 0.3 para 30%)')
    parser.add_argument('--export_best', type=str, default=None, help='Ruta a JSON para exportar el mejor set por símbolo listo para producción')
    parser.add_argument('--search_mode', type=str, default='grid', choices=['grid','random'], help='Estrategia de búsqueda: grid completo o muestreo aleatorio')
    parser.add_argument('--n_trials', type=int, default=None, help='Número de combinaciones aleatorias por símbolo (solo search_mode=random)')
    parser.add_argument('--random_seed', type=int, default=42, help='Semilla para el muestreo aleatorio')
    parser.add_argument('--second_pass', action='store_true', help='Hacer una segunda pasada local alrededor de los mejores por símbolo')
    parser.add_argument('--second_topk', type=int, default=1, help='Cuántos mejores por símbolo usar como base para la segunda pasada')

    args = parser.parse_args()

    symbols_raw = [s.strip() for s in args.symbols.split(',') if s.strip()]
    auto_symbols = (len(symbols_raw) == 1 and symbols_raw[0].lower() in ('auto', 'all'))
    file_symbols = _load_symbols_file(SYMBOLS_FILE) if auto_symbols else []
    if auto_symbols and file_symbols:
        symbols = file_symbols
    else:
        symbols = symbols_raw
    funding_map = _load_funding_events_csv(args.funding_csv)

    if args.portfolio_mode:
        if args.sweep:
            raise SystemExit("--portfolio_mode no es compatible con --sweep en esta versión.")
        trades_map: Dict[str, List[Trade]] = {}
        symbol_results = []
        for sym in symbols:
            df_sym = load_candles(args.data_template, sym)
            bt = Backtester(
                symbol=sym,
                df5m=df_sym,
                initial_equity=args.capital,
                tp_pct=args.tp,
                tp_mode=args.tp_mode,
                tp_atr_mult=args.tp_atr_mult,
                atr_mult_sl=args.atr_mult,
                sl_mode=args.sl_mode,
                sl_pct=args.sl_pct,
                ema_fast=args.ema_fast,
                ema_slow=args.ema_slow,
                rsi_buy=args.rsi_buy,
                rsi_sell=args.rsi_sell,
                adx_min=args.adx_min,
                min_atr_pct=args.min_atr_pct,
                max_atr_pct=args.max_atr_pct,
                be_trigger=args.be_trigger,
                cooldown_bars=args.cooldown,
                logic=args.logic,
                hhll_lookback=args.hhll_lookback,
                time_exit_bars=args.time_exit_bars,
                max_dist_emaslow=args.max_dist_emaslow,
                fresh_cross_max_bars=args.fresh_cross_max_bars,
                require_rsi_cross=args.require_rsi_cross,
                min_ema_spread=args.min_ema_spread,
                require_close_vs_emas=args.require_close_vs_emas,
                min_vol_ratio=args.min_vol_ratio,
                vol_ma_len=args.vol_ma_len,
                adx_slope_len=args.adx_slope_len,
                adx_slope_min=args.adx_slope_min,
                fresh_breakout_only=args.fresh_breakout_only,
                funding_events=_normalize_funding_events_for(sym, funding_map),
            )
            res_sym = bt.run(train_ratio=0.7)
            symbol_results.append(res_sym)
            trades_map[sym.upper()] = bt.trades

        portfolio = simulate_portfolio(
            trades_map,
            initial_equity=args.capital,
            max_positions=args.portfolio_max_positions,
            max_alloc_pct=args.portfolio_alloc_pct,
        )

        print("[PORTFOLIO] Resumen por símbolo:")
        for res in symbol_results:
            print(f"  {res['symbol']}: trades={res['trades']} pnl={res['pnl_net']} winrate={res['winrate_pct']}%")

        print("\n[PORTFOLIO] Resultado agregado:")
        print(f"  Trades ejecutados: {portfolio['trades']} (omitidos: {portfolio['skipped_trades']})")
        print(f"  PnL neto: {portfolio['pnl_net']} | Winrate: {portfolio['winrate_pct']}%")
        print(f"  Max DD: {portfolio['max_dd_pct']} | Sharpe anualizado: {portfolio['sharpe_annual']}")
        print(f"  Cost ratio: {portfolio['cost_ratio']} (comisiones={portfolio['commissions']}, slip={portfolio['slippage_cost']}, funding={portfolio['funding_cost']})")
        return
    # Si es CSV único (sin {symbol}), ajusta lista de símbolos a los realmente disponibles
    csv_unique_mode = ('{symbol}' not in args.data_template)
    sym_map = {}
    available_symbols = []
    if csv_unique_mode:
        raw_syms = _list_symbols_from_csv(args.data_template)
        # Mapa normalizado -> original
        sym_map = {_norm_symbol(s): s for s in raw_syms}
        if auto_symbols and not file_symbols:
            # Usa todos los símbolos del CSV (acotado a 12 por seguridad)
            available_symbols = [sym_map[k] for k in list(sym_map.keys())[:12]]
        else:
            # Filtra los solicitados por disponibilidad real
            for s in symbols:
                ns = _norm_symbol(s)
                if ns in sym_map:
                    available_symbols.append(sym_map[ns])
        if not available_symbols:
            print(f"[WARN] Ninguno de los símbolos solicitados está en {args.data_template}. Usando el primero disponible del CSV si existe.")
            if sym_map:
                available_symbols = [list(sym_map.values())[0]]
        symbols = available_symbols
        print(f"[INFO] Símbolos usados: {symbols}")
    elif auto_symbols and not file_symbols:
        print("[WARN] --symbols=auto requiere pkg/symbols.json cuando data_template tiene {symbol}.")
    os.makedirs(args.out_dir, exist_ok=True)


    # ==========================
    # Modo barrido (grid) opcional vía --sweep <config.json> o --preset <nombre>
    # ==========================
    if args.sweep is not None:
        import json as _json
        with open(args.sweep, 'r') as f:
            cfg = _json.load(f)
        print(f"[SWEEP] Usando config desde JSON: {args.sweep}")
        # Helper para obtener listas desde cfg (si viene escalar, lo volvemos lista)
        def as_list(x):
            if x is None:
                return [None]
            if isinstance(x, list):
                return x
            return [x]

        # Símbolos: si en cfg dice 'auto', usamos la lista ya resuelta
        sweep_symbols = cfg.get('symbols', symbols)
        if isinstance(sweep_symbols, str):
            if sweep_symbols.lower() in ('auto', 'all'):
                sweep_symbols = symbols
            else:
                sweep_symbols = [s.strip() for s in sweep_symbols.split(',') if s.strip()]

        print(f"[SWEEP] data_template: {args.data_template} | out_dir: {args.out_dir}")

        keys = [
            ('tp', as_list(cfg.get('tp', args.tp))),
            ('tp_mode', as_list(cfg.get('tp_mode', args.tp_mode))),
            ('tp_atr_mult', as_list(cfg.get('tp_atr_mult', args.tp_atr_mult))),
            ('ema_fast', as_list(cfg.get('ema_fast', args.ema_fast))),
            ('ema_slow', as_list(cfg.get('ema_slow', args.ema_slow))),
            ('rsi_buy', as_list(cfg.get('rsi_buy', args.rsi_buy))),
            ('rsi_sell', as_list(cfg.get('rsi_sell', args.rsi_sell))),
            ('adx_min', as_list(cfg.get('adx_min', args.adx_min))),
            ('min_atr_pct', as_list(cfg.get('min_atr_pct', args.min_atr_pct))),
            ('max_atr_pct', as_list(cfg.get('max_atr_pct', args.max_atr_pct))),
            ('atr_mult', as_list(cfg.get('atr_mult', args.atr_mult))),
            ('sl_mode', as_list(cfg.get('sl_mode', args.sl_mode))),
            ('sl_pct', as_list(cfg.get('sl_pct', args.sl_pct))),
            ('be_trigger', as_list(cfg.get('be_trigger', args.be_trigger))),
            ('cooldown', as_list(cfg.get('cooldown', args.cooldown))),
            ('logic', as_list(cfg.get('logic', args.logic))),
            ('hhll_lookback', as_list(cfg.get('hhll_lookback', args.hhll_lookback))),
            ('time_exit_bars', as_list(cfg.get('time_exit_bars', args.time_exit_bars))),
            ('max_dist_emaslow', as_list(cfg.get('max_dist_emaslow', args.max_dist_emaslow))), 
            ('fresh_cross_max_bars', as_list(cfg.get('fresh_cross_max_bars', args.fresh_cross_max_bars))),
            ('require_rsi_cross', as_list(cfg.get('require_rsi_cross', args.require_rsi_cross))),
            # Endurecedores nuevos:
            ('min_ema_spread', as_list(cfg.get('min_ema_spread', args.min_ema_spread))),
            ('require_close_vs_emas', as_list(cfg.get('require_close_vs_emas', args.require_close_vs_emas))),
            ('min_vol_ratio', as_list(cfg.get('min_vol_ratio', args.min_vol_ratio))),
            ('vol_ma_len', as_list(cfg.get('vol_ma_len', args.vol_ma_len))),
            ('adx_slope_len', as_list(cfg.get('adx_slope_len', args.adx_slope_len))),
            ('adx_slope_min', as_list(cfg.get('adx_slope_min', args.adx_slope_min))),
            ('fresh_breakout_only', as_list(cfg.get('fresh_breakout_only', args.fresh_breakout_only))),
        ]

        # Modo de búsqueda: grid vs random
        if args.search_mode == 'random':
            import random as _rnd
            _rnd.seed(args.random_seed)
            value_lists = [v for _, v in keys]
            combos = []
            seen = set()
            trials = args.n_trials or 100
            # Genera "trials" combinaciones aleatorias (sin repetir) por símbolo
            # Nota: los símbolos se multiplican después; aquí generamos la cesta base
            while len(combos) < trials:
                pick = tuple(_rnd.choice(v) for v in value_lists)
                if pick not in seen:
                    seen.add(pick)
                    combos.append(pick)
            print(f"[SWEEP] Search=random | trials base={len(combos)}")
        else:
            combos = list(itertools.product(*[v for _, v in keys]))
            print(f"[SWEEP] Search=grid | combos base={len(combos)}")

        # Construye trabajos
        jobs = []
        for sym in sweep_symbols:
            for values in combos:
                params = dict(zip([k for k, _ in keys], values))
                jobs.append((sym, params, args.data_template, args.capital, args.out_dir, funding_map))

        os.makedirs(args.out_dir, exist_ok=True)
        print(f"[SWEEP] Total jobs: {len(jobs)} (symbols={len(sweep_symbols)}, base={len(combos)})")

        # Paraleliza por CPU (cap en 8)
        procs = min(cpu_count(), 8)
        results = []
        total_jobs = len(jobs)
        with Pool(processes=procs) as pool:
            for i, res in enumerate(pool.imap_unordered(run_job, jobs, chunksize=1), start=1):
                results.append(res)
                _print_progress(i, total_jobs)

        import pandas as _pd
        dfres = _pd.DataFrame(results)
        if 'error' in dfres.columns:
            errs = dfres[~dfres['error'].isna()]
            if len(errs):
                print("[SWEEP][WARN] Errores en algunos jobs:")
                print(errs[['symbol','error']].head())
        # ---- Filtros de calidad con diagnóstico ----
        base = dfres[dfres['pnl_net'].notna()].copy()
        print(f"[SWEEP][DIAG] combos con pnl válido: {len(base)} / {len(dfres)}")
        good = base.copy()
        if args.min_trades:
            before = len(good)
            good = good[good['trades'].astype(float) >= args.min_trades]
            print(f"[SWEEP][DIAG] tras min_trades>={args.min_trades}: {len(good)} (−{before-len(good)})")
        if args.min_winrate > 0.0:
            before = len(good)
            good = good[good['winrate_pct'].astype(float) >= args.min_winrate]
            print(f"[SWEEP][DIAG] tras winrate>={args.min_winrate}%: {len(good)} (−{before-len(good)})")
        if args.max_cost_ratio is not None:
            before = len(good)
            good = good[good['cost_ratio'].astype(float) <= args.max_cost_ratio]
            print(f"[SWEEP][DIAG] tras cost_ratio<={args.max_cost_ratio}: {len(good)} (−{before-len(good)})")
        if args.max_dd is not None:
            before = len(good)
            good = good[good['max_dd_pct'].abs().astype(float) <= abs(args.max_dd)]
            print(f"[SWEEP][DIAG] tras max_dd<={abs(args.max_dd)}: {len(good)} (−{before-len(good)})")

        # ---- Ranking por métrica seleccionada ----
        if len(good):
            rb = args.rank_by
            if rb == 'cost_ratio':
                metric = -good.get('cost_ratio', np.nan).astype(float)
            elif rb == 'max_dd_pct':
                metric = -good.get('max_dd_pct', np.nan).abs().astype(float)
            elif rb == 'calmar':
                dd = good.get('max_dd_pct', np.nan).abs().astype(float).replace(0, np.nan)
                metric = good.get('pnl_net', np.nan).astype(float) / dd
            else:
                metric = good.get(rb, np.nan).astype(float)
            metric = metric.where(metric.notna(), other=-np.inf)
            good['__metric__'] = metric
            symbol_best = good.sort_values(['symbol','__metric__'], ascending=[True, False]).groupby('symbol').head(1).copy()

            # Exportar resumen (JSON) y config de producción
            out_json = os.path.join(args.out_dir, 'resumen.json')
            with open(out_json, 'w') as f:
                f.write(good.drop(columns=['__metric__'], errors='ignore').to_json(orient='records', indent=2))
            print(f"[SWEEP] Resumen guardado en {out_json}")

            if args.export_best and len(symbol_best):
                prod = []
                for _, r in symbol_best.iterrows():
                    prod.append({'symbol': r['symbol'], 'params': r['params']})
                outp = args.export_best if os.path.isabs(args.export_best) else os.path.join(args.out_dir, args.export_best)
                try:
                    out_dirname = os.path.dirname(outp)
                    if out_dirname:
                        os.makedirs(out_dirname, exist_ok=True)
                except Exception:
                    pass
                with open(outp, 'w') as f:
                    json.dump(prod, f, indent=2)
                print(f"[SWEEP] Config de producción exportada en {outp}")
                # Copia a pkg/best_prod.json para que el runtime lo lea directo
                try:
                    _pkg_best = os.path.join(os.path.dirname(__file__), 'best_prod.json')
                    with open(_pkg_best, 'w') as _pf:
                        json.dump(prod, _pf, indent=2)
                    print(f"[SWEEP] Copia de producción para pkg en {_pkg_best}")
                except Exception as _e:
                    print(f"[SWEEP][WARN] No se pudo escribir copia en pkg: {_e}")
                # ---- Resumen en consola (ganancia por portafolio usando best por símbolo) ----
                try:
                    tot_pnl = float(symbol_best['pnl_net'].astype(float).sum()) if 'pnl_net' in symbol_best.columns else float('nan')
                    tot_trades = int(symbol_best['trades'].astype(float).sum()) if 'trades' in symbol_best.columns else 0
                    avg_win = float(symbol_best['winrate_pct'].astype(float).mean()) if 'winrate_pct' in symbol_best.columns else float('nan')
                    worst_dd = float(symbol_best['max_dd_pct'].astype(float).min()) if 'max_dd_pct' in symbol_best.columns else float('nan')
                    # worst_dd viene como fracción negativa; mostramos porcentaje absoluto
                    if not np.isnan(worst_dd):
                        worst_dd_str = f"{abs(worst_dd)*100:.2f}%"
                    else:
                        worst_dd_str = "N/A"
                    print(f"[RESUMEN] Portafolio (best por símbolo) → N={len(symbol_best)} | PnL Neto Total={tot_pnl:.2f} | Trades={tot_trades} | Winrate Prom={avg_win:.1f}% | MaxDD peor={worst_dd_str}")
                except Exception as _e:
                    print(f"[RESUMEN][WARN] No se pudo calcular el resumen de portafolio: {_e}")

                if args.portfolio_eval_best:
                    _run_portfolio_from_best(outp, args, funding_map)

                # ---- Segunda pasada local (vecindad) opcional ----
                if args.second_pass and len(symbol_best):
                    print("[SECOND] Iniciando segunda pasada local (vecindad)")
                    import random as _rnd
                    _rnd.seed(int(args.random_seed) + 1)

                    # Armar seeds: top-K por símbolo (según __metric__ ya calculado)
                    try:
                        topk = max(1, int(args.second_topk))
                    except Exception:
                        topk = 1
                    seeds = (
                        good.sort_values(['symbol','__metric__'], ascending=[True, False])
                            .groupby('symbol')
                            .head(topk)
                            .copy()
                    )

                    # Helpers de vecindad
                    tp_list = None
                    try:
                        if isinstance(cfg, dict):
                            tp_list = cfg.get('tp', None)
                    except Exception:
                        tp_list = None
                    def nearest_tp_vals(base):
                        try:
                            b = float(base)
                        except Exception:
                            return [base]
                        if isinstance(tp_list, list) and len(tp_list):
                            s = sorted(set([float(x) for x in tp_list]))
                            if b in s:
                                i = s.index(b)
                                cand = [b]
                                if i-1 >= 0: cand.append(s[i-1])
                                if i+1 < len(s): cand.append(s[i+1])
                                return sorted(set(cand))
                        # fallback ±0.002 con límites sensatos
                        lo, hi = 0.005, 0.03
                        return sorted(set([max(lo, round(b-0.002, 6)), b, min(hi, round(b+0.002, 6))]))

                    def clamp(v, lo, hi):
                        try:
                            vv = float(v)
                        except Exception:
                            return v
                        return max(lo, min(hi, vv))

                    # Construye trabajos de vecindad
                    neigh_jobs = []
                    for _, row in seeds.iterrows():
                        sym = row['symbol']
                        p0 = row['params']
                        if not isinstance(p0, dict):
                            continue
                        # Dominio pequeño alrededor del seed
                        tp_vals = nearest_tp_vals(p0.get('tp', 0.015))
                        rsi_buy0 = int(p0.get('rsi_buy', 55))
                        rsi_sell0 = int(p0.get('rsi_sell', 45))
                        adx0 = int(p0.get('adx_min', 20))
                        min_atr0 = float(p0.get('min_atr_pct', 0.0015))
                        max_atr0 = float(p0.get('max_atr_pct', 0.03))
                        be0 = float(p0.get('be_trigger', 0.0045))
                        logic0 = str(p0.get('logic', 'any'))

                        rsi_buy_vals = sorted(set([clamp(rsi_buy0-2, 45, 65), rsi_buy0, clamp(rsi_buy0+2, 45, 65)]))
                        rsi_sell_vals = sorted(set([clamp(rsi_sell0-2, 35, 60), rsi_sell0, clamp(rsi_sell0+2, 35, 60)]))
                        adx_vals = sorted(set([int(clamp(adx0-4, 15, 50)), int(clamp(adx0, 15, 50)), int(clamp(adx0+4, 15, 50))]))
                        min_atr_vals = sorted(set([round(clamp(min_atr0-0.0005, 0.0008, 0.004), 6), round(min_atr0, 6), round(clamp(min_atr0+0.0005, 0.0008, 0.004), 6)]))
                        max_atr_vals = sorted(set([min(max_atr0, 0.03), 0.04]))
                        atr_mult_vals = [1.8, 2.0, 2.2]
                        be_vals = sorted(set([round(clamp(be0-0.001, 0.0035, 0.0055), 6), round(be0, 6), round(clamp(be0+0.001, 0.0035, 0.0055), 6)]))
                        logic_vals = [logic0, ('strict' if logic0 == 'any' else 'any')]

                        # Muestreo aleatorio de la vecindad (~24 combos por seed)
                        dims = [tp_vals, rsi_buy_vals, rsi_sell_vals, adx_vals, min_atr_vals, max_atr_vals, atr_mult_vals, be_vals, logic_vals]
                        trials_local = 24
                        seen_local = set()
                        while len(seen_local) < trials_local:
                            pick = (
                                _rnd.choice(tp_vals),
                                _rnd.choice(rsi_buy_vals),
                                _rnd.choice(rsi_sell_vals),
                                _rnd.choice(adx_vals),
                                _rnd.choice(min_atr_vals),
                                _rnd.choice(max_atr_vals),
                                _rnd.choice(atr_mult_vals),
                                _rnd.choice(be_vals),
                                _rnd.choice(logic_vals),
                            )
                            if pick in seen_local:
                                continue
                            seen_local.add(pick)
                            tp, rsi_b, rs, adxm, minatr, maxatr, atrm, be, lg = pick
                            p = dict(p0)  # copia del seed
                            p.update({
                                'tp': float(tp),
                                'tp_mode': 'fixed',
                                'tp_atr_mult': 0.0,
                                'rsi_buy': int(rsi_b),
                                'rsi_sell': int(rs),
                                'adx_min': int(adxm),
                                'min_atr_pct': float(minatr),
                                'max_atr_pct': float(maxatr),
                                'atr_mult': float(atrm),
                                'be_trigger': float(be),
                                'logic': str(lg),
                            })
                            neigh_jobs.append((sym, p, args.data_template, args.capital, args.out_dir, funding_map))

                    if len(neigh_jobs):
                        print(f"[SECOND] Vecindad jobs: {len(neigh_jobs)}")
                        procs2 = min(cpu_count(), 8)
                        results2 = []
                        with Pool(processes=procs2) as pool:
                            for i, res in enumerate(pool.imap_unordered(run_job, neigh_jobs, chunksize=1), start=1):
                                results2.append(res)
                                _print_progress(i, len(neigh_jobs), prefix='[SECOND] Progreso')

                        df2 = _pd.DataFrame(results2)
                        df2 = df2[df2['pnl_net'].notna()].copy()
                        # Reaplica filtros de calidad
                        if args.min_trades:
                            df2 = df2[df2['trades'].astype(float) >= args.min_trades]
                        if args.min_winrate > 0.0:
                            df2 = df2[df2['winrate_pct'].astype(float) >= args.min_winrate]
                        if args.max_cost_ratio is not None:
                            df2 = df2[df2['cost_ratio'].astype(float) <= args.max_cost_ratio]
                        if args.max_dd is not None:
                            df2 = df2[df2['max_dd_pct'].abs().astype(float) <= abs(args.max_dd)]

                        if len(df2):
                            # Métrica y comparación contra best inicial
                            rank_key = args.rank_by
                            if rank_key == 'cost_ratio':
                                metric2 = -df2.get('cost_ratio', np.nan).astype(float)
                            elif rank_key == 'max_dd_pct':
                                metric2 = -df2.get('max_dd_pct', np.nan).abs().astype(float)
                            elif rank_key == 'calmar':
                                dd2 = df2.get('max_dd_pct', np.nan).abs().astype(float).replace(0, np.nan)
                                metric2 = df2.get('pnl_net', np.nan).astype(float) / dd2
                            else:
                                metric2 = df2.get(rank_key, np.nan).astype(float)
                            metric2 = metric2.where(metric2.notna(), other=-np.inf)
                            df2['__metric__'] = metric2

                            # Merge por símbolo el mejor entre initial y second
                            current_best = symbol_best.copy()
                            cand_best = df2.sort_values(['symbol','__metric__'], ascending=[True, False]).groupby('symbol').head(1).copy()
                            merged = current_best.merge(cand_best[['symbol','__metric__']], on='symbol', how='left', suffixes=('', '_cand'))
                            take_cand = merged['__metric___cand'].notna() & (merged['__metric___cand'] > merged['__metric__'])
                            # Construye nuevo symbol_best
                            improved = []
                            for sym in current_best['symbol'].unique():
                                c0 = current_best[current_best['symbol'] == sym]
                                c1 = cand_best[cand_best['symbol'] == sym]
                                if len(c1) and (len(c0)==0 or float(c1['__metric__'].iloc[0]) > float(c0['__metric__'].iloc[0])):
                                    improved.append(c1.iloc[0])
                                else:
                                    improved.append(c0.iloc[0])
                            symbol_best = _pd.DataFrame(improved)
                            # Re-exporta best_prod.json final
                            prod2 = []
                            for _, r in symbol_best.iterrows():
                                prod2.append({'symbol': r['symbol'], 'params': r['params']})
                            outp2 = args.export_best if os.path.isabs(args.export_best) else os.path.join(args.out_dir, args.export_best)
                            try:
                                out_dirname2 = os.path.dirname(outp2)
                                if out_dirname2:
                                    os.makedirs(out_dirname2, exist_ok=True)
                            except Exception:
                                pass
                            with open(outp2, 'w') as f:
                                json.dump(prod2, f, indent=2)
                            print(f"[SECOND] Refinamiento completado. Config de producción actualizada en {outp2}")
                            # Copia a pkg/best_prod.json tras refinamiento
                            try:
                                _pkg_best2 = os.path.join(os.path.dirname(__file__), 'best_prod.json')
                                with open(_pkg_best2, 'w') as _pf2:
                                    json.dump(prod2, _pf2, indent=2)
                                print(f"[SECOND] Copia de producción para pkg en {_pkg_best2}")
                            except Exception as _e:
                                print(f"[SECOND][WARN] No se pudo escribir copia en pkg: {_e}")
                            # Resumen consola post-second
                            try:
                                tot_pnl2 = float(symbol_best['pnl_net'].astype(float).sum()) if 'pnl_net' in symbol_best.columns else float('nan')
                                tot_tr2 = int(symbol_best['trades'].astype(float).sum()) if 'trades' in symbol_best.columns else 0
                                avg_win2 = float(symbol_best['winrate_pct'].astype(float).mean()) if 'winrate_pct' in symbol_best.columns else float('nan')
                                worst_dd2 = float(symbol_best['max_dd_pct'].astype(float).min()) if 'max_dd_pct' in symbol_best.columns else float('nan')
                                worst_dd2_str = f"{abs(worst_dd2)*100:.2f}%" if not np.isnan(worst_dd2) else "N/A"
                                print(f"[RESUMEN][SECOND] Portafolio → N={len(symbol_best)} | PnL Neto Total={tot_pnl2:.2f} | Trades={tot_tr2} | Winrate Prom={avg_win2:.1f}% | MaxDD peor={worst_dd2_str}")
                            except Exception as _e:
                                print(f"[SECOND][WARN] No se pudo calcular resumen: {_e}")
                    else:
                        print('[SECOND] No se generaron jobs de vecindad')
            elif args.portfolio_eval_best:
                print('[PORTFOLIO][WARN] --portfolio_eval_best requiere --export_best para poder leer los parámetros.')
        else:
            print('[SWEEP] No hubo resultados válidos')
            # Top-5 por pnl_net ignorando filtros para diagnóstico
            try:
                tmp = dfres[dfres['pnl_net'].notna()].copy()
                if len(tmp):
                    top5 = tmp.sort_values('pnl_net', ascending=False).head(5)
                    print('[SWEEP][DIAG] Top-5 sin filtros:')
                    cols = ['symbol','pnl_net','trades','winrate_pct','cost_ratio','max_dd_pct','params']
                    print(top5[cols].to_string(index=False))
            except Exception as _e:
                print(f"[SWEEP][DIAG][WARN] No se pudo imprimir top-5: {_e}")

        return  # Termina main en modo sweep

    summary = []
    for sym in symbols:
        try:
            df = load_candles(args.data_template, sym)
        except Exception as e:
            print(f"[SKIP] {sym}: {e}")
            continue
        p = {
            'tp_pct': args.tp,
            'tp_mode': args.tp_mode,
            'tp_atr_mult': args.tp_atr_mult,
            'atr_mult_sl': args.atr_mult,
            'sl_mode': args.sl_mode,
            'sl_pct': args.sl_pct,
            'ema_fast': args.ema_fast,
            'ema_slow': args.ema_slow,
            'rsi_buy': args.rsi_buy,
            'rsi_sell': args.rsi_sell,
            'adx_min': args.adx_min,
            'min_atr_pct': args.min_atr_pct,
            'max_atr_pct': args.max_atr_pct,
            'max_dist_emaslow': args.max_dist_emaslow,
            'be_trigger': args.be_trigger,
            'cooldown_bars': args.cooldown,
            'logic': args.logic,
            'fresh_cross_max_bars': args.fresh_cross_max_bars,
            'require_rsi_cross': args.require_rsi_cross,
            # Nuevos endurecedores:
            'min_ema_spread': args.min_ema_spread,
            'require_close_vs_emas': args.require_close_vs_emas,
            'min_vol_ratio': args.min_vol_ratio,
            'vol_ma_len': args.vol_ma_len,
            'adx_slope_len': args.adx_slope_len,
            'adx_slope_min': args.adx_slope_min,
            'fresh_breakout_only': args.fresh_breakout_only,
        }
        
        bt = Backtester(
            symbol=sym,
            df5m=df,
            initial_equity=args.capital,
            tp_pct=p['tp_pct'],
            tp_mode=p['tp_mode'],
            tp_atr_mult=p['tp_atr_mult'],
            atr_mult_sl=p['atr_mult_sl'],
            sl_mode=p['sl_mode'],
            sl_pct=p['sl_pct'],
            ema_fast=p['ema_fast'],
            ema_slow=p['ema_slow'],
            rsi_buy=p['rsi_buy'],
            rsi_sell=p['rsi_sell'],
            adx_min=p['adx_min'],
            min_atr_pct=p['min_atr_pct'],
            max_atr_pct=p['max_atr_pct'],
            be_trigger=p['be_trigger'],
            cooldown_bars=p['cooldown_bars'],
            logic=p['logic'],
            hhll_lookback=args.hhll_lookback,
            time_exit_bars=args.time_exit_bars,
            max_dist_emaslow=p['max_dist_emaslow'],
            fresh_cross_max_bars=p['fresh_cross_max_bars'],
            require_rsi_cross=p['require_rsi_cross'],
            min_ema_spread=p['min_ema_spread'],
            require_close_vs_emas=p['require_close_vs_emas'],
            min_vol_ratio=p['min_vol_ratio'],
            vol_ma_len=p['vol_ma_len'],
            adx_slope_len=p['adx_slope_len'],
            adx_slope_min=p['adx_slope_min'],
            fresh_breakout_only=p['fresh_breakout_only'],
        )
        res = bt.run(train_ratio=0.7)
        summary.append(res)
        print(f"[{sym}] Trades: {res['trades']}, PnL neto: {res['pnl_net']}, Winrate: {res['winrate_pct']}%, CostRatio: {res['cost_ratio']}")

    out_json = os.path.join(args.out_dir, 'resumen.json')
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Resumen guardado en {out_json}")
    # ---- Resumen en consola (ganancia total y métricas básicas) ----
    try:
        if isinstance(summary, list) and len(summary):
            import math as _math
            pnl_vals = [float(r.get('pnl_net', 0.0)) for r in summary if r is not None]
            trades_vals = [int(r.get('trades', 0) or 0) for r in summary if r is not None]
            win_vals = [float(r.get('winrate_pct', float('nan'))) for r in summary if r is not None]
            dd_vals = [float(r.get('max_dd_pct', float('nan'))) for r in summary if r is not None]
            tot_pnl = float(np.nansum(pnl_vals)) if len(pnl_vals) else 0.0
            tot_trades = int(np.nansum(trades_vals)) if len(trades_vals) else 0
            avg_win = float(np.nanmean(win_vals)) if len(win_vals) else float('nan')
            worst_dd = float(np.nanmin(dd_vals)) if len(dd_vals) else float('nan')
            worst_dd_str = f"{abs(worst_dd)*100:.2f}%" if not _math.isnan(worst_dd) else "N/A"
            print(f"[RESUMEN] Portafolio (run directo) → N={len(summary)} | PnL Neto Total={tot_pnl:.2f} | Trades={tot_trades} | Winrate Prom={avg_win:.1f}% | MaxDD peor={worst_dd_str}")
    except Exception as _e:
        print(f"[RESUMEN][WARN] No se pudo calcular resumen: {_e}")


if __name__ == '__main__':
    main()
