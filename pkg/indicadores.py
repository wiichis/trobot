# indicadores.py – versión mínima

import os
from functools import lru_cache

import numpy as np
import pandas as pd
import talib
from pathlib import Path
import json
from .cfg_loader import load_best_symbols
from datetime import datetime, timedelta  # NUEVO: para purgar registros antiguos

# === Parámetros base (alineados al backtest) ===
# === Parámetros base (alineados al backtest) ===
RSI_P, ATR_P, EMA_S, EMA_L, ADX_P = 14, 14, 8, 30, 14
# Escalonado de TP: factores para ladder de take-profit
TP_FACTORS = (0.6, 1.0, 1.6)  # escalonado: 60%, 100%, 160% del tp base
# Multiplicadores legacy (NO usados en prod, mantenidos como fallback si no hay best_prod.json)
TP_M, SL_M, SIGNAL_MIN = 8, 2, 3
# Umbrales legacy (fallback)
RSI_HI, RSI_LO, ADX_MIN = 70, 35, 20

# Umbrales/filtros informativos (pueden quedar como columnas de diagnóstico)
VOL_THRESHOLD = 0.6
VOLAT_THRESHOLD = 0.9855
BASE = "./archivos"

# Defaults antes de leer config
MIN_CONFIRM_5M = 4  # (ya no se usa tras pasar a 5m puro)
BLOCKED_SYMBOLS = []  # puede ser sobreescrita por config

# --- Cargar mejores parámetros del backtest si existen ---
try:
    _cfg_path = Path(__file__).resolve().parent / "best_cfg.json"
    with open(_cfg_path) as fp:
        _best = json.load(fp)
    RSI_P   = _best.get("rsi", RSI_P)
    EMA_S   = _best.get("ema_s", EMA_S)
    EMA_L   = _best.get("ema_l", EMA_L)
    ADX_P   = _best.get("adx", ADX_P)
    TP_M    = _best.get("tp_mult", TP_M)
    SL_M    = _best.get("sl_mult", SL_M)
    RSI_HI  = _best.get("rsi_long", RSI_HI)
    RSI_LO  = _best.get("rsi_short", RSI_LO)
    ADX_MIN = _best.get("adx_min", ADX_MIN)
    VOL_THRESHOLD = _best.get("vol_thr", VOL_THRESHOLD)
    SIGNAL_MIN = _best.get("signal_min", SIGNAL_MIN)
    # Parámetros de confirmación 5m
    MIN_CONFIRM_5M = _best.get("min_confirm_5m", MIN_CONFIRM_5M)
    REL_V_MULT_5M  = _best.get("rel_v_mult_5m", 2.0)
    RSI5_LONG      = _best.get("rsi5_long", 55)
    RSI5_SHORT     = _best.get("rsi5_short", 45)
    RANK_LIMIT_5M  = _best.get("rank_limit_5m", 4)
    # Lista negra opcional desde config
    BLOCKED_SYMBOLS = _best.get("blocklist", BLOCKED_SYMBOLS)
except Exception as e:
    print(f"⚠️  No se pudo cargar best_cfg.json: {e}")

# === Whitelist generada por backtesting (opcional) ===
ALLOWED_SYMBOLS = []
OBS_SYMBOLS = []
BLACKLIST_SYMBOLS = []
try:
    _base_whitelist = load_best_symbols()
    if isinstance(_base_whitelist, (list, tuple)) and _base_whitelist:
        ALLOWED_SYMBOLS = [str(sym).upper() for sym in _base_whitelist if sym]
        print(f"✅ Whitelist activa ({len(ALLOWED_SYMBOLS)}): {', '.join(ALLOWED_SYMBOLS)}")
except Exception as _e:
    # Silencioso en producción; sólo informativo si se requiere debug
    pass

# --- Cargar parámetros por símbolo desde best_prod.json ---
def _load_best_prod_and_params(path: str):
    """
    Lee best_prod.json y devuelve:
      - whitelist: lista de símbolos en MAYÚSCULAS
      - params_map: dict {SYMBOL: {params...}} (vacío si no hay params)
    Soporta esquemas:
      * [{"symbol": "AVAX-USDT", "params": {...}}, ...]
      * ["AVAX-USDT", "BNB-USDT", ...]
      * {"AVAX-USDT": {...}, "BNB-USDT": {...}}
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return [], {}
    wl = []
    pm = {}
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                wl.append(item.upper())
            elif isinstance(item, dict):
                sym = str(item.get("symbol", "")).upper()
                if sym:
                    wl.append(sym)
                    p = item.get("params") or {}
                    pm[sym] = p if isinstance(p, dict) else {}
    elif isinstance(data, dict):
        for k, v in data.items():
            sym = str(k).upper()
            wl.append(sym)
            pm[sym] = v if isinstance(v, dict) else {}
    # de-dup ordenado
    wl = sorted(set(wl))
    return wl, pm

BEST_PROD_PATH = Path(__file__).resolve().parent / "best_prod.json"
PARAMS_BY_SYMBOL = {}
try:
  if BEST_PROD_PATH.exists():
      wl, params_map = _load_best_prod_and_params(str(BEST_PROD_PATH))
      PARAMS_BY_SYMBOL = params_map
      # Si no hay whitelist explícita, usa la derivada del best_prod.json
      if not ALLOWED_SYMBOLS and wl:
          ALLOWED_SYMBOLS = wl
          print(f"✅ Whitelist derivada de best_prod.json ({len(ALLOWED_SYMBOLS)})")
except Exception as _e:
    PARAMS_BY_SYMBOL = {}

def _apply_whitelist(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra por la whitelist si existe y si el DataFrame tiene columna 'symbol'."""
    if isinstance(df, pd.DataFrame) and 'symbol' in df.columns and ALLOWED_SYMBOLS:
        return df[df['symbol'].isin(ALLOWED_SYMBOLS)].copy()
    return df

# Parámetros para confirmación de 5 m
EMA_S_5M, EMA_L_5M, RSI_5M = 3, 7, 14
MAX_PER_SYMBOL = 300  # conservar hasta 300 velas recientes por símbolo

# --- Lista blanca/negra de rendimiento ---

# Días a mantener por símbolo en indicadores (para purga inteligente)
DAYS_KEEP = 10  # Días a mantener por símbolo en indicadores

PRICE_5  = f"{BASE}/cripto_price_5m.csv"
IND_CSV  = f"{BASE}/indicadores.csv"


def _load_cooldowns():
    """Cooldown desactivado: siempre retorna vacío."""
    return {}

# ---------- util ----------
@lru_cache(maxsize=4)
def _read(path, mtime=None):
    """Lee CSV con parseo de fecha y normaliza señales si existen."""
    df = pd.read_csv(
        path,
        parse_dates=["date"],
        na_values=["NA"]
    )
    if 'Long_Signal' in df.columns:
        df['Long_Signal'] = df['Long_Signal'].fillna(False).astype(bool)
    if 'Short_Signal' in df.columns:
        df['Short_Signal'] = df['Short_Signal'].fillna(False).astype(bool)
    return df


# ---------- indicadores ----------
def _calc_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    # Params por símbolo (fallbacks sensatos)
    p = (PARAMS_BY_SYMBOL.get(symbol.upper(), {}) or {})
    ema_f = int(p.get('ema_fast', EMA_S))
    ema_s = int(p.get('ema_slow', EMA_L))
    rsi_buy  = int(p.get('rsi_buy', 56))
    rsi_sell = int(p.get('rsi_sell', 44))
    adx_min  = int(p.get('adx_min', 28))
    min_atr  = float(p.get('min_atr_pct', 0.002))
    max_atr  = float(p.get('max_atr_pct', 0.03))
    tp_pct   = float(p.get('tp', 0.015))
    atr_mult = float(p.get('atr_mult', 2.0))
    logic    = str(p.get('logic', 'strict'))
    hhll_n   = int(p.get('hhll_lookback', 0) or 0)

    # --- Nuevos knobs alineados al backtesting (con defaults "lean") ---
    require_close_vs_emas = bool(p.get('require_close_vs_emas', True))
    min_ema_spread = float(p.get('min_ema_spread', 0.001))
    min_vol_ratio  = float(p.get('min_vol_ratio', 1.1))
    vol_ma_len     = int(p.get('vol_ma_len', 30))
    adx_slope_len  = int(p.get('adx_slope_len', 3))
    adx_slope_min  = float(p.get('adx_slope_min', 0.4))
    fresh_breakout_only = bool(p.get('fresh_breakout_only', False))
    fresh_cross_max_bars = int(p.get('fresh_cross_max_bars', 3))
    max_dist_emaslow = float(p.get('max_dist_emaslow', 0.010))
    require_rsi_cross = bool(p.get('require_rsi_cross', True))

    # Indicadores
    df["RSI"]   = talib.RSI(c, RSI_P)
    df["ATR"]   = talib.ATR(h, l, c, ATR_P)
    df["ATR_pct"] = (df["ATR"] / c).replace([np.inf, -np.inf], np.nan)
    df["EMA_S"] = talib.EMA(c, ema_f)
    df["EMA_L"] = talib.EMA(c, ema_s)
    df["ADX"]   = talib.ADX(h, l, c, ADX_P)

    # --- Features adicionales ---
    # Volumen relativo
    try:
        df["VOL_MA"] = v.rolling(vol_ma_len, min_periods=vol_ma_len).mean()
    except Exception:
        df["VOL_MA"] = np.nan
    df["VOL_OK"] = (v >= (min_vol_ratio * df["VOL_MA"])) if min_vol_ratio > 0 else True

    # Separación mínima EMAs (evita cruces planos)
    # Nota: EMA_S es la rápida y EMA_L la lenta en este módulo
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = df["EMA_L"].replace(0, np.nan).abs()
        df["EMA_SPREAD"] = (df["EMA_S"] - df["EMA_L"]).abs() / denom
    df["EMA_SPREAD_OK"] = (df["EMA_SPREAD"] >= min_ema_spread) if min_ema_spread > 0 else True

    # Pendiente de ADX (fuerza aumentando)
    try:
        df["ADX_SLOPE"] = df["ADX"] - df["ADX"].shift(adx_slope_len)
    except Exception:
        df["ADX_SLOPE"] = np.nan
    df["ADX_SLOPE_OK"] = (df["ADX_SLOPE"] >= adx_slope_min) if adx_slope_min > 0 else True

    # Precio relativo a EMAs
    df["PRICE_LONG_OK"]  = (c > df["EMA_S"]) & (df["EMA_S"] > df["EMA_L"])  # close>EMA_f>EMA_s
    df["PRICE_SHORT_OK"] = (c < df["EMA_S"]) & (df["EMA_S"] < df["EMA_L"])  # close<EMA_f<EMA_s

    # Distancia máxima a EMA lenta (no perseguir)
    with np.errstate(divide='ignore', invalid='ignore'):
        df["DIST_EMA_L"] = (c / df["EMA_L"] - 1.0).abs()
    df["DIST_OK"] = (df["DIST_EMA_L"] <= max_dist_emaslow) if max_dist_emaslow > 0 else True

    # Breakouts HH/LL opcionales
    if hhll_n and hhll_n > 0:
        df["HH"] = h.rolling(window=hhll_n, min_periods=hhll_n).max().shift(1)
        df["LL"] = l.rolling(window=hhll_n, min_periods=hhll_n).min().shift(1)
        long_break  = c > df["HH"]
        short_break = c < df["LL"]
    else:
        long_break = pd.Series(False, index=df.index)
        short_break = pd.Series(False, index=df.index)

    # Condiciones
    trend_long  = df["EMA_S"] > df["EMA_L"]
    trend_short = df["EMA_S"] < df["EMA_L"]
    rsi_long  = df["RSI"] >= rsi_buy
    rsi_short = df["RSI"] <= rsi_sell
    adx_ok    = df["ADX"] >= adx_min
    atr_ok    = (df["ATR_pct"] >= min_atr) & (df["ATR_pct"] <= max_atr)

    # Recencia de cruces RSI
    cross_up = (df["RSI"] >= rsi_buy) & (df["RSI"].shift(1) < rsi_buy)
    cross_dn = (df["RSI"] <= rsi_sell) & (df["RSI"].shift(1) > rsi_sell)
    if fresh_cross_max_bars and fresh_cross_max_bars > 0:
        df["RSI_LONG_RECENT"] = cross_up.rolling(fresh_cross_max_bars, min_periods=1).max().astype(bool)
        df["RSI_SHORT_RECENT"] = cross_dn.rolling(fresh_cross_max_bars, min_periods=1).max().astype(bool)
    else:
        df["RSI_LONG_RECENT"] = rsi_long
        df["RSI_SHORT_RECENT"] = rsi_short

    # Breakouts frescos opcionales
    if hhll_n and hhll_n > 0:
        df["FRESH_LONG_BREAK"] = (c > df["HH"]) & (c.shift(1) <= df["HH"].shift(1))
        df["FRESH_SHORT_BREAK"] = (c < df["LL"]) & (c.shift(1) >= df["LL"].shift(1))
        long_break_use = df["FRESH_LONG_BREAK"] if fresh_breakout_only else long_break
        short_break_use = df["FRESH_SHORT_BREAK"] if fresh_breakout_only else short_break
    else:
        long_break_use = long_break
        short_break_use = short_break

    # Gating base (siempre requerido)
    base_long = trend_long & adx_ok & atr_ok & df["DIST_OK"]
    base_short = trend_short & adx_ok & atr_ok & df["DIST_OK"]
    if require_close_vs_emas:
        base_long = base_long & df["PRICE_LONG_OK"]
        base_short = base_short & df["PRICE_SHORT_OK"]

    # Endurecedores
    gates_long = df["EMA_SPREAD_OK"] & df["VOL_OK"] & df["ADX_SLOPE_OK"]
    gates_short = df["EMA_SPREAD_OK"] & df["VOL_OK"] & df["ADX_SLOPE_OK"]

    # Disparadores (momentum / price action)
    rsi_trig_long = df["RSI_LONG_RECENT"] if require_rsi_cross else rsi_long
    rsi_trig_short = df["RSI_SHORT_RECENT"] if require_rsi_cross else rsi_short

    if logic == 'strict':
        trigger_long = rsi_trig_long & long_break_use
        trigger_short = rsi_trig_short & short_break_use
    else:  # 'any'
        trigger_long = rsi_trig_long | long_break_use
        trigger_short = rsi_trig_short | short_break_use

    df["Long_Signal"]  = (base_long & gates_long & trigger_long)
    df["Short_Signal"] = (base_short & gates_short & trigger_short)

    # Niveles de TP/SL (TP fijo %, SL por ATR) + escalonados
    f1, f2, f3 = TP_FACTORS
    # LONG ladder
    df["TP1_L"] = c * (1.0 + f1 * tp_pct)
    df["TP2_L"] = c * (1.0 + f2 * tp_pct)
    df["TP3_L"] = c * (1.0 + f3 * tp_pct)
    # SHORT ladder
    df["TP1_S"] = c * (1.0 - f1 * tp_pct)
    df["TP2_S"] = c * (1.0 - f2 * tp_pct)
    df["TP3_S"] = c * (1.0 - f3 * tp_pct)
    # TP clásico (compatibilidad) = TP2
    df["TP_L"] = df["TP2_L"]
    df["TP_S"] = df["TP2_S"]
    # SL por ATR
    df["SL_L"] = (c - atr_mult * df["ATR"]).clip(lower=1e-8)
    df["SL_S"] = (c + atr_mult * df["ATR"]).clip(lower=1e-8)

    # Columnas informativas opcionales
    df["Avg_Volume"] = v.rolling(window=20).mean()
    df["Rel_Volume"] = (v / df["Avg_Volume"]).replace([np.inf, -np.inf], np.nan)
    df["Low_Volume"] = df["Rel_Volume"] < VOL_THRESHOLD

    return df


# ---------- main ----------
def _purge_old():
    """Mantiene únicamente los últimos DAYS_KEEP días (+20%) por símbolo en IND_CSV.
    Si el bot se detiene y se corta la data, nunca elimina días incompletos.
    La purga ahora prioriza mantener días completos y sólo elimina si el rango de fechas es mayor al límite de días.
    """
    if not os.path.exists(IND_CSV):
        return
    try:
        df = pd.read_csv(IND_CSV, low_memory=False, parse_dates=["date"])
    except Exception as e:
        print(f"⚠️  No se pudo leer {IND_CSV} para purga: {e}")
        return

    now = df['date'].max()
    symbols = df['symbol'].unique()
    dfs = []
    for symbol in symbols:
        sdf = df[df['symbol'] == symbol].sort_values("date")
        if sdf.empty:
            continue
        min_date = sdf['date'].min()
        max_date = sdf['date'].max()
        days_span = (max_date - min_date).days
        limit_date = max_date - pd.Timedelta(days=DAYS_KEEP)
        # Solo purga si el rango es mayor al límite de días
        if days_span > DAYS_KEEP:
            buffer = int(288 * DAYS_KEEP * 1.2)  # velas 5m por día * días * 20% extra
            sdf = sdf[sdf['date'] >= limit_date]
            # Si por alguna razón quedan más de buffer filas (ejemplo, superposición), recorta por filas
            if len(sdf) > buffer:
                sdf = sdf.tail(buffer)
        dfs.append(sdf)
    new_df = pd.concat(dfs)
    if len(new_df) == len(df):
        return  # nada que borrar
    new_df.to_csv(IND_CSV, index=False)
    print(f"🧹 Purga por símbolo/días: {len(df)-len(new_df)} filas antiguas removidas")

def update_indicators():
    # Cargar el archivo de precios 5m
    if not os.path.exists(PRICE_5):
        print(f"🚫 No existe {PRICE_5}")
        return
    price = _read(PRICE_5, os.path.getmtime(PRICE_5)).sort_values(["symbol", "date"])

    # Whitelist
    if ALLOWED_SYMBOLS:
        before = len(price)
        price = price[price['symbol'].isin(ALLOWED_SYMBOLS)]
        print(f"✅ Whitelist aplicada: {before}→{len(price)} filas, {len(ALLOWED_SYMBOLS)} símbolos")
        if price.empty:
            print("🚫 Whitelist vació el dataset; no hay datos para procesar.")
            return

    # Blacklist
    if BLOCKED_SYMBOLS:
        price = price[~price['symbol'].isin(BLOCKED_SYMBOLS)]
        if price.empty:
            print("🚫 Todos los símbolos fueron bloqueados; no hay datos para procesar.")
            return

    # Calcular por símbolo con parámetros específicos
    outs = []
    for sym, sdf in price.groupby('symbol'):
        try:
            outs.append(_calc_symbol(sdf.copy(), sym))
        except Exception as _e:
            print(f"⚠️  Error calculando {sym}: {_e}")
    if not outs:
        print("🚫 No se generaron indicadores")
        return

    out = pd.concat(outs, ignore_index=True).sort_values(["symbol", "date"]).copy()

    os.makedirs(BASE_DIR := os.path.dirname(IND_CSV), exist_ok=True)
    out = out.rename(columns={
        "SL_L": "Stop_Loss_Long",
        "SL_S": "Stop_Loss_Short",
        "TP_L": "Take_Profit_Long",
        "TP_S": "Take_Profit_Short"
    })

    out.to_csv(IND_CSV, index=False, na_rep="NA")
    _purge_old()


# ---------- utilidad para el bot ----------
def ema_alert(symbol):
    if not os.path.exists(IND_CSV):
        return None, None
    df = _read(IND_CSV, os.path.getmtime(IND_CSV))
    df_symbol = df[df["symbol"] == symbol].sort_values("date")
    if len(df_symbol) < 1:
        return None, None
    # Tomar solo la última vela cerrada
    last_row = df_symbol.iloc[-1]

    def _as_bool(x):
        try:
            return bool(x) if pd.notna(x) else False
        except Exception:
            return False

    ls = _as_bool(last_row.get("Long_Signal", False))
    ss = _as_bool(last_row.get("Short_Signal", False))

    if ls or ss:
        side = "LONG" if ls else "SHORT"
        return last_row.close, f"Alerta de {side}"
    return None, None


if __name__ == "__main__":
    update_indicators()
