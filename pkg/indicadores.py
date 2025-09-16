# indicadores.py – versión mínima

import os
from functools import lru_cache

import numpy as np
import pandas as pd
import talib
from pathlib import Path
import json
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
WHITELIST_PATH = Path(__file__).resolve().parent.parent / "archivos" / "backtesting" / "whitelist.json"
ALLOWED_SYMBOLS = []
OBS_SYMBOLS = []
BLACKLIST_SYMBOLS = []
try:
    if WHITELIST_PATH.exists():
        with open(WHITELIST_PATH, "r", encoding="utf-8") as f:
            _wl = json.load(f) or {}
            ALLOWED_SYMBOLS = _wl.get("whitelist", []) or []
            OBS_SYMBOLS = _wl.get("observation", []) or []
            BLACKLIST_SYMBOLS = _wl.get("blacklist", []) or []
            # Prints informativos (no rompen producción)
            print(f"✅ Whitelist activa ({len(ALLOWED_SYMBOLS)}): {', '.join(ALLOWED_SYMBOLS) if ALLOWED_SYMBOLS else '-'}")
            print(f"👀 En observación ({len(OBS_SYMBOLS)}): {', '.join(OBS_SYMBOLS) if OBS_SYMBOLS else '-'}")
            print(f"⛔ Blacklist ({len(BLACKLIST_SYMBOLS)}): {', '.join(BLACKLIST_SYMBOLS) if BLACKLIST_SYMBOLS else '-'}")
except Exception as _e:
    # Silencioso en producción; solo informativo en debug
    pass

 # --- Cargar parámetros por símbolo desde best_prod.json ---
BEST_PROD_PATH = Path(__file__).resolve().parent / "best_prod.json"
PARAMS_BY_SYMBOL = {}
try:
    if BEST_PROD_PATH.exists():
        with open(BEST_PROD_PATH, "r", encoding="utf-8") as f:
            _bestp = json.load(f) or []
        for item in _bestp:
            sym = str(item.get('symbol', '')).upper()
            pr  = item.get('params') or {}
            if sym:
                PARAMS_BY_SYMBOL[sym] = pr
        # Si no hay whitelist explícita, usa las llaves de best_prod.json
        if not ALLOWED_SYMBOLS and PARAMS_BY_SYMBOL:
            ALLOWED_SYMBOLS = list(PARAMS_BY_SYMBOL.keys())
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
BLOCKED_SYMBOLS = []

# Días a mantener por símbolo en indicadores (para purga inteligente)
DAYS_KEEP = 10  # Días a mantener por símbolo en indicadores

PRICE_5  = f"{BASE}/cripto_price_5m.csv"
IND_CSV  = f"{BASE}/indicadores.csv"


# ---------- util ----------
@lru_cache(maxsize=4)
def _read(path, mtime=None):
    dtypes = {
        "Long_Signal": "boolean",
        "Short_Signal": "boolean"
    }
    return pd.read_csv(
        path,
        parse_dates=["date"],
        dtype=dtypes,
        na_values=["NA"]
    )


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

    # Indicadores
    df["RSI"]   = talib.RSI(c, RSI_P)
    df["ATR"]   = talib.ATR(h, l, c, ATR_P)
    df["ATR_pct"] = (df["ATR"] / c).replace([np.inf, -np.inf], np.nan)
    df["EMA_S"] = talib.EMA(c, ema_f)
    df["EMA_L"] = talib.EMA(c, ema_s)
    df["ADX"]   = talib.ADX(h, l, c, ADX_P)

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

    # Lógica: 'strict' = todas; 'any' = al menos 3 condiciones (incluyendo trend)
    if hhll_n and hhll_n > 0:
        conds_long  = [trend_long, rsi_long, adx_ok, atr_ok, long_break]
        conds_short = [trend_short, rsi_short, adx_ok, atr_ok, short_break]
        need = 5 if logic == 'strict' else 3
    else:
        conds_long  = [trend_long, rsi_long, adx_ok, atr_ok]
        conds_short = [trend_short, rsi_short, adx_ok, atr_ok]
        need = 4 if logic == 'strict' else 3

    df["Long_Signal"]  = (np.sum(conds_long,  axis=0) >= need)
    df["Short_Signal"] = (np.sum(conds_short, axis=0) >= need)

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
    if last_row.Long_Signal or last_row.Short_Signal:
        side = "LONG" if last_row.Long_Signal else "SHORT"
        return last_row.close, f"Alerta de {side}"
    return None, None


if __name__ == "__main__":
    update_indicators()