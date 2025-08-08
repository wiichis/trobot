# indicadores.py – versión mínima

import os
from functools import lru_cache

import numpy as np
import pandas as pd
import talib
from pathlib import Path
import json
from datetime import datetime, timedelta  # NUEVO: para purgar registros antiguos

# === Parámetros ===
# Parámetros ajustados al mejor combo del backtest
RSI_P, ATR_P, EMA_S, EMA_L, ADX_P = 10, 14, 8, 30, 14
TP_M, SL_M, SIGNAL_MIN = 8, 2, 3  # Multiplicadores TP/SL
 # Umbrales adicionales (por si no se sobre‐escriben)
RSI_HI, RSI_LO, ADX_MIN = 70, 35, 20
# Umbrales para filtros de volumen y volatilidad
VOL_THRESHOLD = 0.6       # Rel_Volume < 0.6 ⇒ bajo volumen
VOLAT_THRESHOLD = 0.9855    # Rel_Volatility > 0.9855 ⇒ alta volatilidad
BASE = "./archivos"  # ruta base para todos los CSV/JSON
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
except Exception as e:
    print(f"⚠️  No se pudo cargar best_cfg.json: {e}")
 # Parámetros para confirmación de 5 m
EMA_S_5M, EMA_L_5M, RSI_5M, MIN_CONFIRM_5M = 3, 7, 14, 4
MAX_PER_SYMBOL = 300  # conservar hasta 300 velas recientes por símbolo

# --- Lista blanca/negra de rendimiento ---
# Agrega aquí los símbolos que quieres EXCLUIR del cálculo de indicadores
# Ejemplo: ["BTC-USDT", "DOGE-USDT"]
BLOCKED_SYMBOLS = ["AVAX-USDT", "BNB-USDT"]

# Días a mantener por símbolo en indicadores (para purga inteligente)
DAYS_KEEP = 10  # Días a mantener por símbolo en indicadores

PRICE_30 = f"{BASE}/cripto_price_30m.csv"
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
def _calc(df):
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    df["RSI"]   = talib.RSI(c, RSI_P)
    df["ATR"]   = talib.ATR(h, l, c, ATR_P)
    df["EMA_S"] = talib.EMA(c, EMA_S)
    df["EMA_L"] = talib.EMA(c, EMA_L)
    df["ADX"] = talib.ADX(h, l, c, ADX_P)

    df["TP_L"] = c + df["ATR"] * TP_M
    df["SL_L"] = np.maximum(c - df["ATR"] * SL_M, 1e-8)
    df["TP_S"] = c - df["ATR"] * TP_M
    df["SL_S"] = c + df["ATR"] * SL_M

    # --- Volumen y volatilidad relativos ---
    df["Avg_Volume"] = v.rolling(window=20).mean()
    df["Rel_Volume"] = v / df["Avg_Volume"]
    df["Low_Volume"] = df["Rel_Volume"] < VOL_THRESHOLD

    df["Volatility"] = df["ATR"]  # ATR ya es una medida de volatilidad
    df["Avg_Volatility"] = df["Volatility"].rolling(window=20).mean()
    df["Rel_Volatility"] = df["Volatility"] / df["Avg_Volatility"]
    df["High_Volatility"] = df["Rel_Volatility"] > VOLAT_THRESHOLD

    long_ok  = (
        df["EMA_S"] > df["EMA_L"],
        df["RSI"] < RSI_HI,
        df["ADX"] > ADX_MIN,
        ~df["Low_Volume"]
    )
    short_ok = (
        df["EMA_S"] < df["EMA_L"],
        df["RSI"] > RSI_LO,
        df["ADX"] > ADX_MIN,
        ~df["Low_Volume"]
    )
    SIGNAL_MIN = 4  # Requiere todas las condiciones para señal fuerte

    df["Long_Signal"]  = np.sum(long_ok,  axis=0) >= SIGNAL_MIN
    df["Short_Signal"] = np.sum(short_ok, axis=0) >= SIGNAL_MIN
    return df


# ---------- confirmación 5 m ----------
def _confirm(df30):
    if not os.path.exists(PRICE_5):
        return df30

    df5 = _read(PRICE_5, os.path.getmtime(PRICE_5)).sort_values(["symbol", "date"])
    # --- indicadores por símbolo (evita mezclar pares) ---
    grp_close = df5.groupby("symbol")["close"]
    df5["EMA_S"] = grp_close.transform(lambda s: talib.EMA(s, EMA_S_5M))
    df5["EMA_L"] = grp_close.transform(lambda s: talib.EMA(s, EMA_L_5M))
    df5["RSI"]   = grp_close.transform(lambda s: talib.RSI(s, RSI_5M))
    df5["Rel_V"] = df5["volume"] / df5.groupby("symbol")["volume"].transform(lambda s: s.rolling(20).mean())
    df5["anchor"] = df5["date"].dt.floor("30T")
    df5["rank"]   = df5.groupby(["symbol", "anchor"]).cumcount() + 1

    ok_long  = (df5["EMA_S"] > df5["EMA_L"]) & (df5["RSI"] > 55) & (df5["Rel_V"] > 1.2) & (df5["rank"] <= 4)
    ok_short = (df5["EMA_S"] < df5["EMA_L"]) & (df5["RSI"] < 45) & (df5["Rel_V"] > 1.2) & (df5["rank"] <= 4)
    MIN_CONFIRM_5M = 3  # Requiere todas las condiciones para confirmar

    cnt_l = df5[ok_long ].groupby(["symbol", "anchor"]).size()
    cnt_s = df5[ok_short].groupby(["symbol", "anchor"]).size()

    df30 = df30.merge(cnt_l.rename("cnt_l"), left_on=["symbol", "date"], right_index=True, how="left")
    df30 = df30.merge(cnt_s.rename("cnt_s"), left_on=["symbol", "date"], right_index=True, how="left")
    df30[["cnt_l", "cnt_s"]] = df30[["cnt_l", "cnt_s"]].fillna(0)

    df30["Long_Signal"]  &= df30["cnt_l"] >= MIN_CONFIRM_5M
    df30["Short_Signal"] &= df30["cnt_s"] >= MIN_CONFIRM_5M
    return df30.drop(columns=["cnt_l", "cnt_s"])


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
            buffer = int(48 * DAYS_KEEP * 1.2)  # velas 30m por día * días * 20% extra
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
    # Cargar TODO el archivo de precios
    price = _read(PRICE_30, os.path.getmtime(PRICE_30)).sort_values(["symbol", "date"])
    if price.empty:
        return

    # Filtra símbolos con bajo rendimiento si están listados en BLOCKED_SYMBOLS
    if BLOCKED_SYMBOLS:
        price = price[~price['symbol'].isin(BLOCKED_SYMBOLS)]
        if price.empty:
            print("🚫 Todos los símbolos fueron bloqueados; no hay datos para procesar.")
            return

    out = price.groupby("symbol", group_keys=False).apply(_calc)
    out = _confirm(out)
    out = out.sort_values(["symbol", "date"])

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