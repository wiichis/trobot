# indicadores.py – versión mínima

import os
from functools import lru_cache

import numpy as np
import pandas as pd
import talib
from datetime import datetime, timedelta  # NUEVO: para purgar registros antiguos

# === Parámetros ===
RSI_P, ATR_P, EMA_S, EMA_L, ADX_P = 14, 14, 50, 200, 14
TP_M, SL_M, SIGNAL_MIN = 2.0, 1.0, 3
MAX_PER_SYMBOL = 300  # conservar hasta 300 velas recientes por símbolo

# Días a mantener por símbolo en indicadores (para purga inteligente)
DAYS_KEEP = 10  # Días a mantener por símbolo en indicadores

BASE = "./archivos"
PRICE_30 = f"{BASE}/cripto_price.csv"
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
    _, _, m_h   = talib.MACD(c, 12, 26, 9)
    df["MACD_H"] = m_h
    df["ADX"] = talib.ADX(h, l, c, ADX_P)

    df["TP_L"] = c + df["ATR"] * TP_M
    df["SL_L"] = np.maximum(c - df["ATR"] * SL_M, 1e-8)
    df["TP_S"] = c - df["ATR"] * TP_M
    df["SL_S"] = c + df["ATR"] * SL_M

    long_ok  = (df["EMA_S"] > df["EMA_L"], m_h > 0, df["RSI"] > 45, df["ADX"] > 20)
    short_ok = (df["EMA_S"] < df["EMA_L"], m_h < 0, df["RSI"] < 55, df["ADX"] > 20)
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
    df5["EMA_S"] = grp_close.transform(lambda s: talib.EMA(s, 9))
    df5["EMA_L"] = grp_close.transform(lambda s: talib.EMA(s, 21))
    df5["RSI"]   = grp_close.transform(lambda s: talib.RSI(s, 14))
    df5["Rel_V"] = df5["volume"] / df5.groupby("symbol")["volume"].transform(lambda s: s.rolling(20).mean())
    df5["anchor"] = df5["date"].dt.floor("30T")
    df5["rank"]   = df5.groupby(["symbol", "anchor"]).cumcount() + 1

    ok_long  = (df5["EMA_S"] > df5["EMA_L"]) & (df5["RSI"] > 40) & (df5["Rel_V"] > 1) & (df5["rank"] <= 5)
    ok_short = (df5["EMA_S"] < df5["EMA_L"]) & (df5["RSI"] > 60) & (df5["Rel_V"] > 1) & (df5["rank"] <= 5)

    cnt_l = df5[ok_long ].groupby(["symbol", "anchor"]).size()
    cnt_s = df5[ok_short].groupby(["symbol", "anchor"]).size()

    df30 = df30.merge(cnt_l.rename("cnt_l"), left_on=["symbol", "date"], right_index=True, how="left")
    df30 = df30.merge(cnt_s.rename("cnt_s"), left_on=["symbol", "date"], right_index=True, how="left")
    df30[["cnt_l", "cnt_s"]] = df30[["cnt_l", "cnt_s"]].fillna(0)

    df30["Long_Signal"]  &= df30["cnt_l"] >= 3
    df30["Short_Signal"] &= df30["cnt_s"] >= 3
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

    out = price.groupby("symbol", group_keys=False).apply(_calc)
    out = _confirm(out)
    out = out.sort_values(["symbol", "date"])

    os.makedirs(BASE_DIR := os.path.dirname(IND_CSV), exist_ok=True)
    out = out.rename(columns={"SL_L": "Stop_Loss_Long", "SL_S": "Stop_Loss_Short"})
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