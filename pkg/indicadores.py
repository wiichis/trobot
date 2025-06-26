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
    df5["EMA_S"] = talib.EMA(df5["close"], 9)
    df5["EMA_L"] = talib.EMA(df5["close"], 21)
    df5["RSI"]   = talib.RSI(df5["close"], 14)
    df5["Rel_V"] = df5["volume"] / df5["volume"].rolling(20).mean()
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
    """Mantiene únicamente las últimas MAX_ROWS filas en IND_CSV."""
    if not os.path.exists(IND_CSV):
        return
    try:
        df = pd.read_csv(IND_CSV, low_memory=False)
    except Exception as e:
        print(f"⚠️  No se pudo leer {IND_CSV} para purga: {e}")
        return

    df = df.sort_values(["symbol", "date"])
    new_df = (
        df.groupby("symbol", group_keys=False)
          .tail(MAX_PER_SYMBOL)          # hasta 300 velas por par
    )
    if len(new_df) == len(df):
        return  # nada que borrar
    new_df.to_csv(IND_CSV, index=False)
    print(f"🧹 Purga por símbolo: {len(df)-len(new_df)} filas antiguas removidas")

def update_indicators():
    price = _read(PRICE_30, os.path.getmtime(PRICE_30)).sort_values(["symbol", "date"])
    if os.path.exists(IND_CSV):
        last = pd.read_csv(IND_CSV, usecols=["date"]).date.iloc[-1]
        last_clean = pd.to_datetime(last, utc=True).tz_localize(None)
        price = price[price["date"] > last_clean]
        if price.empty:
            _purge_old()          # asegura limpieza aunque no haya datos nuevos
            return

    out = price.groupby("symbol", group_keys=False).apply(_calc)
    out = _confirm(out)
    out = out.sort_values(["symbol", "date"])

    os.makedirs(BASE_DIR := os.path.dirname(IND_CSV), exist_ok=True)
    header = not os.path.exists(IND_CSV)
    out.to_csv(IND_CSV, mode="a", header=header, index=False, na_rep="NA")
    _purge_old()


# ---------- utilidad para el bot ----------
def ema_alert(symbol):
    if not os.path.exists(IND_CSV):
        return None, None
    df = _read(IND_CSV, os.path.getmtime(IND_CSV))
    last = df[df["symbol"] == symbol].sort_values("date").tail(1)
    if last.empty.any():
        return None, None
    row = last.iloc[0]
    if row.Long_Signal or row.Short_Signal:
        side = "LONG" if row.Long_Signal else "SHORT"
        return row.close, f"{side} {symbol} @ {row.date}"
    return None, None


if __name__ == "__main__":
    update_indicators()