# -*- coding: utf-8 -*-
"""
ta_shared.py - Indicadores compartidos con fallback si TA-Lib no está disponible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import talib as _talib
except Exception:
    _talib = None


def _series_from(out, index):
    if isinstance(out, pd.Series):
        return out
    return pd.Series(out, index=index)


def ema(series: pd.Series, period: int) -> pd.Series:
    if _talib is not None:
        return _series_from(_talib.EMA(series.values, timeperiod=int(period)), series.index)
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    if _talib is not None:
        return _series_from(_talib.RSI(series.values, timeperiod=int(period)), series.index)
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1 / period, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    if _talib is not None:
        return _series_from(
            _talib.ATR(high.values, low.values, close.values, timeperiod=int(period)),
            close.index,
        )
    prev_c = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    if _talib is not None:
        return _series_from(
            _talib.ADX(high.values, low.values, close.values, timeperiod=int(period)),
            close.index,
        )
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = atr(high, low, close, period=1)
    atr_n = tr.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * pd.Series(plus_dm, index=close.index).ewm(alpha=1 / period, adjust=False).mean() / atr_n
    minus_di = 100 * pd.Series(minus_dm, index=close.index).ewm(alpha=1 / period, adjust=False).mean() / atr_n
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    out = dx.ewm(alpha=1 / period, adjust=False).mean()
    return out.fillna(0)
