"""Reglas de los bots de BTC — fuente única para el live (`pkg/btc_bots.py`) y la
simulación (`scripts/btc_lab_temporalidades.py`, `scripts/btc_cuatro_bots.py`).

Cada regla recibe velas CERRADAS (open/high/low/close) y devuelve la posición deseada al
cierre de cada barra: +1 long, -1 short, 0 fuera. Parámetros fijados de antemano el
02/10/2026, sin optimizar (n en barras):
  T1 precio vs media simple de 100: long arriba, short abajo.
  T2 cruce de medias exponenciales 20/50: long si EMA20 > EMA50, short si no.
  T3 Donchian 20/10: entra al cerrar sobre el máximo de 20 barras (bajo el mínimo para
     short); sale al cerrar bajo el mínimo de 10 (sobre el máximo de 10 para short).
  R1 reversión RSI 14: long si RSI < 30, short si RSI > 70; sale al cruzar 50.
Con `solo_long` las señales de short dejan la posición en cero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGLAS = ("T1", "T2", "T3", "R1")


def rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def posiciones(h: pd.DataFrame, regla: str, solo_long: bool) -> pd.Series:
    """Posición deseada al CIERRE de cada barra: +1, -1 o 0."""
    c, hi, lo = h["close"], h["high"], h["low"]
    if regla == "T1":
        p = np.where(c > c.rolling(100).mean(), 1, -1).astype(float)
        p[: 99] = 0
    elif regla == "T2":
        p = np.where(c.ewm(span=20, adjust=False).mean() > c.ewm(span=50, adjust=False).mean(), 1, -1).astype(float)
        p[: 49] = 0
    elif regla == "T3":
        ent_l, ent_s = c > hi.shift(1).rolling(20).max(), c < lo.shift(1).rolling(20).min()
        sal_l, sal_s = c < lo.shift(1).rolling(10).min(), c > hi.shift(1).rolling(10).max()
        p, pos = np.zeros(len(h)), 0.0
        for i in range(len(h)):
            if pos == 1 and sal_l.iat[i]:
                pos = 0.0
            elif pos == -1 and sal_s.iat[i]:
                pos = 0.0
            if pos == 0:
                pos = 1.0 if ent_l.iat[i] else (-1.0 if ent_s.iat[i] else 0.0)
            p[i] = pos
    elif regla == "R1":
        r = rsi(c).to_numpy()
        p, pos = np.zeros(len(h)), 0.0
        for i in range(len(h)):
            if pos == 1 and r[i] >= 50:
                pos = 0.0
            elif pos == -1 and r[i] <= 50:
                pos = 0.0
            if pos == 0:
                pos = 1.0 if r[i] < 30 else (-1.0 if r[i] > 70 else 0.0)
            p[i] = pos
    else:
        raise ValueError(regla)
    if solo_long:
        p = np.where(p > 0, 1.0, 0.0)
    return pd.Series(p, index=h.index)
