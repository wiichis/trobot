"""Reglas del bot BTC por rangos — fuente única para la simulación y el live.

Especificación: `docs/reglas_bot_btc.md` v0.1 (21/10/2026). Este módulo sólo calcula
indicadores y señales de ENTRADA sobre velas 1 h CERRADAS; la gestión de slots, salidas y
cierre mensual vive en `pkg/btc_rangos_motor.py`.

Indicadores (Wilder, en pandas puro, para que el resultado no dependa de si hay TA-Lib):
  RSI(14), ATR(14), ADX(14) con DI+/DI-, Bollinger(20, 2) sobre el close.

Señales (al cierre de la vela t; el motor ejecuta en la apertura de t+1):
  E1  caída: close <= máximo de los highs de las últimas 24 velas × (1 − d), d = 2 % o 3 %.
  E2  close < banda inferior de Bollinger(20, 2) y RSI(14) < 35.
  E3  close <= mínimo de los lows de los últimos N días × 1,01, N = 7 o 14.
  F   filtro: bloquea la entrada si ADX(14) > 25 y DI− > DI+ (tendencia bajista fuerte).
  R   régimen (v0.2, 22/10/2026): sólo se abren slots si el último cierre DIARIO completo
      está sobre su media simple de 100 días. Es la ventana COMPRA de `pkg/btc_alertas.py`
      (regla T1 de `pkg/btc_reglas.py`). Lo abierto se sigue gestionando igual.
  E4  (v0.4) caída variable: close <= máximo de 24 velas × (1 − g × R), g = 0,5 o 1, donde
      R es el rango diario típico (`rango_diario`, debe venir en `ind["rango"]`).

Parámetros fijados el 22/10/2026 a partir de la especificación, sin optimizar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_RSI = 14
N_ATR = 14
N_ADX = 14
BB_N, BB_K = 20, 2.0
E2_RSI_MAX = 35.0
E3_MARGEN = 0.01
F_ADX_MIN = 25.0
N_REGIMEN = 100

ENTRADAS = {
    "E1_2": {"tipo": "E1", "caida": 0.02},
    "E1_3": {"tipo": "E1", "caida": 0.03},
    "E2": {"tipo": "E2"},
    "E3_7": {"tipo": "E3", "dias": 7},
    "E3_14": {"tipo": "E3", "dias": 14},
    "E4_0.5": {"tipo": "E4", "g": 0.5},
    "E4_1": {"tipo": "E4", "g": 1.0},
}
N_RANGO = 14


def _wilder(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(c: pd.Series, n: int = N_RSI) -> pd.Series:
    d = c.diff()
    up = _wilder(d.clip(lower=0), n)
    dn = _wilder(-d.clip(upper=0), n)
    return 100 - 100 / (1 + up / dn)


def true_range(h: pd.DataFrame) -> pd.Series:
    prev = h["close"].shift(1)
    return pd.concat([h["high"] - h["low"], (h["high"] - prev).abs(), (h["low"] - prev).abs()], axis=1).max(axis=1)


def atr(h: pd.DataFrame, n: int = N_ATR) -> pd.Series:
    return _wilder(true_range(h), n)


def adx_di(h: pd.DataFrame, n: int = N_ADX) -> pd.DataFrame:
    up = h["high"].diff()
    dn = -h["low"].diff()
    dm_p = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=h.index)
    dm_m = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=h.index)
    tr = _wilder(true_range(h), n)
    di_p = 100 * _wilder(dm_p, n) / tr
    di_m = 100 * _wilder(dm_m, n) / tr
    dx = 100 * (di_p - di_m).abs() / (di_p + di_m)
    return pd.DataFrame({"adx": _wilder(dx, n), "di_p": di_p, "di_m": di_m})


def indicadores(h: pd.DataFrame) -> pd.DataFrame:
    """Columnas que usan las señales y las salidas, alineadas con `h` (velas cerradas)."""
    c = h["close"]
    media = c.rolling(BB_N).mean()
    desvio = c.rolling(BB_N).std(ddof=0)
    out = pd.DataFrame(index=h.index)
    out["rsi"] = rsi(c)
    out["atr"] = atr(h)
    out["bb_inf"] = media - BB_K * desvio
    out = out.join(adx_di(h))
    out["max_24"] = h["high"].rolling(24).max()
    out["min_7d"] = h["low"].rolling(7 * 24).min()
    out["min_14d"] = h["low"].rolling(14 * 24).min()
    return out


def senal_entrada(h: pd.DataFrame, ind: pd.DataFrame, entrada: str, filtro: bool) -> pd.Series:
    """True en las velas cuyo cierre da señal de compra."""
    p = ENTRADAS[entrada]
    c = h["close"]
    if p["tipo"] == "E1":
        s = c <= ind["max_24"] * (1 - p["caida"])
    elif p["tipo"] == "E2":
        s = (c < ind["bb_inf"]) & (ind["rsi"] < E2_RSI_MAX)
    elif p["tipo"] == "E3":
        s = c <= ind[f"min_{p['dias']}d"] * (1 + E3_MARGEN)
    elif p["tipo"] == "E4":
        s = c <= ind["max_24"] * (1 - p["g"] * ind["rango"])
    else:  # pragma: no cover - ENTRADAS es cerrado
        raise ValueError(entrada)
    if filtro:
        bajista_fuerte = (ind["adx"] > F_ADX_MIN) & (ind["di_m"] > ind["di_p"])
        s = s & ~bajista_fuerte
    return s.fillna(False).astype(bool)


def regimen_compra(cierre_diario: pd.Series, index_1h: pd.DatetimeIndex) -> pd.Series:
    """True en las velas 1 h cuyo último día CERRADO tiene close > media de 100 días.

    `cierre_diario` va indexado por la fecha de apertura del día (00:00 UTC). La vela 1 h que
    abre en t cierra en t + 1 h; en ese momento el último día completo es floor(t + 1 h) − 1 d,
    así que la vela de las 23:00 ya ve el día que termina con ella y ninguna ve el futuro."""
    ok = cierre_diario > cierre_diario.rolling(N_REGIMEN).mean()
    dia = (index_1h + pd.Timedelta(hours=1)).floor("D") - pd.Timedelta(days=1)
    return pd.Series(ok.reindex(dia).fillna(False).to_numpy(dtype=bool), index=index_1h)


def rango_diario(diario: pd.DataFrame, index_1h: pd.DatetimeIndex, n: int = N_RANGO) -> pd.Series:
    """R: mediana de los últimos `n` días CERRADOS de (máximo − mínimo) / cierre, por vela 1 h.

    `diario` (open/high/low/close) va indexado por la fecha de apertura del día. Misma regla de
    frontera que `regimen_compra`: la vela 1 h que abre en t ve hasta el día floor(t + 1 h) − 1 d."""
    r = ((diario["high"] - diario["low"]) / diario["close"]).rolling(n).median()
    dia = (index_1h + pd.Timedelta(hours=1)).floor("D") - pd.Timedelta(days=1)
    return pd.Series(r.reindex(dia).to_numpy(dtype=float), index=index_1h)
