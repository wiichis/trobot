"""Filtro de régimen en el motor del SWEEP: modo 'adx_only' (01/10/2026).

El live usa `htf_mode: adx_only` desde el 20/09 (sólo ADX 1h >= 18, sin dirección de
EMAs). El `Backtester` del sweep sólo implementaba 'ema_adx' (EMA50/EMA200 1h + ADX),
un filtro más restrictivo: un sweep con --htf_filter_enabled optimizaba contra otro
filtro que el que corre en prod. Mismo tipo de hueco que los del parity del 23/09.
"""
import numpy as np
import pandas as pd
import pytest

from pkg.backtesting import Backtester


def _velas(n=6000, seed=7):
    rng = np.random.default_rng(seed)
    # tendencia por tramos + ruido, para que haya cruces, tendencias y laterales
    drift = np.repeat(rng.normal(0, 0.0008, n // 400 + 1), 400)[:n]
    ret = drift + rng.normal(0, 0.002, n)
    close = 100 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, 0.0015, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.0015, n)))
    open_ = np.r_[close[0], close[:-1]]
    vol = rng.lognormal(3, 0.6, n)
    dates = pd.date_range('2026-01-01', periods=n, freq='5min', tz='UTC')
    return pd.DataFrame({'date': dates, 'open': open_, 'high': high, 'low': low,
                         'close': close, 'volume': vol})


def _senales(**kw):
    # filtros de entrada aflojados: el test es sobre el GATE HTF, no sobre la señal 5m
    base = dict(adx_min=5, rsi_buy=51, rsi_sell=49, min_atr_pct=0.0, max_atr_pct=1.0,
                logic='any', fresh_cross_max_bars=500, require_rsi_cross=False,
                min_vol_ratio=0.0, max_dist_emaslow=1.0, min_ema_spread=0.0,
                adx_slope_min=-100.0, require_close_vs_emas=False)
    base.update(kw)
    bt = Backtester('BCH-USDT', _velas(), **base)
    df = bt.df5m
    sig = [bt._entry_signal(r) for _, r in df.iterrows()]
    return df, pd.Series(sig, index=df.index)


@pytest.fixture(scope='module')
def tres():
    _, s0 = _senales()
    dfa, sa = _senales(htf_filter_enabled=True, htf_tf='1h', htf_adx_min=18.0, htf_mode='adx_only')
    _, se = _senales(htf_filter_enabled=True, htf_tf='1h', htf_adx_min=18.0, htf_mode='ema_adx')
    return s0, dfa, sa, se


def test_hay_senales_en_los_datos_sinteticos(tres):
    s0, _, sa, se = tres
    assert s0.notna().sum() > 20
    assert sa.notna().sum() > 0


def test_adx_only_es_la_senal_sin_filtro_recortada_por_adx(tres):
    s0, dfa, sa, _ = tres
    ok = dfa['htf_adx'] >= 18.0
    esperado = s0.where(ok & dfa['htf_adx'].notna())
    pd.testing.assert_series_equal(sa, esperado, check_names=False)


def test_ema_adx_es_mas_restrictivo_que_adx_only(tres):
    _, _, sa, se = tres
    assert (se.notna() & sa.isna()).sum() == 0
    assert se.notna().sum() < sa.notna().sum()


def test_modo_invalido_cae_al_original():
    bt = Backtester('BCH-USDT', _velas(500), htf_mode='raro')
    assert bt.htf_mode == 'ema_adx'
