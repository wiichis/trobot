"""Guard de tick size en el backtest.

Contexto (10/08/2026): CFX no estaba en SYMBOL_TRADING_RULES, así que caía al tick por
defecto de 0.01 — el 24% de su precio. Como los TP se redondean con ROUND_DOWN, el TP
quedaba por debajo del precio de entrada y el trade cerraba como "TP" perdiendo ~13%.
El efecto no se veía como un bug sino como "CFX no tiene edge": arrastró el PnL del
portfolio en las 4 ventanas del cross-val a la vez.
"""
import pytest

from pkg.backtesting import (
    MAX_TICK_PCT_OF_PRICE,
    SYMBOL_TRADING_RULES,
    _assert_tick_size_sane,
    _round_to_tick,
    _tick_size_for,
)

# Precios de referencia al 10/08/2026.
PRECIOS_PORTFOLIO = {
    'APT-USDT': 0.594, 'AVAX-USDT': 6.54, 'BCH-USDT': 214.65, 'BNB-USDT': 602.0,
    'CFX-USDT': 0.0421, 'DYDX-USDT': 0.1143, 'ETH-USDT': 1897.9, 'LINK-USDT': 8.32,
    'ONDO-USDT': 0.3484, 'XMR-USDT': 394.1,
}


@pytest.mark.parametrize('symbol,precio', sorted(PRECIOS_PORTFOLIO.items()))
def test_los_pares_del_portfolio_tienen_tick_sano(symbol, precio):
    assert symbol in SYMBOL_TRADING_RULES, f'{symbol} caería al tick por defecto'
    _assert_tick_size_sane(symbol, precio)


def test_simbolo_barato_sin_entrada_en_la_tabla_falla_ruidosamente():
    assert 'PEPE-USDT' not in SYMBOL_TRADING_RULES
    with pytest.raises(ValueError, match='TICK GUARD'):
        _assert_tick_size_sane('PEPE-USDT', 0.0000082)


def test_simbolo_caro_sin_entrada_no_molesta():
    """El default de 0.01 es legítimo para un par caro; el guard no debe ser ruidoso ahí."""
    _assert_tick_size_sane('ALGO-CARO-USDT', 3000.0)


def test_precio_invalido_no_rompe_el_guard():
    _assert_tick_size_sane('CFX-USDT', 0.0)
    _assert_tick_size_sane('CFX-USDT', None)


def test_reproduce_el_caso_cfx_con_la_tabla_vieja(monkeypatch):
    """Sin la entrada de CFX el sweep se rompe, en vez de devolver un número envenenado."""
    tabla_vieja = {k: v for k, v in SYMBOL_TRADING_RULES.items() if k != 'CFX-USDT'}
    monkeypatch.setattr('pkg.backtesting.SYMBOL_TRADING_RULES', tabla_vieja)
    with pytest.raises(ValueError, match='no está en SYMBOL_TRADING_RULES'):
        _assert_tick_size_sane('CFX-USDT', 0.0421)


def test_el_tick_viejo_de_cfx_ponia_el_tp_debajo_de_la_entrada():
    """El mecanismo exacto de la pérdida falsa, con los números reales del trade."""
    entrada = 0.04533           # entrada real de un long de CFX el 16/07
    tp_deseado = entrada * 1.012  # TP objetivo ~1.2%

    tp_con_tick_viejo = _round_to_tick(tp_deseado, 0.01)
    assert tp_con_tick_viejo == pytest.approx(0.04)
    assert tp_con_tick_viejo < entrada, 'este es el bug: el TP quedaba bajo la entrada'

    tp_con_tick_real = _round_to_tick(tp_deseado, _tick_size_for('CFX-USDT'))
    assert tp_con_tick_real > entrada
    assert tp_con_tick_real == pytest.approx(0.04587, abs=1e-5)


def test_los_ticks_reales_estan_holgados_frente_al_umbral():
    """Los valores del exchange rondan 0.001%-0.03%; el umbral de 0.1% deja margen."""
    peor = max(_tick_size_for(s) / p for s, p in PRECIOS_PORTFOLIO.items())
    assert peor < MAX_TICK_PCT_OF_PRICE / 3
