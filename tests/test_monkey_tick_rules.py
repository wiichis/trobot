"""Reglas de tick en el camino LIVE.

Contexto (10/08/2026): SYMBOL_TRADING_RULES estaba incompleta en producción — le
faltaban APT, AVAX, BCH, ETH, LINK, ONDO y XMR, que caían al tick por defecto de 0.01.
A diferencia del backtest (donde el TP terminaba bajo la entrada), live redondea en
dirección conservadora, así que no invertía los precios; pero los descolocaba:

  - ONDO: entradas LONG a −2.41% del mercado en vez de −0.02%. Como son PostOnly,
    casi nunca llenaban. Figuraba como "par mudo".
  - ONDO: TP efectivo de 3.33% con un TP diseñado de 1.2%.
  - APT: entradas LONG a −0.62%, TP efectivo de 2.75%.
"""
import pytest

from pkg.monkey_bx import (
    MAX_TICK_PCT_OF_PRICE,
    SYMBOL_TRADING_RULES,
    _round_trigger_price,
    _sanitize_entry_limit_price,
    _tick_size_for,
    _warn_if_tick_implausible,
)

# Precios de referencia al 10/08/2026.
PRECIOS_PORTFOLIO = {
    'APT-USDT': 0.5937, 'AVAX-USDT': 6.541, 'BCH-USDT': 214.65, 'BNB-USDT': 602.0,
    'CFX-USDT': 0.04212, 'DYDX-USDT': 0.1143, 'ETH-USDT': 1897.9, 'LINK-USDT': 8.324,
    'ONDO-USDT': 0.3484, 'XMR-USDT': 394.1,
}


@pytest.fixture(autouse=True)
def _reset_warned(monkeypatch):
    monkeypatch.setattr('pkg.monkey_bx._TICK_WARNED', set())


@pytest.mark.parametrize('symbol', sorted(PRECIOS_PORTFOLIO))
def test_todo_el_portfolio_esta_en_la_tabla(symbol):
    assert symbol in SYMBOL_TRADING_RULES, f'{symbol} caería al tick por defecto (0.01)'


@pytest.mark.parametrize('symbol,precio', sorted(PRECIOS_PORTFOLIO.items()))
def test_ningun_par_del_portfolio_dispara_el_aviso(symbol, precio):
    assert _warn_if_tick_implausible(symbol, precio) is False


@pytest.mark.parametrize('symbol,precio', sorted(PRECIOS_PORTFOLIO.items()))
def test_la_entrada_limit_queda_pegada_al_mercado(symbol, precio):
    """El offset diseñado es 2bps; con un tick sano la desviación se mantiene chica."""
    for side in ('LONG', 'SHORT'):
        px = _sanitize_entry_limit_price(precio, symbol, side)
        assert abs(px / precio - 1) < 0.002, f'{symbol} {side}: entrada a {(px/precio-1)*100:.2f}%'
        if side == 'LONG':
            assert px < precio, 'un LONG maker no debe cruzar el book'
        else:
            assert px > precio


@pytest.mark.parametrize('symbol,precio', sorted(PRECIOS_PORTFOLIO.items()))
def test_el_tp_respeta_la_distancia_disenada(symbol, precio):
    """Un TP de 1.2% no debe convertirse en uno de 3.3% por el redondeo."""
    tp = _round_trigger_price(precio * 1.012, symbol, 'LONG', 'TAKE_PROFIT_MARKET')
    assert tp > precio
    assert (tp / precio - 1) < 0.016, f'{symbol}: TP efectivo {(tp/precio-1)*100:.2f}% vs 1.2% diseñado'


def test_simbolo_ausente_y_barato_dispara_el_aviso(monkeypatch):
    monkeypatch.setattr(
        'pkg.monkey_bx.SYMBOL_TRADING_RULES',
        {k: v for k, v in SYMBOL_TRADING_RULES.items() if k != 'ONDO-USDT'},
    )
    assert _warn_if_tick_implausible('ONDO-USDT', 0.3484) is True


def test_el_aviso_se_emite_una_sola_vez_por_simbolo(monkeypatch):
    monkeypatch.setattr(
        'pkg.monkey_bx.SYMBOL_TRADING_RULES',
        {k: v for k, v in SYMBOL_TRADING_RULES.items() if k != 'ONDO-USDT'},
    )
    emitidos = []
    monkeypatch.setattr('pkg.monkey_bx.emit_lifecycle_event',
                        lambda *a, **k: emitidos.append(k) or {})
    for _ in range(5):
        _warn_if_tick_implausible('ONDO-USDT', 0.3484)
    assert len(emitidos) == 1
    assert emitidos[0]['reason'] == 'tick_size_implausible'


def test_precio_invalido_no_rompe_el_aviso():
    for malo in (None, 0, -1, 'x', float('nan')):
        assert _warn_if_tick_implausible('ONDO-USDT', malo) is False


def test_reproduce_el_descolocamiento_de_ondo_con_la_tabla_vieja(monkeypatch):
    """Los números exactos que motivaron el fix."""
    monkeypatch.setattr(
        'pkg.monkey_bx.SYMBOL_TRADING_RULES',
        {k: v for k, v in SYMBOL_TRADING_RULES.items() if k != 'ONDO-USDT'},
    )
    precio = 0.3484
    assert _tick_size_for('ONDO-USDT') == 0.01

    entrada = _sanitize_entry_limit_price(precio, 'ONDO-USDT', 'LONG')
    assert entrada == pytest.approx(0.34)
    assert (entrada / precio - 1) < -0.02, 'entrada a más de 2% bajo el mercado'

    tp = _round_trigger_price(precio * 1.012, 'ONDO-USDT', 'LONG', 'TAKE_PROFIT_MARKET')
    assert tp == pytest.approx(0.36)
    assert (tp / precio - 1) > 0.03, 'TP casi 3x el diseñado'


def test_los_ticks_reales_estan_holgados_frente_al_umbral():
    peor = max(_tick_size_for(s) / p for s, p in PRECIOS_PORTFOLIO.items())
    assert peor < MAX_TICK_PCT_OF_PRICE / 3
