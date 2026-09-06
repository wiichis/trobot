"""El stop no se recoloca en bucle por un desajuste de redondeo.

Contexto (06/09/2026, BCH-USDT SHORT). El job emitió `break_even_activated` **ocho veces
seguidas con el mismo valor**, una cada 5 min, cancelando y reponiendo la misma orden:

    14:19, 14:24, 14:30, 14:35, 14:40, 14:45, 14:50, 14:55  ->  258.608268

Causa: se comparaba el stop CRUDO (258.608268) contra el que devuelve el exchange, que
está redondeado al tick (258.61). `!=` daba siempre True, así que recolocaba para
siempre. No es peligroso pero son llamadas al exchange al pedo, y cada recolocación abre
una ventana breve sin stop entre el cancel y el post.

El candado de monotonía (`protective_stop`) empeoraba esto: fijaba el valor sin redondear,
volviendo el bucle permanente y determinista.
"""
import pytest

from pkg.monkey_bx import _round_trigger_price


class TestRedondeoAlTick:
    def test_el_caso_real_de_bch(self):
        """258.608268 y 258.61 son el MISMO stop; sin redondear parecían distintos."""
        crudo = 258.608268
        redondeado = _round_trigger_price(crudo, 'BCH-USDT', 'SHORT', 'STOP_MARKET')
        assert crudo != 258.61, 'así se comparaba antes: siempre distinto'
        assert redondeado == 258.61, 'redondeado coincide con lo que guarda el exchange'

    def test_es_idempotente(self):
        """Redondear dos veces da lo mismo: sin esto el bucle volvería por otra vía."""
        una = _round_trigger_price(258.608268, 'BCH-USDT', 'SHORT', 'STOP_MARKET')
        dos = _round_trigger_price(una, 'BCH-USDT', 'SHORT', 'STOP_MARKET')
        assert una == dos

    @pytest.mark.parametrize('symbol,side', [
        ('BCH-USDT', 'SHORT'), ('BNB-USDT', 'SHORT'), ('BNB-USDT', 'LONG'),
        ('CFX-USDT', 'LONG'), ('DYDX-USDT', 'SHORT'), ('ONDO-USDT', 'LONG'),
    ])
    def test_idempotente_en_todo_el_portfolio(self, symbol, side):
        base = {'BCH-USDT': 258.608268, 'BNB-USDT': 756.465620, 'CFX-USDT': 0.0421337,
                'DYDX-USDT': 0.1016234, 'ONDO-USDT': 0.348219}[symbol]
        una = _round_trigger_price(base, symbol, side, 'STOP_MARKET')
        assert _round_trigger_price(una, symbol, side, 'STOP_MARKET') == una

    def test_redondea_en_direccion_conservadora(self):
        """Un stop de SHORT se aleja (sube), no se acerca: no dispara antes de tiempo."""
        crudo = 258.608268
        assert _round_trigger_price(crudo, 'BCH-USDT', 'SHORT', 'STOP_MARKET') >= crudo

    def test_long_tambien_conservador(self):
        """Un stop de LONG baja al redondear: tampoco se acerca al precio."""
        crudo = 756.463
        assert _round_trigger_price(crudo, 'BNB-USDT', 'LONG', 'STOP_MARKET') <= crudo


def test_el_redondeo_esta_antes_de_la_comparacion():
    """Guard: si el redondeo se mueve después del `!=`, el bucle vuelve."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / 'pkg' / 'monkey_bx.py').read_text(encoding='utf-8')
    assert src.count('potencial_nuevo_sl = _round_trigger_price(') == 2, 'ambas ramas'
    for op in ('>', '<'):
        i = src.find(f'if potencial_nuevo_sl {op} last_stop_price')
        assert i > 0
        previo = src[max(0, i - 700):i]
        assert '_round_trigger_price(' in previo, f'rama {op}: falta redondear antes'
