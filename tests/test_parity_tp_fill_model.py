"""Modelo de fill de los TP en el parity-sim.

Contexto (25/08/2026): `run_live_parity_portfolio` llenaba un TP con `high >= price`
—simple toque— y además podía llenar TP1, TP2 y TP3 en la MISMA vela. Ninguna de las dos
cosas ocurre en vivo: el live deja órdenes LIMIT maker (`tp_mode: partial_limit_tp`), que
sólo se ejecutan si el mercado atraviesa el nivel, y las manda de a una
(`tp_one_at_a_time: true`), esperando el fill de cada tramo antes de someter el siguiente.

La lógica conservadora (`should_fill_tp_limit`) ya existía y se usaba en `class
Backtester` (el camino del sweep); el parity —que es el que decide todos los A/B— nunca
la invocaba.
"""
import pytest

from pkg.backtesting import LimitFillPolicy, should_fill_tp_limit

POLICY = LimitFillPolicy(buffer_bps=2.0, require_close_confirmation=True)
SOLO_TRADE_THROUGH = LimitFillPolicy(buffer_bps=2.0, require_close_confirmation=False)


class TestLongTPEsUnaVentaLimit:
    """TP de un LONG = orden SELL esperando arriba."""

    def test_tocar_exacto_y_rebotar_no_llena(self):
        """El caso que inflaba la tasa de TP: la vela toca el nivel y se da vuelta.

        La orden está en la cola del book; el toque ejecuta a los de adelante, no a
        nosotros.
        """
        assert should_fill_tp_limit('long', 100.0, bar_high=100.0, bar_low=98.0,
                                    bar_close=98.5, policy=SOLO_TRADE_THROUGH) is False

    def test_atravesar_el_nivel_llena(self):
        """Trade-through: el mercado barrió la cola entera, incluida la nuestra."""
        assert should_fill_tp_limit('long', 100.0, bar_high=100.05, bar_low=98.0,
                                    bar_close=99.0, policy=SOLO_TRADE_THROUGH) is True

    def test_el_buffer_es_de_2bps(self):
        """2 bps sobre 100 = 0.02. Justo por debajo no alcanza; justo encima sí."""
        assert should_fill_tp_limit('long', 100.0, bar_high=100.01, bar_low=99.0,
                                    bar_close=99.5, policy=SOLO_TRADE_THROUGH) is False
        assert should_fill_tp_limit('long', 100.0, bar_high=100.02, bar_low=99.0,
                                    bar_close=99.5, policy=SOLO_TRADE_THROUGH) is True

    def test_el_cierre_confirma_aunque_no_haya_trade_through(self):
        assert should_fill_tp_limit('long', 100.0, bar_high=100.0, bar_low=98.0,
                                    bar_close=100.0, policy=POLICY) is True

    def test_no_tocar_nunca_llena(self):
        assert should_fill_tp_limit('long', 100.0, bar_high=99.9, bar_low=98.0,
                                    bar_close=99.5, policy=POLICY) is False


class TestShortTPEsUnaCompraLimit:
    """TP de un SHORT = orden BUY esperando abajo. Simétrico."""

    def test_tocar_exacto_y_rebotar_no_llena(self):
        assert should_fill_tp_limit('short', 100.0, bar_high=102.0, bar_low=100.0,
                                    bar_close=101.5, policy=SOLO_TRADE_THROUGH) is False

    def test_atravesar_el_nivel_llena(self):
        assert should_fill_tp_limit('short', 100.0, bar_high=102.0, bar_low=99.95,
                                    bar_close=101.0, policy=SOLO_TRADE_THROUGH) is True

    def test_el_cierre_confirma(self):
        assert should_fill_tp_limit('short', 100.0, bar_high=102.0, bar_low=100.0,
                                    bar_close=100.0, policy=POLICY) is True


def test_el_modelo_optimista_y_el_conservador_difieren_justo_en_el_toque():
    """La diferencia entera entre los dos modelos es la vela que toca y rebota."""
    high, low, close, tp = 100.0, 98.0, 98.5, 100.0
    optimista = high >= tp
    conservador = should_fill_tp_limit('long', tp, high, low, close, SOLO_TRADE_THROUGH)
    assert optimista is True and conservador is False


@pytest.mark.parametrize('buffer_bps', [0.0, 1.0, 2.0, 5.0])
def test_el_buffer_escala_con_el_precio_no_es_absoluto(buffer_bps):
    """2 bps son 0.02 sobre 100 pero 0.0000008 sobre CFX a 0.042."""
    pol = LimitFillPolicy(buffer_bps=buffer_bps, require_close_confirmation=False)
    for precio in (0.042, 100.0, 3000.0):
        justo = precio * (1 + buffer_bps / 10000.0)
        assert should_fill_tp_limit('long', precio, justo * 1.000001, precio * 0.99,
                                    precio * 0.995, pol) is True


def test_precios_invalidos_no_rompen():
    assert should_fill_tp_limit('long', 'x', 100.0, 98.0, 99.0, POLICY) is False


class TestUnTramoPorVela:
    """`tp_one_at_a_time`: el live no puede llenar dos tramos en la misma vela."""

    def test_el_parity_lee_la_config_del_live(self):
        from pkg.live_runtime_config import get_tp_mode, get_tp_one_at_a_time
        assert get_tp_mode() == 'partial_limit_tp', 'el live manda TPs LIMIT maker'
        assert get_tp_one_at_a_time() is True

    def test_una_vela_que_barre_los_tres_niveles_solo_llena_el_primero(self):
        """Simula el bucle del parity: sólo el tramo más cercano puede llenar.

        Antes, una vela amplia cerraba la posición entera de una; en vivo entre TP1 y
        TP2 pasan ciclos de 50 s (medido: TP1 02:17, TP2 02:31 en XMR el 22/08).
        """
        plan = [{'price': 100.0, 'qty': 33.0, 'filled': False},
                {'price': 102.0, 'qty': 33.0, 'filled': False},
                {'price': 104.0, 'qty': 34.0, 'filled': False}]
        high, low, close = 105.0, 99.0, 104.5

        hits = []
        for t in plan:
            if t['filled'] or t['qty'] <= 0:
                continue
            if should_fill_tp_limit('long', t['price'], high, low, close, POLICY):
                hits.append(t)
                break  # one_at_a_time
            break

        assert len(hits) == 1
        assert hits[0]['price'] == 100.0

    def test_si_el_tramo_cercano_no_llena_los_lejanos_tampoco_se_evaluan(self):
        """El siguiente tramo ni siquiera se sometió al exchange todavía."""
        plan = [{'price': 100.0, 'qty': 33.0, 'filled': False},
                {'price': 102.0, 'qty': 33.0, 'filled': False}]
        high, low, close = 99.0, 95.0, 98.0

        hits = []
        for t in plan:
            if t['filled'] or t['qty'] <= 0:
                continue
            if should_fill_tp_limit('long', t['price'], high, low, close, POLICY):
                hits.append(t)
            break

        assert hits == []

    def test_los_tramos_ya_llenados_no_bloquean_al_siguiente(self):
        """Con TP1 ya lleno, la evaluación arranca en TP2."""
        plan = [{'price': 100.0, 'qty': 0.0, 'filled': True},
                {'price': 102.0, 'qty': 33.0, 'filled': False}]
        high, low, close = 103.0, 101.0, 102.5

        hits = []
        for t in plan:
            if t['filled'] or t['qty'] <= 0:
                continue
            if should_fill_tp_limit('long', t['price'], high, low, close, POLICY):
                hits.append(t)
            break

        assert len(hits) == 1
        assert hits[0]['price'] == 102.0
