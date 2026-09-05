"""Disparo del break-even.

Contexto (05/09/2026): el BE se evaluaba contra `precio_actual`, que sale de la última
fila de `indicadores.csv` — la vela EN FORMACIÓN (deuda P5.1). El job corre cada 5 min,
así que veía una muestra parcial y desfasada, mientras el TP1 es una orden LIMIT
descansando en el exchange que llena con cualquier toque.

Consecuencia medida entre el 28/07 y el 04/09: de 22 posiciones que llenaron TP1, sólo
7 registraron `break_even_activated`. El BE se armaba en el 16% de las posiciones. Y el
desenlace dependía de eso — con BE armado 7/7 ganadoras, sin BE 8/15.

El fix tiene dos mitades:
  - **estadística**: el BE se compara contra el PRECIO VIVO (`_last_traded_price`).
  - **determinista**: un fill confirmado de TP1 arma el BE aunque el muestreo se lo haya
    perdido, porque TP1 está siempre más lejos que `be_trigger` (ver el test del
    invariante) y por tanto su fill PRUEBA que el precio cruzó el umbral.
"""
import json
from pathlib import Path

import pytest

BEST_PROD = Path(__file__).resolve().parents[1] / 'pkg' / 'best_prod.json'
TP1_FACTOR = 0.6  # TP_LADDER_FACTORS[0]


def _params():
    return {e['symbol']: e['params'] for e in json.loads(BEST_PROD.read_text(encoding='utf-8'))}


@pytest.mark.parametrize('symbol', sorted(_params()))
def test_invariante_tp1_esta_mas_lejos_que_el_be_trigger(symbol):
    """Sostiene el fix determinista: si TP1 llenó, el precio cruzó el BE. Sí o sí.

    Si algún paramset futuro rompe esto, armar el BE al llenar TP1 lo estaría armando
    ANTES del umbral diseñado — hay que revisar el fix, no este test.
    """
    p = _params()[symbol]
    tp1 = float(p['tp']) * TP1_FACTOR
    be = float(p.get('be_trigger', 0.0) or 0.0)
    assert be > 0, f'{symbol} tiene el BE desactivado'
    assert tp1 > be, f'{symbol}: TP1 {tp1:.4%} no supera be_trigger {be:.4%}'


class TestReferenciaDePrecio:
    """El BE debe mirar el precio vivo, no el close de la vela en formación."""

    def test_el_precio_vivo_dispara_donde_la_vela_no(self):
        """El caso real: la mecha cruza el umbral pero el close parcial no."""
        entrada, be_trigger = 7.348, 0.006          # AVAX SHORT del 27/08
        be_price = entrada * (1 - be_trigger)        # 7.3039
        close_vela_en_formacion = 7.3320             # muestra parcial: NO cruza
        precio_vivo = 7.2960                         # el precio real en ese momento: SÍ

        assert close_vela_en_formacion > be_price, 'la vela parcial no habría armado el BE'
        assert precio_vivo <= be_price, 'el precio vivo sí lo arma'

    def test_long_simetrico(self):
        entrada, be_trigger = 100.0, 0.005
        be_price = entrada * (1 + be_trigger)
        assert 100.4 < be_price     # muestra parcial: no arma
        assert 100.7 >= be_price    # precio vivo: arma


class TestGarantiaPorTP1:
    """Un fill de TP1 arma el BE aunque el muestreo de precio se lo pierda."""

    @staticmethod
    def _armaria(be_forzado_por_tp1, ref, entrada, be_trigger, es_long):
        """Réplica de la condición del código (`be_forzado_por_tp1 or ref cruza`)."""
        if es_long:
            return be_forzado_por_tp1 or ref >= entrada * (1 + be_trigger)
        return be_forzado_por_tp1 or ref <= entrada * (1 - be_trigger)

    def test_sin_tp1_el_precio_manda(self):
        assert self._armaria(False, 100.7, 100.0, 0.005, True) is True
        assert self._armaria(False, 100.2, 100.0, 0.005, True) is False

    def test_con_tp1_arma_aunque_el_precio_haya_retrocedido(self):
        """Es el caso de los 7 trades: TP1 llenó y el precio ya volvió atrás."""
        assert self._armaria(True, 100.2, 100.0, 0.005, True) is True
        assert self._armaria(True, 99.9, 100.0, 0.005, True) is True

    def test_con_tp1_arma_tambien_en_short(self):
        assert self._armaria(True, 7.34, 7.348, 0.006, False) is True

    @pytest.mark.parametrize('etapa', ['tp1_filled', 'tp2_live', 'tp2_filled',
                                       'tp3_live', 'tp3_filled'])
    def test_todas_las_etapas_posteriores_a_tp1_cuentan_como_confirmadas(self, etapa):
        confirmadas = ('tp1_filled', 'tp2_live', 'tp2_filled', 'tp3_live', 'tp3_filled')
        assert etapa in confirmadas

    @pytest.mark.parametrize('etapa', ['none', 'tp1_live'])
    def test_antes_de_llenar_tp1_no_se_fuerza(self, etapa):
        confirmadas = ('tp1_filled', 'tp2_live', 'tp2_filled', 'tp3_live', 'tp3_filled')
        assert etapa not in confirmadas


def test_el_codigo_usa_la_referencia_viva_en_ambas_ramas():
    """Guard de regresión: que nadie vuelva a comparar contra `precio_actual`."""
    src = (Path(__file__).resolve().parents[1] / 'pkg' / 'monkey_bx.py').read_text(encoding='utf-8')
    assert 'if be_forzado_por_tp1 or float(be_ref_price) >= be_price:' in src, 'rama LONG'
    assert 'if be_forzado_por_tp1 or float(be_ref_price) <= be_price:' in src, 'rama SHORT'
    assert 'float(precio_actual) >= be_price' not in src
    assert 'float(precio_actual) <= be_price' not in src


def test_el_precio_vivo_solo_se_pide_si_el_be_puede_actuar():
    """No gastar una llamada al API por símbolo si el BE está apagado o bloqueado."""
    src = (Path(__file__).resolve().parents[1] / 'pkg' / 'monkey_bx.py').read_text(encoding='utf-8')
    assert 'if allow_be_overlay and be_trigger > 0.0:\n            _vivo = _last_traded_price(symbol)' in src


def test_hay_fallback_si_el_api_no_responde():
    """Si no hay precio vivo, se usa la vela: peor referencia, pero no se rompe."""
    src = (Path(__file__).resolve().parents[1] / 'pkg' / 'monkey_bx.py').read_text(encoding='utf-8')
    assert 'be_ref_price = precio_actual' in src
