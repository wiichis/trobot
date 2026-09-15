"""Telemetría de la referencia con la que se calcula el nivel de TP.

Contexto (14/09/2026). Medido sobre 75 posiciones reales: el TP1 que el bot somete está
a **0,70× la distancia que dice el paramset**. Despejando el factor de la escalera da
0,40-0,42 contra el 0,60 de `TP_FACTORS[0]`, y **sin correlación con `tp`** (r=+0,13),
o sea que es constante entre pares y no un efecto proporcional.

Lo que ya se descartó: la columna `TP1_L` del CSV es correcta (BCH 253,20 contra close
258,79 = 2,16% = 0,6×0,036 exacto), `_sanitize_tp_limit_price` sólo recorta si el TP
queda del lado equivocado del mercado, `tp_limit_offset_bps` está en 0 y `best_cfg.json`
no existe. La desviación nace **entre leer la fila y someter la orden**.

El sospechoso es que `latest_values` es la última fila = la vela EN FORMACIÓN. No se
puede confirmar con datos históricos porque al cerrar la vela su `close` se sobrescribe.
De ahí esta anotación, que preserva lo que el bot leyó **en el momento de someter**.

⚠️ Lo que estos tests NO prueban: cuál de las hipótesis es la correcta. Eso lo decide la
telemetría en vivo. Acá se fija que la anotación sea fiel y que **jamás rompa una orden**.
"""
import pandas as pd
import pytest

from pkg.monkey_bx import _tp_submit_reference_note


def _fila(symbol='BCH-USDT', close=258.79, date='2026-09-06T12:50:00Z'):
    return pd.DataFrame([{'symbol': symbol, 'close': close, 'date': date,
                          'TP1_L': close * 1.0216, 'TP1_S': close * 0.9784}])


class TestLoQueRegistra:
    def test_registra_close_barra_y_nivel(self):
        n = _tp_submit_reference_note(_fila(), 'BCH-USDT', 253.20014)
        assert 'ref_close=258.79' in n
        assert 'ref_bar=2026-09-06T12:50:00Z' in n
        assert 'ref_lvl=253.20014' in n

    def test_los_tres_campos_permiten_reconstruir_el_factor(self):
        """Con close y nivel se despeja el factor sin depender de nada más."""
        close, tp = 258.79, 0.036
        nivel = close * (1 - 0.6 * tp)            # TP1 de un SHORT, factor correcto
        n = _tp_submit_reference_note(_fila(close=close), 'BCH-USDT', nivel)
        d = dict(p.split('=', 1) for p in n.split(' '))
        f = (1 - float(d['ref_lvl']) / float(d['ref_close'])) / tp
        assert f == pytest.approx(0.6, abs=1e-6)

    def test_el_simbolo_se_normaliza_a_mayusculas(self):
        assert 'ref_close=258.79' in _tp_submit_reference_note(_fila(), 'bch-usdt', 1.0)

    def test_precision_suficiente_para_pares_baratos(self):
        """CFX cotiza ~0,046: con pocos decimales la anotación sería inútil."""
        n = _tp_submit_reference_note(_fila(symbol='CFX-USDT', close=0.046789), 'CFX-USDT', 0.0046123)
        assert 'ref_close=0.046789' in n
        assert 'ref_lvl=0.0046123' in n


class TestNuncaRompe:
    """La propiedad de seguridad: observar no puede costar una orden."""

    def test_simbolo_ausente_devuelve_vacio(self):
        assert _tp_submit_reference_note(_fila(), 'OTRO-USDT', 1.0) == ''

    def test_dataframe_vacio(self):
        vacio = pd.DataFrame(columns=['symbol', 'close', 'date'])
        assert _tp_submit_reference_note(vacio, 'BCH-USDT', 1.0) == ''

    def test_sin_columna_close_registra_lo_que_puede(self):
        df = pd.DataFrame([{'symbol': 'BCH-USDT', 'date': '2026-09-06T12:50:00Z'}])
        n = _tp_submit_reference_note(df, 'BCH-USDT', 253.2)
        assert 'ref_close' not in n
        assert 'ref_bar=' in n and 'ref_lvl=' in n

    @pytest.mark.parametrize('malo', [None, float('nan'), 'no-es-numero'])
    def test_nivel_invalido_no_rompe(self, malo):
        n = _tp_submit_reference_note(_fila(), 'BCH-USDT', malo)
        assert isinstance(n, str) and 'ref_lvl' not in n

    def test_close_nan_se_omite(self):
        n = _tp_submit_reference_note(_fila(close=float('nan')), 'BCH-USDT', 253.2)
        assert 'ref_close' not in n and 'ref_lvl=253.2' in n

    def test_objeto_que_explota_al_filtrar(self):
        """Si `latest_values` no es lo esperado, se traga y devuelve vacío."""
        class Explota:
            def __getitem__(self, k):
                raise RuntimeError('boom')

        assert _tp_submit_reference_note(Explota(), 'BCH-USDT', 1.0) == ''

    def test_none_como_latest_values(self):
        assert _tp_submit_reference_note(None, 'BCH-USDT', 1.0) == ''


def test_las_dos_ramas_anotan_el_submit():
    """Guard: LONG y SHORT deben registrar la referencia, en el evento y en el ledger."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / 'pkg' / 'monkey_bx.py').read_text(encoding='utf-8')
    assert src.count('_tp_submit_reference_note(') == 3, 'definición + las dos ramas'
    assert src.count('notes=_tp_ref_note') == 2, 'ledger en ambas ramas'
    assert src.count('tp_ref=_tp_ref_note') == 2, 'lifecycle en ambas ramas'


def test_la_anotacion_se_calcula_solo_tras_un_submit_exitoso():
    """No anotar órdenes que el exchange rechazó — la lección del 06/09 (`7b5cf72`)."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / 'pkg' / 'monkey_bx.py').read_text(encoding='utf-8')
    i = 0
    hallados = 0
    while True:
        i = src.find('_tp_ref_note = _tp_submit_reference_note(', i)
        if i < 0:
            break
        assert 'if ok:' in src[max(0, i - 400):i], 'el gate de éxito va antes de anotar'
        hallados += 1
        i += 1
    assert hallados == 2
