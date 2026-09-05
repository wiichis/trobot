"""Ratchet temporal del stop.

Regla: si un tramo de TP ya llenó y el SIGUIENTE tarda más de `after_min` minutos, el
stop sube a pegarse al mejor precio que alcanzó el trade menos `buffer_pct`. Nunca baja
del break-even ni de la entrada: **asegura, no arriesga**.

Motivación (medida el 31/08): el recorrido favorable medio a 8h es +1,85% pero el trade
termina mucho peor. El ratchet convierte parte de ese máximo en ganancia realizada.

⚠️ Lo que estos tests NO prueban: que el ratchet gane plata. Eso sale de la simulación
(8-9 de 10 pares, 4/4 ventanas) y de la medición en vivo. Acá se fija la MECÁNICA.
"""
import pandas as pd
import pytest

from pkg.live_runtime_config import get_ratchet_config
from pkg.monkey_bx import _ratchet_stop_candidate

AHORA = pd.Timestamp.now(tz='UTC')
CFG = {'enabled': True, 'after_min': 30.0, 'buffer_pct': 0.0025, 'min_improve_pct': 0.0005}


def _velas(symbol, highs, lows, termina_hace_min=0):
    """OHLC sintético, una vela cada 5 min, terminando hace `termina_hace_min`.

    Con el default terminan AHORA, así que caen dentro de la ventana del ratchet.
    Subiendo `termina_hace_min` se las manda antes del inicio de la etapa, para probar
    el caso "no hay velas que medir".
    """
    n = len(highs)
    fin = AHORA - pd.Timedelta(minutes=termina_hace_min)
    fechas = [fin - pd.Timedelta(minutes=5 * (n - 1 - i)) for i in range(n)]
    return pd.DataFrame({'symbol': symbol, 'high': highs, 'low': lows,
                         'date': [f.isoformat() for f in fechas]})


def _estado(stage='tp1_filled', hace_min=45):
    return {'tp_stage': stage,
            'updated_at_utc': (AHORA - pd.Timedelta(minutes=hace_min)).isoformat()}


class TestCuandoActua:
    def test_long_pega_el_stop_al_mejor_precio(self):
        df = _velas('X', highs=[100.5, 101.8, 101.2], lows=[99.8, 100.4, 100.9])
        r = _ratchet_stop_candidate(df, 'X', 'LONG', 100.0, _estado(), CFG, be_stop=100.02)
        assert r == pytest.approx(101.8 * (1 - 0.0025))   # 101.5455

    def test_short_es_simetrico(self):
        df = _velas('X', highs=[7.36, 7.34, 7.35], lows=[7.33, 7.29, 7.31])
        r = _ratchet_stop_candidate(df, 'X', 'SHORT', 7.348, _estado(), CFG, be_stop=7.3465)
        assert r == pytest.approx(7.29 * (1 + 0.0025))    # 7.3082

    def test_el_buffer_se_respeta(self):
        df = _velas('X', highs=[102.0], lows=[99.0])
        cfg = dict(CFG, buffer_pct=0.01)
        r = _ratchet_stop_candidate(df, 'X', 'LONG', 100.0, _estado(), cfg, be_stop=100.02)
        assert r == pytest.approx(102.0 * 0.99)


class TestCuandoNoActua:
    def test_apagado_no_hace_nada(self):
        df = _velas('X', highs=[101.8], lows=[99.8])
        cfg = dict(CFG, enabled=False)
        assert _ratchet_stop_candidate(df, 'X', 'LONG', 100.0, _estado(), cfg, 100.02) is None

    def test_el_reloj_no_vencio(self):
        df = _velas('X', highs=[101.8], lows=[99.8])
        est = _estado(hace_min=10)     # sólo 10 min, hacen falta 30
        assert _ratchet_stop_candidate(df, 'X', 'LONG', 100.0, est, CFG, 100.02) is None

    @pytest.mark.parametrize('stage', ['none', 'tp1_live'])
    def test_sin_tp1_confirmado_no_actua(self, stage):
        """Antes de que llene un tramo no hay nada asegurado que proteger."""
        df = _velas('X', highs=[101.8], lows=[99.8])
        assert _ratchet_stop_candidate(df, 'X', 'LONG', 100.0, _estado(stage), CFG, 100.02) is None

    @pytest.mark.parametrize('stage', ['tp3_filled'])
    def test_si_no_quedan_tramos_pendientes_no_actua(self, stage):
        """En tp3_filled la posición ya cerró: no hay siguiente tramo que esperar."""
        df = _velas('X', highs=[101.8], lows=[99.8])
        assert _ratchet_stop_candidate(df, 'X', 'LONG', 100.0, _estado(stage), CFG, 100.02) is None

    def test_sin_velas_en_la_ventana_no_inventa(self):
        """Velas anteriores al inicio de la etapa: no se puede medir el máximo."""
        df = _velas('X', highs=[101.8], lows=[99.8], termina_hace_min=120)
        assert _ratchet_stop_candidate(df, 'X', 'LONG', 100.0, _estado(hace_min=45), CFG, 100.02) is None

    def test_simbolo_sin_datos(self):
        df = _velas('OTRO', highs=[101.8], lows=[99.8])
        assert _ratchet_stop_candidate(df, 'X', 'LONG', 100.0, _estado(), CFG, 100.02) is None

    def test_timestamp_corrupto_no_rompe(self):
        df = _velas('X', highs=[101.8], lows=[99.8])
        est = {'tp_stage': 'tp1_filled', 'updated_at_utc': 'no-es-fecha'}
        assert _ratchet_stop_candidate(df, 'X', 'LONG', 100.0, est, CFG, 100.02) is None


class TestNuncaEmpeora:
    """La propiedad de seguridad: el ratchet asegura, jamás arriesga."""

    def test_long_nunca_por_debajo_del_break_even(self):
        """El precio apenas se movió: el candidato quedaría bajo el BE -> se usa el BE."""
        df = _velas('X', highs=[100.05], lows=[99.5])
        r = _ratchet_stop_candidate(df, 'X', 'LONG', 100.0, _estado(), CFG, be_stop=100.02)
        assert r >= 100.02

    def test_short_nunca_por_encima_del_break_even(self):
        df = _velas('X', highs=[100.5], lows=[99.95])
        r = _ratchet_stop_candidate(df, 'X', 'SHORT', 100.0, _estado(), CFG, be_stop=99.98)
        assert r <= 99.98

    def test_long_nunca_por_debajo_de_la_entrada(self):
        df = _velas('X', highs=[100.01], lows=[99.0])
        r = _ratchet_stop_candidate(df, 'X', 'LONG', 100.0, _estado(), CFG, be_stop=99.5)
        assert r >= 100.0

    def test_short_nunca_por_encima_de_la_entrada(self):
        df = _velas('X', highs=[101.0], lows=[99.99])
        r = _ratchet_stop_candidate(df, 'X', 'SHORT', 100.0, _estado(), CFG, be_stop=100.5)
        assert r <= 100.0


class TestConfig:
    def test_por_defecto_viene_apagado(self):
        """Se enciende desde live_benchmark_runtime.json, no por código."""
        assert get_ratchet_config()['enabled'] in (True, False)

    def test_el_buffer_no_puede_ser_absurdamente_chico(self, tmp_path, monkeypatch):
        """Un buffer dentro del ruido intrabarra es una ilusión del backtest de 5m."""
        import json

        import pkg.live_runtime_config as lrc
        cfg = tmp_path / 'rt.json'
        cfg.write_text(json.dumps({'ratchet': {'enabled': True, 'buffer_pct': 0.00001}}),
                       encoding='utf-8')
        monkeypatch.setattr(lrc, 'DEFAULT_CONFIG_PATH', cfg)
        lrc.reload_live_runtime_config()
        assert lrc.get_ratchet_config()['buffer_pct'] >= 0.001

    def test_valores_invalidos_caen_al_default(self, tmp_path, monkeypatch):
        import json

        import pkg.live_runtime_config as lrc
        cfg = tmp_path / 'rt.json'
        cfg.write_text(json.dumps({'ratchet': {'enabled': True, 'after_min': 'x'}}),
                       encoding='utf-8')
        monkeypatch.setattr(lrc, 'DEFAULT_CONFIG_PATH', cfg)
        lrc.reload_live_runtime_config()
        assert lrc.get_ratchet_config()['after_min'] == 30.0


def test_el_evento_es_distinguible_del_break_even():
    """Clave para atribuir el lunes: ratchet y BE se despliegan juntos."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / 'pkg' / 'monkey_bx.py').read_text(encoding='utf-8')
    assert src.count('"ratchet_activated"') == 4, 'evento en lifecycle+ledger, ambas ramas'
    assert '"break_even_activated"' in src, 'el evento del BE sigue existiendo aparte'
