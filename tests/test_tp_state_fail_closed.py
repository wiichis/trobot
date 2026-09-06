"""Un error de lectura del estado NO debe borrar el estado.

Contexto (06/09/2026). `_load_state_df` devolvía un DataFrame VACÍO en silencio cuando
`pd.read_csv` fallaba. La cadena que eso desataba:

  1. lectura falla -> frame vacío
  2. `upsert_tp_state` no encuentra la fila -> la crea desde `_default_row`
     (`tp_stage=none`, `tp_mode=legacy_market_tp`)
  3. `_save_state_df` escribe un archivo con ESA SOLA FILA
  4. el seguimiento de TP de TODOS los demás pares desaparece

Un único error de lectura transitorio destruía todo el estado sin dejar rastro. Es la
firma que dejó BCH-USDT: llenó TP1 (0,14 -> 0,10, +0,1456 verificado en el exchange),
el ladder dejó de seguirse, la fila quedó con valores por defecto, y no hubo ningún
`clear_tp_state` que lo explicara.

Ahora se falla CERRADO: si el archivo existe y no se puede leer, se levanta
`TpStateUnreadable`, se emite un CRITICAL, y NADIE escribe.
"""
import pytest

import pkg.tp_stage_state as tps


@pytest.fixture(autouse=True)
def _aislado(tmp_path, monkeypatch):
    monkeypatch.setattr(tps, 'TP_STAGE_STATE_CSV', tmp_path / 'tp_stage_state.csv')


def _poblar():
    tps.upsert_tp_state('BCH-USDT', 'SHORT', tp_stage='tp2_live', tp_mode='partial_limit_tp')
    tps.upsert_tp_state('BNB-USDT', 'SHORT', tp_stage='tp1_live', tp_mode='partial_limit_tp')
    tps.upsert_tp_state('XMR-USDT', 'LONG', tp_stage='none', tp_mode='partial_limit_tp')


def _romper_lectura(monkeypatch):
    def _boom(*a, **k):
        raise OSError('lectura fallida')
    monkeypatch.setattr(tps.pd, 'read_csv', _boom)


class TestNoSeBorraNada:
    def test_un_upsert_con_lectura_rota_no_toca_el_archivo(self, monkeypatch):
        _poblar()
        antes = tps.TP_STAGE_STATE_CSV.read_text(encoding='utf-8')
        _romper_lectura(monkeypatch)
        tps.upsert_tp_state('BCH-USDT', 'SHORT', break_even_state='active')
        assert tps.TP_STAGE_STATE_CSV.read_text(encoding='utf-8') == antes

    def test_el_upsert_reporta_que_no_persistio(self, monkeypatch):
        _poblar()
        _romper_lectura(monkeypatch)
        out = tps.upsert_tp_state('BCH-USDT', 'SHORT', break_even_state='active')
        assert out['persist_ok'] is False
        assert 'read_failed' in out['persist_error']

    def test_clear_no_borra_a_ciegas(self, monkeypatch):
        _poblar()
        antes = tps.TP_STAGE_STATE_CSV.read_text(encoding='utf-8')
        _romper_lectura(monkeypatch)
        tps.clear_tp_state('BCH-USDT', 'SHORT', source='x')
        assert tps.TP_STAGE_STATE_CSV.read_text(encoding='utf-8') == antes

    def test_el_estado_sobrevive_a_un_fallo_transitorio(self, monkeypatch):
        """Lo que pasaba antes: se perdían los OTROS pares, no sólo el que se tocaba."""
        _poblar()
        real = tps.pd.read_csv
        roto = {'si': True}

        def _quizas_falla(*a, **k):
            if roto['si']:
                raise OSError('lectura fallida')
            return real(*a, **k)

        monkeypatch.setattr(tps.pd, 'read_csv', _quizas_falla)
        tps.upsert_tp_state('BCH-USDT', 'SHORT', break_even_state='active')

        roto['si'] = False   # el fallo era transitorio; la lectura se recupera
        assert tps.get_tp_state('BNB-USDT', 'SHORT')['tp_stage'] == 'tp1_live'
        assert tps.get_tp_state('XMR-USDT', 'LONG')['tp_mode'] == 'partial_limit_tp'
        assert tps.get_tp_state('BCH-USDT', 'SHORT')['tp_stage'] == 'tp2_live'


class TestLecturaDegradada:
    def test_get_tp_state_no_tumba_al_llamador(self, monkeypatch):
        _poblar()
        _romper_lectura(monkeypatch)
        st = tps.get_tp_state('BCH-USDT', 'SHORT')
        assert st['tp_stage'] == 'none', 'devuelve default para no romper'

    def test_avisa_con_CRITICAL(self, monkeypatch):
        _poblar()
        capt = []
        monkeypatch.setattr(tps, '_emit_state_event',
                            lambda cat, sev='WARN', **f: capt.append((cat, sev)))
        _romper_lectura(monkeypatch)
        tps.upsert_tp_state('BCH-USDT', 'SHORT', break_even_state='active')
        assert ('tp_state_read_failed', 'CRITICAL') in capt

    def test_marca_la_persistencia_como_no_sana(self, monkeypatch):
        _poblar()
        _romper_lectura(monkeypatch)
        tps.upsert_tp_state('BCH-USDT', 'SHORT', break_even_state='active')
        assert tps.is_tp_state_persist_healthy() is False


def test_archivo_inexistente_sigue_siendo_normal(tmp_path, monkeypatch):
    """Arrancar sin archivo NO es un error: es el primer arranque."""
    monkeypatch.setattr(tps, 'TP_STAGE_STATE_CSV', tmp_path / 'no_existe.csv')
    st = tps.upsert_tp_state('X-USDT', 'LONG', tp_stage='none')
    assert st['tp_stage'] == 'none'
    assert st['persist_ok'] is True
