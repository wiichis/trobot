"""El stop protector nunca retrocede, y el reloj del ratchet no se auto-bloquea.

Contexto (06/09/2026, BNB-USDT SHORT, reportado por el usuario). Se vieron estos eventos:

    13:44  ratchet -> 752.73
    14:14  ratchet -> 749.69
    14:45  ratchet -> 743.25
    14:50  BE      -> 756.47   ← el stop RETROCEDIÓ

Dos bugs encadenados:

A) **El reloj se reiniciaba solo.** El ratchet medía su espera contra `updated_at_utc`,
   que `upsert_tp_state` reescribe en CADA llamada — incluida `set_break_even_state`, que
   corre cada ciclo. Así que al disparar se auto-bloqueaba 30 min: por eso los disparos
   caen exactamente cada 30 min, en diente de sierra.

B) **El indicador pisaba al ratchet.** `potencial_nuevo_sl` se recalcula desde cero cada
   ciclo desde `Stop_Loss_Short` (756,547 en ese momento), más flojo que el 743,25 que el
   ratchet ya había puesto. El valor del ratchet no se persistía, y `last_stop_price`
   sale de un CSV que puede venir desfasado.
"""
import pandas as pd
import pytest

import pkg.tp_stage_state as tps


@pytest.fixture(autouse=True)
def _estado_aislado(tmp_path, monkeypatch):
    monkeypatch.setattr(tps, 'TP_STAGE_STATE_CSV', tmp_path / 'tp_stage_state.csv')


class TestMonotonia:
    """El caso real: 743,25 no puede volver a 756,47."""

    def test_short_no_retrocede(self):
        tps.bump_protective_stop('BNB-USDT', 'SHORT', 752.73, is_long=False)
        tps.bump_protective_stop('BNB-USDT', 'SHORT', 749.69, is_long=False)
        tps.bump_protective_stop('BNB-USDT', 'SHORT', 743.25, is_long=False)
        # el indicador propone algo más flojo
        assert tps.bump_protective_stop('BNB-USDT', 'SHORT', 756.47, is_long=False) == 743.25
        assert tps.get_protective_stop('BNB-USDT', 'SHORT') == 743.25

    def test_long_no_retrocede(self):
        tps.bump_protective_stop('X-USDT', 'LONG', 100.5, is_long=True)
        assert tps.bump_protective_stop('X-USDT', 'LONG', 99.0, is_long=True) == 100.5

    def test_si_mejora_avanza(self):
        tps.bump_protective_stop('X-USDT', 'SHORT', 750.0, is_long=False)
        assert tps.bump_protective_stop('X-USDT', 'SHORT', 740.0, is_long=False) == 740.0

    def test_sin_valor_previo_acepta_el_primero(self):
        assert tps.bump_protective_stop('X-USDT', 'LONG', 100.0, is_long=True) == 100.0

    def test_none_no_rompe_ni_borra(self):
        tps.bump_protective_stop('X-USDT', 'LONG', 100.0, is_long=True)
        assert tps.bump_protective_stop('X-USDT', 'LONG', None, is_long=True) == 100.0

    def test_se_resetea_al_abrir_posicion(self):
        """Heredarlo de la posición anterior pinaría el candado en un precio ajeno."""
        tps.bump_protective_stop('X-USDT', 'SHORT', 740.0, is_long=False)
        tps.upsert_tp_state('X-USDT', 'SHORT', tp_stage='none')   # apertura
        assert tps.get_protective_stop('X-USDT', 'SHORT') is None

    def test_al_cerrar_se_borra_la_fila(self):
        tps.bump_protective_stop('X-USDT', 'SHORT', 740.0, is_long=False)
        tps.clear_tp_state('X-USDT', 'SHORT')
        assert tps.get_protective_stop('X-USDT', 'SHORT') is None


class TestRelojDeEtapa:
    """`stage_since_utc` sólo avanza cuando la ETAPA cambia."""

    def test_escribir_otros_campos_no_reinicia_el_reloj(self):
        tps.upsert_tp_state('X-USDT', 'SHORT', tp_stage='tp1_filled')
        t0 = tps.get_stage_since_utc('X-USDT', 'SHORT')
        # esto es lo que corre cada ciclo y rompía el reloj
        tps.set_break_even_state('X-USDT', 'SHORT', 'active')
        tps.upsert_tp_state('X-USDT', 'SHORT', sl_guard_until_utc='x')
        assert tps.get_stage_since_utc('X-USDT', 'SHORT') == t0

    def test_cambiar_de_etapa_si_reinicia_el_reloj(self):
        tps.upsert_tp_state('X-USDT', 'SHORT', tp_stage='tp1_filled')
        t0 = tps.get_stage_since_utc('X-USDT', 'SHORT')
        tps.upsert_tp_state('X-USDT', 'SHORT', tp_stage='tp2_live')
        assert tps.get_stage_since_utc('X-USDT', 'SHORT') != t0

    def test_reescribir_la_MISMA_etapa_no_reinicia(self):
        tps.upsert_tp_state('X-USDT', 'SHORT', tp_stage='tp2_live')
        t0 = tps.get_stage_since_utc('X-USDT', 'SHORT')
        tps.upsert_tp_state('X-USDT', 'SHORT', tp_stage='tp2_live')
        assert tps.get_stage_since_utc('X-USDT', 'SHORT') == t0

    def test_el_ratchet_usa_stage_since_no_updated_at(self):
        import inspect

        import pkg.monkey_bx as mb
        src = inspect.getsource(mb._ratchet_stop_candidate)
        codigo = '\n'.join(ln.split('#')[0] for ln in src.splitlines())
        assert 'stage_since_utc' in codigo, 'el reloj debe ser stage_since_utc'

    def test_el_ratchet_ya_no_se_autobloquea(self):
        """Simula el ciclo real: dispara, escribe estado, y el reloj NO se reinicia."""
        tps.upsert_tp_state('X-USDT', 'SHORT', tp_stage='tp2_live')
        t0 = pd.to_datetime(tps.get_stage_since_utc('X-USDT', 'SHORT'), utc=True)
        for _ in range(5):                      # cinco ciclos del job
            tps.set_break_even_state('X-USDT', 'SHORT', 'active')
            tps.bump_protective_stop('X-USDT', 'SHORT', 740.0, is_long=False)
        t1 = pd.to_datetime(tps.get_stage_since_utc('X-USDT', 'SHORT'), utc=True)
        assert t1 == t0


def test_el_candado_esta_cableado_en_ambas_ramas():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / 'pkg' / 'monkey_bx.py').read_text(encoding='utf-8')
    assert src.count('_prot = get_protective_stop(symbol, positionSide)') == 2
    assert src.count('bump_protective_stop(symbol, positionSide,') == 2
