"""Telemetría de transiciones del estado de TP.

Contexto (06/09/2026). BCH-USDT SHORT llenó TP1 (posición 0,14 → 0,10, +0,1456 real
confirmado contra el exchange) y el bot no lo registró: `tp_stage` quedó en `none`, el
ladder no avanzó, y el ratchet y el BE-tras-TP1 quedaron ciegos.

Al reconstruir el origen se descartaron con evidencia `clear_tp_state`, el fallback a
legacy por fallo de persistencia, la carrera con el hilo horario y las filas duplicadas.
**No se pudo cerrar**: la fila lleva los valores de `_default_row`, así que fue
materializada por un upsert que no encontró ninguna existente — pero nadie registra
cuándo una fila se crea, se borra o cambia de `tp_mode`, así que la ventana entre las
12:48 y las 14:19 es inobservable después del hecho.

Estos tres eventos convierten el próximo caso en evidencia en vez de arqueología. NO
cambian comportamiento: sólo observan.
"""
import pytest

import pkg.tp_stage_state as tps


@pytest.fixture(autouse=True)
def _aislado(tmp_path, monkeypatch):
    monkeypatch.setattr(tps, 'TP_STAGE_STATE_CSV', tmp_path / 'tp_stage_state.csv')


@pytest.fixture
def eventos(monkeypatch):
    capt = []
    monkeypatch.setattr(tps, '_emit_state_event',
                        lambda cat, sev='WARN', **f: capt.append({'cat': cat, 'sev': sev, **f}))
    return capt


class TestFilaRecreada:
    """El síntoma exacto del caso BCH: alguien materializa la fila desde defaults."""

    def test_avisa_si_un_upsert_incidental_crea_la_fila(self, eventos):
        # set_break_even_state / bump_protective_stop asumen que la fila YA existe
        tps.set_break_even_state('BCH-USDT', 'SHORT', 'active')
        assert [e for e in eventos if e['cat'] == 'tp_state_row_recreated']
        ev = [e for e in eventos if e['cat'] == 'tp_state_row_recreated'][0]
        assert ev['sev'] == 'CRITICAL'
        assert ev['symbol'] == 'BCH-USDT'

    def test_el_protective_stop_tambien_lo_dispara(self, eventos):
        tps.bump_protective_stop('BCH-USDT', 'SHORT', 258.6, is_long=False)
        assert [e for e in eventos if e['cat'] == 'tp_state_row_recreated']

    def test_la_apertura_normal_NO_avisa(self, eventos):
        """Crear la fila al abrir posición es legítimo: declara la etapa."""
        tps.upsert_tp_state('BCH-USDT', 'SHORT', tp_stage='none', tp_mode='partial_limit_tp')
        assert not [e for e in eventos if e['cat'] == 'tp_state_row_recreated']

    def test_sobre_una_fila_existente_no_avisa(self, eventos):
        tps.upsert_tp_state('BCH-USDT', 'SHORT', tp_stage='none', tp_mode='partial_limit_tp')
        eventos.clear()
        tps.set_break_even_state('BCH-USDT', 'SHORT', 'active')
        assert not [e for e in eventos if e['cat'] == 'tp_state_row_recreated']


class TestCambioDeModo:
    def test_avisa_al_cambiar_de_modo(self, eventos):
        tps.upsert_tp_state('X-USDT', 'SHORT', tp_stage='none', tp_mode='partial_limit_tp')
        eventos.clear()
        tps.upsert_tp_state('X-USDT', 'SHORT', tp_mode='legacy_market_tp')
        ev = [e for e in eventos if e['cat'] == 'tp_mode_changed']
        assert len(ev) == 1
        assert ev[0]['tp_mode_anterior'] == 'partial_limit_tp'
        assert ev[0]['tp_mode_nuevo'] == 'legacy_market_tp'

    def test_reescribir_el_mismo_modo_no_avisa(self, eventos):
        tps.upsert_tp_state('X-USDT', 'SHORT', tp_stage='none', tp_mode='partial_limit_tp')
        eventos.clear()
        tps.upsert_tp_state('X-USDT', 'SHORT', tp_mode='partial_limit_tp')
        assert not [e for e in eventos if e['cat'] == 'tp_mode_changed']


class TestBorrado:
    def test_registra_el_origen(self, eventos):
        tps.upsert_tp_state('X-USDT', 'SHORT', tp_stage='tp2_live', tp_mode='partial_limit_tp')
        eventos.clear()
        tps.clear_tp_state('X-USDT', 'SHORT', source='sl_watch_stop_loss_inferido')
        ev = [e for e in eventos if e['cat'] == 'tp_state_cleared']
        assert len(ev) == 1
        assert ev[0]['source'] == 'sl_watch_stop_loss_inferido'
        assert ev[0]['tp_stage_previo'] == 'tp2_live', 'guarda qué se perdió'

    def test_sin_origen_queda_marcado(self, eventos):
        tps.upsert_tp_state('X-USDT', 'SHORT', tp_stage='none')
        eventos.clear()
        tps.clear_tp_state('X-USDT', 'SHORT')
        assert [e for e in eventos if e['cat'] == 'tp_state_cleared'][0]['source'] == 'sin_origen'

    def test_borrar_algo_inexistente_no_avisa(self, eventos):
        tps.clear_tp_state('NO-EXISTE', 'SHORT', source='x')
        assert not eventos

    def test_el_unico_llamador_declara_su_origen(self):
        import inspect

        import pkg.monkey_bx as mb
        src = inspect.getsource(mb)
        assert 'clear_tp_state(symbol, position_side, source=' in src


def test_la_telemetria_nunca_rompe_la_persistencia(monkeypatch):
    """Si el emisor falla, el estado se guarda igual. Observar no puede costar datos."""
    def _explota(*a, **k):
        raise RuntimeError('telegram caido')

    monkeypatch.setattr('pkg.lifecycle_events.emit_lifecycle_event', _explota)
    tps.set_break_even_state('X-USDT', 'SHORT', 'active')      # dispara el emisor
    assert tps.get_tp_state('X-USDT', 'SHORT')['break_even_state'] == 'active'
