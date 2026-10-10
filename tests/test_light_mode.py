"""Modo ligero (10/10/2026): sólo salta indicadores cuando el bot está ocioso."""
import json

import pytest

import pkg.light_mode as lm


@pytest.fixture
def env(tmp_path, monkeypatch):
    cfg = tmp_path / "light.json"
    cfg.write_text(json.dumps({"enabled": True, "max_saltos_seguidos": 3}))
    pend = tmp_path / "pend.csv"; pend.write_text("symbol,orderId,type,stopPrice,time\n")
    tp = tmp_path / "tp.csv"; tp.write_text("symbol,tp_stage\nBNB-USDT,none\n")
    monkeypatch.setattr(lm, "CONFIG_PATH", cfg); monkeypatch.setattr(lm, "PENDING_PREV", pend); monkeypatch.setattr(lm, "TP_STATE", tp)
    monkeypatch.setattr(lm, "_saltos_seguidos", 0); monkeypatch.setattr(lm, "_primer_ciclo", True); monkeypatch.setattr(lm, "_ultimo_estado", None)
    return cfg, pend, tp


PAUSADAS = lambda: False
HABILITADAS = lambda: True


def test_primer_ciclo_calcula_y_luego_salta(env):
    assert lm.skip_indicators(PAUSADAS) is False       # el primer ciclo siempre corre
    assert lm.skip_indicators(PAUSADAS) is True


def test_cada_cuatro_ciclos_calcula_igual(env):
    lm.skip_indicators(PAUSADAS)
    r = [lm.skip_indicators(PAUSADAS) for _ in range(8)]
    assert r == [True, True, True, False, True, True, True, False]     # tope de 3 saltos seguidos


def test_no_salta_con_entradas_habilitadas(env):
    lm.skip_indicators(HABILITADAS)
    assert all(lm.skip_indicators(HABILITADAS) is False for _ in range(5))


def test_no_salta_con_ordenes_pendientes(env):
    _, pend, _ = env
    lm.skip_indicators(PAUSADAS)
    pend.write_text("symbol,orderId,type,stopPrice,time\nBNB-USDT,123,STOP_MARKET,700,1\n")
    assert lm.skip_indicators(PAUSADAS) is False


def test_no_salta_con_posicion_en_el_estado_de_tp(env):
    _, _, tp = env
    lm.skip_indicators(PAUSADAS)
    tp.write_text("symbol,tp_stage\nBNB-USDT,tp1_live\n")
    assert lm.skip_indicators(PAUSADAS, lambda s: True) is False        # el exchange dice abierta
    assert lm.skip_indicators(PAUSADAS, lambda s: None) is False        # no se pudo leer: no se arriesga


def test_fila_huerfana_se_descarta_solo_si_el_exchange_confirma_plano(env):
    _, _, tp = env
    tp.write_text("symbol,tp_stage\nBNB-USDT,tp2_live\nCFX-USDT,none\n")
    lm.skip_indicators(PAUSADAS, lambda s: False)                       # primer ciclo
    assert lm.skip_indicators(PAUSADAS, lambda s: False) is True


def test_lectura_estricta_de_posicion(monkeypatch):
    import types, sys
    def fake(resp):
        m = types.ModuleType("pkg.bingx"); m.perpetual_swap_positions = lambda s: resp
        monkeypatch.setitem(sys.modules, "pkg.bingx", m); monkeypatch.setattr(__import__("pkg"), "bingx", m, raising=False)
    fake(json.dumps({"code": 0, "data": []})); assert lm._posicion_abierta("X-USDT") is False
    fake(json.dumps({"code": 0, "data": [{"positionAmt": "-0.5"}]})); assert lm._posicion_abierta("X-USDT") is True
    fake(json.dumps({"code": 100413, "msg": "Null apiKey"})); assert lm._posicion_abierta("X-USDT") is None   # error de auth != plana
    fake("no es json"); assert lm._posicion_abierta("X-USDT") is None


def test_al_volver_a_estar_activo_reinicia_el_contador(env):
    _, pend, _ = env
    lm.skip_indicators(PAUSADAS); assert lm.skip_indicators(PAUSADAS) is True
    pend.write_text("symbol,orderId,type,stopPrice,time\nX-USDT,1,LIMIT,1,1\n")
    assert lm.skip_indicators(PAUSADAS) is False
    pend.write_text("symbol,orderId,type,stopPrice,time\n")
    assert lm.skip_indicators(PAUSADAS) is True


@pytest.mark.parametrize("contenido", ["\x00\x01basura", "a,b\n1,2\n"])
def test_ante_archivo_raro_no_salta(env, contenido):
    _, _, tp = env
    lm.skip_indicators(PAUSADAS)
    tp.write_text(contenido)
    assert lm.skip_indicators(PAUSADAS) is False


def test_archivo_de_estado_ausente_o_config_apagada(env):
    cfg, pend, tp = env
    lm.skip_indicators(PAUSADAS)
    pend.unlink()
    assert lm.skip_indicators(PAUSADAS) is False           # sin foto de órdenes: no se arriesga
    pend.write_text("symbol,orderId,type,stopPrice,time\n")
    cfg.write_text(json.dumps({"enabled": False}))
    assert lm.skip_indicators(PAUSADAS) is False


def test_excepcion_interna_no_salta(env):
    def roto(): raise RuntimeError("x")
    assert lm.skip_indicators(roto) is False
