"""Telemetría del fail-closed del filtro de régimen HTF.

Cuando el filtro está pedido y sus features no se pueden calcular, indicadores.py
bloquea TODAS las señales. Es lo correcto (no operar a ciegas), pero sin aviso deja el
portfolio mudo sin alarma — la clase de fallo que ya costó meses en este proyecto.
"""
import numpy as np
import pandas as pd
import pytest

import pkg.indicadores as ind


@pytest.fixture(autouse=True)
def _limpiar_estado():
    ind._HTF_ESTADO_PREVIO.clear()
    yield
    ind._HTF_ESTADO_PREVIO.clear()


@pytest.fixture
def espia(monkeypatch):
    vistos = []
    monkeypatch.setattr(ind, "emit_lifecycle_event",
                        lambda cat, sev="INFO", **kw: vistos.append((cat, sev, kw)))
    return vistos


def test_el_fallo_emite_critical(espia):
    ind._htf_telemetria("BCH-USDT", "resample_vacio", htf_tf="1h")
    assert len(espia) == 1
    cat, sev, kw = espia[0]
    assert cat == "htf_regime_fail_closed"
    assert sev == "CRITICAL"
    assert kw["symbol"] == "BCH-USDT"
    assert kw["motivo"] == "resample_vacio"


def test_no_repite_mientras_el_estado_no_cambie(espia):
    """El job corre cada 5 min: un evento por ciclo serían ~288/día por par."""
    for _ in range(50):
        ind._htf_telemetria("BCH-USDT", "resample_vacio")
    assert len(espia) == 1


def test_avisa_cuando_se_recupera(espia):
    ind._htf_telemetria("BCH-USDT", "resample_vacio")
    ind._htf_telemetria("BCH-USDT", "ok", htf_adx=21.5)
    assert [e[0] for e in espia] == ["htf_regime_fail_closed", "htf_regime_recuperado"]
    assert espia[1][1] == "INFO"


def test_el_arranque_sano_no_hace_ruido(espia):
    """Sin fallo previo, un 'ok' es operación normal y no merece evento."""
    ind._htf_telemetria("BCH-USDT", "ok", htf_adx=21.5)
    assert espia == []


def test_un_cambio_de_motivo_vuelve_a_avisar(espia):
    ind._htf_telemetria("BCH-USDT", "resample_vacio")
    ind._htf_telemetria("BCH-USDT", "excepcion:KeyError")
    assert len(espia) == 2
    assert espia[1][2]["motivo"] == "excepcion:KeyError"


def test_el_estado_es_por_simbolo(espia):
    ind._htf_telemetria("BCH-USDT", "resample_vacio")
    ind._htf_telemetria("BNB-USDT", "resample_vacio")
    assert len(espia) == 2
    assert {e[2]["symbol"] for e in espia} == {"BCH-USDT", "BNB-USDT"}


def test_observar_no_puede_romper_el_calculo(monkeypatch):
    """Si la telemetría explota, los indicadores tienen que seguir."""
    def explota(*a, **k):
        raise RuntimeError("telegram caido")
    monkeypatch.setattr(ind, "emit_lifecycle_event", explota)
    ind._htf_telemetria("BCH-USDT", "resample_vacio")  # no debe propagar


def test_el_adx_nan_en_la_ultima_barra_tambien_avisa(espia):
    """Segundo camino silencioso: las features existen pero la barra a evaluar es NaN,
    y `.fillna(False)` la bloquea igual."""
    ind._htf_telemetria("BCH-USDT", "adx_nan_en_ultima_barra", htf_tf="1h")
    assert espia[0][0] == "htf_regime_fail_closed"
    assert espia[0][2]["motivo"] == "adx_nan_en_ultima_barra"


def test_el_evento_dice_que_se_bloquea_todo(espia):
    ind._htf_telemetria("BCH-USDT", "resample_vacio")
    assert "TODAS" in espia[0][2]["detalle"]
