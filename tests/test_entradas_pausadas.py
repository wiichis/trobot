"""Interruptor de entradas nuevas (02/10/2026).

La estrategia actual no tiene edge medido (scripts/edge_horizonte.py) y el usuario decidió
dejar de abrir posiciones mientras se investigan otras. El interruptor corta SÓLO las
entradas: la gestión de las posiciones abiertas (colocando_TK_SL,
unrealized_profit_positions) no lo consulta.
"""
import json

import pytest

import pkg.live_runtime_config as lrc
import pkg.monkey_bx as mbx


def _cfg(tmp_path, monkeypatch, contenido):
    p = tmp_path / "rt.json"
    p.write_text(json.dumps(contenido), encoding="utf-8")
    monkeypatch.setattr(lrc, "DEFAULT_CONFIG_PATH", p)
    lrc.reload_live_runtime_config()


@pytest.fixture(autouse=True)
def _recargar_al_final():
    yield
    lrc.reload_live_runtime_config()


@pytest.mark.parametrize("contenido, esperado", [
    ({}, True),                                    # sin bloque: como siempre
    ({"entries": {"enabled": True}}, True),
    ({"entries": {"enabled": False}}, False),
    ({"entries": {"enabled": "false"}}, False),
    ({"entries": {"enabled": "off"}}, False),
    ({"entries": "raro"}, True),
])
def test_getter(tmp_path, monkeypatch, contenido, esperado):
    _cfg(tmp_path, monkeypatch, contenido)
    assert lrc.are_entries_enabled() is esperado


def test_pausado_no_evalua_senales_y_avisa_una_sola_vez(monkeypatch):
    llamadas = {"pendientes": 0, "eventos": []}

    def pendientes():
        llamadas["pendientes"] += 1

    def no_deberia(*a, **k):
        raise AssertionError("con las entradas pausadas no se evalúan señales")

    monkeypatch.setattr(mbx, "obteniendo_ordenes_pendientes", pendientes)
    monkeypatch.setattr(mbx, "are_entries_enabled", lambda: False)
    monkeypatch.setattr(mbx.pkg.price_bingx_5m, "currencies_list", no_deberia)
    monkeypatch.setattr(mbx, "emit_lifecycle_event", lambda cat, sev, **f: llamadas["eventos"].append(cat))
    monkeypatch.setattr(mbx, "_ENTRADAS_PAUSADAS_AVISADO", False)

    mbx.colocando_ordenes()
    mbx.colocando_ordenes()

    assert llamadas["pendientes"] == 2           # el registro de órdenes se sigue refrescando
    assert llamadas["eventos"] == ["entries_paused"]


def test_habilitado_sigue_el_camino_normal(monkeypatch):
    llegó = {}
    monkeypatch.setattr(mbx, "obteniendo_ordenes_pendientes", lambda: None)
    monkeypatch.setattr(mbx, "are_entries_enabled", lambda: True)
    monkeypatch.setattr(mbx, "is_entry_hour_allowed_utc", lambda: llegó.setdefault("gate", True) and False)
    mbx.colocando_ordenes()
    assert llegó.get("gate")
