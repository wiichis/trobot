"""Confirmación de fills de TP consultando la orden en el exchange (modo exchange_state).

Contexto (02/10/2026). Cuando una orden TP LIMIT desaparecía del book, el bot decidía si
se había llenado mirando si la posición se redujo. Consultadas en BingX las 157 órdenes
TP del 28/07 al 30/09: llenaron 66 y la inferencia vio 41. Las que se perdían:

- los 3 TP3: su cantidad es todo lo que queda, así que al llenar la posición queda en
  cero y la reducción no se puede medir. Se registraban como trail/be/stop_loss.
- 13 TP1 que llenaron justo antes de un stop (posición en cero al detectarlo).
- 7 que llenaron con la posición viva pero con la base de cantidad desfasada (entrada
  llenada en varias veces): el bot volvía a someter el mismo tramo.

Las respuestas grabadas son las reales del 02/10 (LINK TP3 del 15/08, AVAX TP3 del
23/08, AVAX TP1 del 09/09).
"""
import json

import pandas as pd
import pytest

from tests.conftest import make_orders_df
from tests.exchange_falso import montar

LINK_TP3 = "2088352128828440576"
AVAX_TP3 = "2091388360835727360"
AVAX_TP1 = "2097399327612731392"

RESP_LINK_TP3_FILLED = json.dumps({"code": 0, "msg": "", "data": {"order": {
    "symbol": "LINK-USDT", "orderId": 2088352128828440576, "side": "SELL",
    "positionSide": "LONG", "type": "LIMIT", "status": "FILLED", "origQty": "1.5",
    "executedQty": "1.5", "price": "9.159", "avgPrice": "9.159",
    "updateTime": 1786756932000}}})
RESP_AVAX_TP3_CANCELLED = json.dumps({"code": 0, "msg": "", "data": {"order": {
    "symbol": "AVAX-USDT", "orderId": 2091388360835727360, "side": "BUY",
    "positionSide": "SHORT", "type": "LIMIT", "status": "CANCELLED", "origQty": "3",
    "executedQty": "0", "price": "7.153", "avgPrice": "0.000",
    "updateTime": 1787491150000}}})
RESP_AVAX_TP1_FILLED = json.dumps({"code": 0, "msg": "", "data": {"order": {
    "symbol": "AVAX-USDT", "orderId": 2097399327612731392, "side": "BUY",
    "positionSide": "SHORT", "type": "LIMIT", "status": "FILLED", "origQty": "1",
    "executedQty": "1", "price": "7.922", "avgPrice": "7.922",
    "updateTime": 1788947324000}}})


def _responde(monkeypatch, mb, respuesta):
    llamadas = []

    def _q(symbol, order_id, timeout=10):
        llamadas.append((symbol, str(order_id)))
        return respuesta
    monkeypatch.setattr(mb.pkg.bingx, "query_order", _q)
    return llamadas


def _gone(symbol, oid, side, pside, price):
    return make_orders_df([{"symbol": symbol, "orderId": int(oid), "type": "LIMIT",
                            "side": side, "positionSide": pside, "price": price}])


def _seed(tps, symbol, pside, idx, oid, qty, price, base):
    tps.set_tp_submitted(symbol, pside, tp_idx=idx, order_id=oid, qty=qty, price=price,
                         submit_position_qty=base, tp_mode="partial_limit_tp",
                         fill_confirmation_mode="exchange_state")


@pytest.fixture
def modo_exchange(monkeypatch):
    import pkg.monkey_bx as mb
    monkeypatch.setattr(mb, "get_tp_fill_confirmation_mode", lambda: "exchange_state")
    cierres = []
    monkeypatch.setattr(mb, "_record_trade_closed",
                        lambda s, p, reason, stop_price=None: cierres.append((s, p, reason)))
    return cierres


class TestParseoRespuestaReal:
    def test_filled(self, monkeypatch):
        import pkg.monkey_bx as mb
        _responde(monkeypatch, mb, RESP_LINK_TP3_FILLED)
        st = mb._query_order_status("LINK-USDT", LINK_TP3)
        assert st["status"] == "FILLED"
        assert st["executed_qty"] == 1.5
        assert st["avg_price"] == 9.159

    def test_cancelled_doble_l_y_simple(self, monkeypatch):
        import pkg.monkey_bx as mb
        _responde(monkeypatch, mb, RESP_AVAX_TP3_CANCELLED)
        assert mb._query_order_status("AVAX-USDT", AVAX_TP3)["status"] == "CANCELLED"
        _responde(monkeypatch, mb, RESP_AVAX_TP3_CANCELLED.replace("CANCELLED", "CANCELED"))
        assert mb._query_order_status("AVAX-USDT", AVAX_TP3)["status"] == "CANCELLED"

    @pytest.mark.parametrize("respuesta", [
        json.dumps({"code": 80016, "msg": "order not exist", "data": {}}),
        json.dumps({"code": 0, "msg": "", "data": {}}),
        "<html>502 Bad Gateway</html>",
    ])
    def test_ante_la_duda_none(self, monkeypatch, respuesta):
        import pkg.monkey_bx as mb
        _responde(monkeypatch, mb, respuesta)
        assert mb._query_order_status("LINK-USDT", LINK_TP3) is None

    def test_id_distinto_es_none(self, monkeypatch):
        import pkg.monkey_bx as mb
        _responde(monkeypatch, mb, RESP_LINK_TP3_FILLED)
        assert mb._query_order_status("LINK-USDT", "2088352128828440832") is None

    def test_error_de_red_es_none(self, monkeypatch):
        import pkg.monkey_bx as mb

        def _boom(*a, **k):
            raise TimeoutError("read timed out")
        monkeypatch.setattr(mb.pkg.bingx, "query_order", _boom)
        assert mb._query_order_status("LINK-USDT", LINK_TP3) is None


class TestTransiciones:
    def test_tp3_lleno_con_posicion_en_cero_se_confirma_y_cierra_como_tp3(
        self, isolated_workspace, runtime_event_spy, modo_exchange, monkeypatch
    ):
        """LINK 15/08: TP3 llenó a 9,159 y quedó registrado como stop."""
        import pkg.monkey_bx as mb
        import pkg.tp_stage_state as tps

        _seed(tps, "LINK-USDT", "LONG", 3, LINK_TP3, 1.5, 9.159, 1.5)
        _responde(monkeypatch, mb, RESP_LINK_TP3_FILLED)
        monkeypatch.setattr(mb, "total_positions", lambda _s: (None, None, None, None, None))

        mb._log_pending_order_transitions(
            _gone("LINK-USDT", LINK_TP3, "SELL", "LONG", 9.159), make_orders_df([]))

        filled = [e for e in runtime_event_spy["lifecycle"] if e["category"] == "tp3_filled"]
        assert len(filled) == 1
        assert filled[0]["source"] == "exchange_order_status"
        assert filled[0]["fill_price"] == 9.159
        led = [e for e in runtime_event_spy["ledger"] if e["event_type"] == "tp3_filled"][0]
        assert led["data_quality"] == "actual"
        assert led["actual_fill_price"] == 9.159
        assert modo_exchange == [("LINK-USDT", "LONG", "tp3")]
        st = tps.get_tp_state("LINK-USDT", "LONG")
        assert st["tp_stage"] == "tp3_filled"
        assert st["tp3_order_id"] == ""

    def test_tp3_cancelado_con_posicion_en_cero_no_es_cierre_por_tp(
        self, isolated_workspace, runtime_event_spy, modo_exchange, monkeypatch
    ):
        """AVAX 23/08: el stop cerró la posición y BingX canceló el TP3."""
        import pkg.monkey_bx as mb
        import pkg.tp_stage_state as tps

        _seed(tps, "AVAX-USDT", "SHORT", 3, AVAX_TP3, 3.0, 7.153, 3.0)
        _responde(monkeypatch, mb, RESP_AVAX_TP3_CANCELLED)
        monkeypatch.setattr(mb, "total_positions", lambda _s: (None, None, None, None, None))

        mb._log_pending_order_transitions(
            _gone("AVAX-USDT", AVAX_TP3, "BUY", "SHORT", 7.153), make_orders_df([]))

        assert [e for e in runtime_event_spy["lifecycle"] if e["category"].startswith("tp")] == []
        assert modo_exchange == []
        flat = [e for e in runtime_event_spy["ledger"] if e["event_type"] == "tp3_gone_position_flat"]
        assert len(flat) == 1 and flat[0]["data_quality"] == "actual"
        assert tps.get_tp_state("AVAX-USDT", "SHORT")["tp3_order_id"] == ""

    def test_tp1_lleno_con_base_desfasada_avanza_el_stage(
        self, isolated_workspace, runtime_event_spy, modo_exchange, monkeypatch
    ):
        """AVAX 09/09: la posición no 'se redujo' respecto de la base (la entrada siguió
        llenando) y el bot volvió a someter TP1. Con el exchange, el stage avanza."""
        import pkg.monkey_bx as mb
        import pkg.tp_stage_state as tps

        _seed(tps, "AVAX-USDT", "SHORT", 1, AVAX_TP1, 1.0, 7.922, 3.0)
        _responde(monkeypatch, mb, RESP_AVAX_TP1_FILLED)
        monkeypatch.setattr(mb, "total_positions", lambda _s: ("AVAX-USDT", "SHORT", 8.0, 3.0, 0.0))

        mb._log_pending_order_transitions(
            _gone("AVAX-USDT", AVAX_TP1, "BUY", "SHORT", 7.922), make_orders_df([]))

        st = tps.get_tp_state("AVAX-USDT", "SHORT")
        assert st["tp_stage"] == "tp1_filled"
        assert mb._next_tp_idx_from_stage(st["tp_stage"]) == 2
        assert not any(e["category"] == "tp1_failed" for e in runtime_event_spy["lifecycle"])
        assert modo_exchange == []   # TP1 no cierra la posición

    def test_orden_todavia_abierta_es_un_gone_falso(
        self, isolated_workspace, runtime_event_spy, modo_exchange, monkeypatch
    ):
        import pkg.monkey_bx as mb
        import pkg.tp_stage_state as tps

        _seed(tps, "AVAX-USDT", "SHORT", 1, AVAX_TP1, 1.0, 7.922, 3.0)
        _responde(monkeypatch, mb, RESP_AVAX_TP1_FILLED.replace(
            '"status": "FILLED"', '"status": "NEW"').replace('"executedQty": "1"', '"executedQty": "0"'))
        monkeypatch.setattr(mb, "total_positions", lambda _s: ("AVAX-USDT", "SHORT", 8.0, 3.0, 0.0))

        mb._log_pending_order_transitions(
            _gone("AVAX-USDT", AVAX_TP1, "BUY", "SHORT", 7.922), make_orders_df([]))

        assert [e["event_type"] for e in runtime_event_spy["ledger"]] == ["tp1_gone_but_open"]
        assert runtime_event_spy["lifecycle"] == []
        st = tps.get_tp_state("AVAX-USDT", "SHORT")
        assert st["tp_stage"] == "tp1_live"
        assert st["tp1_order_id"] == AVAX_TP1

    def test_si_la_consulta_falla_cae_a_la_inferencia(
        self, isolated_workspace, runtime_event_spy, modo_exchange, monkeypatch
    ):
        import pkg.monkey_bx as mb
        import pkg.tp_stage_state as tps

        _seed(tps, "AVAX-USDT", "SHORT", 1, AVAX_TP1, 1.0, 7.922, 3.0)
        # query_order queda bloqueada por el conftest (responde code=-1).
        monkeypatch.setattr(mb, "total_positions", lambda _s: ("AVAX-USDT", "SHORT", 8.0, 2.0, 0.0))

        mb._log_pending_order_transitions(
            _gone("AVAX-USDT", AVAX_TP1, "BUY", "SHORT", 7.922), make_orders_df([]))

        filled = [e for e in runtime_event_spy["lifecycle"] if e["category"] == "tp1_filled"]
        assert len(filled) == 1
        assert filled[0]["source"] == "exchange_state_unavailable_fallback_inferred"
        assert filled[0]["confirmation_mode"] == "inferred"

    def test_en_modo_inferred_no_consulta_el_exchange(
        self, isolated_workspace, runtime_event_spy, monkeypatch
    ):
        import pkg.monkey_bx as mb
        import pkg.tp_stage_state as tps

        monkeypatch.setattr(mb, "get_tp_fill_confirmation_mode", lambda: "inferred")
        llamadas = _responde(monkeypatch, mb, RESP_AVAX_TP1_FILLED)
        _seed(tps, "AVAX-USDT", "SHORT", 1, AVAX_TP1, 1.0, 7.922, 3.0)
        monkeypatch.setattr(mb, "total_positions", lambda _s: ("AVAX-USDT", "SHORT", 8.0, 2.0, 0.0))

        mb._log_pending_order_transitions(
            _gone("AVAX-USDT", AVAX_TP1, "BUY", "SHORT", 7.922), make_orders_df([]))

        assert llamadas == []


def test_ciclo_completo_tp3_lleno_no_queda_registrado_como_stop(
    isolated_workspace, runtime_event_spy, monkeypatch
):
    """Exchange falso + CSV reales: snapshot -> TP3 llena y BingX cancela el SL ->
    snapshot -> transiciones -> SL watch. El cierre queda como tp3 y el watch no lo
    duplica como stop (era lo que pasaba con los 3 TP3 llenos de agosto)."""
    import pkg.tp_stage_state as tps

    ctx = montar(isolated_workspace, monkeypatch, runtime_event_spy, symbol="LINK-USDT",
                 params={}, precio_entrada=9.0, indicadores={"close": 9.0})
    mb, ex = ctx.mb, ctx.ex
    monkeypatch.setattr(mb, "get_tp_fill_confirmation_mode", lambda: "exchange_state")
    monkeypatch.setattr(mb.pkg.bingx, "hystory_PnL", lambda: json.dumps({"code": 0, "data": []}))

    ex.posicion("LONG", 1.5, avg=9.0)
    sl = ex.agregar("STOP_MARKET", "SELL", "LONG", 1.5, stop=8.9)
    tp3 = ex.agregar("LIMIT", "SELL", "LONG", 1.5, price=9.159)
    _seed(tps, "LINK-USDT", "LONG", 3, tp3, 1.5, 9.159, 1.5)
    mb._append_sl_watch("LINK-USDT", 8.9, "LONG", sl)
    mb.obteniendo_ordenes_pendientes()

    ex.llenar_cierre(tp3)
    ex.cancel_order("LINK-USDT", sl)       # BingX cancela el SL al quedar en cero
    mb.obteniendo_ordenes_pendientes()
    mb.sync_cooldowns_from_sl_fills()

    cats = [e["category"] for e in runtime_event_spy["lifecycle"]]
    assert "tp3_filled" in cats
    assert "stop_loss_hit" not in cats
    cierres = pd.read_csv(isolated_workspace / "archivos" / "trade_closed_log.csv")
    assert cierres["close_reason"].tolist() == ["tp3"]
