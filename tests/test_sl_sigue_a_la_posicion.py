"""El SL tiene que cubrir la posición entera aunque la entrada llene en varias veces.

Caso real BNB-USDT 07/09/2026 (order_submit_log + order_lifecycle_log de prod):

    02:43:21  entrada LIMIT BUY 0,05 @753,85
    02:43:58  llena 0,01 → SL 0,01 @740,77 y TP1 0,01
    02:48:15  la entrada desaparece del book: llenó el resto → posición 0,05
    07:43:18  el SL de 0,01 dispara y cierra el 20 %. El 80 % sigue abierto, y el bot le
              pone un stop nuevo más lejos (727,98). Cierra a las 15:58 en −0,93.

En los 4 fills parciales reales desde el 01/08 el remanente llenó en 4-9 min y el stop
quedó con la cantidad del primer fill: nadie lo redimensiona salvo que el BE o el trailing
muevan el precio.
"""
from __future__ import annotations

import pandas as pd
import pytest

from tests.exchange_falso import montar

BNB = "BNB-USDT"
# El paramset real de BNB (pkg/best_prod.json): TP1 efectivo 0,42·tp = 1,26%.
BNB_PARAMS = {"tp": 0.03, "tp_mode": "fixed", "sl_mode": "percent", "sl_pct": 0.018,
              "be_trigger": 0.006, "cooldown": 10, "peso": 0.20}
SL_REAL = 740.77


@pytest.fixture
def bnb(isolated_workspace, runtime_event_spy, temp_tp_state, monkeypatch):
    return montar(isolated_workspace, monkeypatch, runtime_event_spy, symbol=BNB,
                  params=BNB_PARAMS, precio_entrada=753.85, indicadores={
        "date": "2026-09-07 02:40:00+00:00", "close": 753.85,
        "TP1_L": 767.42, "TP2_L": 776.47, "TP3_L": 790.03, "Stop_Loss_Long": SL_REAL,
        "TP1_S": 740.28, "TP2_S": 731.23, "TP3_S": 717.67, "Stop_Loss_Short": 767.42,
        "Take_Profit_Long": 776.47, "Take_Profit_Short": 731.23,
    })


def _cola(base, symbol, side, *, counter=0, entry_order_id=""):
    pd.DataFrame([{"symbol": symbol, "tipo": side, "counter": counter,
                   "entry_order_id": entry_order_id}]).to_csv(
        base / "archivos" / "position_id_register.csv", index=False)


def _leer_cola():
    df = pd.read_csv("./archivos/position_id_register.csv", dtype={"entry_order_id": str})
    return df


def _categorias(ctx):
    return [e["category"] for e in ctx.eventos["lifecycle"]]


def _entrada_parcial_protegida(ctx):
    """02:43 — entrada de 0,05, llena 0,01, el ciclo coloca SL 0,01 y TP1 0,01."""
    ex = ctx.ex
    ex.price[BNB] = 753.85
    entrada = ex.agregar("LIMIT", "BUY", "LONG", 0.05, price=753.85, oid=2096791401911025664)
    _cola(ctx.base, BNB, "LONG", entry_order_id=entrada)
    ex.llenar_entrada(entrada, 0.01)
    ctx.mb.colocando_TK_SL()
    (stop, o), = ex.stops().items()
    assert o["origQty"] == pytest.approx(0.01) and o["stopPrice"] == pytest.approx(SL_REAL)
    return entrada, stop


class TestCasoBNB:

    def test_el_remanente_llena_y_el_sl_pasa_a_cubrir_todo(self, bnb):
        ex = bnb.ex
        entrada, stop_viejo = _entrada_parcial_protegida(bnb)
        assert _leer_cola()["symbol"].tolist() == [BNB], "remanente vivo: la fila espera"

        # 02:48 — el resto llena.
        ex.llenar_entrada(entrada, 0.04)
        bnb.mb.colocando_TK_SL()

        (stop_nuevo, o), = ex.stops().items()
        assert o["origQty"] == pytest.approx(0.05)
        assert o["stopPrice"] == pytest.approx(SL_REAL), "mismo precio, sólo cambia la cantidad"
        assert stop_viejo in ex.cancel_calls and stop_viejo not in ex.orders
        # El nuevo se colocó ANTES de cancelar el viejo: nunca hubo un instante sin stop.
        posts = [i for i, p in enumerate(ex.posts) if p["type"] == "STOP_MARKET"]
        assert len(posts) == 2
        ev = [e for e in bnb.eventos["lifecycle"] if e["category"] == "stop_resized"]
        assert len(ev) == 1
        assert ev[0]["qty_anterior"] == pytest.approx(0.01) and ev[0]["qty"] == pytest.approx(0.05)
        # El watch del SL apunta al stop nuevo: si no, su fill no se registraría.
        watch = pd.read_csv("./archivos/sl_watch.csv", dtype={"orderId": str})
        assert watch["orderId"].tolist() == [stop_nuevo]

        bnb.mb.colocando_TK_SL()
        assert _leer_cola().empty, "entrada resuelta y protegida: sale de la cola"
        assert len(ex.stops()) == 1 and len([p for p in ex.posts if p["type"] == "STOP_MARKET"]) == 2

    def test_cuando_dispara_cierra_la_posicion_entera(self, bnb):
        """07:43 — en prod cerró el 20%. Ahora cierra todo, y el cierre queda registrado."""
        ex = bnb.ex
        entrada, _ = _entrada_parcial_protegida(bnb)
        ex.llenar_entrada(entrada, 0.04)
        bnb.mb.colocando_TK_SL()
        (stop_nuevo, _o), = ex.stops().items()

        ex.llenar_cierre(stop_nuevo)          # el precio toca 740,77
        assert "LONG" not in ex.pos or ex.pos["LONG"]["qty"] == pytest.approx(0.0)
        bnb.mb.obteniendo_ordenes_pendientes()
        bnb.mb.sync_cooldowns_from_sl_fills()
        hit = [e for e in bnb.eventos["lifecycle"] if e["category"] == "stop_loss_hit"]
        assert len(hit) == 1 and hit[0]["order_id"] == stop_nuevo

    def test_red_de_seguridad_en_unrealized_profit_positions(self, bnb):
        """Una posición que ya salió de la cola: el job de cada 5 min la corrige."""
        ex = bnb.ex
        ex.price[BNB] = 750.0
        ex.posicion("LONG", 0.05)
        stop_viejo = ex.agregar("STOP_MARKET", "SELL", "LONG", 0.01, stop=SL_REAL)
        bnb.mb.obteniendo_ordenes_pendientes()
        # El SL del indicador queda por debajo del actual: el precio del stop no se mueve.
        ind = pd.read_csv("./archivos/indicadores.csv")
        ind["Stop_Loss_Long"] = 735.0
        ind.to_csv("./archivos/indicadores.csv", index=False)

        bnb.mb.unrealized_profit_positions()

        (stop_nuevo, o), = ex.stops().items()
        assert stop_nuevo != stop_viejo and o["origQty"] == pytest.approx(0.05)
        assert o["stopPrice"] == pytest.approx(SL_REAL)
        ev = [e for e in bnb.eventos["lifecycle"] if e["category"] == "stop_resized"]
        assert ev and ev[0]["source"] == "unrealized_profit_positions"


class TestAjusteSeguro:
    """Ante cualquier duda, no tocar el stop que hay."""

    def test_cubierto_no_hace_nada(self, bnb):
        ex = bnb.ex
        ex.posicion("LONG", 0.05)
        ex.agregar("STOP_MARKET", "SELL", "LONG", 0.05, stop=SL_REAL)
        bnb.mb.obteniendo_ordenes_pendientes()
        assert bnb.mb._ajustar_cantidad_sl(BNB, "LONG", 0.05, source="t") == "cubierto"
        assert ex.posts == [] and ex.cancel_calls == []

    def test_sin_cantidad_en_el_registro_no_toca_nada(self, bnb):
        """Si el registro no trae `origQty`, no se puede saber cuánto cubre el stop."""
        ex = bnb.ex
        ex.posicion("LONG", 0.05)
        ex.agregar("STOP_MARKET", "SELL", "LONG", 0.01, stop=SL_REAL)
        bnb.mb.obteniendo_ordenes_pendientes()
        reg = pd.read_csv("./archivos/order_id_register.csv").drop(columns=["origQty"])
        reg.to_csv("./archivos/order_id_register.csv", index=False)
        assert bnb.mb._ajustar_cantidad_sl(BNB, "LONG", 0.05, source="t") == "sin_cantidad"
        assert ex.posts == [] and ex.cancel_calls == []

    def test_cantidad_cero_es_dato_desconocido_no_stop_vacio(self, bnb):
        """Si viniera origQty 0, tratarlo como corto recolocaría el stop en cada ciclo."""
        ex = bnb.ex
        ex.posicion("LONG", 0.05)
        ex.agregar("STOP_MARKET", "SELL", "LONG", 0.0, stop=SL_REAL)
        bnb.mb.obteniendo_ordenes_pendientes()
        assert bnb.mb._ajustar_cantidad_sl(BNB, "LONG", 0.05, source="t") == "sin_cantidad"
        assert ex.posts == [] and ex.cancel_calls == []

    def test_registro_viejo_se_confirma_contra_el_exchange(self, bnb):
        """El registro dice 0,01 pero el exchange ya tiene el stop bueno: no se duplica."""
        ex = bnb.ex
        ex.posicion("LONG", 0.05)
        viejo = ex.agregar("STOP_MARKET", "SELL", "LONG", 0.01, stop=SL_REAL)
        bnb.mb.obteniendo_ordenes_pendientes()
        del ex.orders[viejo]
        ex.agregar("STOP_MARKET", "SELL", "LONG", 0.05, stop=SL_REAL)
        assert bnb.mb._ajustar_cantidad_sl(BNB, "LONG", 0.05, source="t") == "cubierto"
        assert ex.posts == [] and ex.cancel_calls == []

    def test_si_el_exchange_rechaza_el_nuevo_queda_el_viejo(self, bnb, monkeypatch):
        ex = bnb.ex
        ex.posicion("LONG", 0.05)
        viejo = ex.agregar("STOP_MARKET", "SELL", "LONG", 0.01, stop=SL_REAL)
        bnb.mb.obteniendo_ordenes_pendientes()
        monkeypatch.setattr(bnb.mb.pkg.bingx, "post_order", lambda *a, **k:
                            '{"code": 110412, "msg": "stop price invalid", "data": {}}')

        assert bnb.mb._ajustar_cantidad_sl(BNB, "LONG", 0.05, source="t") == "fallido"

        assert viejo in ex.orders and ex.cancel_calls == [], "nunca cancelar sin reemplazo"
        ev = [e for e in bnb.eventos["lifecycle"] if e["category"] == "stop_resize_failed"]
        assert len(ev) == 1 and ev[0]["severity"] == "CRITICAL"

    def test_no_toca_el_stop_del_otro_lado(self, bnb):
        """En Hedge mode puede haber un SHORT del mismo par: su stop no cuenta."""
        ex = bnb.ex
        ex.posicion("LONG", 0.05)
        ex.agregar("STOP_MARKET", "BUY", "SHORT", 0.05, stop=767.42)
        bnb.mb.obteniendo_ordenes_pendientes()
        assert bnb.mb._ajustar_cantidad_sl(BNB, "LONG", 0.05, source="t") == "sin_stop"
        assert ex.posts == [] and ex.cancel_calls == []


class TestBaseDelLadder:

    def test_sube_si_la_posicion_crece_antes_de_tp1(self, temp_tp_state):
        import pkg.monkey_bx as mb
        import pkg.tp_stage_state as tps
        tps.upsert_tp_state("ONDO-USDT", "SHORT", tp_mode="partial_limit_tp", tp_stage="none")
        tps.set_tp_submitted("ONDO-USDT", "SHORT", tp_idx=1, order_id="1", qty=22.99,
                             price=0.4237, submit_position_qty=22.99, tp_mode="partial_limit_tp")
        st = mb._reanclar_base_del_ladder("ONDO-USDT", "SHORT", tps.get_tp_state("ONDO-USDT", "SHORT"), 34.03)
        assert float(st["tp1_submit_position_qty"]) == pytest.approx(34.03)
        # Y con eso el fill de TP1 se puede confirmar: reducción 22,99 ≥ 60% del tramo.
        assert 34.03 - 11.04 >= 0.6 * 22.99 > 22.99 - 11.04

    @pytest.mark.parametrize("stage", ["tp1_filled", "tp2_live", "tp2_filled", "tp3_live"])
    def test_no_se_mueve_despues_de_un_tramo(self, temp_tp_state, stage):
        """Reescribirla con la posición viva fue el bug del 27/07."""
        import pkg.monkey_bx as mb
        import pkg.tp_stage_state as tps
        tps.upsert_tp_state("X-USDT", "LONG", tp_mode="partial_limit_tp", tp_stage=stage,
                            tp1_submit_position_qty=30.0)
        st = mb._reanclar_base_del_ladder("X-USDT", "LONG", tps.get_tp_state("X-USDT", "LONG"), 45.0)
        assert float(st["tp1_submit_position_qty"]) == pytest.approx(30.0)

    def test_una_reduccion_no_la_baja(self, temp_tp_state):
        import pkg.monkey_bx as mb
        import pkg.tp_stage_state as tps
        tps.upsert_tp_state("X-USDT", "LONG", tp_mode="partial_limit_tp", tp_stage="tp1_live",
                            tp1_submit_position_qty=30.0)
        st = mb._reanclar_base_del_ladder("X-USDT", "LONG", tps.get_tp_state("X-USDT", "LONG"), 12.0)
        assert float(st["tp1_submit_position_qty"]) == pytest.approx(30.0)


def test_short_no_pisa_el_watch_del_stop_movido(isolated_workspace, runtime_event_spy,
                                                temp_tp_state, monkeypatch):
    """La rama SHORT reescribía el watch cada ciclo con el precio del INDICADOR.

    Con el stop ya movido por el BE (0,4275 contra 0,4309 del indicador) no casaba,
    quedaba con orderId vacío, y `sync_cooldowns_from_sl_fills` lo descartaba como
    huérfano: el cierre por stop dejaba de registrarse.
    """
    sym = "ONDO-USDT"
    ctx = montar(isolated_workspace, monkeypatch, runtime_event_spy, symbol=sym,
                 params={"tp": 0.025, "tp_mode": "fixed", "sl_pct": 0.01, "be_trigger": 0.006,
                         "cooldown": 10, "peso": 0.13},
                 precio_entrada=0.4281, indicadores={
        "date": "2026-09-24 09:40:00+00:00", "close": 0.4250,
        "TP1_S": 0.42168, "TP2_S": 0.41740, "TP3_S": 0.41097, "Stop_Loss_Short": 0.4309,
        "TP1_L": 0.43252, "TP2_L": 0.43725, "TP3_L": 0.44372, "Stop_Loss_Long": 0.4223,
    })
    ex = ctx.ex
    ex.price[sym] = 0.4250
    ex.posicion("SHORT", 34.03)
    stop = ex.agregar("STOP_MARKET", "BUY", "SHORT", 34.03, stop=0.4275)
    pd.DataFrame([{"symbol": sym, "stop_price": 0.4275, "position_side": "SHORT",
                   "orderId": stop, "ts": "2026-09-24T09:36:44"}]).to_csv(
        "./archivos/sl_watch.csv", index=False)
    import pkg.tp_stage_state as tps
    tps.upsert_tp_state(sym, "SHORT", tp_mode="partial_limit_tp", tp_stage="none")
    _cola(ctx.base, sym, "SHORT")
    ctx.mb.obteniendo_ordenes_pendientes()

    ctx.mb.colocando_TK_SL()

    watch = pd.read_csv("./archivos/sl_watch.csv", dtype={"orderId": str})
    assert watch["orderId"].tolist() == [stop]
    assert watch["stop_price"].tolist() == [pytest.approx(0.4275)]
