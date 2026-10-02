"""Timeout de la cola de protección con la posición ABIERTA (ONDO-USDT, 24/09/2026).

Lo que pasó en prod (order_submit_log + order_lifecycle_log + execution_ledger):

    08:53:17  entrada LIMIT SELL 57,02 @0,4281 (peso 0,13 → ~24 USDT de notional)
    08:54:31  llena 22,99 (40 %) → SL 22,99 y TP1 por el 100 % (un tercio no llega a 7 USDT)
    ~09:03    el remanente llena 11,04 más → posición 34,03
    09:33:18  TP1 llena → quedan 11,04 (4,65 USDT): ningún TP entra al exchange
    09:34     tp1_failed below_min_close_notional; la fila vuelve a la cola y cuenta
    09:53     contador 20 → se cancela `orderId.iloc[0]` del símbolo = el STOP_MARKET,
              con aviso "⛔ ONDO — Orden cancelada / No se ejecutó a tiempo"
    09:54     el STOP se recoloca: ~1 min sin stop. Se repite 10:12, 10:32 y 10:51.

Los tests corren contra un exchange falso que responde con la FORMA REAL de BingX, y
dejan correr el ciclo de persistencia de verdad: openOrders → order_id_register.csv →
releer; cola → position_id_register.csv → releer; estado de TP → tp_stage_state.csv.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

SYM = "ONDO-USDT"
# Ids reales del caso del 24/09.
ENTRADA = "2103045092158734336"
TP1 = "2103045465791528960"
STOP = "2103056025279414272"   # el que el timeout de las 09:53 canceló

# Lo que el ladder necesita del paramset real de ONDO (pkg/best_prod.json al 24/09):
# TP1 = 0,4281 × (1 − 0,6·0,025), al 70 % del camino → 0,42360495, el ref_lvl del ledger.
ONDO_PARAMS = {"tp": 0.025, "tp_mode": "fixed", "sl_mode": "percent", "sl_pct": 0.01,
               "be_trigger": 0.006, "cooldown": 10, "peso": 0.13}

AVISO_TIMEOUT = "No se ejecutó a tiempo"


class ExchangeFalso:
    """BingX con el estado mínimo para el caso: órdenes, posiciones y precio."""

    def __init__(self):
        self.orders = {}
        self.pos = {}
        self.price = {}
        self.cancel_calls = []
        self.posts = []
        self.cancel_ocupado = set()   # órdenes vivas que el exchange no deja cancelar
        self._seq = 2103070000000000000

    # ── estado ──
    def agregar(self, otype, side, pside, qty, *, price=0.0, stop=0.0, oid=None, executed=0.0):
        if oid is None:
            self._seq += 256
            oid = self._seq
        self.orders[str(oid)] = dict(
            symbol=SYM, orderId=int(oid), side=side, positionSide=pside, type=otype,
            origQty=float(qty), price=float(price), stopPrice=float(stop),
            executedQty=float(executed), time=1758707804138,
        )
        return str(oid)

    def posicion(self, pside, qty, avg=0.4281):
        self.pos[pside] = {"qty": round(float(qty), 8), "avg": float(avg)}

    def llenar_entrada(self, oid, qty):
        o = self.orders[oid]
        o["executedQty"] = round(o["executedQty"] + qty, 8)
        actual = self.pos.get(o["positionSide"], {}).get("qty", 0.0)
        self.posicion(o["positionSide"], actual + qty, avg=o["price"])
        if o["executedQty"] >= o["origQty"] - 1e-9:
            del self.orders[oid]

    def llenar_cierre(self, oid):
        o = self.orders.pop(oid)
        p = self.pos[o["positionSide"]]
        p["qty"] = round(p["qty"] - o["origQty"], 8)

    def stops(self):
        return {k: o for k, o in self.orders.items() if o["type"] == "STOP_MARKET"}

    # ── API de pkg.bingx, con la forma de las respuestas reales ──
    def query_pending_orders(self):
        orders = [{
            "symbol": o["symbol"], "orderId": o["orderId"], "side": o["side"],
            "positionSide": o["positionSide"], "type": o["type"],
            "origQty": f"{o['origQty']}", "price": f"{o['price']}",
            "executedQty": f"{o['executedQty']}", "avgPrice": "0.0000", "cumQuote": "0",
            "stopPrice": f"{o['stopPrice']}" if o["stopPrice"] else "",
            "profit": "0.0000", "commission": "0.000000",
            "status": "PARTIALLY_FILLED" if o["executedQty"] else "NEW",
            "time": o["time"], "updateTime": o["time"], "clientOrderId": "",
            "leverage": "5X", "workingType": "MARK_PRICE", "onlyOnePosition": False,
            "reduceOnly": False, "postOnly": o["type"] == "LIMIT", "stopGuaranteed": "false",
            "triggerOrderId": 0, "trailingStopRate": 0, "trailingStopDistance": 0,
        } for o in self.orders.values()]
        return json.dumps({"code": 0, "msg": "", "data": {"orders": orders}})

    # BingX rechaza los cierres TP/LIMIT/MARKET bajo ~6,4 USDT (101485/110422, 08/07),
    # pero acepta el STOP_MARKET: en prod el SL de 11,04 × 0,4275 ≈ 4,7 USDT entró 9 veces.
    MIN_CIERRE_USDT = 6.4

    def post_order(self, symbol, quantity, price, stopPrice, position_side, type, side, **kw):
        self.posts.append(dict(type=type, side=side, position_side=position_side,
                               qty=float(quantity), price=float(price), stop=float(stopPrice)))
        cierra = (position_side == "LONG") == (side == "SELL")
        if cierra and type in ("LIMIT", "TAKE_PROFIT_MARKET", "MARKET"):
            ref = {"LIMIT": float(price), "TAKE_PROFIT_MARKET": float(stopPrice)}.get(type, self.price.get(SYM, 0.0))
            if float(quantity) * ref < self.MIN_CIERRE_USDT:
                return json.dumps({"code": 101485, "msg": "The minimum order amount is 6.4 USDT", "data": {}})
        oid = self.agregar(type, side, position_side, quantity, price=price, stop=stopPrice)
        return json.dumps({"code": 0, "msg": "", "data": {"order": {
            "orderId": int(oid), "orderID": oid, "symbol": symbol,
            "positionSide": position_side, "side": side, "type": type,
            "price": float(price), "quantity": float(quantity), "stopPrice": float(stopPrice),
            "workingType": "MARK_PRICE", "timeInForce": kw.get("timeInForce", "GTC")}}})

    def cancel_order(self, symbol, order_Id):
        oid = str(order_Id)
        self.cancel_calls.append(oid)
        if oid in self.cancel_ocupado:
            return json.dumps({"code": 109500, "msg": "system busy", "data": {}})
        o = self.orders.pop(oid, None)
        if o is None:
            # Ya llenó o expiró. El código exacto no importa: sólo code == 0 es éxito.
            return json.dumps({"code": 80018, "msg": "order not exist", "data": {}})
        return json.dumps({"code": 0, "msg": "", "data": {"order": {
            "symbol": symbol, "orderId": o["orderId"], "side": o["side"],
            "positionSide": o["positionSide"], "type": o["type"],
            "origQty": f"{o['origQty']}", "executedQty": f"{o['executedQty']}",
            "status": "CANCELLED"}}})

    def perpetual_swap_positions(self, symbol):
        data = [{
            "symbol": SYM, "positionId": "1", "positionSide": side, "isolated": False,
            "positionAmt": f"{p['qty']}", "availableAmt": f"{p['qty']}",
            "unrealizedProfit": "0.0", "realisedProfit": "0.0", "initialMargin": "1.0",
            "avgPrice": f"{p['avg']}", "leverage": 5,
            "markPrice": f"{self.price.get(SYM, p['avg'])}", "updateTime": 0,
        } for side, p in self.pos.items() if symbol == SYM and p["qty"] > 0]
        return json.dumps({"code": 0, "msg": "", "data": data})

    def last_price_trading_par(self, symbol):
        return json.dumps({"code": 0, "msg": "", "data": {"symbol": symbol, "price": f"{self.price[symbol]}"}})


@pytest.fixture
def ondo(isolated_workspace, runtime_event_spy, temp_tp_state, monkeypatch):
    import pkg.monkey_bx as mb

    ex = ExchangeFalso()
    for name in ("query_pending_orders", "post_order", "cancel_order",
                 "perpetual_swap_positions", "last_price_trading_par"):
        monkeypatch.setattr(mb.pkg.bingx, name, getattr(ex, name))
    best = isolated_workspace / "best_prod.json"
    best.write_text(json.dumps([{"symbol": SYM, "params": ONDO_PARAMS}]))
    monkeypatch.setattr(mb, "BEST_PROD_PATH", str(best))
    mensajes = []
    monkeypatch.setattr(mb, "bot_send_text", lambda m: mensajes.append(str(m)))
    monkeypatch.setattr(mb, "_TP_SKIP_EMIT_MEMO", {})
    pd.DataFrame([{
        "symbol": SYM, "date": "2026-09-24 08:50:00+00:00", "close": 0.4266,
        "TP1_S": 0.42168, "TP2_S": 0.41740, "TP3_S": 0.41097, "Stop_Loss_Short": 0.4309,
        "TP1_L": 0.43252, "TP2_L": 0.43725, "TP3_L": 0.44372, "Stop_Loss_Long": 0.4223,
        "Take_Profit_Short": 0.4159, "Take_Profit_Long": 0.4373,
    }]).to_csv(isolated_workspace / "archivos" / "indicadores.csv", index=False)
    return SimpleNamespace(mb=mb, ex=ex, mensajes=mensajes, eventos=runtime_event_spy,
                           base=isolated_workspace)


def _cola(base, *, counter, entry_order_id="", side="SHORT"):
    pd.DataFrame([{"symbol": SYM, "tipo": side, "counter": counter,
                   "entry_order_id": entry_order_id}]).to_csv(
        base / "archivos" / "position_id_register.csv", index=False)


def _leer_cola():
    """Lo que quedó en disco, leído sin pasar por el código bajo prueba."""
    df = pd.read_csv("./archivos/position_id_register.csv", dtype={"entry_order_id": str})
    if "entry_order_id" in df.columns:
        df["entry_order_id"] = df["entry_order_id"].fillna("")
    return df


def _categorias(ctx):
    return [e["category"] for e in ctx.eventos["lifecycle"]]


def _avisos_timeout(ctx):
    return [m for m in ctx.mensajes if AVISO_TIMEOUT in m]


def _tp1_llenado_sin_confirmar(tps):
    """El estado de TP real tras las 09:33: TP1 llenó pero el stage sigue en tp1_live.

    La base del reparto quedó en 22,99 (el primer fill) y la posición creció a 34,03, así
    que la reducción medida (22,99 − 11,04 = 11,95) no llega al 60 % del tramo (13,79) y
    el fill nunca se confirma. Es consecuencia del remanente: se conserva tal cual.
    """
    tps.upsert_tp_state(SYM, "SHORT", tp_mode="partial_limit_tp", tp_stage="none",
                        tp_fill_confirmation_mode="inferred")
    tps.set_tp_submitted(SYM, "SHORT", tp_idx=1, order_id=TP1, qty=22.99, price=0.4237,
                         submit_position_qty=22.99, tp_mode="partial_limit_tp",
                         fill_confirmation_mode="inferred")


# ─────────────────────────────────────────────────────────────────────────────
# (1) El timeout con la posición abierta no toca la protección
# ─────────────────────────────────────────────────────────────────────────────

class TestTimeoutConPosicionAbierta:

    def test_caso_ondo_0953_no_cancela_el_stop(self, ondo):
        """El estado exacto de las 09:53: posición 11,04, sólo el STOP en el book."""
        import pkg.tp_stage_state as tps
        ex = ondo.ex
        ex.posicion("SHORT", 11.04)
        ex.price[SYM] = 0.4210
        ex.agregar("STOP_MARKET", "BUY", "SHORT", 11.04, stop=0.4275, oid=STOP)
        _tp1_llenado_sin_confirmar(tps)
        _cola(ondo.base, counter=19, entry_order_id=ENTRADA)   # este ciclo llega a 20
        ondo.mb.obteniendo_ordenes_pendientes()                 # registro por el camino real

        ondo.mb.colocando_TK_SL()

        assert STOP not in ex.cancel_calls
        assert STOP in ex.orders, "el stop de la posición abierta sigue en el book"
        # Sólo se intentó cancelar la ENTRADA (ya no existía: el exchange lo rechaza).
        assert ex.cancel_calls == [ENTRADA]
        assert not [p for p in ex.posts if p["type"] == "STOP_MARKET"], \
            "no hizo falta recolocar el stop: nunca se fue"
        assert _avisos_timeout(ondo) == []
        assert "entry_order_canceled_or_expired" not in _categorias(ondo)
        assert SYM not in _leer_cola()["symbol"].tolist()

    def test_cancela_solo_el_remanente_aunque_el_tp_sea_la_primera_orden(self, ondo):
        """Sin SL en el book (p. ej. rechazado) el timeout sí llega con el TP vivo.

        El código viejo cancelaba `iloc[0]`: aquí, el TP1. Ahora sólo la entrada, y como
        falta el SL la fila sigue en este ciclo para colocarlo.
        """
        import pkg.tp_stage_state as tps
        ex = ondo.ex
        ex.posicion("SHORT", 34.03)
        ex.price[SYM] = 0.4250
        ex.agregar("LIMIT", "BUY", "SHORT", 22.99, price=0.4237, oid=TP1)          # 1ª fila
        ex.agregar("LIMIT", "SELL", "SHORT", 57.02, price=0.4281, oid=ENTRADA, executed=34.03)
        _tp1_llenado_sin_confirmar(tps)
        _cola(ondo.base, counter=19, entry_order_id=ENTRADA)
        ondo.mb.obteniendo_ordenes_pendientes()

        ondo.mb.colocando_TK_SL()

        assert ex.cancel_calls == [ENTRADA]
        assert TP1 in ex.orders
        stops = [p for p in ex.posts if p["type"] == "STOP_MARKET"]
        assert len(stops) == 1 and stops[0]["qty"] == pytest.approx(34.03), \
            "el SL faltante se coloca en el mismo ciclo, por la posición entera"
        assert _avisos_timeout(ondo) == []
        rem = [e for e in ondo.eventos["lifecycle"] if e["category"] == "entry_remainder_canceled"]
        assert len(rem) == 1 and rem[0]["qty"] == pytest.approx(34.03)
        assert rem[0]["order_id"] == ENTRADA
        # Con SL y TP1 vivos quedó protegida: sale de la cola por el camino normal.
        assert SYM not in _leer_cola()["symbol"].tolist()

    def test_si_falta_el_sl_no_suelta_la_fila(self, ondo, monkeypatch):
        """Una posición sin stop no puede salir de la cola de protección."""
        import pkg.tp_stage_state as tps
        ex = ondo.ex
        ex.posicion("SHORT", 34.03)
        ex.price[SYM] = 0.4250
        ex.agregar("LIMIT", "BUY", "SHORT", 22.99, price=0.4237, oid=TP1)
        _tp1_llenado_sin_confirmar(tps)
        _cola(ondo.base, counter=19, entry_order_id=ENTRADA)
        ondo.mb.obteniendo_ordenes_pendientes()
        rechazos = []

        def _sl_rechazado(*args, **kwargs):
            rechazos.append(args)
            return json.dumps({"code": 110412, "msg": "stop price invalid", "data": {}})
        monkeypatch.setattr(ondo.mb.pkg.bingx, "post_order", _sl_rechazado)   # rechaza todo

        ondo.mb.colocando_TK_SL()

        assert rechazos, "intentó colocar el SL"
        cola = _leer_cola()
        assert cola["symbol"].tolist() == [SYM]
        assert int(cola["counter"].iloc[0]) == 0, "el plazo vuelve a empezar"
        assert ex.cancel_calls == [ENTRADA] and TP1 in ex.orders


# ─────────────────────────────────────────────────────────────────────────────
# El propósito original del timeout sigue intacto: la entrada que no llenó
# ─────────────────────────────────────────────────────────────────────────────

class TestTimeoutSinPosicion:

    def test_entrada_que_no_lleno_se_cancela_y_avisa(self, ondo):
        ex = ondo.ex
        ex.price[SYM] = 0.4250
        ex.agregar("LIMIT", "SELL", "SHORT", 57.02, price=0.4281, oid=ENTRADA)
        _cola(ondo.base, counter=19, entry_order_id=ENTRADA)
        ondo.mb.obteniendo_ordenes_pendientes()

        ondo.mb.colocando_TK_SL()

        assert ex.cancel_calls == [ENTRADA] and ENTRADA not in ex.orders
        assert _avisos_timeout(ondo) == ["⛔ *ONDO* — Orden cancelada\n_No se ejecutó a tiempo_"]
        ev = [e for e in ondo.eventos["lifecycle"] if e["category"] == "entry_order_canceled_or_expired"]
        assert len(ev) == 1 and ev[0]["order_id"] == ENTRADA and ev[0]["reason"] == "protection_timeout"
        led = [e for e in ondo.eventos["ledger"] if e["event_type"] == "entry_order_canceled_or_expired"]
        assert led[0]["order_id"] == ENTRADA
        assert _leer_cola().empty

    def test_si_el_exchange_no_la_cancelo_no_dice_cancelada(self, ondo):
        """Antes `canceled = True` con sólo no lanzar excepción, sin leer la respuesta."""
        ondo.ex.price[SYM] = 0.4250
        _cola(ondo.base, counter=19, entry_order_id=ENTRADA)   # ya expiró: no está en el book
        ondo.mb.obteniendo_ordenes_pendientes()

        ondo.mb.colocando_TK_SL()

        assert ondo.ex.cancel_calls == [ENTRADA]
        assert _avisos_timeout(ondo) == ["⏳ *ONDO* — Orden expirada\n_No se ejecutó a tiempo_"]
        assert _leer_cola().empty

    def test_entrada_listada_que_no_se_pudo_cancelar_sigue_en_cola(self, ondo):
        """Si sigue pendiente puede llenar: ni "expirada" ni soltar la fila."""
        ex = ondo.ex
        ex.price[SYM] = 0.4250
        ex.agregar("LIMIT", "SELL", "SHORT", 57.02, price=0.4281, oid=ENTRADA)
        ex.cancel_ocupado.add(ENTRADA)
        _cola(ondo.base, counter=19, entry_order_id=ENTRADA)
        ondo.mb.obteniendo_ordenes_pendientes()

        ondo.mb.colocando_TK_SL()

        assert _avisos_timeout(ondo) == []
        cola = _leer_cola()
        assert cola["symbol"].tolist() == [SYM] and int(cola["counter"].iloc[0]) == 0

    def test_posicion_ilegible_no_afirma_nada(self, ondo, monkeypatch):
        """`total_positions` confunde "no hay posición" con "no pude leer"."""
        ex = ondo.ex
        ex.price[SYM] = 0.4250
        ex.agregar("STOP_MARKET", "BUY", "SHORT", 11.04, stop=0.4275, oid=STOP)
        _cola(ondo.base, counter=19, entry_order_id=ENTRADA)
        ondo.mb.obteniendo_ordenes_pendientes()
        monkeypatch.setattr(ondo.mb.pkg.bingx, "perpetual_swap_positions", lambda _s: json.dumps(
            {"code": 100410, "msg": "rate limit", "data": None}))

        ondo.mb.colocando_TK_SL()

        assert STOP in ex.orders and STOP not in ex.cancel_calls
        assert _avisos_timeout(ondo) == []
        warn = [e for e in ondo.eventos["lifecycle"] if e["category"] == "execution_quality_warning"]
        assert warn and warn[0]["reason"] == "protection_timeout_position_unreadable"
        assert _leer_cola()["symbol"].tolist() == [SYM]


# ─────────────────────────────────────────────────────────────────────────────
# (2) Con SL y ningún TP posible, la posición está protegida
# ─────────────────────────────────────────────────────────────────────────────

class TestProtegidaSoloConSL:

    def _estado_0934(self, ondo):
        import pkg.tp_stage_state as tps
        ex = ondo.ex
        ex.posicion("SHORT", 11.04)
        ex.price[SYM] = 0.4210
        ex.agregar("STOP_MARKET", "BUY", "SHORT", 11.04, stop=0.4275, oid=STOP)
        _tp1_llenado_sin_confirmar(tps)

    def test_sale_de_la_cola_y_el_contador_no_avanza(self, ondo):
        """09:34 en adelante, 30 ciclos (25 min): nunca llega al timeout."""
        self._estado_0934(ondo)
        _cola(ondo.base, counter=0)
        for _ in range(30):
            ondo.mb.obteniendo_ordenes_pendientes()   # lo refrescan los otros jobs
            ondo.mb.colocando_TK_SL()
            assert _leer_cola().empty

        assert ondo.ex.cancel_calls == []
        assert not [p for p in ondo.ex.posts if p["type"] in ("LIMIT", "STOP_MARKET")]
        assert _avisos_timeout(ondo) == []
        sl_only = [e for e in ondo.eventos["lifecycle"] if e["category"] == "protection_sl_only"]
        assert len(sl_only) == 1, "una vez por hora, no en cada ciclo"
        assert sl_only[0]["reason"] == "tp_below_min_close_notional"

    def test_con_remanente_de_entrada_vivo_sigue_en_cola(self, ondo):
        """El próximo fill puede volver posible el TP (AVAX 19/08: 1,0 → 6,0 en 7 min)."""
        self._estado_0934(ondo)
        ondo.ex.agregar("LIMIT", "SELL", "SHORT", 57.02, price=0.4281, oid=ENTRADA, executed=34.03)
        _cola(ondo.base, counter=0, entry_order_id=ENTRADA)
        ondo.mb.obteniendo_ordenes_pendientes()

        ondo.mb.colocando_TK_SL()

        assert _leer_cola()["symbol"].tolist() == [SYM]
        assert "protection_sl_only" not in _categorias(ondo)

    def test_sin_sl_no_cuenta_como_protegida(self, ondo, monkeypatch):
        self._estado_0934(ondo)
        del ondo.ex.orders[STOP]
        _cola(ondo.base, counter=0)
        ondo.mb.obteniendo_ordenes_pendientes()
        monkeypatch.setattr(ondo.mb.pkg.bingx, "post_order", lambda *a, **k: json.dumps(
            {"code": 110412, "msg": "stop price invalid", "data": {}}))

        ondo.mb.colocando_TK_SL()

        assert _leer_cola()["symbol"].tolist() == [SYM]
        assert "protection_sl_only" not in _categorias(ondo)

    def test_tp_posible_que_falla_no_cuenta_como_protegida(self, ondo, monkeypatch):
        """Sólo el notional mínimo justifica no tener TP; un rechazo del exchange, no."""
        import pkg.tp_stage_state as tps
        ex = ondo.ex
        ex.posicion("SHORT", 34.03)
        ex.price[SYM] = 0.4250
        ex.agregar("STOP_MARKET", "BUY", "SHORT", 34.03, stop=0.4309, oid=STOP)
        tps.upsert_tp_state(SYM, "SHORT", tp_mode="partial_limit_tp", tp_stage="none",
                            tp_fill_confirmation_mode="inferred")
        _cola(ondo.base, counter=0)
        ondo.mb.obteniendo_ordenes_pendientes()
        monkeypatch.setattr(ondo.mb.pkg.bingx, "post_order", lambda *a, **k: json.dumps(
            {"code": 101400, "msg": "rejected", "data": {}}))

        ondo.mb.colocando_TK_SL()

        assert _leer_cola()["symbol"].tolist() == [SYM]
        assert "protection_sl_only" not in _categorias(ondo)


# ─────────────────────────────────────────────────────────────────────────────
# El caso completo, desde la entrada
# ─────────────────────────────────────────────────────────────────────────────

def test_replay_ondo_desde_la_entrada(ondo, monkeypatch):
    """colocando_ordenes → fill parcial → crece → TP1 → 40 ciclos de protección.

    A diferencia de prod, aquí el remanente SÍ figura en openOrders (como en los otros 3
    fills parciales medidos). Mientras está vivo la posición puede crecer, así que no se
    da por protegida; al vencer el plazo se cancela él, y sólo él.
    """
    import pkg.monkey_bx as mb
    ex = ondo.ex
    ex.price[SYM] = 0.4281
    monkeypatch.setattr(mb.pkg.price_bingx_5m, "currencies_list", lambda: [SYM])
    monkeypatch.setattr(mb.pkg.indicadores, "ema_alert", lambda _c: (0.4281, "Alerta de SHORT"))
    monkeypatch.setattr(mb, "total_monkey", lambda: 185.0)

    mb.colocando_ordenes()
    entrada = [oid for oid, o in ex.orders.items() if o["type"] == "LIMIT" and o["side"] == "SELL"]
    assert len(entrada) == 1
    entrada = entrada[0]
    assert _leer_cola()["entry_order_id"].tolist() == [entrada], "el id exacto, sin pasar por float"
    qty_total = ex.orders[entrada]["origQty"]

    # 08:54 — llena 22,99 y se protege: SL + TP1 por todo lo llenado.
    ex.llenar_entrada(entrada, 22.99)
    ex.price[SYM] = 0.4266
    mb.colocando_TK_SL()
    assert _leer_cola().empty
    stop_inicial = list(ex.stops())
    assert len(stop_inicial) == 1
    tp1 = [oid for oid, o in ex.orders.items() if o["type"] == "LIMIT" and o["side"] == "BUY"]
    assert len(tp1) == 1 and ex.orders[tp1[0]]["origQty"] == pytest.approx(22.99)

    # ~09:03 el remanente llena 11,04 más; 09:33 TP1 llena → quedan 11,04.
    ex.llenar_entrada(entrada, 11.04)
    ex.llenar_cierre(tp1[0])
    ex.price[SYM] = 0.4210
    assert entrada in ex.orders, "quedan 22,99 de la entrada en el book"

    for _ in range(40):
        mb.obteniendo_ordenes_pendientes()
        mb.colocando_TK_SL()

    assert ex.cancel_calls == [entrada], "el remanente, una vez; nada más"
    assert entrada not in ex.orders
    assert set(stop_inicial) <= set(ex.stops()), "el stop nunca se tocó"
    assert _avisos_timeout(ondo) == []
    cats = _categorias(ondo)
    assert cats.count("entry_remainder_canceled") == 1
    assert "entry_order_canceled_or_expired" not in cats
    assert "protection_sl_only" in cats
    assert _leer_cola().empty
    assert ex.pos["SHORT"]["qty"] == pytest.approx(11.04)
    assert qty_total > 34.03


# ─────────────────────────────────────────────────────────────────────────────
# Persistencia: el id de la entrada sobrevive el CSV
# ─────────────────────────────────────────────────────────────────────────────

def test_entry_order_id_no_pasa_por_float(isolated_workspace):
    """Una fila de arranque en frío (sin id) basta para que pandas lea la columna como float."""
    import pkg.monkey_bx as mb
    path = isolated_workspace / "archivos" / "position_id_register.csv"
    pd.DataFrame([
        {"symbol": SYM, "tipo": "SHORT", "counter": 3, "entry_order_id": ENTRADA},
        {"symbol": "BNB-USDT", "tipo": "LONG", "counter": 0, "entry_order_id": ""},
    ]).to_csv(path, index=False)

    crudo = pd.read_csv(path)["entry_order_id"].iloc[0]
    assert mb._norm_order_id(crudo) != ENTRADA, "lo que pasaría sin dtype=str"

    cola = mb._load_position_queue()
    assert cola["entry_order_id"].tolist() == [ENTRADA, ""]
    cola.to_csv(path, index=False)   # lo que hace colocando_TK_SL al final del ciclo
    assert mb._load_position_queue()["entry_order_id"].tolist() == [ENTRADA, ""]


def test_cola_vieja_sin_columna_se_lee(isolated_workspace):
    """El archivo de prod al desplegar no tiene `entry_order_id`."""
    import pkg.monkey_bx as mb
    (isolated_workspace / "archivos" / "position_id_register.csv").write_text(
        "symbol,tipo,counter\nONDO-USDT,SHORT,7\n")
    cola = mb._load_position_queue()
    assert cola["entry_order_id"].tolist() == [""]
    assert int(cola["counter"].iloc[0]) == 7


def test_entradas_del_registro_ignoran_stop_y_tps():
    import pkg.monkey_bx as mb
    df = pd.DataFrame([
        {"symbol": SYM, "orderId": int(STOP), "type": "STOP_MARKET", "side": "BUY", "positionSide": "SHORT"},
        {"symbol": SYM, "orderId": int(TP1), "type": "LIMIT", "side": "BUY", "positionSide": "SHORT"},
        {"symbol": SYM, "orderId": int(ENTRADA), "type": "LIMIT", "side": "SELL", "positionSide": "SHORT"},
    ])
    assert mb._entry_orders_in_register(df, "SHORT") == [ENTRADA]
    assert mb._entry_orders_in_register(df, "LONG") == []


def test_estado_posicion_con_tipo_nan_no_inventa_sin_posicion(monkeypatch):
    """`str(nan).upper()` es 'NAN': filtrar por ese lado diría "sin posición" sobre una viva."""
    import pkg.monkey_bx as mb
    ex = ExchangeFalso()
    ex.posicion("SHORT", 11.04)
    monkeypatch.setattr(mb.pkg.bingx, "perpetual_swap_positions", ex.perpetual_swap_positions)
    assert mb._estado_posicion(SYM, "NAN") == ("abierta", pytest.approx(11.04))
    assert mb._estado_posicion(SYM, "LONG") == ("sin_posicion", 0.0)
