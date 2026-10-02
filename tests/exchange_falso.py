"""Exchange falso para los tests del camino de ejecución.

Mínimo estado (órdenes, posiciones, precio) y respuestas con la FORMA REAL de BingX, para
que todo lo que está arriba —openOrders → order_id_register.csv → releer, la cola, el
estado de TP— corra de verdad. Ver "Bordes con el mundo real" en CLAUDE.md.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd


class ExchangeFalso:
    """BingX con el estado mínimo para el caso: órdenes, posiciones y precio."""

    def __init__(self, sym, precio_entrada=None):
        self.sym = sym
        self.precio_entrada = precio_entrada   # avgPrice por defecto de las posiciones
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
            symbol=self.sym, orderId=int(oid), side=side, positionSide=pside, type=otype,
            origQty=float(qty), price=float(price), stopPrice=float(stop),
            executedQty=float(executed), time=1758707804138,
        )
        return str(oid)

    def posicion(self, pside, qty, avg=None):
        avg = self.precio_entrada if avg is None else avg
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
            ref = {"LIMIT": float(price), "TAKE_PROFIT_MARKET": float(stopPrice)}.get(type, self.price.get(self.sym, 0.0))
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
            "symbol": self.sym, "positionId": "1", "positionSide": side, "isolated": False,
            "positionAmt": f"{p['qty']}", "availableAmt": f"{p['qty']}",
            "unrealizedProfit": "0.0", "realisedProfit": "0.0", "initialMargin": "1.0",
            "avgPrice": f"{p['avg']}", "leverage": 5,
            "markPrice": f"{self.price.get(self.sym, p['avg'])}", "updateTime": 0,
        } for side, p in self.pos.items() if symbol == self.sym and p["qty"] > 0]
        return json.dumps({"code": 0, "msg": "", "data": data})

    def last_price_trading_par(self, symbol):
        return json.dumps({"code": 0, "msg": "", "data": {"symbol": symbol, "price": f"{self.price[symbol]}"}})


def montar(base, monkeypatch, spy, *, symbol, params, indicadores, precio_entrada=None):
    """Conecta el exchange falso, un best_prod de un par y su fila de indicadores."""
    import pkg.monkey_bx as mb

    ex = ExchangeFalso(symbol, precio_entrada)
    for name in ("query_pending_orders", "post_order", "cancel_order",
                 "perpetual_swap_positions", "last_price_trading_par"):
        monkeypatch.setattr(mb.pkg.bingx, name, getattr(ex, name))
    best = base / "best_prod.json"
    best.write_text(json.dumps([{"symbol": symbol, "params": params}]))
    monkeypatch.setattr(mb, "BEST_PROD_PATH", str(best))
    mensajes = []
    monkeypatch.setattr(mb, "bot_send_text", lambda m: mensajes.append(str(m)))
    monkeypatch.setattr(mb, "_TP_SKIP_EMIT_MEMO", {})
    pd.DataFrame([dict(symbol=symbol, **indicadores)]).to_csv(
        base / "archivos" / "indicadores.csv", index=False)
    return SimpleNamespace(mb=mb, ex=ex, mensajes=mensajes, eventos=spy, base=base)
