"""Cliente mínimo de BingX SPOT para el bot BTC por rangos en modo real (24/10/2026).

Endpoints (verificados contra la implementación de ccxt, 24/10/2026; la documentación oficial es
una SPA que no se deja leer):
  GET  /openApi/spot/v1/common/symbols        reglas del par (stepSize, tickSize, minQty, minNotional)
  GET  /openApi/spot/v1/account/balance       saldos (free / locked por asset)
  GET  /openApi/spot/v1/ticker/price          último precio
  POST /openApi/spot/v1/trade/order           MARKET (quoteOrderQty) / LIMIT GTC (quantity + price)
  GET  /openApi/spot/v1/trade/query           estado de una orden (orderId o clientOrderID)
  POST /openApi/spot/v1/trade/cancel          cancelar
  GET  /openApi/spot/v1/user/commissionRate   comisiones de la cuenta

Firma: igual que `pkg/bingx.py` (HMAC-SHA256 del query string, `X-BX-APIKEY`).

Bordes con el mundo real (lecciones del bot de 5 min):
- Toda respuesta con `code != 0` levanta `ErrorExchange`: sin claves el API devuelve un error que
  se confunde con "no hay nada".
- Los ids de orden se guardan SIEMPRE como texto (el float de un CSV pierde dígitos).
- Las cantidades y precios se mandan como texto ya redondeado al step / tick del exchange; nada de
  valores por defecto en las reglas del contrato.
"""
from __future__ import annotations

import hmac
import json
import math
import os
import time
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from hashlib import sha256
from typing import Callable, Dict, Optional

import requests

API = "https://open-api.bingx.com"


class ErrorExchange(Exception):
    pass


@dataclass(frozen=True)
class Reglas:
    symbol: str
    step: float
    tick: float
    min_qty: float
    min_notional: float
    compra_api: bool
    venta_api: bool

    def qty_abajo(self, q: float) -> str:
        return _al_paso(q, self.step, ROUND_FLOOR)

    def precio_arriba(self, p: float) -> str:
        return _al_paso(p, self.tick, ROUND_CEILING)


def _al_paso(x: float, paso: float, modo) -> str:
    d, p = Decimal(repr(x)), Decimal(repr(paso))
    return str(((d / p).to_integral_value(rounding=modo) * p).normalize().quantize(p))


@dataclass
class Orden:
    order_id: str
    client_id: str
    side: str
    tipo: str
    estado: str           # NEW / PENDING / PARTIALLY_FILLED / FILLED / CANCELED / FAILED
    precio: float         # precio límite (0 en MARKET)
    qty: float            # origQty
    ejecutado: float      # executedQty (base)
    quote: float          # cummulativeQuoteQty (USDT brutos)
    fee: float            # comisión, positiva
    fee_asset: str

    @property
    def precio_medio(self) -> float:
        return self.quote / self.ejecutado if self.ejecutado > 0 else 0.0

    @property
    def terminada(self) -> bool:
        return self.estado in ("FILLED", "CANCELED", "CANCELLED", "FAILED")


def orden_desde_json(d: Dict) -> Orden:
    def f(k):
        v = d.get(k)
        return float(v) if v not in (None, "") else 0.0
    return Orden(order_id=str(d.get("orderId", "")), client_id=str(d.get("clientOrderID") or d.get("clientOrderId") or ""),
                 side=str(d.get("side", "")), tipo=str(d.get("type", "")), estado=str(d.get("status", "")),
                 precio=f("price"), qty=f("origQty"), ejecutado=f("executedQty"), quote=f("cummulativeQuoteQty"),
                 fee=abs(f("fee")), fee_asset=str(d.get("feeAsset") or ""))


Transporte = Callable[[str, str, Dict[str, str], bool], Dict]


class ClienteSpot:
    """`transporte(metodo, path, params, firmado) -> dict` se inyecta en los tests."""

    def __init__(self, api_key: str = "", secret: str = "", transporte: Optional[Transporte] = None, timeout: int = 15):
        self.api_key, self.secret, self.timeout = api_key, secret, timeout
        self._transporte = transporte or self._http

    @classmethod
    def desde_credenciales(cls) -> "ClienteSpot":
        """Clave propia del bot spot (sólo trading spot, sin retiro, IP del server), en
        `/etc/default/trobot`. Sin respaldo en la clave de perpetuos: si falta, no opera."""
        key = os.getenv("TROBOT_BINGX_SPOT_APIKEY", "").strip()
        secret = os.getenv("TROBOT_BINGX_SPOT_SECRETKEY", "").strip()
        if not key or not secret:
            raise ErrorExchange("faltan TROBOT_BINGX_SPOT_APIKEY / TROBOT_BINGX_SPOT_SECRETKEY en el entorno")
        return cls(key, secret)

    def _http(self, metodo: str, path: str, params: Dict[str, str], firmado: bool) -> Dict:
        headers = {}
        if firmado:
            params = dict(params, timestamp=str(int(time.time() * 1000)))
        qs = "&".join(f"{k}={params[k]}" for k in sorted(params))
        url = f"{API}{path}?{qs}"
        if firmado:
            url += "&signature=" + hmac.new(self.secret.encode(), qs.encode(), sha256).hexdigest()
            headers["X-BX-APIKEY"] = self.api_key
        r = requests.request(metodo, url, headers=headers, timeout=self.timeout)
        return json.loads(r.text)

    def _llamar(self, metodo: str, path: str, params: Dict[str, str], firmado: bool = True) -> Dict:
        d = self._transporte(metodo, path, params, firmado)
        if not isinstance(d, dict) or d.get("code") != 0:
            code = d.get("code") if isinstance(d, dict) else None
            msg = d.get("msg") if isinstance(d, dict) else str(d)[:200]
            raise ErrorExchange(f"{path} code={code} msg={msg}")
        return d.get("data")

    # --- lectura -----------------------------------------------------------------
    def reglas(self, symbol: str) -> Reglas:
        data = self._llamar("GET", "/openApi/spot/v1/common/symbols", {"symbol": symbol})
        sims = [x for x in (data or {}).get("symbols", []) if x.get("symbol") == symbol]
        if not sims:
            raise ErrorExchange(f"sin reglas para {symbol}")
        x = sims[0]
        faltan = [k for k in ("stepSize", "tickSize", "minQty", "minNotional") if not x.get(k)]
        if faltan:  # nada de valores por defecto en las reglas del contrato
            raise ErrorExchange(f"reglas de {symbol} sin {faltan}")
        return Reglas(symbol, float(x["stepSize"]), float(x["tickSize"]), float(x["minQty"]),
                      float(x["minNotional"]), bool(x.get("apiStateBuy", False)), bool(x.get("apiStateSell", False)))

    def saldos(self) -> Dict[str, Dict[str, float]]:
        data = self._llamar("GET", "/openApi/spot/v1/account/balance", {})
        out = {}
        for b in (data or {}).get("balances", []):
            out[str(b["asset"])] = {"free": float(b.get("free") or 0), "locked": float(b.get("locked") or 0)}
        return out

    def precio(self, symbol: str) -> float:
        data = self._llamar("GET", "/openApi/spot/v1/ticker/price", {"symbol": symbol}, firmado=False)
        fila = data[0] if isinstance(data, list) else data
        trades = fila.get("trades") if isinstance(fila, dict) else None
        p = float(trades[0]["price"]) if trades else float(fila["price"])
        if not math.isfinite(p) or p <= 0:
            raise ErrorExchange(f"precio inválido {fila}")
        return p

    def libro(self, symbol: str) -> Dict[str, float]:
        """Mejor compra / venta del libro: para medir el deslizamiento real de cada orden."""
        data = self._llamar("GET", "/openApi/spot/v1/ticker/bookTicker", {"symbol": symbol}, firmado=False)
        fila = data[0] if isinstance(data, list) else data
        return {"bid": float(fila["bidPrice"]), "ask": float(fila["askPrice"])}

    def comisiones(self, symbol: str) -> Dict[str, float]:
        data = self._llamar("GET", "/openApi/spot/v1/user/commissionRate", {"symbol": symbol})
        return {"taker": float(data["takerCommissionRate"]), "maker": float(data["makerCommissionRate"])}

    def consultar(self, symbol: str, order_id: str = "", client_id: str = "") -> Orden:
        params = {"symbol": symbol}
        if order_id:
            params["orderId"] = str(order_id)
        else:
            params["clientOrderID"] = client_id
        return orden_desde_json(self._llamar("GET", "/openApi/spot/v1/trade/query", params))

    # --- órdenes ------------------------------------------------------------------
    def comprar_mercado(self, symbol: str, usdt: str, client_id: str) -> Orden:
        d = self._llamar("POST", "/openApi/spot/v1/trade/order",
                         {"symbol": symbol, "side": "BUY", "type": "MARKET", "quoteOrderQty": usdt,
                          "newClientOrderId": client_id})
        return orden_desde_json(dict(d, clientOrderID=client_id))

    def vender_limite(self, symbol: str, qty: str, precio: str, client_id: str) -> Orden:
        d = self._llamar("POST", "/openApi/spot/v1/trade/order",
                         {"symbol": symbol, "side": "SELL", "type": "LIMIT", "timeInForce": "GTC",
                          "quantity": qty, "price": precio, "newClientOrderId": client_id})
        return orden_desde_json(dict(d, clientOrderID=client_id))

    def cancelar(self, symbol: str, order_id: str) -> None:
        self._llamar("POST", "/openApi/spot/v1/trade/cancel", {"symbol": symbol, "orderId": str(order_id)})

    def esperar_terminada(self, symbol: str, order_id: str, intentos: int = 6, pausa: float = 1.0) -> Orden:
        """Una orden de mercado se llena en milisegundos, pero el query puede tardar en reflejarlo."""
        o = self.consultar(symbol, order_id=order_id)
        for _ in range(intentos - 1):
            if o.terminada:
                break
            time.sleep(pausa)
            o = self.consultar(symbol, order_id=order_id)
        return o
