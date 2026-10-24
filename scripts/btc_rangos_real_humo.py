"""Prueba de humo del modo real (24/10/2026): coloca una COMPRA LÍMITE al 50 % del precio, por la
cantidad mínima del par (~7 USDT a hoy), la consulta y la cancela. Esa orden no se puede llenar.
Prueba la firma de los POST, el alta de la orden, la búsqueda por clientOrderID y la cancelación
antes de que el bot real mande su primera orden de verdad.

Corre en el server con la clave spot cargada:
  sudo bash -c 'set -a; . /etc/default/trobot; set +a; cd /home/ubuntu/TRobot && \
    PYTHONPATH=. env/bin/python3 scripts/btc_rangos_real_humo.py'
"""
from __future__ import annotations

import sys
import time

from pkg.bingx_spot import ClienteSpot, ErrorExchange, orden_desde_json

SYMBOL = "BTC-USDT"


def main() -> int:
    cli = ClienteSpot.desde_credenciales()
    reglas = cli.reglas(SYMBOL)
    precio = cli.precio(SYMBOL)
    sal = cli.saldos()
    libre = sal.get("USDT", {}).get("free", 0.0)
    limite = reglas.precio_arriba(precio * 0.5)
    qty = reglas.qty_abajo(reglas.min_qty * 1.05 + reglas.step)
    nocional = float(limite) * float(qty)
    print(f"precio {precio} · USDT libres en spot {libre:.2f} · orden de prueba: compra {qty} BTC a {limite} "
          f"({nocional:.2f} USDT)")
    if libre < nocional:
        print("No hay USDT suficientes en spot para la prueba.")
        return 1
    cid = f"BRHUMO{int(time.time())}"
    d = cli._llamar("POST", "/openApi/spot/v1/trade/order",
                    {"symbol": SYMBOL, "side": "BUY", "type": "LIMIT", "timeInForce": "GTC", "quantity": qty,
                     "price": limite, "newClientOrderId": cid})
    o = orden_desde_json(dict(d, clientOrderID=cid))
    print("creada:", o.order_id, o.estado)
    try:
        q = cli.consultar(SYMBOL, client_id=cid)
        print("consulta por clientOrderID:", q.order_id, q.estado, "precio", q.precio, "qty", q.qty)
        assert q.order_id == o.order_id, "la consulta por clientOrderID devolvió otra orden"
    finally:
        cli.cancelar(SYMBOL, o.order_id)
    f = cli.esperar_terminada(SYMBOL, o.order_id)
    print("después de cancelar:", f.estado, "ejecutado", f.ejecutado)
    if f.estado not in ("CANCELED", "CANCELLED") or f.ejecutado != 0:
        print("⚠️ la orden no quedó cancelada limpia: revisar en BingX")
        return 1
    print("OK: alta, consulta y cancelación funcionan con la clave spot")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ErrorExchange as exc:
        print("ERROR del exchange:", exc)
        sys.exit(1)
