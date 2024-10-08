import pkg
import time
import requests
import hmac
from hashlib import sha256


APIURL = "https://open-api.bingx.com"
APIKEY = pkg.credentials.APIKEY
SECRETKEY = pkg.credentials.SECRETKEY

#Generar Firma
def get_sign(api_secret, payload):
    signature = hmac.new(api_secret.encode("utf-8"), payload.encode("utf-8"), digestmod=sha256).hexdigest()
    #print("sign=" + signature)
    return signature


#Enviar Requerimiento
def send_request(methed, path, urlpa, payload):
    url = "%s%s?%s&signature=%s" % (APIURL, path, urlpa, get_sign(SECRETKEY, urlpa))
    #print(url)

    headers = {
        'X-BX-APIKEY': APIKEY,
    }
    response = requests.request(methed, url, headers=headers, data=payload)
    return response.text


#Definir Parametros
def praseParam(paramsMap):
    sortedKeys = sorted(paramsMap)
    paramsStr = "&".join(["%s=%s" % (x, paramsMap[x]) for x in sortedKeys])
    return paramsStr


#Cancelar una orden
def cancel_order(symbol, order_Id):
    payload = {}
    path = '/openApi/swap/v2/trade/order'
    methed = "DELETE"
    paramsMap = {
        "orderId": order_Id,
        "timestamp": int(time.time() * 1000),
        "symbol": symbol,
    }
    paramsStr = praseParam(paramsMap)
    return send_request(methed, path, paramsStr, payload)


#Cancelar todas las ordenes
def cancel_all_orders(symbol):
    payload = {}
    path = '/openApi/swap/v2/trade/allOpenOrders'
    methed = "DELETE"
    paramsMap = {
        "timestamp": int(time.time() * 1000),
        "symbol": symbol,
    }
    paramsStr = praseParam(paramsMap)
    return send_request(methed, path, paramsStr, payload)

#Cerrar todas las posiciones
def one_clickLclose_all_positions():
    payload = {}
    path = '/openApi/swap/v2/trade/closeAllPositions'
    methed = "POST"
    paramsMap = {
        "timestamp": int(time.time() * 1000),
    }
    paramsStr = praseParam(paramsMap)
    return send_request(methed, path, paramsStr, payload)


#Obtner el balance 
def get_balance():
    payload = {}
    path = '/openApi/swap/v2/user/balance'
    methed = "GET"
    paramsMap = {
        "timestamp": int(time.time() * 1000),
    }
    paramsStr = praseParam(paramsMap)
    return send_request(methed, path, paramsStr, payload)

# Colocar una orden
def post_order(symbol, quantity, price, stopPrice, position_side, type, side):
    payload = {}
    path = '/openApi/swap/v2/trade/order'
    methed = "POST"
    paramsMap = {
        "symbol": symbol,
        "quantity": quantity,
        "price": price,
        "stopPrice": stopPrice,
        "positionSide": position_side,
        "type": type,
        "side": side,
        "timestamp": int(time.time() * 1000)
    }
    paramsStr = praseParam(paramsMap)
    return send_request(methed, path, paramsStr, payload)


#Obtener todas las prosiciones
def perpetual_swap_positions(symbol):
    payload = {}
    path = '/openApi/swap/v2/user/positions'
    methed = "GET"
    paramsMap = {
        "symbol": symbol,
        "timestamp": int(time.time() * 1000),
    }
    paramsStr = praseParam(paramsMap)
    return send_request(methed, path, paramsStr, payload)


#Consultar Ordenes Pendientes
def query_pending_orders():
    payload = {}
    path = '/openApi/swap/v2/trade/openOrders'
    methed = "GET"
    paramsMap = {
        #"symbol": symbol,
        "timestamp": int(time.time() * 1000),
    }
    paramsStr = praseParam(paramsMap)
    return send_request(methed, path, paramsStr, payload)


#Obtener el prcio de un PAR
def last_price_trading_par(symbol):
    payload = {}
    path = '/openApi/swap/v2/quote/price'
    methed = "GET"
    paramsMap = {
        "symbol": symbol
    }
    paramsStr = praseParam(paramsMap)
    return send_request(methed, path, paramsStr, payload)



#Obtener el historico de PnL
def hystory_PnL():
    payload = {}
    path = '/openApi/swap/v2/user/income'
    methed = "GET"
    paramsMap = {
        "symbol": "",
        "incomeType": "",
        "startTime": 0,
        "endTime": 0,
        "limit": 0,
        "timestamp": int(time.time() * 1000),
        "recvWindow": 0
    }
    paramsStr = praseParam(paramsMap)
    return send_request(methed, path, paramsStr, payload)


# Obtener el valor de múltiples velas
def get_candle(symbol, interval, limit=2):
    payload = {}
    path = '/openApi/swap/v3/quote/klines'
    method = "GET"
    paramsMap = {
        "symbol": symbol,
        "interval": interval,
        "limit": str(limit),  # Número de velas a obtener
        "timestamp": str(int(time.time() * 1000)) 
    }
    paramsStr = praseParam(paramsMap)
    return send_request(method, path, paramsStr, payload)



