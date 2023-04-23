import pkg
import time
import requests
import hmac
from hashlib import sha256


APIURL = "https://open-api.bingx.com"
APIKEY = pkg.credentials.APIKEY
SECRETKEY = pkg.credentials.SECRETKEY


def get_sign(api_secret, payload):
    signature = hmac.new(api_secret.encode("utf-8"), payload.encode("utf-8"), digestmod=sha256).hexdigest()
    print("sign=" + signature)
    return signature


def send_request(methed, path, urlpa, payload):
    url = "%s%s?%s&signature=%s" % (APIURL, path, urlpa, get_sign(SECRETKEY, urlpa))
    print(url)

    headers = {
        'X-BX-APIKEY': APIKEY,
    }
    response = requests.request(methed, url, headers=headers, data=payload)
    return response.text


def praseParam(paramsMap):
    sortedKeys = sorted(paramsMap)
    paramsStr = "&".join(["%s=%s" % (x, paramsMap[x]) for x in sortedKeys])
    return paramsStr


def cancel_order(symbol, positionId):
    payload = {}
    path = '/openApi/swap/v2/trade/order'
    methed = "DELETE"
    paramsMap = {
        "orderId": positionId,
        "timestamp": int(time.time() * 1000),
        "symbol": symbol,
    }
    paramsStr = praseParam(paramsMap)
    return send_request(methed, path, paramsStr, payload)


def one_clickLclose_all_positions():
    payload = {}
    path = '/openApi/swap/v2/trade/closeAllPositions'
    methed = "POST"
    paramsMap = {
        "timestamp": int(time.time() * 1000),
    }
    paramsStr = praseParam(paramsMap)
    return send_request(methed, path, paramsStr, payload)


def get_balance():
    payload = {}
    path = '/openApi/swap/v2/user/balance'
    methed = "GET"
    paramsMap = {
        "timestamp": int(time.time() * 1000),
    }
    paramsStr = praseParam(paramsMap)
    return send_request(methed, path, paramsStr, payload)


def post_market_order():
    payload = {}
    path = '/openApi/swap/v2/trade/order'
    methed = "POST"
    paramsMap = {
        "side": "BUY",
        "positionSide": "LONG",
        "quantity": 5,
        "symbol": "LINK-USDT",
        "type": "MARKET",
        "timestamp": int(time.time() * 1000),
    }
    paramsStr = praseParam(paramsMap)
    return send_request(methed, path, paramsStr, payload)


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

def last_price_trading_par(symbol):
    payload = {}
    path = '/openApi/swap/v2/quote/price'
    methed = "GET"
    paramsMap = {
        "symbol": symbol
    }
    paramsStr = praseParam(paramsMap)
    return send_request(methed, path, paramsStr, payload)