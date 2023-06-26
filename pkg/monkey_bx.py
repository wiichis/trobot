import pkg
import pandas as pd
from datetime import datetime
import requests
import json

#Funcion Enviar Mensajes
def bot_send_text(bot_message):

    bot_token = pkg.credentials.token
    bot_chatID = pkg.credentials.chatID
    send_text = pkg.credentials.send + bot_message
    response = requests.get(send_text)

    return response

# Calucando El Valor de las inversiones
def total_monkey():
    monkey = pkg.bingx.get_balance()
    monkey = json.loads(monkey)
    balance = monkey['data']['balance']['balance']
    return balance

# Obteniendo las Posiciones
def total_positions(currencie):
    positions = pkg.bingx.perpetual_swap_positions(currencie)
    positions = json.loads(positions)
    symbol = positions['data'][0]['symbol'].split('-')[0]
    symbolPair = positions['data'][0]['symbol']
    positionId = positions['data'][0]['positionId']
    return symbol,symbolPair,positionId

#Obteniendo Ordenes Pendientes
def obteniendo_ordenes_pendientes():
    ordenes = pkg.bingx.query_pending_orders()
    ordenes = json.loads(ordenes)

    orders = ordenes['data']['orders']

    for order in orders:
        print(f"Symbol: {order['symbol']}")
        print(f"Order ID: {order['orderId']}")
        print(f"Side: {order['side']}")
        print(f"Position side: {order['positionSide']}")
        print(f"Type: {order['type']}")
        print(f"Original quantity: {order['origQty']}")
        print(f"Price: {order['price']}")
        print(f"Executed quantity: {order['executedQty']}")
        print(f"Average price: {order['avgPrice']}")
        print(f"Cumulative quote: {order['cumQuote']}")
        print(f"Stop price: {order['stopPrice']}")
        print(f"Profit: {order['profit']}")
        print(f"Commission: {order['commission']}")
        print(f"Status: {order['status']}")
        print(f"Time: {order['time']}")
        print(f"Update time: {order['updateTime']}")
        


#Cerrando Ordenes
def close_orders(currencie):
    symbol,symbolPair,positionId = total_positions(currencie)
    close = pkg.bingx.cancel_order(symbolPair,positionId)

def saving_operations():
    currencies = pkg.api.currencies_list()
    date = datetime.now()
    df = pd.read_csv('./archivos/monkey_register.csv')
    #Limpieza y orden
    df.sort_values('date', inplace=True)
    df.to_csv('./archivos/monkey_register.csv', index=False)

    for currencie in currencies:
        total_money = total_monkey()
        trade = 5
        if total_money > trade:
            try:
                price_last, stop_lose, profit, tipo, envelope_superior, envelope_inferior = pkg.indicadores.ema_alert(currencie)
                currency_amount = trade / price_last
                if tipo == '=== Alerta de LONG ===':
                    pkg.bingx.placeOrder(currencie + "-USDT", 'Bid', price_last, trade,'Market','Open', profit, stop_lose )

    
            except:
                continue
        else:
            continue