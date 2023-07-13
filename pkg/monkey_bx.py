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
        datenow = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        data = {'datenow': [datenow], 'balance': [balance]}
        df = pd.DataFrame(data)

        with open('./archivos/ganancias.csv', 'a', newline='') as archivo_csv:
            df.to_csv(archivo_csv, header=archivo_csv.tell() == 0, index=False)
        
        return balance


def monkey_result():
    #Obteniendo ultimo resultado
    balance_actual = float(total_monkey())
    # Cargar los datos desde el archivo CSV en un DataFrame de pandas
    df = pd.read_csv('./archivos/ganancias.csv')

   # Convertir la columna 'date' en formato datetime
    df['date'] = pd.to_datetime(df['date'])

    # Filtrar los datos para obtener solo el día actual y la hora actual
    fecha_actual = datetime.now().date()
    hora_actual = datetime.now().hour
    df_dia_actual = df[df['date'].dt.date == fecha_actual]
    df_hora_actual = df_dia_actual[df_dia_actual['date'].dt.hour == hora_actual]

    # Obtener el balance al inicio y al final del día actual
    balance_inicial_dia = df_dia_actual['balance'].iloc[0]
    balance_final_dia = df_dia_actual['balance'].iloc[-1]

    # Obtener el balance al inicio y al final de la hora actual
    balance_inicial_hora = df_hora_actual['balance'].iloc[0]
    balance_final_hora = df_hora_actual['balance'].iloc[-1]

    # Calcular la diferencia del día actual y la diferencia de la hora actual
    diferencia_dia = balance_final_dia - balance_inicial_dia
    diferencia_hora = balance_final_hora - balance_inicial_hora

    return balance_actual, diferencia_hora, diferencia_dia


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

    #Crear un DataFrame con los datos
    df = pd.DataFrame(orders)

    # Verificar si la columna 'stopPrice' está presente en el DataFrame
    if 'stopPrice' not in df.columns:
        return
    
    # Agregar las columnas adicionales

    df['Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df['OrderID Stop Loss'] = ""
    df['OrderID Take Profit'] = ""
    df['Stop Lose Value'] = df['stopPrice']
    df['Profit'] = df['profit']

    # Recorrer cada fila y asignar los valores adecuados
    for i, row in df.iterrows():
        if row['type'] == 'STOP_MARKET':
            df.at[i, 'OrderID Stop Loss'] = row['orderId']
            df.at[i, 'Stop Lose Value'] = row['stopPrice']
        elif row['type'] == 'TAKE_PROFIT_MARKET':
            df.at[i, 'OrderID Take Profit'] = row['orderId']
        df.at[i, 'Currencie'] = row['symbol']

    # Seleccionar las columnas deseadas
    columns = ['Date', 'Currencie', 'OrderID Stop Loss', 'OrderID Take Profit', 'Stop Lose Value', 'Profit']
    df = df[columns]

    # Guardar los datos en un archivo CSV
    csv_file = './archivos/order_id_register.csv'
    df.to_csv(csv_file, index=False)

    return orders       


def colocando_ordenes():
    currencies = pkg.api.currencies_list()
    df = pd.read_csv('./archivos/order_id_register.csv')

    for currencie in currencies:
        # Verificar si la moneda ya está en el DataFrame
        if currencie in df['Currencie'].values:
            continue

        contador = len(df)
        contador = int(contador)
        max_contador = len(currencies)
        max_contador = int(max_contador)
        
        try:
            price_last, stop_lose, profit, tipo = pkg.indicadores.ema_alert(currencie)
            if tipo == '=== Alerta de LONG ===':
                total_money = float(total_monkey())
                trade = total_money / max_contador
                currency_amount = trade / price_last
                #Colocando orden de compra
                response = json.loads(pkg.bingx.post_order(currencie, "MARKET", "BUY", "BOTH", price_last, currency_amount, 0))
                if 'orderId' in response['data']['order']:
                    #Enviando Mensajes
                    alert = f' 🚨 🤖 🚨 \n *{tipo}* \n 🚧 *{currencie}* \n *Precio Actual:* {round(price_last,3)} \n *Stop Loss* en: {round(stop_lose,3)} \n *Profit* en: {round(profit,3)}\n *Trade: * {round(trade,2)}\n *Contador* {contador}'
                    bot_send_text(alert)
                    
                    # Configurar la orden de stop loss
                    pkg.bingx.post_order(currencie, "STOP_MARKET", "SELL", "BOTH", 0, currency_amount, stop_lose)

                    # Configurar la orden de take profit
                    pkg.bingx.post_order(currencie, "TAKE_PROFIT_MARKET", "SELL", "BOTH", 0, currency_amount, profit)
                else:
                    print("Error al realizar la orden de compra:", response['msg']) 

            elif tipo == '=== Alerta de SHORT ===':
                total_money = float(total_monkey())
                # Colocando orden de venta
                response = json.loads(pkg.bingx.post_order(currencie, "MARKET", "SELL", "BOTH", price_last, currency_amount, 0))
                if 'orderId' in response['data']['order']:
                    # Enviando Mensajes
                    alert = f' 🚨 🤖 🚨 \n *{tipo}* \n 🚧 *{currencie}* \n *Precio Actual:* {round(price_last,3)} \n *Stop Loss* en: {round(stop_lose,3)} \n *Profit* en: {round(profit,3)}\n *Trade: * {round(trade,2)}\n *Contador* {contador}'
                    bot_send_text(alert)

                    # Configurar la orden de stop loss
                    pkg.bingx.post_order(currencie, "STOP_MARKET", "BUY", "BOTH", 0, currency_amount, stop_lose)

                    # Configurar la orden de take profit
                    pkg.bingx.post_order(currencie, "TAKE_PROFIT_MARKET", "BUY", "BOTH", 0, currency_amount, profit)
                else:
                    print("Error al realizar la orden de venta:", response['msg'])  

        except:
            continue
        else:
            continue


def prueba_short():
    currencie = "BTC-USDT"
    price_last = 30579
    currency_amount = 0.00015
    stop_lose = 30650
    profit = 30520
    tipo = "Alerta de SHORT"


    response = json.loads(pkg.bingx.post_order(currencie, "MARKET", "SELL", "BOTH", 0, currency_amount, 0))
    print(response)
    if 'orderId' in response['data']['order']:
        # Enviando Mensajes
        alert = f' 🚨 🤖 🚨 \n *{tipo}* \n 🚧 *{currencie}* \n *Precio Actual:* {round(price_last,3)} \n *Stop Loss* en: {round(stop_lose,3)} \n *Profit* en: {round(profit,3)}\n'
        bot_send_text(alert)

        # Configurar la orden de stop loss
        pkg.bingx.post_order(currencie, "STOP_MARKET", "BUY", "BOTH", 0, currency_amount, stop_lose)

        # Configurar la orden de take profit
        pkg.bingx.post_order(currencie, "TAKE_PROFIT_MARKET", "BUY", "BOTH", 0, currency_amount, profit)
    else:
        print("Error al realizar la orden de venta:", response['msg'])  



def cerrando_ordenes():
    # Leer los datos desde un archivo CSV
    df = pd.read_csv('./archivos/order_id_register.csv')

    # Contar los valores y filtrar los que tienen una aparición
    value_counts = df['Currencie'].value_counts()
    filtered_counts = value_counts[value_counts == 1]

    # Evaluar los valores filtrados que tienen 'OrderID Stop Loss' vacío
    for currencie in filtered_counts.index:
        filtered_df = df[df['Currencie'] == currencie]
        if filtered_df['OrderID Stop Loss'].isnull().all():
            order_id_take_profit = int(filtered_df['OrderID Take Profit'].iloc[0]) # Obtener el valor del primer registro
            #Enviadndo Mensajes
            alert = f' 💸 🤖 💸 \n *Perdida SHORT* \n 🚧' + currencie
            bot_send_text(alert)
            #Canelando orden pendiente
            pkg.bingx.cancel_order(currencie, order_id_take_profit)
        else:
            order_id_stop_loss = int(filtered_df['OrderID Stop Loss'].iloc[0])  # Obtener el valor del primer registro
            #Enviadndo Mensajes
            alert = f' 💵 🤖 💵 \n *Ganancia SHORT* \n 🚧' + currencie
            bot_send_text(alert)
            #Cancelando orden pendiente
            pkg.bingx.cancel_order(currencie, order_id_stop_loss)

