import pkg
import pandas as pd
from datetime import datetime, timedelta
import requests
import json
import time

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

def resultado_PnL():
    df_data = pd.read_csv('./archivos/PnL.csv')
    npl = pkg.bingx.hystory_PnL()
    npl = json.loads(npl)
    df = pd.DataFrame(npl['data'])
    
    # Convertir el campo "time" en formato de fecha y hora
    df['time'] = pd.to_datetime(df['time'], unit='ms')

    # Concatenar el DataFrame existente con los nuevos datos
    df_concat = pd.concat([df_data, df])

    # Eliminar duplicados basados en todas las columnas
    df_unique = df_concat.drop_duplicates()

    # Limitar el número de líneas a 10,000 (manteniendo las más recientes)
    df_limited = df_unique.tail(10000)

    # Guardar el DataFrame actualizado en el archivo CSV
    df_limited.to_csv('./archivos/PnL.csv', index=False)


def monkey_result():
    # Obteniendo último resultado
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
    
    # Calcular la fecha de una semana atrás
    fecha_semana_pasada = datetime.now().date() - timedelta(days=7)
    
    # Filtrar los datos para obtener la semana pasada
    df_semana_pasada = df[df['date'].dt.date >= fecha_semana_pasada]
    
    # Obtener el balance al inicio y al final de la semana pasada
    balance_inicial_semana = df_semana_pasada['balance'].iloc[0]
    balance_final_semana = df_semana_pasada['balance'].iloc[-1]
    
    # Calcular la diferencia de la semana pasada
    diferencia_semana = balance_final_semana - balance_inicial_semana
    
    return balance_actual, diferencia_hora, diferencia_dia, diferencia_semana


# Obteniendo las Posiciones
def total_positions(symbol):
    positions = pkg.bingx.perpetual_swap_positions(symbol)
    positions = json.loads(positions)
    #Capturando el error en caso de que no haya nada
    if not positions['data']:
        return None, None, None, None

    symbol = positions['data'][0]['symbol']
    positionSide = positions['data'][0]['positionSide']
    price = float(positions['data'][0]['avgPrice'])
    positionAmt = positions['data'][0]['positionAmt']
    return symbol, positionSide, price, positionAmt


#Obteniendo Ordenes Pendientes
def obteniendo_ordenes_pendientes():
    ordenes = pkg.bingx.query_pending_orders()
    ordenes = json.loads(ordenes)
    orders = ordenes['data']['orders']

    #Crear un DataFrame con los datos
    df = pd.DataFrame(orders)

    # Verificar si la columna 'stopPrice' está presente en el DataFrame
    if 'stopPrice' not in df.columns:
        df = pd.DataFrame({
            'Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'Currencie': ['CERO'],
            'OrderID Stop Loss': [''],
            'OrderID Take Profit': [''],
            'Stop Lose Value': [100],
            'Profit': [100]
        })

        # Guardar los datos en un archivo CSV
        csv_file = './archivos/order_id_register.csv'
        df.to_csv(csv_file, index=False)
        return      
    
    # Agregar las columnas adicionales
    df['Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df['OrderID Stop Loss'] = "0"
    df['OrderID Take Profit'] = "0"
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
    df_positions = pd.read_csv('./archivos/position_id_register.csv')

    for currencie in currencies:
        # Verificar si la moneda ya está en el DataFrame
        if currencie in df['Currencie'].values:
            continue

        contador = len(df)
        contador = int(contador)
        max_contador = len(currencies)
        max_contador = int(max_contador)
        
        try:
            price_last, tipo = pkg.indicadores.ema_alert(currencie)
            if tipo == '=== Alerta de LONG ===':
                total_money = float(total_monkey())
                trade = total_money / max_contador
                currency_amount = trade / price_last
                #Colocando orden de compra
                pkg.bingx.post_order(currencie,currency_amount,price_last,0,"LONG", "LIMIT", "BUY")
                
                #Guardando las posiciones
                df_posiciones = {'symbol': currencie, 'tipo': 'LONG'}
                print(df_posiciones)
                df_positions.loc[len(df_positions)] = df_posiciones

                #Enviando Mensajes
                alert = f' 🚨 🤖 🚨 \n *{tipo}* \n 🚧 *{currencie}* \n *Precio Actual:* {round(price_last,3)} \n *Stop Loss* en: {round(price_last * 0.998,3)} \n *Profit* en: {round(price_last * 1.006,3)}\n *Trade: * {round(trade,2)}\n *Contador* {contador}'
                bot_send_text(alert)

            elif tipo == '=== Alerta de SHORT ===':
                total_money = float(total_monkey())
                trade = total_money / max_contador
                currency_amount = trade / price_last
                # Colocando orden de venta
                response = pkg.bingx.post_order(currencie,currency_amount,price_last,price_last,"SHORT","LIMIT", "SELL")
                print(response)

                #Guardando las posiciones
                df_posiciones = pd.DataFrame({'symbol': currencie, 'tipo': 'SHORT'})
                df_positions = pd.concat([df_positions, df_posiciones], ignore_index=True)

                # Enviando Mensajes
                alert = f' 🚨 🤖 🚨 \n *{tipo}* \n 🚧 *{currencie}* \n *Precio Actual:* {round(price_last,3)} \n *Stop Loss* en: {round(price_last * 1.002,3)} \n *Profit* en: {round(0.994,3)}\n *Trade: * {round(trade,2)}\n *Contador* {contador}'
                bot_send_text(alert)
            
            #Guardando Posiciones
            df_positions.to_csv('./archivos/position_id_register.csv', index=False)
        except:
            continue
        else:
            continue

def prueba():
    long_stop_lose = 0.9983
    long_profit = 1.005

    currencies = pkg.api.currencies_list()
    for currencie in currencies:
        #configurara SL TK Trayendo las posisciones
        symbol,price,positionAmt =  total_positions(currencie)
        if symbol is None:
            continue

        # Convertir price a float antes de la multiplicación
        price = float(price)

        # Configurar la orden de stop loss
        pkg.bingx.post_order(symbol, "STOP_MARKET", "SELL", "BOTH", 0, positionAmt, price * long_stop_lose)

        # Configurar la orden de take profit
        pkg.bingx.post_order(symbol, "TAKE_PROFIT_MARKET", "SELL", "BOTH", 0, positionAmt, price * long_profit)


def prueba_short():
    #compra = pkg.bingx.post_order('TRX-USDT', "MARKET", "BUY", "LONG", 0, 50, 0)
    response = pkg.bingx.post_order("BTC-USDT", 1, 29320, 29320, "SHORT", "LIMIT", "SELL")

    print("prueba", response)


def colocando_TK_SL():
    #Configuracion SL TP
    long_stop_lose = 0.998
    long_profit = 1.006
    short_stop_lose = 1.002
    short_profit = 0.994

    #obteniendo posiciones sin SL o TP
    df_posiciones = pd.read_csv('./archivos/position_id_register.csv')
    
    for index, row in df_posiciones.iterrows():
        symbol = row['symbol']
 
        #Obteniendo el valor de las posiciones reales
        symbol, positionSide, price, positionAmt = total_positions(symbol)

        if positionSide == 'LONG':
            # Configurar la orden de stop loss
            pkg.bingx.post_order(symbol, "STOP_MARKET", "SELL", "BOTH", 0, positionAmt, price * long_stop_lose)
            time.sleep(1)
            # Configurar la orden de take profit
            pkg.bingx.post_order(symbol, "TAKE_PROFIT_MARKET", "SELL", "BOTH", 0, positionAmt, price * long_profit)

            #Borrando linea
            df_posiciones.drop(index, inplace=True)

        elif positionSide == 'SHORT':
            # Configurar la orden de stop loss
            pkg.bingx.post_order(symbol, "STOP_MARKET", "BUY", "BOTH", 0, positionAmt, price * short_stop_lose)
            time.sleep(1)
            # Configurar la orden de take profit
            pkg.bingx.post_order(symbol, "TAKE_PROFIT_MARKET", "BUY", "BOTH", 0, positionAmt, price * short_profit)

            #Borrando linea
            df_posiciones.drop(index, inplace=True)

    #Guardando Posiciones
    df_posiciones.to_csv('./archivos/position_id_register.csv', index=False)





  