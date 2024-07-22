import pkg
import pandas as pd
from datetime import datetime, timedelta
import requests
import json
import time

#Obtener los valores de SL TP


def get_last_take_profit_stop_loss(symbol):
    # Leer el archivo CSV
    df = pd.read_csv('./archivos/indicadores.csv')

    # Filtrar por el símbolo de criptomoneda especificado
    filtered_df = df[df['symbol'] == symbol]

    if not filtered_df.empty:
        # Obtener los últimos valores de Take_Profit y Stop_Loss
        take_profit = filtered_df['Take_Profit'].iloc[-1]
        stop_loss = filtered_df['Stop_Loss'].iloc[-1]

        long_stop_lose = 1 - stop_loss
        long_profit = 1 + take_profit
        short_stop_lose = 1 + stop_loss
        short_profit = 1 - take_profit


        return long_stop_lose, long_profit, short_stop_lose, short_profit
    else:
        # Si no hay datos para ese símbolo, devolver None
        return None, None, None, None



#Funcion Enviar Mensajes
def bot_send_text(bot_message):

    bot_token = pkg.credentials.token
    bot_chatID = pkg.credentials.chatID
    send_text = pkg.credentials.send + bot_message
    response = requests.get(send_text)
 
    return response

def total_monkey():
    # Obtiene el balance actual
    monkey = pkg.bingx.get_balance()
    monkey = json.loads(monkey)

    balance = 0

    try:
        balance = monkey['data']['balance']['balance']
    except KeyError as e:
        print(f"Clave no encontrada: {e}")
    datenow = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    data = {'date': [datenow], 'balance': [balance]}
    df_new = pd.DataFrame(data)

    # Ruta del archivo
    file_path = './archivos/ganancias.csv'
    
    # Añade los nuevos datos al final y se queda con las últimas 10000 filas
    df_old = pd.read_csv(file_path)
    df_total = pd.concat([df_old, df_new])
    df_total = df_total.tail(10000)  # Mantiene solo las últimas 10000 filas

    # Guarda el DataFrame en un archivo csv
    df_total.to_csv(file_path, index=False)
    
    return balance

def resultado_PnL():
    df_data = pd.read_csv('./archivos/PnL.csv')
    npl = pkg.bingx.hystory_PnL()
    npl = json.loads(npl)

    # Verificar si 'data' tiene datos
    if 'data' in npl and npl['data']:
        df = pd.DataFrame(npl['data'])

        # Verificar si la columna 'time' ya está en formato datetime
        if 'time' in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df['time']):
                print("La columna 'time' ya está en formato datetime")
            else:
                df['time'] = pd.to_datetime(df['time'], unit='ms')
        else:
            raise KeyError("La columna 'time' no está presente en los datos obtenidos.")
        
        df_concat = pd.concat([df_data, df])
        df_unique = df_concat.drop_duplicates()
        df_limited = df_unique.tail(10000)
        df_limited.to_csv('./archivos/PnL.csv', index=False)
    else:
        print("No hay datos nuevos para procesar en 'npl'.")



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
    # Obteniendo las posiciones desde la API o fuente de datos
    positions = pkg.bingx.perpetual_swap_positions(symbol)
    
    try:
        # Intenta decodificar el JSON
        positions = json.loads(positions)
    except json.JSONDecodeError:
        # Si hay un error en la decodificación, retorna None para todos los campos
        print("Error decodificando JSON. Posible respuesta vacía o mal formada.")
        return None, None, None, None, None

    # Verifica si 'data' está en la respuesta y que no está vacía
    if 'data' in positions and positions['data']:
        # Extrae los datos necesarios
        symbol = positions['data'][0]['symbol']
        positionSide = positions['data'][0]['positionSide']
        price = float(positions['data'][0]['avgPrice'])
        positionAmt = positions['data'][0]['positionAmt']
        unrealizedProfit = positions['data'][0]['unrealizedProfit']
        return symbol, positionSide, price, positionAmt, unrealizedProfit
    else:
        # Retorna None si 'data' no está presente o está vacía
        return None, None, None, None, None


#Obteniendo Ordenes Pendientes
def obteniendo_ordenes_pendientes():
    try:
        ordenes_raw = pkg.bingx.query_pending_orders()
        ordenes = json.loads(ordenes_raw)
    except json.JSONDecodeError as e:
        print("Error decodificando JSON:", e)
        return []

    data = ordenes.get('data', {})
    orders = data.get('orders', [])

    if not orders:
        df = pd.DataFrame([{'symbol': 'zero', 'orderId': 0, 'side': 'both'}])
    else:
        df = pd.DataFrame(orders)

    try:
        csv_file = './archivos/order_id_register.csv'
        df.to_csv(csv_file, index=False)
    except Exception as e:
        print("Error al guardar en CSV:", e)

    return orders


def colocando_ordenes():
    currencies = pkg.api.currencies_list()
    df = pd.read_csv('./archivos/order_id_register.csv')
    df_positions = pd.read_csv('./archivos/position_id_register.csv')

    for currency in currencies:
        # Verificar si la moneda ya está en el DataFrame
        if currency in df['symbol'].values:
            continue

        contador =  float(len(df))
        max_contador = float(len(currencies))
        
        try:
            price_last, tipo = pkg.indicadores.ema_alert(currency)
            total_money = float(total_monkey())
            trade = total_money / max_contador
            currency_amount = trade / float(price_last)

            if tipo == '=== Alerta de LONG ===':
                # Colocando orden de compra
                pkg.bingx.post_order(currency, currency_amount, price_last, 0, "LONG", "LIMIT", "BUY")
                
                # Guardando las posiciones
                nueva_fila = pd.DataFrame({'symbol': [currency], 'tipo': ['LONG'], 'counter': [0]})
                df_positions = pd.concat([df_positions, nueva_fila], ignore_index=True)

                # Enviando Mensajes
                alert = f'🚨 🤖 🚨 \n *{tipo}* \n 🚧 *{currency}* \n *Precio Actual:* {round(price_last, 3)} \n *Stop Loss* en: {round(price_last * 0.998, 3)} \n *Profit* en: {round(price_last * 1.006, 3)}\n *Trade:* {round(trade, 2)}\n *Contador:* {contador}'
                bot_send_text(alert)

            elif tipo == '=== Alerta de SHORT ===':
                # Colocando orden de venta
                pkg.bingx.post_order(currency, currency_amount, price_last, 0, "SHORT", "LIMIT", "SELL")

                # Guardando las posiciones
                nueva_fila = pd.DataFrame({'symbol': [currency], 'tipo': ['SHORT'], 'counter': [0]})
                df_positions = pd.concat([df_positions, nueva_fila], ignore_index=True)

                # Enviando Mensajes
                alert = f'🚨 🤖 🚨 \n *{tipo}* \n 🚧 *{currency}* \n *Precio Actual:* {round(price_last, 3)} \n *Stop Loss* en: {round(price_last * 1.002, 3)} \n *Profit* en: {round(price_last * 0.994, 3)}\n *Trade:* {round(trade, 2)}\n *Contador:* {contador}'
                bot_send_text(alert)

        except Exception as e:
            pass
            #print(f"Error al procesar {currency}: {e}")

    # Guardando Posiciones fuera del bucle
    df_positions.to_csv('./archivos/position_id_register.csv', index=False)


def colocando_TK_SL():
    #obteniendo posiciones sin SL o TP
    df_posiciones = pd.read_csv('./archivos/position_id_register.csv')
    df_posiciones['counter'] += 1


    #Oteniendo ordenes pendientes
    df_ordenes = pd.read_csv('./archivos/order_id_register.csv')

    for index, row in df_posiciones.iterrows():
        symbol = row['symbol']
        counter = row['counter']

        #aqui necesitamos el codigo de order_id
        if counter >= 30:
            # Filtrar el valor orderId del symbol 
            try:
                orderId = df_ordenes[df_ordenes['symbol'] == symbol]['orderId'].iloc[0]
                pkg.bingx.cancel_order(symbol, orderId)
                df_posiciones.drop(index, inplace=True)
            except:
                pass
        #Obteniendo el valor de las posiciones reales
        try:
            long_stop_lose, long_profit, short_stop_lose, short_profit = get_last_take_profit_stop_loss(symbol)
            symbol, positionSide, price, positionAmt,unrealizedProfit = total_positions(symbol)

            if positionSide == 'LONG':
                # Configurar la orden de stop loss
                pkg.bingx.post_order(symbol, positionAmt, 0,  price * long_stop_lose, "LONG", "STOP_MARKET", "SELL")
                time.sleep(1)
                # Configurar la orden de take profit
                pkg.bingx.post_order(symbol, positionAmt, 0, price * long_profit, "LONG", "TAKE_PROFIT_MARKET", "SELL")

                #Borrando linea
                df_posiciones.drop(index, inplace=True)

            elif positionSide == 'SHORT':
                # Configurar la orden de stop loss
                pkg.bingx.post_order(symbol, positionAmt, 0, price * short_stop_lose, "SHORT", "STOP_MARKET", "BUY")
                time.sleep(1)
                # Configurar la orden de take profit
                pkg.bingx.post_order(symbol, positionAmt, 0, price * short_profit, "SHORT", "TAKE_PROFIT_MARKET", "BUY")

                #Borrando linea
                df_posiciones.drop(index, inplace=True)

        except:
            pass

    #Guardando Posiciones
    df_posiciones.to_csv('./archivos/position_id_register.csv', index=False)


#Cerrando Posiciones antiguas
#Filtrando Posiciones con mas de 30 min
def filtrando_posiciones_antiguas() -> pd.DataFrame:
    try:
        # Cargar los datos
        data = pd.read_csv('./archivos/order_id_register.csv')
        
        # Ajustar por zona horaria sumando 5 horas al tiempo actual
        #Tiempo Server AWS
        current_time = pd.Timestamp.now()  - timedelta(hours=9)
        #Tiempo Mac
        #current_time = pd.Timestamp.now()  + timedelta(hours=5)
        
        # Comprobar si la columna 'symbol' está en el DataFrame
        if 'symbol' not in data.columns:
            raise KeyError("La columna 'symbol' no se encuentra en el DataFrame.")
        
        # Filtro de columnas
        data_filtered = data[['symbol','orderId','type','time', 'stopPrice']].copy()
        data_filtered['time'] = pd.to_datetime(data_filtered['time'], unit='ms')
        
        # Calcular la diferencia de tiempo
        data_filtered['time_difference'] = (current_time - data_filtered['time']).dt.total_seconds() / 60
        
        # Filtrar entradas con más de 2 minutos de diferencia
        data_filtered = data_filtered[(data_filtered['time_difference'] > 1) & (data_filtered['type'] == 'STOP_MARKET')]
        
        # Remover duplicados basado en 'symbol'
        data_filtered = data_filtered.drop_duplicates(subset='symbol')


        return data_filtered

    except FileNotFoundError:
        return pd.DataFrame(columns=['symbol', 'time', 'stopPrice'])  # Retorna un DataFrame con las columnas esperadas pero vacío

    except KeyError as e:
        return pd.DataFrame(columns=['symbol', 'time', 'stopPrice'])  # Retorna un DataFrame con las columnas esperadas pero vacío



#Consultando margen actual
def unrealized_profit_positions():
    #Obteniendo de indicadores los campos SL y last_price
    df_indicadores = pd.read_csv('./archivos/indicadores.csv')

    # Agrupar por 'symbol' y obtener la última fila de cada grupo
    latest_values = df_indicadores.groupby('symbol').last().reset_index()

    # Seleccionar solo las columnas requeridas: 'symbol', 'close', 'Stop_Loss'
    latest_values = latest_values[['symbol', 'close', 'Stop_Loss']]

    # Obteniendo data filtrada de la funcion anterior
    data_filtered = filtrando_posiciones_antiguas()

    # Extraer las columnas 'symbol' en listas
    symbols = data_filtered['symbol'].tolist()
    
    currencies = pkg.api.currencies_list()
    max_contador = int(len(currencies))
    total_money = float(total_monkey())
    trade = (total_money / max_contador)
    
    for symbol in symbols:
        #obteniendo de indicadores el Ultimo Precio y % de Sl
        symbol_data = latest_values[latest_values['symbol'] == symbol]
        precio_actual = symbol_data['close'].iloc[0]
        stop_loss = symbol_data['Stop_Loss'].iloc[0]

        result = total_positions(symbol)
        
        # Verificar si result recibió valores None, lo cual indica que no hay datos
        if result[0] is None:
            print(f"No hay datos de posición para el símbolo: {symbol}")
            continue  # Saltar a la siguiente iteración del bucle

        # Desempaquetar el resultado ya que ahora estamos seguros de que tenemos datos
        symbol_result, positionSide, price, positionAmt, unrealizedProfit = result

        # Obtener el último valor de 'stopPrice' para un símbolo específico
        filtered_data = data_filtered[data_filtered['symbol'] == symbol]
        last_stop_price = filtered_data['stopPrice'].iloc[-1]
        orderId = filtered_data['orderId'].iloc[-1]

        if positionSide == 'LONG':
            # Lógica para posición larga
            potencial_nuevo_sl = precio_actual * (1 - stop_loss )
            if potencial_nuevo_sl > last_stop_price:
                pkg.bingx.cancel_order(symbol, orderId)
                time.sleep(1)
                pkg.bingx.post_order(symbol, positionAmt, 0, potencial_nuevo_sl, "LONG", "STOP_MARKET", "SELL")
        elif positionSide == 'SHORT':
            # Lógica para posición corta
            potencial_nuevo_sl = precio_actual * (1 + stop_loss)
            if potencial_nuevo_sl < last_stop_price:
                pkg.bingx.cancel_order(symbol, orderId)
                time.sleep(1)
                pkg.bingx.post_order(symbol, positionAmt, 0, potencial_nuevo_sl, "SHORT", "STOP_MARKET", "BUY")
                
        
        
            



  
