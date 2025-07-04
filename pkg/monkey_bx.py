import pkg
import pandas as pd
from datetime import datetime, timedelta
import requests
import json
import time
import os

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

        return take_profit, stop_loss
    else:
        # Si no hay datos para ese símbolo, devolver None
        return None, None


#Funcion Enviar Mensajes
def bot_send_text(bot_message):

    bot_token = pkg.credentials.token
    bot_chatID = pkg.credentials.chatID
    send_text = pkg.credentials.send + bot_message
    response = requests.get(send_text)
 
    return response

def total_monkey():
    """
    Devuelve el balance de la cuenta.
    Maneja respuestas vacías o JSON mal formado para evitar que el bot se caiga.
    """
    raw = pkg.bingx.get_balance()

    # ── Validaciones defensivas ──────────────────────────────────────────
    if not raw or not raw.strip():
        print("Error: get_balance() devolvió cadena vacía.")
        return 0.0

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"Error decodificando JSON: {e} → {raw[:120]!r}")
        return 0.0
    # ─────────────────────────────────────────────────────────────────────

    balance = 0.0
    try:
        balance = float(data['data']['balance']['balance'])
    except (KeyError, ValueError, TypeError) as e:
        print(f"Error al obtener el balance: {e}")
        balance = 0.0  # Valor predeterminado en caso de error

    # Registrar histórico de balances
    datenow = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df_new = pd.DataFrame({'date': [datenow], 'balance': [balance]})

    file_path = './archivos/ganancias.csv'
    try:
        df_old = pd.read_csv(file_path)
        df_total = pd.concat([df_old, df_new]).tail(10000)
    except FileNotFoundError:
        df_total = df_new

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
    pkg.monkey_bx.obteniendo_ordenes_pendientes()
    currencies = pkg.api.currencies_list()
    df_orders = pd.read_csv('./archivos/order_id_register.csv')
    df_positions = pd.read_csv('./archivos/position_id_register.csv')

    # Cargar los pesos actualizados
    PESOS_ACTUALIZADOS_PATH = './archivos/pesos_actualizados.csv'
    if os.path.exists(PESOS_ACTUALIZADOS_PATH):
        df_pesos = pd.read_csv(PESOS_ACTUALIZADOS_PATH)
    else:
        # Si no hay pesos actualizados, asignar pesos iguales
        df_pesos = pd.DataFrame({
            'symbol': currencies,
            'peso_actualizado': [1.0 / len(currencies)] * len(currencies)
        })

    # Aplicar límite máximo al peso individual (40% en normalización al 100%)
    LIMITE_PESO_INDIVIDUAL = 0.40

    # Aplicar límite máximo al peso en el DataFrame de pesos
    df_pesos['peso_ajustado'] = df_pesos['peso_actualizado'].clip(
        upper=LIMITE_PESO_INDIVIDUAL
    )

    # Normalizar los pesos ajustados al 100%
    suma_pesos_ajustados = df_pesos['peso_ajustado'].sum()
    df_pesos['peso_normalizado'] = df_pesos['peso_ajustado'] / suma_pesos_ajustados

    # Normalizar los pesos al 200%
    df_pesos['peso_normalizado'] *= 2

    # Aplicar límite máximo al peso individual al 200% (80%)
    LIMITE_PESO_INDIVIDUAL_200 = LIMITE_PESO_INDIVIDUAL * 2
    df_pesos['peso_normalizado'] = df_pesos['peso_normalizado'].clip(
        upper=LIMITE_PESO_INDIVIDUAL_200
    )

    # Obtener el total de fondos disponibles
    total_money = float(pkg.monkey_bx.total_monkey())
    capital_disponible = total_money

    # Lista para almacenar las monedas con señales activas
    active_currencies = []

    for currency in currencies:
        # Verificar si la moneda ya está en el DataFrame de órdenes o de posiciones
        if currency in df_orders['symbol'].values or currency in df_positions['symbol'].values:
            continue

        try:
            # Obtener el peso normalizado de la moneda
            peso = df_pesos.loc[df_pesos['symbol'] == currency, 'peso_normalizado'].values
            if len(peso) == 0:
                peso = (1.0 / len(currencies)) * 2  # Peso por defecto normalizado al 200%
                peso = min(peso, LIMITE_PESO_INDIVIDUAL_200)
            else:
                peso = peso[0]

            # Obtener precio y tipo de alerta
            price_last, tipo = pkg.indicadores.ema_alert(currency)

            # Verificar si se obtuvo una alerta válida (nueva condición)
            if "Alerta de LONG" not in str(tipo) and "Alerta de SHORT" not in str(tipo):
                continue  # No hay alerta para esta moneda

            # Asegurar que price_last es numérico
            try:
                price_last = float(price_last)
            except (TypeError, ValueError):
                continue

            # Añadir a la lista de monedas activas
            active_currencies.append({
                'symbol': currency,
                'tipo': tipo,
                'price_last': price_last,
                'peso': peso
            })

        except Exception as e:
            pass  # Manejo de excepciones

    # Si no hay monedas activas, terminar la función
    if not active_currencies:
        # Silenciado: print("No hay señales activas en este momento.")
        return

    # Calcular la suma de los pesos de las monedas activas
    suma_pesos_activos = sum(item['peso'] for item in active_currencies)

    # Reajustar los pesos para que la suma de los pesos activos no exceda 200%
    factor_ajuste = min(2.0, suma_pesos_activos) / suma_pesos_activos
    for item in active_currencies:
        item['peso_ajustado'] = item['peso'] * factor_ajuste

    # Calcular el capital asignado a cada moneda y verificar que no exceda el capital disponible
    total_capital_asignado = 0
    for item in active_currencies:
        peso = item['peso_ajustado']
        trade = total_money * peso
        # Verificar si el capital asignado excede el capital disponible
        if total_capital_asignado + trade > capital_disponible:
            # Ajustar el trade para no exceder el capital disponible
            trade = capital_disponible - total_capital_asignado
            # Si no queda capital disponible, no colocar la orden
            if trade <= 0:
                print(f"No hay capital disponible para {item['symbol']}.")
                continue
        total_capital_asignado += trade
        item['trade'] = trade

    # Colocar las órdenes
    for item in active_currencies:
        if 'trade' not in item:
            continue  # Si no se asignó capital, pasar a la siguiente moneda

        currency = item['symbol']
        tipo = item['tipo']
        price_last = item['price_last']
        trade = item['trade']
        peso = item['peso_ajustado']

        currency_amount = trade / price_last

        # Definir factores de stop loss y profit
        if "LONG" in str(tipo):
            stop_loss_factor = 0.998
            profit_factor = 1.006
            order_side = "BUY"
            position_side = "LONG"
        elif "SHORT" in str(tipo):
            stop_loss_factor = 1.002
            profit_factor = 0.994
            order_side = "SELL"
            position_side = "SHORT"
        else:
            continue  # Si no es ninguno, no colocar la orden

        # Colocando la orden
        pkg.bingx.post_order(
            currency, currency_amount, 0, 0, position_side, "MARKET", order_side
        )

        # Guardando las posiciones
        nueva_fila = pd.DataFrame({
            'symbol': [currency],
            'tipo': [position_side],
            'counter': [0]
        })
        df_positions = pd.concat([df_positions, nueva_fila], ignore_index=True)

        # Cálculo porcentajes para mostrar variación TP/SL
        if "LONG" in str(tipo):
            profit_pct = (profit_factor - 1) * 100         # Ej: +0.60 %
            sl_pct = (stop_loss_factor - 1) * 100          # Ej: -0.20 %
        else:  # SHORT
            profit_pct = (1 - profit_factor) * 100         # Negativo
            sl_pct = (1 - stop_loss_factor) * 100

        # ---------- NUEVO FORMATO DE ALERTA MARKDOWN ----------
        alert = (
            "💎 *TRADE ALERT* 💎\n"
            f"`{currency}` | {'🟢 LONG' if 'LONG' in tipo else '🔴 SHORT'}\n\n"
            f"*Entrada:* `{round(price_last, 4)}`\n"
            f"*TP:* `{round(price_last * profit_factor, 4)}` (+{profit_pct:.2f}%)\n"
            f"*SL:* `{round(price_last * stop_loss_factor, 4)}` ({sl_pct:.2f}%)\n\n"
            f"💰 *Capital:* `${round(trade, 2)}` — *{peso*100:.2f}%*"
        )
        # ------------------------------------------------------
        pkg.monkey_bx.bot_send_text(alert)

    # Guardando Posiciones fuera del bucle
    df_positions.to_csv('./archivos/position_id_register.csv', index=False)

def colocando_TK_SL():
    # Obteniendo posiciones sin SL o TP
    df_posiciones = pd.read_csv('./archivos/position_id_register.csv')
    # Asegurarse de que exista la columna 'counter'
    if 'counter' not in df_posiciones.columns:
        df_posiciones['counter'] = 0
    if not df_posiciones.empty:
        df_posiciones['counter'] += 1

    # Obteniendo órdenes pendientes
    df_ordenes = pd.read_csv('./archivos/order_id_register.csv')
    # --- Normalizar columnas y evitar KeyError ---------------------------
    # Asegurar que exista la columna 'type'.
    # Algunos registros de BingX devuelven 'orderType' o ninguna de las dos.
    if 'type' not in df_ordenes.columns:
        if 'orderType' in df_ordenes.columns:
            df_ordenes = df_ordenes.rename(columns={'orderType': 'type'})
        else:
            # Crear columna vacía para evitar KeyError posteriores
            df_ordenes['type'] = ''
    # --------------------------------------------------------------------

    # Leer los últimos valores de indicadores
    df_indicadores = pd.read_csv('./archivos/indicadores.csv')
    latest_values = df_indicadores.groupby('symbol').last().reset_index()

    # Agrega un pequeño delay antes de buscar la posición en BingX para dar tiempo a que la posición MARKET aparezca
    time.sleep(2)
    for index, row in df_posiciones.iterrows():
        symbol = row['symbol']
        counter = row['counter']

        # Verificar si ya existen órdenes SL y TP pendientes para este símbolo
        symbol_orders = df_ordenes[df_ordenes['symbol'] == symbol]
        sl_exists = not symbol_orders[symbol_orders['type'] == 'STOP_MARKET'].empty
        tp_exists = not symbol_orders[symbol_orders['type'] == 'TAKE_PROFIT_MARKET'].empty

        # Si ambos existen, ya está protegido: eliminamos la entrada y seguimos
        if sl_exists and tp_exists:
            df_posiciones.drop(index, inplace=True)
            continue

        # Verificar si se debe cancelar la orden después de cierto tiempo
        if counter >= 20:
            try:
                orderId = df_ordenes[df_ordenes['symbol'] == symbol]['orderId'].iloc[0]
                pkg.bingx.cancel_order(symbol, orderId)
                df_posiciones.drop(index, inplace=True)
                df_posiciones.to_csv('./archivos/position_id_register.csv', index=False)  # Guarda inmediatamente
                print(f"Orden cancelada por timeout para {symbol}.")
                pkg.monkey_bx.bot_send_text(
                    f"❌ Orden cancelada por timeout para {symbol}. No se ejecutó en el tiempo límite."
                )
                continue  # Evita múltiples mensajes y cancelaciones para la misma orden
            except Exception as e:
                print(f"Error al cancelar la orden para {symbol}: {e}")
                pass

        # Obteniendo el valor de las posiciones reales
        try:
            # Obtener los detalles de la posición actual
            result = total_positions(symbol)
            if result[0] is None:
                print(f"Posición para {symbol} aún no aparece en el exchange. Reintentando en el próximo ciclo.")
                continue

            symbol_result, positionSide, price, positionAmt, unrealizedProfit = result

            # Obtener los niveles de Take Profit y Stop Loss según el lado de la posición
            symbol_data = latest_values[latest_values['symbol'] == symbol]

            if positionSide == 'LONG':
                take_profit = symbol_data['Take_Profit_Long'].iloc[0]
                stop_loss = symbol_data['Stop_Loss_Long'].iloc[0]
                # Comprobar qué órdenes faltan
                exito_sl = sl_exists
                exito_tp = tp_exists

                if not sl_exists:
                    try:
                        pkg.bingx.post_order(symbol, positionAmt, 0, stop_loss, "LONG", "STOP_MARKET", "SELL")
                        exito_sl = True
                        time.sleep(1)
                    except Exception as e:
                        print(f"Error colocando SL para {symbol}: {e}")

                if not tp_exists:
                    try:
                        pkg.bingx.post_order(symbol, positionAmt, 0, take_profit, "LONG", "TAKE_PROFIT_MARKET", "SELL")
                        exito_tp = True
                    except Exception as e:
                        print(f"Error colocando TP para {symbol}: {e}")

                # Si ambos existen ahora, eliminamos la marca para este símbolo
                if exito_sl and exito_tp:
                    df_posiciones.drop(index, inplace=True)

            elif positionSide == 'SHORT':
                take_profit = symbol_data['Take_Profit_Short'].iloc[0]
                stop_loss = symbol_data['Stop_Loss_Short'].iloc[0]
                # Comprobar qué órdenes faltan
                exito_sl = sl_exists
                exito_tp = tp_exists

                if not sl_exists:
                    try:
                        pkg.bingx.post_order(symbol, positionAmt, 0, stop_loss, "SHORT", "STOP_MARKET", "BUY")
                        exito_sl = True
                        time.sleep(1)
                    except Exception as e:
                        print(f"Error colocando SL para {symbol}: {e}")

                if not tp_exists:
                    try:
                        pkg.bingx.post_order(symbol, positionAmt, 0, take_profit, "SHORT", "TAKE_PROFIT_MARKET", "BUY")
                        exito_tp = True
                    except Exception as e:
                        print(f"Error colocando TP para {symbol}: {e}")

                # Si ambos existen ahora, eliminamos la marca para este símbolo
                if exito_sl and exito_tp:
                    df_posiciones.drop(index, inplace=True)

        except Exception as e:
            print(f"Error al configurar SL/TP para {symbol}: {e}")
            pass

    # Guardando Posiciones
    df_posiciones.to_csv('./archivos/position_id_register.csv', index=False)


#Cerrando Posiciones antiguas
def filtrando_posiciones_antiguas() -> pd.DataFrame:
    try:
        # Cargar los datos
        data = pd.read_csv('./archivos/order_id_register.csv')
        
        # Ajustar por zona horaria sumando 5 horas al tiempo actual
        # Tiempo Server AWS
        current_time = pd.Timestamp.now() - timedelta(hours=9)
        # Tiempo Mac
        # current_time = pd.Timestamp.now() + timedelta(hours=5)
        
        # Comprobar si la columna 'symbol' está en el DataFrame
        if 'symbol' not in data.columns:
            raise KeyError("La columna 'symbol' no se encuentra en el DataFrame.")
        
        # Filtro de columnas
        data_filtered = data[['symbol', 'orderId', 'type', 'time', 'stopPrice']].copy()
        data_filtered['time'] = pd.to_datetime(data_filtered['time'], unit='ms')
        
        # Calcular la diferencia de tiempo
        data_filtered['time_difference'] = (current_time - data_filtered['time']).dt.total_seconds() / 60
        
        # Filtrar entradas con más de 1 minuto de diferencia y de tipo 'STOP_MARKET'
        data_filtered = data_filtered[(data_filtered['time_difference'] > 1) & (data_filtered['type'] == 'STOP_MARKET')]
        
        # Remover duplicados basado en 'symbol'
        data_filtered = data_filtered.drop_duplicates(subset='symbol')
        
        # Resetear el índice
        data_filtered.reset_index(drop=True, inplace=True)
        
        return data_filtered

    except FileNotFoundError:
        return pd.DataFrame(columns=['symbol', 'orderId', 'type', 'time', 'stopPrice'])  # Retorna un DataFrame con las columnas esperadas pero vacío

    except KeyError as e:
        return pd.DataFrame(columns=['symbol', 'orderId', 'type', 'time', 'stopPrice'])  # Retorna un DataFrame con las columnas esperadas pero vacío
    


def unrealized_profit_positions():
    # Cargar los indicadores
    df_indicadores = pd.read_csv('./archivos/indicadores.csv')
    
    # Verificar que las columnas necesarias existen
    required_columns = ['symbol', 'close', 'Stop_Loss_Long', 'Stop_Loss_Short']
    missing_columns = [col for col in required_columns if col not in df_indicadores.columns]
    if missing_columns:
        print(f"Las siguientes columnas faltan en 'indicadores.csv': {missing_columns}")
        return
    
    # Convertir símbolos a mayúsculas para asegurar coincidencia
    df_indicadores['symbol'] = df_indicadores['symbol'].str.upper()
    
    # Agrupar por 'symbol' y obtener la última fila de cada grupo
    latest_values = df_indicadores.groupby('symbol').last().reset_index()
    
    # Obtener datos filtrados de la función anterior
    data_filtered = filtrando_posiciones_antiguas()
    
    # Verificar si 'data_filtered' no está vacío
    if data_filtered.empty:
        # Silenciado: print("No hay posiciones antiguas para procesar.")
        return
    
    # Convertir símbolos a mayúsculas
    data_filtered['symbol'] = data_filtered['symbol'].str.upper()
    
    # Extraer la lista de símbolos
    symbols = data_filtered['symbol'].tolist()
    
    for symbol in symbols:
        # Silenciado: print(f"Procesando símbolo: {symbol}")
        
        # Obtener datos del símbolo en 'latest_values'
        symbol_data = latest_values[latest_values['symbol'] == symbol]
        
        # Verificar si 'symbol_data' está vacío
        if symbol_data.empty:
            # Silenciado: print(f"No hay datos de indicadores para el símbolo: {symbol}")
            continue
        
        # Obtener el precio actual
        precio_actual = symbol_data['close'].iloc[0]
        
        # Obtener datos de posición utilizando 'total_positions'
        result = total_positions(symbol)
        
        # Verificar si 'result' es None o no tiene suficientes datos
        if not result or result[0] is None:
            # Silenciado: print(f"No hay datos de posición para el símbolo: {symbol}")
            continue  # Saltar a la siguiente iteración del bucle
        
        # Desempaquetar el resultado
        symbol_result, positionSide, price, positionAmt, unrealizedProfit = result
        
        # Obtener el último valor de 'stopPrice' y 'orderId' para el símbolo
        filtered_data = data_filtered[data_filtered['symbol'] == symbol]
        
        # Verificar si 'filtered_data' está vacío
        if filtered_data.empty:
            # Silenciado: print(f"No se encontraron datos de 'order_id_register.csv' para el símbolo: {symbol}")
            continue
        
        # Acceder de forma segura a 'stopPrice' y 'orderId'
        last_stop_price = filtered_data['stopPrice'].iloc[-1]
        orderId = filtered_data['orderId'].iloc[-1]
        
        # Asegurar que 'positionAmt' es numérico
        try:
            positionAmt = float(positionAmt)
        except ValueError:
            # Silenciado: print(f"Cantidad de posición inválida para {symbol}: {positionAmt}")
            continue
        
        if positionSide == 'LONG':
            stop_loss = symbol_data['Stop_Loss_Long'].iloc[0]
            potencial_nuevo_sl = stop_loss
            if potencial_nuevo_sl > last_stop_price and potencial_nuevo_sl != last_stop_price:
                try:
                    pkg.bingx.cancel_order(symbol, orderId)
                    time.sleep(1)
                    pkg.bingx.post_order(symbol, positionAmt, 0, potencial_nuevo_sl, "LONG", "STOP_MARKET", "SELL")
                    print(f"Stop Loss actualizado para {symbol} (LONG) a {potencial_nuevo_sl}")
                except Exception as e:
                    print(f"Error al actualizar el Stop Loss para {symbol}: {e}")
            else:
                print(f"SL actual para {symbol} (LONG) es suficientemente bueno, no se modifica.")

        elif positionSide == 'SHORT':
            stop_loss = symbol_data['Stop_Loss_Short'].iloc[0]
            potencial_nuevo_sl = stop_loss
            if potencial_nuevo_sl < last_stop_price and potencial_nuevo_sl != last_stop_price:
                try:
                    pkg.bingx.cancel_order(symbol, orderId)
                    time.sleep(1)
                    pkg.bingx.post_order(symbol, positionAmt, 0, potencial_nuevo_sl, "SHORT", "STOP_MARKET", "BUY")
                    print(f"Stop Loss actualizado para {symbol} (SHORT) a {potencial_nuevo_sl}")
                except Exception as e:
                    print(f"Error al actualizar el Stop Loss para {symbol}: {e}")
            else:
                print(f"SL actual para {symbol} (SHORT) es suficientemente bueno, no se modifica.")
        else:
            # Silenciado: print(f"positionSide desconocido para {symbol}: {positionSide}")
            continue
  
