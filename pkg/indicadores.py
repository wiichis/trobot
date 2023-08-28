import pandas as pd
import numpy as np
from datetime import datetime
import talib


def heikin_ashi(df):
    df['HA_Close'] = round((df['price'] + df['price'].shift(1)) / 2, 8)
    df['HA_Open'] = round((df['HA_Close'] + df['HA_Close'].shift(1)) / 2, 8)
    df['HA_High'] = round(df[['price', 'HA_Open', 'HA_Close']].max(axis=1), 6)
    df['HA_Low'] = round(df[['price', 'HA_Open', 'HA_Close']].min(axis=1), 6)
    return df

  # Esto es un estado global que se utilizará dentro de la función

def detect_trend_change(row, previous_row):

    # Si el cuerpo de la vela actual es alcista y el anterior es bajista
    if row['HA_Close'] > row['HA_Open'] and previous_row['HA_Close'] < previous_row['HA_Open']:
        result = f"Cambio a Alcista"

    # Si el cuerpo de la vela actual es bajista y el anterior es alcista
    elif row['HA_Close'] < row['HA_Open'] and previous_row['HA_Close'] > previous_row['HA_Open']:
        result = f"Cambio a Bajista"
 
    else:
        result = None

    return result





def emas_indicator():
    global previous_row
    # Importar datos de precios
    df = pd.read_csv('./archivos/cripto_price.csv')
    df = df.iloc[-750:] 
   
    grouped = df.groupby('symbol', group_keys=False)
  
    # Calcular el EMA de período 50 y el EMA de período 21 para cada grupo
    ema50 = grouped['price'].transform(lambda x: talib.EMA(x, timeperiod=50))
    ema21 = grouped['price'].transform(lambda x: talib.EMA(x, timeperiod=21))
    df['ema50'] = ema50
    df['ema21'] = ema21

    # Calcular el RSI de período 14 para cada grupo
    rsi = grouped['price'].transform(lambda x: talib.RSI(x, timeperiod=14))
    df['rsi'] = rsi

     # Calcular la media móvil exponencial
    ancho_banda = 5
    periodo = int(ancho_banda / 2)
    smoothing = grouped['price'].transform(lambda x: talib.EMA(x, timeperiod=periodo))

        # Calcular los residuos
    residuos = grouped['price'].transform(lambda x: x - x.mean())

        # Calcular el envelope
    envelope_superior = smoothing + 2 * residuos
    envelope_inferior = smoothing - 2 * residuos
    price = df['price']
    df['envelope_superior'] = envelope_superior
    df['envelope_inferior'] = envelope_inferior

    # Calcula Heikin Ashi para cada moneda
    #df = grouped.apply(lambda group: heikin_ashi(group))

    # Logica de Alertas
    # Ordena el DataFrame primero por 'symbol' y luego por 'date'
    # df = df.sort_values(by=['symbol', 'date'])

    # alertas = []
    # previous_row = None
    # previous_symbol = None

    # for index, row in df.iterrows():
    #     if row['symbol'] != previous_symbol:
    #         previous_row = None  # Reiniciar previous_row si cambiamos de símbolo
    #         previous_symbol = row['symbol']

    #     alerta = detect_trend_change(row, previous_row)
    #     alertas.append(alerta)
    #     previous_row = row  # Actualizar previous_row para la siguiente iteración

    # df['alerta'] = alertas

    #Calcular la columna 'type' utilizando los valores de EMA, RSI, Envelope para cada fila
    df.loc[(ema50 > ema21) & (rsi < 30) & (envelope_inferior >= price),'type'] = 'LONG'
    df.loc[(ema50 < ema21) & (rsi > 70) & (envelope_superior <= price),'type'] = 'SHORT'
            
    cruce_emas = df.groupby('symbol').tail(20).reset_index()
    cruce_emas = cruce_emas.sort_values(['symbol', 'date'])
    cruce_emas.to_csv('./archivos/indicadores.csv', index=False)

    return cruce_emas

def ema_alert(currencie):
    cruce_emas = emas_indicator()
    df_filterd = cruce_emas[cruce_emas['symbol'] ==  currencie]
    price_last = df_filterd['price'].iloc[-1]
    type = df_filterd['type'].iloc[-1]
       
    if type == 'LONG':
        tipo = '=== Alerta de LONG ==='
        return price_last, tipo
    elif type == 'SHORT':
        tipo = '=== Alerta de SHORT ==='
        return price_last, tipo
    else:
        pass



#prueba = emas_indicator()