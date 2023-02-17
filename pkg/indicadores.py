import pandas as pd
import numpy as np
from datetime import datetime
import talib


def emas_indicator():
    # Importar datos de precios
    df = pd.read_csv('./archivos/cripto_price.csv')
    df = df.iloc[-5000:] 
   
    # Establecer la columna 'date' como el índice del DataFrame
    df.set_index('date', inplace=True)
    grouped = df.groupby('symbol')

    # Calcular el EMA de período 50 y el EMA de período 21 para cada grupo
    ema50 = grouped['price'].transform(lambda x: talib.EMA(x, timeperiod=50))
    ema21 = grouped['price'].transform(lambda x: talib.EMA(x, timeperiod=21))
    df['ema50'] = ema50
    df['ema21'] = ema21

    # Calcular el RSI de período 14 para cada grupo
    rsi = grouped['price'].transform(lambda x: talib.RSI(x, timeperiod=14))
    df['rsi'] = rsi

    # # Calcular Nadaraya-Watson Enveloper con Tablib
    # ancho_banda = 5
    # periodo = int(ancho_banda / 2)
    # smoothing = grouped['price'].transform(lambda x: talib.EMA(x, timeperiod=periodo))

    # # Restamos la media móvil a los datos originales
    # residuos = grouped['price'] - smoothing

    # # Calculamos el Nadaraya-Watson Envelope a partir de los residuos
    # envelope = grouped['price'].transform(lambda x: talib.EMA(np.abs(residuos), timeperiod=periodo))
    # envelope_superior = smoothing + ancho_banda * envelope
    # envelope_inferior = smoothing - ancho_banda * envelope
    # df['envelope_superior'] = envelope_superior
    # df['envelope_inferior'] = envelope_inferior

    # Calcular la columna 'type' utilizando los valores de EMA y RSI para cada fila
    df['type'] = 'NONE'
    df.loc[(ema50 > ema21) & (rsi < 30), 'type'] = 'LONG'
    df.loc[(ema50 < ema21) & (rsi > 70), 'type'] = 'SHORT'

            
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
        stop_lose = price_last * 0.99
        profit = price_last * 1.03
        tipo = '=== Alerta de LONG ==='
        return price_last, stop_lose, profit, tipo
    elif type == 'SHORT':
        stop_lose = price_last * 1.01
        profit = price_last * 0.97
        tipo = '=== Alerta de SHORT ==='
        return price_last, stop_lose, profit, tipo
           


