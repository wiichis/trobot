import pandas as pd
import numpy as np
from datetime import datetime
import talib



def emas_indicator():
    # Importar datos de precios
    df = pd.read_csv('./archivos/cripto_price.csv')
    df = df.iloc[-5000:] 
   
    grouped = df.groupby('symbol')
  
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


    #Calular la inclinacion de la tendencia
   # Define una función lambda para calcular la pendiente
    get_slope = lambda x: np.polyfit(range(len(x.iloc[-4::-5])), x.iloc[-4::-5], 1)[0]

    # Calcula la pendiente para cada grupo
    slope = grouped['price'].transform(get_slope)

    # Agrega la pendiente como una nueva columna en el DataFrame
    df['pendiente'] = slope
   

    #Calcular la columna 'type' utilizando los valores de EMA y RSI para cada fila
    df['type'] = 'NONE'
    df.loc[(ema50 > ema21) & (rsi < 30) & (envelope_inferior >= price) & slope > 0, 'type'] = 'LONG'
    df.loc[(ema50 < ema21) & (rsi > 70) & (envelope_superior <= price) & slope < 0,'type'] = 'SHORT'
            
    cruce_emas = df.groupby('symbol').tail(20).reset_index()
    cruce_emas = cruce_emas.sort_values(['symbol', 'date'])
    cruce_emas.to_csv('./archivos/indicadores.csv', index=False)

    return cruce_emas

def ema_alert(currencie):
    cruce_emas = emas_indicator()
    df_filterd = cruce_emas[cruce_emas['symbol'] ==  currencie]
    price_last = df_filterd['price'].iloc[-1]
    type = df_filterd['type'].iloc[-1]
    envelope_superior = df_filterd['envelope_superior'].iloc[-1]
    envelope_inferior = df_filterd['envelope_inferior'].iloc[-1]
       
    if type == 'LONG':
        stop_lose = price_last * 0.9983
        profit = price_last * 1.005
        tipo = '=== Alerta de LONG ==='
        return price_last, stop_lose, profit, tipo, envelope_superior, envelope_inferior
    elif type == 'SHORT':
        stop_lose = price_last * 1.0017
        profit = price_last * 0.995
        tipo = '=== Alerta de SHORT ==='
        return price_last, stop_lose, profit, tipo, envelope_superior, envelope_inferior
    else:
        pass

