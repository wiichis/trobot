import pandas as pd
from datetime import datetime
import talib


def emas_indicator():
    # Importar datos de precios
    df = pd.read_csv('./archivos/cripto_price.csv')
    df_month = df.iloc[-19000:] #Un Mes
    
    #Convertir date en un datetime
    df_month = df_month.copy()
    df_month['date'] = pd.to_datetime(df_month['date'])
   
    # Establecer la columna 'date' como el índice del DataFrame
    df_month.set_index('date', inplace=True)
    grouped = df_month.groupby('symbol')

    # Calcular el EMA de período 50 y el EMA de período 21 para cada grupo
    ema50 = grouped['price'].transform(lambda x: talib.EMA(x, timeperiod=50))
    ema21 = grouped['price'].transform(lambda x: talib.EMA(x, timeperiod=21))
    df_month['ema50'] = ema50
    df_month['ema21'] = ema21

    # Calcular el RSI de período 14 para cada grupo
    rsi = grouped['price'].transform(lambda x: talib.RSI(x, timeperiod=14))
    df_month['rsi'] = rsi

    # Calcular la columna 'type' utilizando los valores de EMA y RSI para cada fila
    df_month['type'] = 'NONE'
    df_month.loc[(ema50 > ema21) & (rsi < 30), 'type'] = 'LONG'
    df_month.loc[(ema50 < ema21) & (rsi > 70), 'type'] = 'SHORT'
            
    cruce_emas = df_month.groupby('symbol').tail(20).reset_index()
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
           


