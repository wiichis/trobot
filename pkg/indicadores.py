import pandas as pd
import numpy as np
from datetime import datetime
import talib

def calcular_vela_subida_bajada(precios):

    #Tomar los ultimos 15 ciclos
    precios = precios.iloc[-15:]

    apertura = precios.iloc[0]
    cierre = precios.iloc[-1]
    maximo = precios.max()
    minimo = precios.min()
    porcentaje_mecha = (maximo - minimo) / apertura * 100
    subida_bajada = 'Subida' if cierre > apertura else 'Bajada'

    alerta = None
    if abs(cierre - apertura) > apertura * 0.012:  # Diferencia de precio mayor al 1%
        if porcentaje_mecha > 20:  # Porcentaje de mecha mayor al 20%
            alerta = 'Retorno Envolvente'
        else:
            alerta = f'{subida_bajada} Envolvente'

    # Mensaje predeterminado si no se cumple ninguna condición
    if alerta is None:
        alerta = 'Sin alerta'

    return alerta


def emas_indicator():
    # Importar datos de precios
    df = pd.read_csv('./archivos/cripto_price.csv')
    df = df.iloc[-20000:] 
   
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

    # Calcular vela envolvente
    alertas = grouped['price'].transform(lambda x: calcular_vela_subida_bajada(x))
    df['alerta'] = alertas
    
    #Calcular la columna 'type' utilizando los valores de EMA, RSI, Envelope para cada fila
    df['type'] = 'NONE'
    df.loc[(ema50 > ema21) & (rsi < 30) & (envelope_inferior >= price) & (alertas != 'Bajada Envolvente'),'type'] = 'LONG'
    df.loc[(ema50 < ema21) & (rsi > 70) & (envelope_superior <= price)& (alertas != 'Subida Envolvente'),'type'] = 'SHORT'
            
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

