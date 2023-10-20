import pandas as pd
import numpy as np
from datetime import datetime
import talib

#Calculo de ATR de Velas
def calculate_atr(data, period=14):
    high = data['high']
    low = data['low']
    close = data['close']

    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    return atr

#Identificar Velas Largas
def is_long_candle(data, multiplier=2):   #Multiplier es el tamaño de las velas
    candle_range = data['high'] - data['low']
    atr = calculate_atr(data)
    return candle_range > multiplier * atr

def analizar_tendencia_criptomonedas(df, umbral=0, umbral_alerta=45):
    df['Diferencia'] = df['close'] - df['open']
    df['Tendencia'] = 0
    df.loc[df['Diferencia'] > umbral, 'Tendencia'] = 1
    df.loc[df['Diferencia'] < -umbral, 'Tendencia'] = -1

    # Agrupar por símbolo y calcular la tendencia acumulada
    df['Tendencia_acumulada'] = df.groupby('symbol')['Tendencia'].cumsum()
    
    # Crear una columna que indica si se cumple la condición de tendencia acumulada mayor a umbral_alerta
    df['Alerta'] = abs(df['Tendencia_acumulada']) > umbral_alerta
    
    # Eliminar columnas temporales
    df.drop(['Diferencia'], axis=1, inplace=True)
    
    return df



def emas_indicator():
    # Importar datos de precios
    df = pd.read_csv('./archivos/cripto_price.csv', parse_dates=["date"])
    df = df.iloc[-2000:] 
   
    grouped = df.groupby('symbol')

    # Calcular el EMA de período 50 y el EMA de período 21 para cada grupo
    ema50 = grouped['price'].transform(lambda x: talib.EMA(x, timeperiod=50))
    ema21 = grouped['price'].transform(lambda x: talib.EMA(x, timeperiod=21))
    df['ema50'] = round(ema50, 4)
    df['ema21'] = round(ema21, 4)

    # Calcular el RSI de período 14 para cada grupo
    rsi = grouped['price'].transform(lambda x: talib.RSI(x, timeperiod=14))
    df['rsi'] = round(rsi, 4)

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
    df['envelope_superior'] = round(envelope_superior, 4)
    df['envelope_inferior'] = round(envelope_inferior,4 )

        # Creamos una columna que identifica a qué intervalo de 5 minutos pertenece cada fila
    df["time_bin"] = df["date"].dt.floor('15T')

    # Agrupamos por símbolo y time_bin y generamos los valores para cada vela
    grouped_velas = df.groupby(['symbol', 'time_bin'])
    df = df.join(grouped_velas['price'].agg(open='first', high='max', low='min', close='last'), on=['symbol', 'time_bin'])

    ohlc = grouped_velas['price'].agg(open='first', high='max', low='min', close='last')
    ohlc.reset_index(inplace=True)

     # Calculamos el ATR y si la vela es larga para cada grupo y lo añadimos al DataFrame ohlc
    ohlc['ATR'] = round(ohlc.groupby('symbol').apply(lambda x: calculate_atr(x)).reset_index(level=0, drop=True),5)
    ohlc['IsLongCandle'] = ohlc.groupby('symbol').apply(lambda x: is_long_candle(x)).reset_index(level=0, drop=True)

    # Merge ohlc de vuelta con df para agregar las nuevas columnas calculadas
    merged_df = pd.merge(df, ohlc, how='left', on=['symbol', 'time_bin'], suffixes=('', '_ohlc'))

    # Direccion de las velas consecutivas
    umbral = 0  # Ajusta el umbral según tus necesidades
    umbral_alerta = 45  # Ajusta el número de velas consecutivas para considerar
    merged_df = analizar_tendencia_criptomonedas(merged_df, umbral, umbral_alerta)




    #Calcular la columna 'type' utilizando los valores de EMA y RSI para cada fila
    merged_df['type'] = 'NONE'
    merged_df.loc[(ema50 > ema21) & (rsi < 30) & (envelope_inferior >= price) & (merged_df['IsLongCandle'] == False ) & (merged_df['Alerta'] == False ),'type'] = 'LONG'
    merged_df.loc[(ema50 < ema21) & (rsi > 70) & (envelope_superior <= price) & (merged_df['IsLongCandle'] == False ) & (merged_df['Alerta'] == False ),'type'] = 'SHORT'


            
    cruce_emas = merged_df.groupby('symbol').tail(35).reset_index()
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

