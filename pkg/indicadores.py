import pandas as pd
import numpy as np
from datetime import datetime
import talib


def calculate_pct(df):
    df['cambio_pct'] = round(df['price'].pct_change() * 100, 2)
    return df

def calculate_volatility_alert(df):
    alerts = []
    window_size = 45
    for i in range(len(df)):
        if i < window_size - 1:  #reduciendo el % de cambio
            alerts.append('Baja')  # o podrías usar NaN o alguna otra etiqueta para indicar que la ventana aún no es lo suficientemente grande
        else:
            window = df['cambio_pct'].iloc[i-window_size+1:i+1]
            alert = 'Alta' if any(abs(x) > 0.25 for x in window) else 'Baja'
            alerts.append(alert)
    df['volatility_alert'] = alerts
    return df

def calculate_type(row):
    if (
        row['ema50'] > row['ema21'] and 
        row['rsi'] < 30 and 
        row['envelope_inferior'] >= row['price'] and 
        row['volatility_alert'] == 'Baja'
    ):
        return 'LONG'
    
    if (
        row['ema50'] < row['ema21'] and 
        row['rsi'] > 70 and 
        row['envelope_superior'] <= row['price'] and 
        row['volatility_alert'] == 'Baja'
    ):
        return 'SHORT'
    
    return None

def calculate_rsi(df):
    df['rsi'] = round(talib.RSI(df['price'], timeperiod=14),1)
    return df

def calculate_ema(df, timeperiod, column_name):
    df[column_name] = talib.EMA(df['price'], timeperiod=timeperiod)
    df[column_name] = df[column_name].round(3)
    return df

def calculate_envelope(df):
    ancho_banda = 0.00586  # 5% como ejemplo
    periodo = 100  # Puedes ajustar este valor según tus necesidades
    df['sma'] = talib.SMA(df['price'], timeperiod=periodo)  # Usamos SMA como ejemplo, pero puedes usar EMA si prefieres

    # Calculamos el envelope superior e inferior como un porcentaje de la media móvil
    df['envelope_superior'] = round(df['sma'] * (1 + ancho_banda),3)
    df['envelope_inferior'] = round(df['sma'] * (1 - ancho_banda),3)
    return df


def apply_all_indicators(df):
    df = df.groupby('symbol', group_keys=False).apply(lambda x: x.tail(110)).reset_index(drop=True)
    df = df.groupby('symbol', group_keys=False).apply(calculate_pct)
    df = df.groupby('symbol', group_keys=False).apply(calculate_volatility_alert)
    df = df.groupby('symbol', group_keys=False).apply(calculate_rsi)
    df = df.groupby('symbol', group_keys=False).apply(lambda x: calculate_ema(x, 50, 'ema50'))
    df = df.groupby('symbol', group_keys=False).apply(lambda x: calculate_ema(x, 21, 'ema21'))
    df = df.groupby('symbol', group_keys=False).apply(calculate_envelope)

    df.reset_index(drop=True, inplace=True)
    return df



# Utilizar la función
def indicadores():
    df = pd.read_csv('./archivos/cripto_price.csv')
    df_indicators = apply_all_indicators(df)
    # Eliminar las columnas que no quieres guardar
    df_indicators.drop(['sma'], axis=1, inplace=True)
    df_indicators['type'] = df_indicators.apply(calculate_type, axis=1)
    df_indicators.to_csv('./archivos/indicadores.csv', index=False)

    return df_indicators



def ema_alert(currencie):
    cruce_emas = indicadores()
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



