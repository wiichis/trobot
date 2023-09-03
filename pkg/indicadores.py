import pandas as pd
import numpy as np
from datetime import datetime
import talib

last_trend = None

def detect_trend_change(row, previous_row):
    global last_trend
    if row['HA_Close'] > row['HA_Open'] and previous_row['HA_Close'] < previous_row['HA_Open']:
        last_trend = "Cambio a Alcista"
    elif row['HA_Close'] < row['HA_Open'] and previous_row['HA_Close'] > previous_row['HA_Open']:
        last_trend = "Cambio a Bajista"
    return last_trend

def heikin_ashi(df):
    df['HA_Close'] = round((df['price'] + df['price'].shift(1)) / 2, 8)
    df['HA_Open'] = round((df['HA_Close'] + df['HA_Close'].shift(1)) / 2, 8)
    df['HA_High'] = round(df[['price', 'HA_Open', 'HA_Close']].max(axis=1), 6)
    df['HA_Low'] = round(df[['price', 'HA_Open', 'HA_Close']].min(axis=1), 6)
    return df

def calculate_type(row):
    if (
        row['ema50'] > row['ema21'] and 
        row['rsi'] < 30 and 
        row['envelope_inferior'] >= row['price'] and 
        row['trend_change'] == 'Cambio a Alcista'
    ):
        return 'LONG'
    
    if (
        row['ema50'] < row['ema21'] and 
        row['rsi'] > 70 and 
        row['envelope_superior'] <= row['price'] and 
        row['trend_change'] == 'Cambio a Bajista'
    ):
        return 'SHORT'
    
    return None

def calculate_rsi(df):
    df['rsi'] = talib.RSI(df['price'], timeperiod=14)
    return df

def calculate_ema(df, timeperiod, column_name):
    df[column_name] = talib.EMA(df['price'], timeperiod=timeperiod)
    return df

def calculate_envelope(df):
    ancho_banda = 5
    periodo = int(ancho_banda / 2)
    df['smoothing'] = talib.EMA(df['price'], timeperiod=periodo)

    df['residuos'] = df['price'] - df['price'].mean()
    df['envelope_superior'] = df['smoothing'] + 2 * df['residuos']
    df['envelope_inferior'] = df['smoothing'] - 2 * df['residuos']
    return df

def apply_all_indicators(df):
    df = df.groupby('symbol', group_keys=False).apply(lambda x: x.tail(100)).reset_index(drop=True)
    df = df.groupby('symbol', group_keys=False).apply(heikin_ashi)
    df['trend_change'] = df.apply(lambda row: detect_trend_change(row, df.iloc[df.index.get_loc(row.name) - 1] if df.index.get_loc(row.name) > 0 else row), axis=1)
    df = df.groupby('symbol', group_keys=False).apply(calculate_rsi)
    df = df.groupby('symbol', group_keys=False).apply(lambda x: calculate_ema(x, 50, 'ema50'))
    df = df.groupby('symbol', group_keys=False).apply(lambda x: calculate_ema(x, 21, 'ema21'))
    df = df.groupby('symbol', group_keys=False).apply(calculate_envelope)

    df.reset_index(drop=True, inplace=True)
    return df

def calculate_type(row):
    if (
        row['ema50'] > row['ema21'] and 
        row['rsi'] < 30 and 
        row['envelope_inferior'] >= row['price'] and 
        row['trend_change'] == 'Cambio a Alcista'
    ):
        return 'LONG'
    
    if (
        row['ema50'] < row['ema21'] and 
        row['rsi'] > 70 and 
        row['envelope_superior'] <= row['price'] and 
        row['trend_change'] == 'Cambio a Bajista'
    ):
        return 'SHORT'
    
    return None


# Utilizar la función
def indicadores():
    df = pd.read_csv('./archivos/cripto_price.csv')
    df_indicators = apply_all_indicators(df)
    # Eliminar las columnas que no quieres guardar
    df_indicators.drop(['HA_Close', 'HA_Open', 'HA_High', 'HA_Low', 'smoothing', 'residuos'], axis=1, inplace=True)
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



