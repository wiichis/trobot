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

def detect_trend_change(row, previous_row):
    if row['HA_Close'] > row['HA_Open'] and previous_row['HA_Close'] < previous_row['HA_Open']:
        return "Cambio a Alcista"
    elif row['HA_Close'] < row['HA_Open'] and previous_row['HA_Close'] > previous_row['HA_Open']:
        return "Cambio a Bajista"
    else:
        return None


def calculate_indicators(group):
    group = heikin_ashi(group)
    group['rsi'] = talib.RSI(group['price'], timeperiod=14)

    ancho_banda = 5
    periodo = int(ancho_banda / 2)
    group['smoothing'] = talib.EMA(group['price'], timeperiod=periodo)

    group['residuos'] = group['price'] - group['price'].mean()
    group['envelope_superior'] = group['smoothing'] + 2 * group['residuos']
    group['envelope_inferior'] = group['smoothing'] - 2 * group['residuos']

    group['trend_change'] = group.apply(lambda row: detect_trend_change(row, group.iloc[group.index.get_loc(row.name) - 1]), axis=1)
    
    return group

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



def emas_indicator():
    df = pd.read_csv('./archivos/cripto_price.csv')
    
    # Uso de .copy() para evitar SettingWithCopyWarning
    df_ema = df.iloc[-750:].copy()
  
    df_ema['ema50'] = df_ema.groupby('symbol')['price'].transform(lambda x: talib.EMA(x, timeperiod=50))
    df_ema['ema21'] = df_ema.groupby('symbol')['price'].transform(lambda x: talib.EMA(x, timeperiod=21))
  
    # Uso de group_keys=True para evitar FutureWarning
    df_indicators = df_ema.groupby('symbol', group_keys=True).apply(calculate_indicators)
    
    # Restablecer el índice para evitar ambigüedad
    df_indicators.reset_index(drop=True, inplace=True)
    
    last_10_rows = df_indicators.groupby('symbol').tail(10).reset_index()
    last_10_rows = last_10_rows.sort_values(['symbol', 'date'])


    # Insertar columna 'type' con valores predeterminados como 'NEUTRAL'
    df_ema['type'] = 'NEUTRAL'

    # Calcular el tipo de comercio usando el método 'apply' y la función 'calculate_type'
    last_10_rows['type'] = last_10_rows.apply(calculate_type, axis=1)

    # Eliminar las columnas que no quieres guardar
    last_10_rows.drop(['HA_Close', 'HA_Open', 'HA_High', 'HA_Low', 'smoothing', 'residuos'], axis=1, inplace=True)
    last_10_rows.to_csv('./archivos/indicadores.csv', index=False)

    return last_10_rows


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



prueba = emas_indicator()