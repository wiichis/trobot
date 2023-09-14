import pandas as pd
import numpy as np
from datetime import datetime
import talib


def calculate_pct(df):
    df['cambio_pct'] = df['price'].pct_change() * 100
    return df

def calculate_volatility_alert(df):
    alerts = []
    window_size = 20
    for i in range(len(df)):
        if i < window_size - 0.8:  #reduciendo el % de cambio
            alerts.append('Baja')  # o podrías usar NaN o alguna otra etiqueta para indicar que la ventana aún no es lo suficientemente grande
        else:
            window = df['cambio_pct'].iloc[i-window_size+1:i+1]
            alert = 'Alta' if any(abs(x) > 1.5 for x in window) else 'Baja'
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
    #Calulando residuos para shorts
    df['residuos_short'] = df['price'] - df['smoothing'] 
    df['envelope_superior'] = df['smoothing'] + 2 * df['residuos_short']
    df['envelope_inferior'] = df['smoothing'] - 2 * df['residuos']
    return df

def apply_all_indicators(df):
    df = df.groupby('symbol', group_keys=False).apply(lambda x: x.tail(65)).reset_index(drop=True)
    df = df.groupby('symbol', group_keys=False).apply(calculate_pct)
    df = df.groupby('symbol', group_keys=False).apply(calculate_volatility_alert)
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
        row['envelope_inferior'] >= row['price'] #and 
        #row['trend_change'] == 'Cambio a Alcista'
    ):
        return 'LONG'
    
    if (
        row['ema50'] < row['ema21'] and 
        row['rsi'] > 70 and 
        row['envelope_superior'] <= row['price'] #and 
        #row['trend_change'] == 'Cambio a Bajista'
    ):
        return 'SHORT'
    
    return None


# Utilizar la función
def indicadores():
    df = pd.read_csv('./archivos/cripto_price.csv')
    df_indicators = apply_all_indicators(df)
    # Eliminar las columnas que no quieres guardar
    df_indicators.drop(['smoothing', 'residuos'], axis=1, inplace=True)
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



