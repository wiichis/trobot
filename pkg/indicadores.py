import pandas as pd
import numpy as np
#from datetime import datetime
#import talib

def indicator():
    # Cargar los datos
    crypto_data = pd.read_csv('./archivos/cripto_price.csv')
    crypto_data['date'] = pd.to_datetime(crypto_data['date'])
    crypto_data.sort_values(by='date', inplace=True)

    # Agrupar los datos en velas de una hora freq='H' freq='30T'
    hourly_data = crypto_data.groupby(['symbol', pd.Grouper(key='date', freq='H')]).agg(
        open_price=('price', 'first'),
        high_price=('price', 'max'),
        low_price=('price', 'min'),
        close_price=('price', 'last'),
        line_count=('price', 'size')  # Contador de líneas por vela
    ).reset_index()

    # Función para calcular el RSI
    def calculate_rsi(data, window=14):
        delta = data.diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0

        roll_up = up.rolling(window).mean()
        roll_down = down.abs().rolling(window).mean()

        rs = roll_up / roll_down
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi
    
    # Calcular la SMA de 20 períodos
    hourly_data['SMA_20'] = hourly_data.groupby('symbol')['close_price'].transform(lambda x: x.rolling(window=20).mean())

    # Calcular el RSI de 20 períodos
    rsi_results = hourly_data.groupby('symbol', group_keys=True)['close_price'].apply(lambda x: calculate_rsi(x, window=20)).reset_index(level=0, drop=True)
    hourly_data['RSI_20'] = rsi_results



    # Calcular el MACD y su línea de señal
    ema_12 = hourly_data.groupby('symbol')['close_price'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema_26 = hourly_data.groupby('symbol')['close_price'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    hourly_data['MACD'] = ema_12 - ema_26
    hourly_data['MACD_Signal'] = hourly_data.groupby('symbol')['MACD'].transform(lambda x: x.ewm(span=9, adjust=False).mean())



    # Función para detectar cruces del MACD
    def detect_macd_cross(df):
        # Considerar solo velas con más de 12 líneas
        df = df[df['line_count'] > 10]

        bullish_cross = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
        bearish_cross = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))
        return bullish_cross, bearish_cross

    # (Este es el lugar donde se añadirían los cálculos de SMA, RSI, MACD, etc., siguiendo los pasos previos)
    # Aplicar la función para detectar cruces del MACD
    hourly_data['Bullish_MACD_Cross'] = False
    hourly_data['Bearish_MACD_Cross'] = False
    for symbol, group in hourly_data.groupby('symbol'):
        bullish_cross, bearish_cross = detect_macd_cross(group)
        hourly_data.loc[group.index, 'Bullish_MACD_Cross'] = bullish_cross
        hourly_data.loc[group.index, 'Bearish_MACD_Cross'] = bearish_cross

   # Calcular la volatilidad como la desviación estándar de los cambios en el precio de cierre
    hourly_data['Volatility'] = hourly_data.groupby('symbol')['close_price'].transform(lambda x: x.pct_change().rolling(window=20).std())

    hourly_data['Long_Signal'] = (
        (hourly_data['close_price'] > hourly_data['SMA_20']) & 
        hourly_data['Bullish_MACD_Cross'] & 
        (hourly_data['RSI_20'] < 70) &
        (hourly_data['Volatility'] > 0.005)  # Condición de volatilidad añadida
    )
    hourly_data['Short_Signal'] = (
        (hourly_data['close_price'] < hourly_data['SMA_20']) & 
        hourly_data['Bearish_MACD_Cross'] & 
        (hourly_data['RSI_20'] > 30) &
        (hourly_data['Volatility'] > 0.005)  # Condición de volatilidad añadida
    )


    # Establecer el Take Profit y Stop Loss
    # Take Profit es 3 veces la volatilidad y Stop Loss es la volatilidad
    hourly_data['Take_Profit'] = 3 * hourly_data['Volatility']
    hourly_data['Stop_Loss'] = hourly_data['Volatility']

    hourly_data.to_csv('./archivos/indicadores.csv', index=False)
    return hourly_data

def ema_alert(currencie):
    cruce_emas = indicator()
    df_filtered = cruce_emas[cruce_emas['symbol'] == currencie]
    
    # Verificar si df_filtered está vacío
    if df_filtered.empty:
        print(f"No hay datos para {currencie}")
        return None, None

    price_last = df_filtered['close_price'].iloc[-1]

    if df_filtered['Long_Signal'].iloc[-1]:
        tipo = '=== Alerta de LONG ==='
        return price_last, tipo
    elif df_filtered['Short_Signal'].iloc[-1]:
        tipo = '=== Alerta de SHORT ==='
        return price_last, tipo
    else:
        # Manejar el caso donde no hay señal de LONG o SHORT
        return None, None