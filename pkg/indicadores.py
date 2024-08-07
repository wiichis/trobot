import pandas as pd
import numpy as np

def load_data(filepath='./archivos/cripto_price.csv'):
    try:
        crypto_data = pd.read_csv(filepath, parse_dates=['date'])
        crypto_data.sort_values(by='date', inplace=True)
        return crypto_data
    except Exception as e:
        print(f"Error leyendo el archivo: {e}")
        return None

def filter_duplicates(crypto_data):
    # Eliminar duplicados exactos y aquellos con volumen igual a 0
    crypto_data = crypto_data.drop_duplicates()
    crypto_data = crypto_data[crypto_data['volume'] > 0]
    return crypto_data

def calculate_rsi(data, window=11):
    delta = data.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_atr(df, window=13):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

def detect_macd_cross(df):
    bullish_cross = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
    bearish_cross = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))
    bullish_cross = bullish_cross & (df['MACD'] > 0)
    bearish_cross = bearish_cross & (df['MACD'] < 0)
    df['Bullish_MACD_Cross'] = bullish_cross
    df['Bearish_MACD_Cross'] = bearish_cross
    return df

def calculate_indicators(crypto_data):
    # Asegurarse de que los datos estén ordenados cronológicamente dentro de cada símbolo
    crypto_data.sort_values(by=['symbol', 'date'], inplace=True)
    
    complete_data = crypto_data[crypto_data['volume'] > 0].copy()
    complete_data['SMA_10'] = complete_data.groupby('symbol')['close'].transform(lambda x: x.rolling(window=10).mean())
    complete_data['RSI_9'] = complete_data.groupby('symbol')['close'].transform(lambda x: calculate_rsi(x, window=9))
    complete_data['EMA_9'] = complete_data.groupby('symbol')['close'].transform(lambda x: x.ewm(span=5, adjust=False).mean())
    complete_data['EMA_21'] = complete_data.groupby('symbol')['close'].transform(lambda x: x.ewm(span=22, adjust=False).mean())
    complete_data['MACD'] = complete_data['EMA_9'] - complete_data['EMA_21']
    complete_data['MACD_Signal'] = complete_data.groupby('symbol')['MACD'].transform(lambda x: x.ewm(span=7, adjust=False).mean())
    complete_data = complete_data.groupby('symbol', group_keys=False).apply(detect_macd_cross).reset_index(drop=True)
    complete_data['ATR'] = complete_data.groupby('symbol').apply(lambda x: calculate_atr(x)).reset_index(drop=True)
    complete_data['Avg_Volume'] = complete_data.groupby('symbol')['volume'].transform(lambda x: x.rolling(window=20).mean())
    complete_data['Rel_Volume'] = complete_data['volume'] / complete_data['Avg_Volume']
    
    complete_data['Long_Signal'] = (
        (complete_data['close'] > complete_data['SMA_10']) & 
        complete_data['Bullish_MACD_Cross'] & 
        (complete_data['RSI_9'] < 70) &
        (complete_data['Rel_Volume'] > 1.5)
    )
    complete_data['Short_Signal'] = (
        (complete_data['close'] < complete_data['SMA_10']) & 
        complete_data['Bearish_MACD_Cross'] & 
        (complete_data['RSI_9'] > 30) &
        (complete_data['Rel_Volume'] > 1.5)
    )

    complete_data['Take_Profit'] = 3 * complete_data['ATR']
    complete_data['Stop_Loss'] = complete_data['ATR']

    # Ordenar por 'symbol' y 'date' antes de guardar
    complete_data = complete_data.sort_values(by=['symbol', 'date'])
    complete_data.to_csv('./archivos/indicadores.csv', index=False)
    
    return complete_data

def ema_alert(currencie, data_path='./archivos/cripto_price.csv'):
    try:
        crypto_data = load_data(data_path)
        if crypto_data is None:
            return None, None

        crypto_data = filter_duplicates(crypto_data)
        cruce_emas = calculate_indicators(crypto_data)
        
        # Crear una copia explícita del DataFrame filtrado
        df_filtered = cruce_emas[cruce_emas['symbol'] == currencie].copy()

        if df_filtered.empty:
            print(f"No hay datos para {currencie}")
            return None, None

        # Asegurarse de que los datos estén ordenados cronológicamente
        df_filtered.sort_values(by='date', inplace=True)
        price_last = df_filtered['close'].iloc[-1]

        # Imprimir en consola el símbolo, la fecha, el precio de apertura y cierre de la vela evaluada
        # symbol = df_filtered['symbol'].iloc[-1]
        # date = df_filtered['date'].iloc[-1]
        # open_price = df_filtered['open'].iloc[-1]
        # close_price = df_filtered['close'].iloc[-1]
        # print(f"Evaluando: Symbol={symbol}, Date={date}, Open={open_price}, Close={close_price}")

        if df_filtered['Long_Signal'].iloc[-1]:
            return price_last, '=== Alerta de LONG ==='
        elif df_filtered['Short_Signal'].iloc[-1]:
            return price_last, '=== Alerta de SHORT ==='
        else:
            return None, None
    except Exception as e:
        print(f"Error en ema_alert: {e}")
        return None, None

