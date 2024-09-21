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
    avg_gain = pd.Series(gain, index=data.index).rolling(window=window).mean()
    avg_loss = pd.Series(loss, index=data.index).rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df, window=14):
    # Calcular los rangos requeridos para el True Range (TR)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())

    # Calcular el True Range (TR)
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Calcular el ATR usando una media móvil exponencial (EMA)
    atr = tr.ewm(span=window, adjust=False).mean()

    return atr

def detect_macd_cross(df):  #revisar si esta funcion aun se usa
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
    
    # Calcular los indicadores necesarios
    complete_data['RSI_11'] = complete_data.groupby('symbol')['close'].transform(lambda x: calculate_rsi(x, window=11))
    complete_data['ATR'] = complete_data.groupby('symbol').apply(lambda x: calculate_atr(x, window=14)).reset_index(level=0, drop=True)
    complete_data['Avg_Volume'] = complete_data.groupby('symbol')['volume'].transform(lambda x: x.rolling(window=20).mean())
    complete_data['Rel_Volume'] = complete_data['volume'] / complete_data['Avg_Volume']
    
    # Configurar las señales de compra y venta basadas en RSI y Volumen Relativo
    complete_data['Long_Signal'] = (
        (complete_data['RSI_11'] < 20) & 
        (complete_data['Rel_Volume'] > 1.2)
    )
    complete_data['Short_Signal'] = (
        (complete_data['RSI_11'] > 70) & 
        (complete_data['Rel_Volume'] > 1.2)
    )

    # Definir porcentajes para TP y SL
    tp_percentage = 0.011  # 5% de ganancia esperada
    sl_percentage = 0.004  # 2% de pérdida máxima aceptada

    # Calcular TP y SL basados en porcentajes del precio de cierre
    complete_data['Take_Profit'] = complete_data['close'] * tp_percentage
    complete_data['Stop_Loss'] = complete_data['close'] * sl_percentage

    # Ordenar por 'symbol' y 'date' antes de guardar
    complete_data.sort_values(by=['symbol', 'date'], inplace=True)
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

        if df_filtered['Long_Signal'].iloc[-1]:
            return price_last, '=== Alerta de LONG ==='
        elif df_filtered['Short_Signal'].iloc[-1]:
            return price_last, '=== Alerta de SHORT ==='
        else:
            return None, None
    except Exception as e:
        print(f"Error en ema_alert: {e}")
        return None, None