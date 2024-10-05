import pandas as pd
import numpy as np
import talib  # Asegúrate de que 'talib' esté correctamente instalado

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
    crypto_data = crypto_data.reset_index(drop=True)
    return crypto_data

def calculate_indicators(data, output_filepath='./archivos/indicadores.csv'):
    # Ordenar el DataFrame por símbolo y fecha
    data.sort_values(by=['symbol', 'date'], inplace=True)
    data = data.copy()
    
    symbols = data['symbol'].unique()
    
    for symbol in symbols:
        df_symbol = data[data['symbol'] == symbol].copy()
        
        # Validación y limpieza de datos
        df_symbol = df_symbol[df_symbol['close'] > 0]
        df_symbol['close'] = df_symbol['close'].ffill()
        df_symbol['open'] = df_symbol['open'].ffill()
        df_symbol['high'] = df_symbol['high'].ffill()
        df_symbol['low'] = df_symbol['low'].ffill()
        df_symbol['volume'] = df_symbol['volume'].fillna(0)
        
        # Verificar si hay suficientes datos
        required_periods = 26  # Período más largo utilizado
        if len(df_symbol) < required_periods:
            continue  # Saltar al siguiente símbolo si no hay suficientes datos
        
        try:
            # Calcular indicadores técnicos
            df_symbol['RSI_11'] = talib.RSI(df_symbol['close'], timeperiod=11)
            df_symbol['ATR'] = talib.ATR(
                df_symbol['high'], df_symbol['low'], df_symbol['close'], timeperiod=14)
            df_symbol['OBV'] = talib.OBV(df_symbol['close'], df_symbol['volume'])
            df_symbol['OBV_Slope'] = df_symbol['OBV'].diff()
            
            # Medias Móviles Exponenciales
            df_symbol['EMA_Short'] = talib.EMA(df_symbol['close'], timeperiod=12)
            df_symbol['EMA_Long'] = talib.EMA(df_symbol['close'], timeperiod=26)
            
            # MACD
            df_symbol['MACD'], df_symbol['MACD_Signal'], df_symbol['MACD_Hist'] = talib.MACD(
                df_symbol['close'], fastperiod=12, slowperiod=26, signalperiod=9)
            df_symbol['MACD_Bullish'] = (
                (df_symbol['MACD'] > df_symbol['MACD_Signal']) &
                (df_symbol['MACD'].shift(1) <= df_symbol['MACD_Signal'].shift(1))
            ).astype('boolean')
            df_symbol['MACD_Bearish'] = (
                (df_symbol['MACD'] < df_symbol['MACD_Signal']) &
                (df_symbol['MACD'].shift(1) >= df_symbol['MACD_Signal'].shift(1))
            ).astype('boolean')
            
            # Patrones de velas
            df_symbol['Hammer'] = talib.CDLHAMMER(
                df_symbol['open'], df_symbol['high'], df_symbol['low'], df_symbol['close'])
            df_symbol['ShootingStar'] = talib.CDLSHOOTINGSTAR(
                df_symbol['open'], df_symbol['high'], df_symbol['low'], df_symbol['close'])
            
            # Calcular Volumen Promedio y Volumen Relativo
            df_symbol['Avg_Volume'] = df_symbol['volume'].rolling(window=20).mean()
            df_symbol['Rel_Volume'] = df_symbol['volume'] / df_symbol['Avg_Volume']
            
            # Calcular ADX
            df_symbol['ADX'] = talib.ADX(
                df_symbol['high'], df_symbol['low'], df_symbol['close'], timeperiod=14)
            
            # Definir multiplicadores para el ATR
            tp_multiplier = 6
            sl_multiplier = 2
            
            # Calcular TP y SL basados en ATR para posiciones LARGAS
            df_symbol['Take_Profit_Long'] = df_symbol['close'] + (df_symbol['ATR'] * tp_multiplier)
            df_symbol['Stop_Loss_Long'] = df_symbol['close'] - (df_symbol['ATR'] * sl_multiplier)
            df_symbol['Stop_Loss_Long'] = df_symbol['Stop_Loss_Long'].clip(lower=1e-8)
            
            # Calcular TP y SL basados en ATR para posiciones CORTAS
            df_symbol['Take_Profit_Short'] = df_symbol['close'] - (df_symbol['ATR'] * tp_multiplier)
            df_symbol['Stop_Loss_Short'] = df_symbol['close'] + (df_symbol['ATR'] * sl_multiplier)
            df_symbol['Take_Profit_Short'] = df_symbol['Take_Profit_Short'].clip(lower=1e-8)
            
            # Reemplazar valores NaN
            df_symbol = df_symbol.ffill().fillna(0)
            
            # Actualizar el DataFrame principal
            data.loc[df_symbol.index, df_symbol.columns] = df_symbol
            
        except Exception as e:
            continue  # Si ocurre un error, saltar al siguiente símbolo
    
    # Definir señales de tendencia
    data['Trend_Up'] = data['EMA_Short'] > data['EMA_Long']
    data['Trend_Down'] = data['EMA_Short'] < data['EMA_Long']

    # Asegurarse de que no haya valores NaN en los indicadores
    data['Hammer'] = data['Hammer'].fillna(0)
    data['ShootingStar'] = data['ShootingStar'].fillna(0)
    data['MACD_Bullish'] = data['MACD_Bullish'].fillna(False).astype('boolean')
    data['MACD_Bearish'] = data['MACD_Bearish'].fillna(False).astype('boolean')
    data['ADX'] = data['ADX'].fillna(0)
    
    # Convertir columnas booleanas al tipo correcto
    data['Trend_Up'] = data['Trend_Up'].astype('boolean')
    data['Trend_Down'] = data['Trend_Down'].astype('boolean')

    # Señales combinadas
    data['Long_Signal'] = (
        ((data['Hammer'] != 0) & data['Trend_Up']) |
        (data['MACD_Bullish'] & (data['ADX'] > 25))
    ).astype('boolean')
    
    data['Short_Signal'] = (
        ((data['ShootingStar'] != 0) & data['Trend_Down']) |
        (data['MACD_Bearish'] & (data['ADX'] > 25))
    ).astype('boolean')

    # Restablecer índices si es necesario y retornar el DataFrame actualizado
    data.reset_index(drop=True, inplace=True)
    
    # Guardar el DataFrame resultante en 'indicadores.csv'
    data.to_csv(output_filepath, index=False)
    
    return data


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
    

    
    