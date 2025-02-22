import pandas as pd
import numpy as np
import talib  # Asegúrate de que 'ta-lib' esté correctamente instalado

# =============================
# SECCIÓN DE VARIABLES
# =============================

# Parámetros de indicadores
RSI_PERIOD = 14  # Período del RSI
ATR_PERIOD = 14  # Período del ATR
EMA_SHORT_PERIOD = 8  # Período de la EMA corta
EMA_LONG_PERIOD = 20  # Período de la EMA larga
ADX_PERIOD = 7  # Período del ADX

# Multiplicadores para TP y SL basados en ATR
TP_MULTIPLIER = 5  # Multiplicador para el Take Profit
SL_MULTIPLIER = 2.5  # Multiplicador para el Stop Loss

# Umbrales para filtrar ruido del mercado
VOLUME_THRESHOLD = 0.68  # Umbral para volumen bajo (78% del volumen promedio)
VOLATILITY_THRESHOLD = 1.07  # Umbral para volatilidad alta (107% de la volatilidad promedio)

# Niveles de RSI para señales
RSI_OVERSOLD = 30  # Nivel de sobreventa para RSI
RSI_OVERBOUGHT = 65  # Nivel de sobrecompra para RSI

# =============================
# FIN DE LA SECCIÓN DE VARIABLES
# =============================

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

def calculate_indicators(
    data,
    rsi_period=14,
    atr_period=14,
    ema_short_period=12,
    ema_long_period=26,
    adx_period=14,
    tp_multiplier=4,
    sl_multiplier=1.2,
    volume_threshold=0.78,
    volatility_threshold=1.07,
    rsi_oversold=34,
    rsi_overbought=70,
    output_filepath='./archivos/indicadores.csv'
):
    # Ordenar el DataFrame por símbolo y fecha
    data = data.sort_values(by=['symbol', 'date']).copy()
    
    symbols = data['symbol'].unique()
    
    processed_symbols = []  # Lista para almacenar los DataFrames procesados
    
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
        required_periods = max(rsi_period, atr_period, ema_long_period, adx_period, 26)
        if len(df_symbol) < required_periods:
            continue  # Saltar al siguiente símbolo si no hay suficientes datos
        
        try:
            # Calcular indicadores técnicos
            df_symbol['RSI'] = talib.RSI(df_symbol['close'], timeperiod=rsi_period)
            df_symbol['ATR'] = talib.ATR(
                df_symbol['high'], df_symbol['low'], df_symbol['close'], timeperiod=atr_period)
            df_symbol['OBV'] = talib.OBV(df_symbol['close'], df_symbol['volume'])
            df_symbol['OBV_Slope'] = df_symbol['OBV'].diff()
            
            # Medias Móviles Exponenciales
            df_symbol['EMA_Short'] = talib.EMA(df_symbol['close'], timeperiod=ema_short_period)
            df_symbol['EMA_Long'] = talib.EMA(df_symbol['close'], timeperiod=ema_long_period)
            
            # MACD
            df_symbol['MACD'], df_symbol['MACD_Signal'], df_symbol['MACD_Hist'] = talib.MACD(
                df_symbol['close'], fastperiod=12, slowperiod=26, signalperiod=9)
            df_symbol['MACD_Bullish'] = (
                (df_symbol['MACD'] > df_symbol['MACD_Signal']) &
                (df_symbol['MACD'].shift(1) <= df_symbol['MACD_Signal'].shift(1))
            )
            df_symbol['MACD_Bearish'] = (
                (df_symbol['MACD'] < df_symbol['MACD_Signal']) &
                (df_symbol['MACD'].shift(1) >= df_symbol['MACD_Signal'].shift(1))
            )
            
            # Patrones de velas
            df_symbol['Hammer'] = talib.CDLHAMMER(
                df_symbol['open'], df_symbol['high'], df_symbol['low'], df_symbol['close'])
            df_symbol['ShootingStar'] = talib.CDLSHOOTINGSTAR(
                df_symbol['open'], df_symbol['high'], df_symbol['low'], df_symbol['close'])
            
            # Calcular Volumen Promedio y Volumen Relativo
            df_symbol['Avg_Volume'] = df_symbol['volume'].rolling(window=20).mean()
            df_symbol['Rel_Volume'] = df_symbol['volume'] / df_symbol['Avg_Volume']
            
            # Calcular Volatilidad (usando el ATR)
            df_symbol['Volatility'] = df_symbol['ATR']
            df_symbol['Avg_Volatility'] = df_symbol['Volatility'].rolling(window=20).mean()
            df_symbol['Rel_Volatility'] = df_symbol['Volatility'] / df_symbol['Avg_Volatility']
            
            # Calcular ADX
            df_symbol['ADX'] = talib.ADX(
                df_symbol['high'], df_symbol['low'], df_symbol['close'], timeperiod=adx_period)
            
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
            
            # Añadir df_symbol procesado a la lista
            processed_symbols.append(df_symbol)
                
        except Exception as e:
            print(f"Error al calcular indicadores para {symbol}: {e}")
            continue  # Si ocurre un error, saltar al siguiente símbolo
    
    # Concatenar todos los df_symbol procesados
    if processed_symbols:
        data = pd.concat(processed_symbols, ignore_index=True)
    else:
        data = pd.DataFrame()  # Si no hay símbolos procesados, devolver un DataFrame vacío

    # Definir señales de tendencia
    data['Trend_Up'] = data['EMA_Short'] > data['EMA_Long']
    data['Trend_Down'] = data['EMA_Short'] < data['EMA_Long']

    # Convertir columnas booleanas al tipo 'bool' de NumPy
    data['Trend_Up'] = data['Trend_Up'].astype('bool')
    data['Trend_Down'] = data['Trend_Down'].astype('bool')
    
    # Asegurarse de que no haya valores NaN en los indicadores
    data['Hammer'] = data['Hammer'].fillna(0)
    data['ShootingStar'] = data['ShootingStar'].fillna(0)
    data['MACD_Bullish'] = data['MACD_Bullish'].fillna(False).astype('bool')
    data['MACD_Bearish'] = data['MACD_Bearish'].fillna(False).astype('bool')
    data['ADX'] = data['ADX'].fillna(0)
    
    # Filtrar períodos de bajo volumen y alta volatilidad
    data['Low_Volume'] = data['Rel_Volume'] < volume_threshold
    data['High_Volatility'] = data['Rel_Volatility'] > volatility_threshold
    data['Low_Volume'] = data['Low_Volume'].astype('bool')
    data['High_Volatility'] = data['High_Volatility'].astype('bool')

    # Definir tendencia a largo plazo
    data['EMA_Long_Term'] = talib.EMA(data['close'], timeperiod=50)
    data['Trend_Up_Long_Term'] = data['EMA_Short'] > data['EMA_Long_Term']
    data['Trend_Down_Long_Term'] = data['EMA_Short'] < data['EMA_Long_Term']
    data['Trend_Up_Long_Term'] = data['Trend_Up_Long_Term'].astype('bool')
    data['Trend_Down_Long_Term'] = data['Trend_Down_Long_Term'].astype('bool')
    
    # Señales combinadas con filtrado de ruido y análisis de RSI
    data['Long_Signal'] = (
        (
            ((data['Hammer'] != 0) & data['Trend_Up'] & data['Trend_Up_Long_Term'] & (data['RSI'] < rsi_oversold)) |
            (data['MACD_Bullish'] & (data['ADX'] > 25) & data['Trend_Up_Long_Term'] & (data['RSI'] < rsi_overbought))
        ) &
        (~data['Low_Volume']) &
        (~data['High_Volatility'])
    ).astype('bool')
    
    data['Short_Signal'] = (
        (
            ((data['ShootingStar'] != 0) & data['Trend_Down'] & data['Trend_Down_Long_Term'] & (data['RSI'] > rsi_overbought)) |
            (data['MACD_Bearish'] & (data['ADX'] > 25) & data['Trend_Down_Long_Term'] & (data['RSI'] > rsi_oversold))
        ) &
        (~data['Low_Volume']) &
        (~data['High_Volatility'])
    ).astype('bool')
    
    # Restablecer índices y ordenar por símbolo y fecha
    data = data.sort_values(by=['symbol', 'date']).reset_index(drop=True)
    
    # Guardar el DataFrame resultante en 'indicadores.csv'
    data.to_csv(output_filepath, index=False)
    
    return data

def ema_alert(currencie, data_path='./archivos/cripto_price.csv'):
    try:
        crypto_data = load_data(data_path)
        if crypto_data is None:
            return None, None

        crypto_data = filter_duplicates(crypto_data)
        
        # Asegurarse de pasar los parámetros actualizados
        cruce_emas = calculate_indicators(
            crypto_data,
            rsi_period=RSI_PERIOD,
            atr_period=ATR_PERIOD,
            ema_short_period=EMA_SHORT_PERIOD,
            ema_long_period=EMA_LONG_PERIOD,
            adx_period=ADX_PERIOD,
            tp_multiplier=TP_MULTIPLIER,
            sl_multiplier=SL_MULTIPLIER,
            volume_threshold=VOLUME_THRESHOLD,
            volatility_threshold=VOLATILITY_THRESHOLD,
            rsi_oversold=RSI_OVERSOLD,
            rsi_overbought=RSI_OVERBOUGHT
        )
        
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