import pandas as pd
import numpy as np
import talib  # Asegúrate de que 'ta-lib' esté correctamente instalado
import concurrent.futures

# =============================
# SECCIÓN DE VARIABLES
# =============================

# Parámetros de indicadores
RSI_PERIOD = 14  # Período del RSI
ATR_PERIOD = 10  # Período del ATR
EMA_SHORT_PERIOD = 10  # Período de la EMA corta
EMA_LONG_PERIOD = 18  # Período de la EMA larga
ADX_PERIOD = 7  # Período del ADX

# Multiplicadores para TP y SL basados en ATR
TP_MULTIPLIER = 15  # Multiplicador para el Take Profit
SL_MULTIPLIER = 2  # Multiplicador para el Stop Loss

# Umbrales para filtrar ruido del mercado
VOLUME_THRESHOLD = 0.5  # Umbral para volumen bajo (78% del volumen promedio)
VOLATILITY_THRESHOLD = 1.2  # Umbral para volatilidad alta (107% de la volatilidad promedio)

# Niveles de RSI para señales
RSI_OVERSOLD = 34  # Nivel de sobreventa para RSI
RSI_OVERBOUGHT = 69  # Nivel de sobrecompra para RSI

# Lista de monedas deshabilitadas para ignorar en el cálculo de indicadores
DISABLED_COINS = ["ADA-USDT", "SHIB-USDT", "BTC-USDT", "AVAX-USDT", "CFX-USDT", "LTC-USDT", "DOT-USDT"]

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
    data = data.sort_values(by=['symbol', 'date']).copy()
    symbols = [s for s in data['symbol'].unique() if s not in DISABLED_COINS]

    def _process_symbol(symbol):
        df_symbol = data[data['symbol'] == symbol].copy()
        df_symbol = df_symbol[df_symbol['close'] > 0]
        df_symbol[['close','open','high','low']] = df_symbol[['close','open','high','low']].ffill()
        df_symbol['volume'] = df_symbol['volume'].fillna(0)
        required_periods = max(rsi_period, atr_period, ema_long_period, adx_period, 26)
        if len(df_symbol) < required_periods:
            return None
        try:
            df_symbol['RSI'] = talib.RSI(df_symbol['close'], timeperiod=rsi_period)
            df_symbol['ATR'] = talib.ATR(df_symbol['high'], df_symbol['low'], df_symbol['close'], timeperiod=atr_period)
            df_symbol['OBV'] = talib.OBV(df_symbol['close'], df_symbol['volume'])
            df_symbol['OBV_Slope'] = df_symbol['OBV'].diff()
            df_symbol['EMA_Short'] = talib.EMA(df_symbol['close'], timeperiod=ema_short_period)
            df_symbol['EMA_Long'] = talib.EMA(df_symbol['close'], timeperiod=ema_long_period)
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
            df_symbol['Hammer'] = talib.CDLHAMMER(
                df_symbol['open'], df_symbol['high'], df_symbol['low'], df_symbol['close'])
            df_symbol['ShootingStar'] = talib.CDLSHOOTINGSTAR(
                df_symbol['open'], df_symbol['high'], df_symbol['low'], df_symbol['close'])
            df_symbol['Avg_Volume'] = df_symbol['volume'].rolling(window=20).mean()
            df_symbol['Rel_Volume'] = df_symbol['volume'] / df_symbol['Avg_Volume']
            df_symbol['Volatility'] = df_symbol['ATR']
            df_symbol['Avg_Volatility'] = df_symbol['Volatility'].rolling(window=20).mean()
            df_symbol['Rel_Volatility'] = df_symbol['Volatility'] / df_symbol['Avg_Volatility']
            df_symbol['ADX'] = talib.ADX(
                df_symbol['high'], df_symbol['low'], df_symbol['close'], timeperiod=adx_period)
            df_symbol['Take_Profit_Long'] = df_symbol['close'] + (df_symbol['ATR'] * tp_multiplier)
            df_symbol['Stop_Loss_Long'] = df_symbol['close'] - (df_symbol['ATR'] * sl_multiplier)
            df_symbol['Stop_Loss_Long'] = df_symbol['Stop_Loss_Long'].clip(lower=1e-8)
            df_symbol['Take_Profit_Short'] = df_symbol['close'] - (df_symbol['ATR'] * tp_multiplier)
            df_symbol['Stop_Loss_Short'] = df_symbol['close'] + (df_symbol['ATR'] * sl_multiplier)
            df_symbol['Take_Profit_Short'] = df_symbol['Take_Profit_Short'].clip(lower=1e-8)
            df_symbol = df_symbol.ffill().fillna(0)
            return df_symbol
        except Exception as e:
            print(f"Error al calcular indicadores para {symbol}: {e}")
            return None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        processed = [df for df in executor.map(_process_symbol, symbols) if df is not None]
    data = pd.concat(processed, ignore_index=True) if processed else pd.DataFrame()

    data['Trend_Up'] = (data['EMA_Short'] > data['EMA_Long']).astype('bool')
    data['Trend_Down'] = (data['EMA_Short'] < data['EMA_Long']).astype('bool')

    data['Hammer'] = data['Hammer'].fillna(0)
    data['ShootingStar'] = data['ShootingStar'].fillna(0)
    data['MACD_Bullish'] = data['MACD_Bullish'].fillna(False).astype('bool')
    data['MACD_Bearish'] = data['MACD_Bearish'].fillna(False).astype('bool')
    data['ADX'] = data['ADX'].fillna(0)

    data['Low_Volume'] = (data['volume'] / data['volume'].rolling(20).mean()) < volume_threshold
    data['High_Volatility'] = (data['ATR'] / data['ATR'].rolling(20).mean()) > volatility_threshold
    data['Low_Volume'] = data['Low_Volume'].astype('bool')
    data['High_Volatility'] = data['High_Volatility'].astype('bool')

    data['EMA_Long_Term'] = talib.EMA(data['close'], timeperiod=50)
    data['Trend_Up_Long_Term'] = (data['EMA_Short'] > data['EMA_Long_Term']).astype('bool')
    data['Trend_Down_Long_Term'] = (data['EMA_Short'] < data['EMA_Long_Term']).astype('bool')

    # Reintegrar filtro de ruido: descartar sólo si ambas condiciones coinciden
    signal_long = (
        ((data['Hammer'] != 0) & data['Trend_Up'] & data['Trend_Up_Long_Term'] & (data['RSI'] < rsi_oversold)) |
        (data['MACD_Bullish'] & (data['ADX'] > 25) & data['Trend_Up_Long_Term'] & (data['RSI'] < rsi_overbought))
    )
    data['Long_Signal'] = (signal_long & ~(data['Low_Volume'] & data['High_Volatility'])).astype('bool')

    # Reintegrar filtro de ruido: descartar sólo si ambas condiciones coinciden
    signal_short = (
        ((data['ShootingStar'] != 0) & data['Trend_Down'] & data['Trend_Down_Long_Term'] & (data['RSI'] > rsi_overbought)) |
        (data['MACD_Bearish'] & (data['ADX'] > 25) & data['Trend_Down_Long_Term'] & (data['RSI'] > rsi_oversold))
    )
    data['Short_Signal'] = (signal_short & ~(data['Low_Volume'] & data['High_Volatility'])).astype('bool')

    # DEBUG: conteo de señales crudas y filtradas
    orig_long = int(signal_long.sum())
    orig_short = int(signal_short.sum())
    filt_long = int(data['Long_Signal'].sum())
    filt_short = int(data['Short_Signal'].sum())
    print(f"DEBUG INDICATORS: Señales crudas LONG={orig_long}, SHORT={orig_short}; filtradas LONG={filt_long}, SHORT={filt_short}")

    data = data.sort_values(by=['symbol', 'date']).reset_index(drop=True)
    # Optimización de memoria
    for col in data.select_dtypes(include=['float64']).columns:
        data[col] = pd.to_numeric(data[col], downcast='float')
    for col in data.select_dtypes(include=['int64']).columns:
        data[col] = pd.to_numeric(data[col], downcast='integer')
    data.drop(columns=[c for c in ['ATR','OBV','OBV_Slope','Avg_Volume','Rel_Volume','Volatility','Avg_Volatility','Rel_Volatility'] if c in data.columns], inplace=True)
    data.to_csv(output_filepath, index=False)
    return data

def ema_alert(currencie, data_path='./archivos/cripto_price.csv'):
    try:
        if currencie in DISABLED_COINS:
            return None, None
        crypto_data = load_data(data_path)
        if crypto_data is None:
            return None, None
        crypto_data = filter_duplicates(crypto_data)
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
        df_filtered = cruce_emas[cruce_emas['symbol'] == currencie].copy()
        if df_filtered.empty:
            return None, None
        df_filtered.sort_values(by='date', inplace=True)
        signals = df_filtered[df_filtered['Long_Signal'] | df_filtered['Short_Signal']]
        if not signals.empty:
            last_signal = signals.iloc[-1]
            sig_type = 'LONG' if last_signal['Long_Signal'] else 'SHORT'
            alert_msg = f"=== Alerta de {sig_type} en {last_signal['date']} (precio: {last_signal['close']}) ==="
            return last_signal['close'], alert_msg
        else:
            return None, None
    except Exception as e:
        print(f"Error en ema_alert: {e}")
        return None, None