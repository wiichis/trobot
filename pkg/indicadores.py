#
# ========== PARÁMETROS PRINCIPALES ==========
# Velas 30 minutos
RSI_PERIOD = 14
ATR_PERIOD = 14
EMA_SHORT_PERIOD = 50
EMA_LONG_PERIOD = 200
ADX_PERIOD = 14

TP_MULTIPLIER = 2.0   # ATR × 2 take‑profit
SL_MULTIPLIER = 1.0   # ATR × 1 stop‑loss

# Umbrales heredados (aún usados en validaciones y confirmaciones)
VOLUME_THRESHOLD = 1.0468
VOLATILITY_THRESHOLD = 1.2649
RSI_OVERSOLD = 42
RSI_OVERBOUGHT = 60



# Bollinger Bands
BB_PERIOD = 20
BB_STD = 2

# Velas 5 minutos
FIVE_MIN_DATA_PATH = './archivos/cripto_price_5m.csv'
FIVE_MIN_EMA_SHORT = 9
FIVE_MIN_EMA_LONG = 21
FIVE_MIN_RSI = 14
FIVE_MIN_MIN_CONFIRM = 1

import pandas as pd
import numpy as np
import talib  # Asegúrate de que 'ta-lib' esté correctamente instalado
import concurrent.futures
import os  # Permite activar el modo de simulación de señales

DISABLED_COINS = []

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
    rsi_overbought=RSI_OVERBOUGHT,
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

            # Bollinger Bands
            df_symbol['BB_upper'], df_symbol['BB_middle'], df_symbol['BB_lower'] = talib.BBANDS(
                df_symbol['close'], timeperiod=BB_PERIOD, nbdevup=BB_STD, nbdevdn=BB_STD, matype=0)

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

            # Señales balanceadas (se requiere al menos 3 de 4 condiciones)
            long_cond_trend = df_symbol['EMA_Short'] > df_symbol['EMA_Long']
            long_cond_macd  = df_symbol['MACD_Hist'] > 0
            long_cond_rsi   = df_symbol['RSI'] > 45
            long_cond_adx   = df_symbol['ADX'] > 20
            conditions_long = (
                long_cond_trend,
                long_cond_macd,
                long_cond_rsi,
                long_cond_adx
            )
            df_symbol['Long_Signal'] = (np.sum(conditions_long, axis=0) >= 4)

            short_cond_trend = df_symbol['EMA_Short'] < df_symbol['EMA_Long']
            short_cond_macd  = df_symbol['MACD_Hist'] < 0
            short_cond_rsi   = df_symbol['RSI'] < 55
            short_cond_adx   = df_symbol['ADX'] > 20
            conditions_short = (
                short_cond_trend,
                short_cond_macd,
                short_cond_rsi,
                short_cond_adx
            )
            df_symbol['Short_Signal'] = (np.sum(conditions_short, axis=0) >= 4)
            return df_symbol
        except Exception as e:
            print(f"Error al calcular indicadores para {symbol}: {e}")
            return None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        processed = [df for df in executor.map(_process_symbol, symbols) if df is not None]
    data = pd.concat(processed, ignore_index=True) if processed else pd.DataFrame()

    data = data.sort_values(by=['symbol', 'date']).reset_index(drop=True)

    # Confirmación con velas de 5 minutos
    data = get_5m_confirmation(data)

    # Optimización de memoria
    for col in data.select_dtypes(include=['float64']).columns:
        data[col] = pd.to_numeric(data[col], downcast='float')
    for col in data.select_dtypes(include=['int64']).columns:
        data[col] = pd.to_numeric(data[col], downcast='integer')

    # Eliminar columnas intermedias no necesarias para producción
    drop_cols = ['OBV', 'OBV_Slope', 'Avg_Volume', 'Rel_Volume', 'Volatility',
                 'Avg_Volatility', 'Rel_Volatility',
                 'Hammer', 'ShootingStar', 'MACD_Bullish', 'MACD_Bearish']
    data.drop(columns=[c for c in drop_cols if c in data.columns], inplace=True)

    data.to_csv(output_filepath, index=False)
    return data


# Confirmación de señales usando velas de 5 minutos
def get_5m_confirmation(df_30m,
                        five_min_data_path=FIVE_MIN_DATA_PATH,
                        ema_short=FIVE_MIN_EMA_SHORT,
                        ema_long=FIVE_MIN_EMA_LONG,
                        rsi_5m=FIVE_MIN_RSI,
                        min_confirm_5m=FIVE_MIN_MIN_CONFIRM):
    """
    Para cada señal Long/Short en df_30m, busca confirmación en las 5 velas completas de 5m siguientes.
    Solo mantiene la señal si hay al menos 1 de 5 velas con cruce EMA + RSI 40/60 + volumen relativo > 1.1
    """
    try:
        df_5m = pd.read_csv(five_min_data_path, parse_dates=['date'])
        df_5m.sort_values(by=['symbol', 'date'], inplace=True)
    except Exception as e:
        print(f"Error cargando archivo de 5m: {e}")
        # Si no se puede cargar, desactiva todas las señales
        df_30m['Long_Signal'] = False
        df_30m['Short_Signal'] = False
        return df_30m

    # Calcula EMAs en 5m si no existen
    if f'EMA_{ema_short}' not in df_5m.columns or f'EMA_{ema_long}' not in df_5m.columns:
        df_5m[f'EMA_{ema_short}'] = talib.EMA(df_5m['close'], timeperiod=ema_short)
        df_5m[f'EMA_{ema_long}'] = talib.EMA(df_5m['close'], timeperiod=ema_long)

    # Calcula RSI en 5m si no existe
    if 'RSI' not in df_5m.columns:
        df_5m['RSI'] = talib.RSI(df_5m['close'], timeperiod=rsi_5m)

    # Calcula volumen relativo en 5 m (media móvil de 20 velas)
    if 'Rel_Volume' not in df_5m.columns:
        df_5m['Avg_Volume'] = df_5m['volume'].rolling(window=20).mean()
        df_5m['Rel_Volume'] = df_5m['volume'] / df_5m['Avg_Volume']

    # Para cada símbolo, procesa señales
    df_30m['Long_Signal_5m_confirm'] = False
    df_30m['Short_Signal_5m_confirm'] = False
    for symbol in df_30m['symbol'].unique():
        df_30m_sym = df_30m[df_30m['symbol'] == symbol]
        df_5m_sym = df_5m[df_5m['symbol'] == symbol]
        if df_5m_sym.empty:
            continue
        # Indexar por fecha para eficiencia
        df_5m_sym = df_5m_sym.set_index('date')
        # Asegurar que el índice sea Timestamp para evitar comparaciones str > Timestamp
        if df_5m_sym.index.dtype == 'object':
            df_5m_sym.index = pd.to_datetime(df_5m_sym.index, errors='coerce')
        for idx, row in df_30m_sym.iterrows():
            dt_30m = row['date']
            # Asegurar que dt_30m sea Timestamp para evitar comparaciones str > Timestamp
            if isinstance(dt_30m, str):
                dt_30m = pd.to_datetime(dt_30m, errors='coerce')
            # Solo señales activas
            if not (row.get('Long_Signal', False) or row.get('Short_Signal', False)):
                continue
            # Busca las 5 velas de 5m posteriores a la fecha de la vela de 30m (solo velas completas)
            mask = (df_5m_sym.index > dt_30m)
            df_next_5 = df_5m_sym[mask].head(5)
            if len(df_next_5) < 5:
                continue  # no hay suficientes velas para confirmar
            ema_s = df_next_5[f'EMA_{ema_short}']
            ema_l = df_next_5[f'EMA_{ema_long}']
            rsi_vals = df_next_5['RSI']

            # Para Long: contar velas con cruce alcista, RSI > 40 y volumen relativo > 1.1
            if row.get('Long_Signal', False):
                cross_mask = (ema_s > ema_l)
                rsi_mask   = (rsi_vals > 40)
                vol_mask   = (df_next_5['Rel_Volume'] > 1.1)
                valid_count = (cross_mask & rsi_mask & vol_mask).sum()
                if valid_count >= min_confirm_5m:
                    df_30m.at[idx, 'Long_Signal_5m_confirm'] = True

            # Para Short: contar velas con cruce bajista, RSI > 60 y volumen relativo > 1.1
            if row.get('Short_Signal', False):
                cross_mask = (ema_s < ema_l)
                rsi_mask   = (rsi_vals > 60)
                vol_mask   = (df_next_5['Rel_Volume'] > 1.1)
                valid_count = (cross_mask & rsi_mask & vol_mask).sum()
                if valid_count >= min_confirm_5m:
                    df_30m.at[idx, 'Short_Signal_5m_confirm'] = True

    # Solo activa la señal si también hay confirmación en 5m
    df_30m['Long_Signal'] = df_30m['Long_Signal'] & df_30m['Long_Signal_5m_confirm']
    df_30m['Short_Signal'] = df_30m['Short_Signal'] & df_30m['Short_Signal_5m_confirm']
    # Limpia columnas auxiliares
    df_30m.drop(['Long_Signal_5m_confirm', 'Short_Signal_5m_confirm'], axis=1, inplace=True)
    return df_30m

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
        last_n = 1  # Solo analizamos la última vela cerrada
        recent_rows = df_filtered.tail(last_n)
        recent_signals = recent_rows[recent_rows['Long_Signal'] | recent_rows['Short_Signal']]
        if not recent_signals.empty:
            last_signal = recent_signals.iloc[-1]
            sig_type = 'LONG' if last_signal['Long_Signal'] else 'SHORT'
            alert_msg = f"=== Alerta de {sig_type} en {last_signal['date']} (precio: {last_signal['close']}) ==="
            return last_signal['close'], alert_msg
        else:
            return None, None
    except Exception as e:
        print(f"Error en ema_alert: {e}")
        return None, None