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
FIVE_MIN_MIN_CONFIRM = 3

import pandas as pd
import numpy as np
import talib  # Asegúrate de que 'ta-lib' esté correctamente instalado
import concurrent.futures
import os  # Permite activar el modo de simulación de señales
from functools import lru_cache  # NUEVO: cache de lectura CSV

DISABLED_COINS = []

# =========  utilidades de carga con caché =========
@lru_cache(maxsize=8)
def _cached_read_csv(path: str):
    """Lee un CSV y lo deja en caché; limpiar con _cached_read_csv.cache_clear()"""
    df = pd.read_csv(path, parse_dates=['date'])
    # ── Garantizar que 'date' sea datetime ─────────────────────────────────
    if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
        try:
            df['date'] = pd.to_datetime(df['date'])
        except Exception:
            pass  # Si falla dejamos el valor original
    # ──────────────────────────────────────────────────────────────────────
    return df

def load_data(filepath='./archivos/cripto_price.csv'):
    try:
        crypto_data = _cached_read_csv(filepath).copy()
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


# Confirmación de señales usando velas de 5 minutos (vectorizado + caché)
def get_5m_confirmation(df_30m,
                        five_min_data_path=FIVE_MIN_DATA_PATH,
                        ema_short=FIVE_MIN_EMA_SHORT,
                        ema_long=FIVE_MIN_EMA_LONG,
                        rsi_5m=FIVE_MIN_RSI,
                        min_confirm_5m=FIVE_MIN_MIN_CONFIRM):
    """
    Proyecta cada vela 30 m a sus 5 velas posteriores de 5 m y confirma la señal
    sin bucles por fila (más rápido). Requiere ≥ min_confirm_5m velas válidas.
    """
    try:
        df_5m = _cached_read_csv(five_min_data_path).copy()
    except Exception as e:
        print(f"Error cargando archivo 5m: {e}")
        df_30m['Long_Signal'] = False
        df_30m['Short_Signal'] = False
        return df_30m

    # Orden y cálculos técnicos
    df_5m.sort_values(['symbol', 'date'], inplace=True)
    for col, p in {f'EMA_{ema_short}': ema_short, f'EMA_{ema_long}': ema_long}.items():
        if col not in df_5m.columns:
            df_5m[col] = talib.EMA(df_5m['close'], timeperiod=p)
    if 'RSI' not in df_5m.columns:
        df_5m['RSI'] = talib.RSI(df_5m['close'], timeperiod=rsi_5m)
    if 'Rel_Volume' not in df_5m.columns:
        df_5m['Rel_Volume'] = df_5m['volume'] / df_5m['volume'].rolling(20).mean()

    # Ancla cada vela 5 m a su vela 30 m
    df_5m['anchor'] = df_5m['date'].dt.floor('30T')
    df_5m['rank'] = df_5m.groupby(['symbol', 'anchor']).cumcount() + 1  # 1 a 6

    m_long = (
        (df_5m[f'EMA_{ema_short}'] > df_5m[f'EMA_{ema_long}']) &
        (df_5m['RSI'] > 40) &
        (df_5m['Rel_Volume'] > 1.0) &
        (df_5m['rank'] <= 5)
    )
    m_short = (
        (df_5m[f'EMA_{ema_short}'] < df_5m[f'EMA_{ema_long}']) &
        (df_5m['RSI'] > 60) &
        (df_5m['Rel_Volume'] > 1.0) &
        (df_5m['rank'] <= 5)
    )

    cnt_long = df_5m[m_long].groupby(['symbol', 'anchor']).size().rename('cnt_long')
    cnt_short = df_5m[m_short].groupby(['symbol', 'anchor']).size().rename('cnt_short')

    df_30m = df_30m.merge(cnt_long, left_on=['symbol', 'date'],
                          right_index=True, how='left')
    df_30m = df_30m.merge(cnt_short, left_on=['symbol', 'date'],
                          right_index=True, how='left')
    df_30m[['cnt_long', 'cnt_short']] = df_30m[['cnt_long', 'cnt_short']].fillna(0)

    df_30m['Long_Signal'] &= df_30m['cnt_long'] >= min_confirm_5m
    df_30m['Short_Signal'] &= df_30m['cnt_short'] >= min_confirm_5m
    return df_30m.drop(columns=['cnt_long', 'cnt_short'])

def ema_alert(symbol, csv_path='./archivos/indicadores.csv'):
    """
    Devuelve alerta si la última vela del símbolo en el archivo de indicadores
    tiene señal Long o Short.  Lee el CSV procesado y evita recalcular todo.
    """
    try:
        # Evita alertas para pares temporalmente deshabilitados
        if symbol in DISABLED_COINS:
            return None, None

        # Carga el CSV (con parse_dates en _cached_read_csv)
        df = _cached_read_csv(csv_path).copy()
        df_symbol = df[df['symbol'] == symbol].copy()  # evita SettingWithCopyWarning

        if df_symbol.empty:
            return None, None

        # Toma la vela más reciente
        df_symbol = df_symbol.sort_values('date')
        last = df_symbol.iloc[-1]

        if last['Long_Signal'] or last['Short_Signal']:
            sig_type = 'LONG' if last['Long_Signal'] else 'SHORT'
            price = last['close']
            date = last['date']
            alert_msg = f"=== Alerta de {sig_type} en {date} (precio: {price}) ==="
            return price, alert_msg

        return None, None
    except Exception as e:
        print(f"Error en ema_alert: {e}")
        return None, None