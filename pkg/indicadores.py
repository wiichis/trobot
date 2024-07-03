import pandas as pd
import numpy as np

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

def bollinger_bands(data, window=20):
    sma = data.rolling(window).mean()
    std = data.rolling(window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band, lower_band

def indicator():
    crypto_data = pd.read_csv('./archivos/cripto_price.csv')
    crypto_data['date'] = pd.to_datetime(crypto_data['date'])
    crypto_data.sort_values(by='date', inplace=True)

    half_hour_data = crypto_data.groupby(['symbol', pd.Grouper(key='date', freq='30T')]).agg(
        open_price=('price', 'first'),
        high_price=('price', 'max'),
        low_price=('price', 'min'),
        close_price=('price', 'last'),
        line_count=('price', 'size')
    ).reset_index()

    half_hour_data['SMA_50'] = half_hour_data.groupby('symbol')['close_price'].transform(lambda x: x.rolling(window=50).mean())
    half_hour_data['EMA_50'] = half_hour_data.groupby('symbol')['close_price'].transform(lambda x: x.ewm(span=50, adjust=False).mean())
    half_hour_data['EMA_200'] = half_hour_data.groupby('symbol')['close_price'].transform(lambda x: x.ewm(span=200, adjust=False).mean())

    rsi_results = half_hour_data.groupby('symbol', group_keys=True)['close_price'].apply(lambda x: calculate_rsi(x, window=30)).reset_index(level=0, drop=True)
    half_hour_data['RSI_30'] = rsi_results

    ema_12 = half_hour_data.groupby('symbol')['close_price'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema_26 = half_hour_data.groupby('symbol')['close_price'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    half_hour_data['MACD'] = ema_12 - ema_26
    half_hour_data['MACD_Signal'] = half_hour_data.groupby('symbol')['MACD'].transform(lambda x: x.ewm(span=9, adjust=False).mean())

    half_hour_data['Upper_Band'], half_hour_data['Lower_Band'] = zip(*half_hour_data.groupby('symbol')['close_price'].apply(lambda x: bollinger_bands(x, window=20)))
    half_hour_data['Upper_Band'] = half_hour_data['Upper_Band'].reset_index(level=0, drop=True)
    half_hour_data['Lower_Band'] = half_hour_data['Lower_Band'].reset_index(level=0, drop=True)

    def detect_macd_cross(df):
        df = df[df['line_count'] > 5]

        bullish_cross = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
        bearish_cross = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))
        return bullish_cross, bearish_cross

    half_hour_data['Bullish_MACD_Cross'] = False
    half_hour_data['Bearish_MACD_Cross'] = False
    for symbol, group in half_hour_data.groupby('symbol'):
        bullish_cross, bearish_cross = detect_macd_cross(group)
        half_hour_data.loc[group.index, 'Bullish_MACD_Cross'] = bullish_cross
        half_hour_data.loc[group.index, 'Bearish_MACD_Cross'] = bearish_cross

    half_hour_data['Volatility'] = half_hour_data.groupby('symbol')['close_price'].transform(lambda x: x.pct_change().rolling(window=20).std())

    # Calcular el percentil 75 de la volatilidad para cada moneda
    volatility_thresholds = half_hour_data.groupby('symbol')['Volatility'].quantile(0.75)

    # Calcular indicadores en el marco de 4 horas
    four_hour_data = crypto_data.groupby(['symbol', pd.Grouper(key='date', freq='4H')]).agg(
        close_price=('price', 'last')
    ).reset_index()

    four_hour_data['SMA_50'] = four_hour_data.groupby('symbol')['close_price'].transform(lambda x: x.rolling(window=50).mean())
    four_hour_data['EMA_50'] = four_hour_data.groupby('symbol')['close_price'].transform(lambda x: x.ewm(span=50, adjust=False).mean())
    four_hour_data['EMA_200'] = four_hour_data.groupby('symbol')['close_price'].transform(lambda x: x.ewm(span=200, adjust=False).mean())

    # Filtrar señales con confirmación múltiple y confluencia de indicadores
    half_hour_data['Long_Signal'] = (
        (half_hour_data['close_price'] > half_hour_data['SMA_50']) & 
        half_hour_data['Bullish_MACD_Cross'] & 
        (half_hour_data['RSI_30'] < 70) &
        (half_hour_data['Volatility'] > half_hour_data['symbol'].map(volatility_thresholds)) &
        (half_hour_data['EMA_50'] > half_hour_data['EMA_200']) &
        (half_hour_data['close_price'] < half_hour_data['Upper_Band']) &
        (four_hour_data['EMA_50'].iloc[-1] > four_hour_data['EMA_200'].iloc[-1])
    )
    half_hour_data['Short_Signal'] = (
        (half_hour_data['close_price'] < half_hour_data['SMA_50']) & 
        half_hour_data['Bearish_MACD_Cross'] & 
        (half_hour_data['RSI_30'] > 30) &
        (half_hour_data['Volatility'] > half_hour_data['symbol'].map(volatility_thresholds)) &
        (half_hour_data['EMA_50'] < half_hour_data['EMA_200']) &
        (half_hour_data['close_price'] > half_hour_data['Lower_Band']) &
        (four_hour_data['EMA_50'].iloc[-1] < four_hour_data['EMA_200'].iloc[-1])
    )

    half_hour_data['Take_Profit'] = 3 * half_hour_data['Volatility']
    half_hour_data['Stop_Loss'] = half_hour_data['Volatility']

    half_hour_data.to_csv('./archivos/indicadores.csv', index=False)
    return half_hour_data

def ema_alert(currencie):
    cruce_emas = indicator()
    df_filtered = cruce_emas[cruce_emas['symbol'] == currencie]
    
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
        return None, None
