import pandas as pd
import numpy as np

def indicator():
    # Cargar los datos
    crypto_data = pd.read_csv('./archivos/cripto_price.csv', parse_dates=['date'])
    crypto_data.sort_values(by='date', inplace=True)

    # Agrupar los datos en velas de 30 minutos
    half_hour_data = crypto_data.groupby(['symbol', pd.Grouper(key='date', freq='30min')]).agg(
        open_price=('price', 'first'),
        high_price=('price', 'max'),
        low_price=('price', 'min'),
        close_price=('price', 'last'),
        line_count=('price', 'size')
    ).reset_index()

    # Función para calcular el RSI
    def calculate_rsi(data, window=14):
        delta = data.diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=window).mean()
        avg_loss = pd.Series(loss).rolling(window=window).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    # Calcular la SMA de 20 períodos
    half_hour_data['SMA_20'] = half_hour_data.groupby('symbol')['close_price'].transform(lambda x: x.rolling(window=20).mean())

    # Calcular el RSI de 14 períodos
    half_hour_data['RSI_14'] = half_hour_data.groupby('symbol')['close_price'].transform(lambda x: calculate_rsi(x, window=14))

    # Calcular el MACD y su línea de señal
    half_hour_data['EMA_12'] = half_hour_data.groupby('symbol')['close_price'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    half_hour_data['EMA_26'] = half_hour_data.groupby('symbol')['close_price'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    half_hour_data['MACD'] = half_hour_data['EMA_12'] - half_hour_data['EMA_26']
    half_hour_data['MACD_Signal'] = half_hour_data.groupby('symbol')['MACD'].transform(lambda x: x.ewm(span=9, adjust=False).mean())

    # Detectar cruces del MACD con filtro
    def detect_macd_cross(df):
        bullish_cross = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
        bearish_cross = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))
        bullish_cross = bullish_cross & (df['MACD'] > 0)  # Filtro adicional
        bearish_cross = bearish_cross & (df['MACD'] < 0)  # Filtro adicional
        df['Bullish_MACD_Cross'] = bullish_cross
        df['Bearish_MACD_Cross'] = bearish_cross
        return df

    half_hour_data = half_hour_data.groupby('symbol').apply(detect_macd_cross).reset_index(drop=True)

    # Calcular la volatilidad como la desviación estándar de los cambios en el precio de cierre
    half_hour_data['Volatility'] = half_hour_data.groupby('symbol')['close_price'].transform(lambda x: x.pct_change().rolling(window=20).std())

    # Señales de compra y venta
    half_hour_data['Long_Signal'] = (
        (half_hour_data['close_price'] > half_hour_data['SMA_20']) & 
        half_hour_data['Bullish_MACD_Cross'] & 
        (half_hour_data['RSI_14'] < 70) &
        (half_hour_data['Volatility'] > 0.00)
    )
    half_hour_data['Short_Signal'] = (
        (half_hour_data['close_price'] < half_hour_data['SMA_20']) & 
        half_hour_data['Bearish_MACD_Cross'] & 
        (half_hour_data['RSI_14'] > 30) &
        (half_hour_data['Volatility'] > 0.00)
    )

    # Establecer el Take Profit y Stop Loss
    half_hour_data['Take_Profit'] = 3 * half_hour_data['Volatility']
    half_hour_data['Stop_Loss'] = half_hour_data['Volatility']

    # Guardar los datos en el archivo CSV
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
        return price_last, '=== Alerta de LONG ==='
    elif df_filtered['Short_Signal'].iloc[-1]:
        return price_last, '=== Alerta de SHORT ==='
    else:
        return None, None
