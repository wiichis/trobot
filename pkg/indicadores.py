import pandas as pd
import numpy as np
#from datetime import datetime
#import talib

def indicator():
    # Cargar los datos
    crypto_data = pd.read_csv('./archivos/cripto_price.csv')
    crypto_data['date'] = pd.to_datetime(crypto_data['date'])
    crypto_data.sort_values(by='date', inplace=True)

    # Agrupar los datos en velas de una hora
    hourly_data = crypto_data.groupby(['symbol', pd.Grouper(key='date', freq='H')]).agg(
        open_price=('price', 'first'),
        high_price=('price', 'max'),
        low_price=('price', 'min'),
        close_price=('price', 'last')
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

    # Identificar señales de Long y Short
    hourly_data['Long_Signal'] = (
        (hourly_data['close_price'] > hourly_data['SMA_20']) & 
        hourly_data['Bullish_MACD_Cross'] & 
        (hourly_data['RSI_20'] < 70)
    )
    hourly_data['Short_Signal'] = (
        (hourly_data['close_price'] < hourly_data['SMA_20']) & 
        hourly_data['Bearish_MACD_Cross'] & 
        (hourly_data['RSI_20'] > 30)
    )

    # Calcular la volatilidad como la desviación estándar de los cambios en el precio de cierre
    hourly_data['Volatility'] = hourly_data.groupby('symbol')['close_price'].transform(lambda x: x.pct_change().rolling(window=20).std())

    # Establecer el Take Profit y Stop Loss
    # Take Profit es 3 veces la volatilidad y Stop Loss es la volatilidad
    # Establecer el Take Profit y Stop Loss
    hourly_data['Take_Profit'] = 3 * hourly_data['Volatility']
    hourly_data['Stop_Loss'] = hourly_data['Volatility']

    hourly_data.to_csv('./archivos/indicadores.csv', index=False)
    return hourly_data

def ema_alert(currencie):
    cruce_emas = indicator()
    df_filtered = cruce_emas[cruce_emas['symbol'] == currencie]
    
    if not df_filtered.empty:
        price_last = df_filtered['close_price'].iloc[-1]

        type = 'LONG' if df_filtered['Long_Signal'].iloc[-1] else ('SHORT' if df_filtered['Short_Signal'].iloc[-1] else None)
        
        if type == 'LONG':
            tipo = '=== Alerta de LONG ==='
            return price_last, tipo
        elif type == 'SHORT':
            tipo = '=== Alerta de SHORT ==='
            return price_last, tipo
        else:
            # No hay señal de Long o Short
            return price_last, 'No hay señal'
    else:
        # No hay datos disponibles para la moneda especificada
        return None, 'No hay datos disponibles'

# Uso de la función:
price, alert_type = ema_alert('BTC-USDT')






































# #Calculo de ATR de Velas
# def calculate_atr(data, period=14):
#     high = data['high']
#     low = data['low']
#     close = data['close']

#     tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
#     atr = tr.rolling(period).mean()

#     return atr

# #Identificar Velas Largas
# def is_long_candle(data, multiplier=1.9):   #Multiplier es el tamaño de las velas
#     candle_range = data['high'] - data['low']
#     atr = calculate_atr(data)
#     return candle_range > multiplier * atr

# def analizar_tendencia_criptomonedas(df, umbral, umbral_alerta):
#     df_2 = df.iloc[-550:].copy()
#     df_2['Diferencia'] = df_2['close'] - df_2['open']
#     df_2['Tendencia'] = 0
#     df_2.loc[df_2['Diferencia'] > umbral, 'Tendencia'] = 1
#     df_2.loc[df_2['Diferencia'] < -umbral, 'Tendencia'] = -1

#     # Agrupar por símbolo y calcular la tendencia acumulada
#     df_2['Tendencia_acumulada'] = df_2.groupby('symbol')['Tendencia'].cumsum()
    
#     # Crear una columna que indica si se cumple la condición de tendencia acumulada mayor a umbral_alerta
#     df_2['Alerta'] = abs(df_2['Tendencia_acumulada']) > umbral_alerta
    
#     # Eliminar columnas temporales
#     df_2.drop(['Diferencia'], axis=1, inplace=True)
    
#     return df_2



# def emas_indicator():
#     # Importar datos de precios
#     df = pd.read_csv('./archivos/cripto_price.csv', parse_dates=["date"])
#     df = df.iloc[-20000:] 
   
#     grouped = df.groupby('symbol')

#     # Calcular el EMA de período 50 y el EMA de período 21 para cada grupo
#     ema50 = grouped['price'].transform(lambda x: talib.EMA(x, timeperiod=50))
#     ema21 = grouped['price'].transform(lambda x: talib.EMA(x, timeperiod=21))
#     df['ema50'] = round(ema50, 4)
#     df['ema21'] = round(ema21, 4)

#     # Calcular el RSI de período 14 para cada grupo
#     rsi = grouped['price'].transform(lambda x: talib.RSI(x, timeperiod=14))
#     df['rsi'] = round(rsi, 4)

#      # Calcular la media móvil exponencial
#     ancho_banda = 5
#     periodo = int(ancho_banda / 2)
#     smoothing = grouped['price'].transform(lambda x: talib.EMA(x, timeperiod=periodo))

#         # Calcular los residuos
#     residuos = grouped['price'].transform(lambda x: x - x.mean())

#         # Calcular el envelope
#     envelope_superior = smoothing + 2 * residuos
#     envelope_inferior = smoothing - 2 * residuos
#     price = df['price']
#     df['envelope_superior'] = round(envelope_superior, 4)
#     df['envelope_inferior'] = round(envelope_inferior,4 )

#         # Creamos una columna que identifica a qué intervalo de 5 minutos pertenece cada fila
#     df["time_bin"] = df["date"].dt.floor('15T')

#     # Agrupamos por símbolo y time_bin y generamos los valores para cada vela
#     grouped_velas = df.groupby(['symbol', 'time_bin'])
#     df = df.join(grouped_velas['price'].agg(open='first', high='max', low='min', close='last'), on=['symbol', 'time_bin'])

#     ohlc = grouped_velas['price'].agg(open='first', high='max', low='min', close='last')
#     ohlc.reset_index(inplace=True)

#      # Calculamos el ATR y si la vela es larga para cada grupo y lo añadimos al DataFrame ohlc
#     ohlc['ATR'] = round(ohlc.groupby('symbol').apply(lambda x: calculate_atr(x)).reset_index(level=0, drop=True),5)
#     ohlc['IsLongCandle'] = ohlc.groupby('symbol').apply(lambda x: is_long_candle(x)).reset_index(level=0, drop=True)

#     # Merge ohlc de vuelta con df para agregar las nuevas columnas calculadas
#     merged_df = pd.merge(df, ohlc, how='left', on=['symbol', 'time_bin'], suffixes=('', '_ohlc'))

#     # Direccion de las velas consecutivas
#     umbral = 0  # Ajusta el umbral según tus necesidades
#     umbral_alerta = 27  # Ajusta el número de velas consecutivas para considerar
#     merged_df = analizar_tendencia_criptomonedas(merged_df, umbral, umbral_alerta)




#     #Calcular la columna 'type' utilizando los valores de EMA y RSI para cada fila
#     merged_df['type'] = 'NONE'
#     merged_df.loc[(ema50 > ema21) & (rsi < 30) & (envelope_inferior >= price) & (merged_df['IsLongCandle'] == False ) & (merged_df['Alerta'] == False ),'type'] = 'LONG'
#     merged_df.loc[(ema50 < ema21) & (rsi > 70) & (envelope_superior <= price) & (merged_df['IsLongCandle'] == False ) & (merged_df['Alerta'] == False ),'type'] = 'SHORT'


            
#     cruce_emas = merged_df.groupby('symbol').tail(40).reset_index()
#     cruce_emas = cruce_emas.sort_values(['symbol', 'date'])
#     cruce_emas.to_csv('./archivos/indicadores.csv', index=False)

#     return cruce_emas

# def ema_alert(currencie):
#     cruce_emas = emas_indicator()
#     df_filterd = cruce_emas[cruce_emas['symbol'] ==  currencie]
#     price_last = df_filterd['price'].iloc[-1]
#     type = df_filterd['type'].iloc[-1]
       
#     if type == 'LONG':
#         tipo = '=== Alerta de LONG ==='
#         return price_last, tipo
#     elif type == 'SHORT':
#         tipo = '=== Alerta de SHORT ==='
#         return price_last, tipo
#     else:
#         pass

