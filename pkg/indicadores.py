import pandas as pd
import talib



def emas_indicator():
    # Importar datos de precios
    df = pd.read_csv('./archivos/cripto_price.csv')
    df_month = df.iloc[-190000:] #Un Mes
    
    #Convertir date en un datetime
    df_month['date'] = pd.to_datetime(df_month['date'])

    # Establecer la columna 'date' como el índice del DataFrame
    df_month.set_index('date', inplace=True)
    grouped = df_month.groupby("symbol")

    results = []

    for symbol, group in grouped:
        # Calcular el EMA de período 50 y el EMA de período 21
        group['EMA50'] = talib.EMA(group['price'], timeperiod=50)
        group['EMA21'] = talib.EMA(group['price'], timeperiod=21)
        group['RSI'] = talib.RSI(group['price'], timeperiod=14)
        # Guardar los resultados en un diccionario
        results.append(group)
            
    df_results = pd.concat(results)
    cruce_emas = df_results.groupby('symbol').tail(10).reset_index()
           
    return cruce_emas

def ema_alert(currencie):
    cruce_emas = emas_indicator()
    df_filterd = cruce_emas[cruce_emas['symbol'] ==  currencie]
    # ema50 = df_filterd['EMA50'].iloc[0]
    # ema50_last = df_filterd['EMA50'].iloc[1]
    # ema21 = df_filterd['EMA21'].iloc[0]
    # ema21_last = df_filterd['EMA21'].iloc[1]
    price_last = df_filterd['price'].iloc[-1]
    rsi_last = df_filterd['RSI'].iloc[-1]
    ema50_mean = df_filterd['EMA50'].mean()
    ema21_mean = df_filterd['EMA21'].mean()
       
    if ema50_mean < ema21_mean:
        #if ema50 > ema21:
            if rsi_last < 30:
                stop_lose = price_last * 0.99
                profit = price_last * 1.03
                tipo = '=== Alerta de LONG ==='
                return price_last, stop_lose, profit, tipo
    elif ema50_mean > ema21_mean:
        #if ema50 < ema21:
            if rsi_last > 70:
                stop_lose = price_last * 1.01
                profit = price_last * 0.97
                tipo = '=== Alerta de SHORT ==='
                return price_last, stop_lose, profit, tipo
           

