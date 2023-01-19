import pandas as pd
import talib



# Importar datos de precios
def emas_indicator():
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
        group["EMA50"] = talib.EMA(group["price"], timeperiod=50)
        group["EMA21"] = talib.EMA(group["price"], timeperiod=21)
        # Guardar los resultados en un diccionario
        results.append(group)
    
    df_results = pd.concat(results)
    cruce_emas = df_results.groupby('symbol').tail(2).reset_index()
    
    return cruce_emas


def ema_alert(currencie):
    cruce_emas = emas_indicator()
    df_filterd = cruce_emas[cruce_emas['symbol'] ==  currencie]
    ema50 = df_filterd['EMA50'].iloc[0]
    ema50_last = df_filterd['EMA50'].iloc[1]
    ema21 = df_filterd['EMA21'].iloc[0]
    ema21_last = df_filterd['EMA21'].iloc[1]
    price = df_filterd['price'].iloc[0]
    price_last = df_filterd['price'].iloc[1]
    
    if ema50_last < ema21_last:
        if ema50 > ema21:
            stop_lose = price_last * 0.99
            profit = price_last * 1.03
            tipo = 'LONG:'
            return price_last, stop_lose, profit, tipo
    elif ema50_last > ema21_last:
        if ema50 < ema21:
            stop_lose = price_last * 1.01
            profit = price_last * 0.97
            tipo = 'SHORT:'
            return price_last, stop_lose, profit, tipo
           
