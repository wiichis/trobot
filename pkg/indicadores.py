import pandas as pd
import numpy as np



# Importar datos de precios
def emas_indicator():
    df = pd.read_csv('./archivos/cripto_price.csv')
    df_month = df.iloc[-190000:] #Un Mes
    grouped = df_month.groupby("symbol")

    results = []
    for symbol, group in grouped:
        #Utiliza el método rolling() para calcular la media móvil exponencial de 50 períodos:
        group["EMA50"] = group["price"].rolling(window=50).mean()

        #Utiliza el método rolling() para calcular la media móvil exponencial de 21 períodos:
        group["EMA21"] = group["price"].rolling(window=21).mean()

        group['Upper'] = group['EMA21'] + (group['EMA21'] - group['EMA50'])
        group['Lower'] = group['EMA21'] - (group['EMA21'] - group['EMA50'])

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
            tipo = 'Alerta de Entrada:'
            print(stop_lose, profit, tipo)
            return stop_lose, profit, tipo
    elif ema50_last > ema21_last:
        if ema50 < ema21:
            stop_lose = price_last * 1.01
            profit = price_last * 0.97
            tipo = 'Alerta de Short:'
            print(stop_lose, profit, tipo)
            return stop_lose, profit, tipo
           
#prueba = emas_indicator()
prueba1 = ema_alert('XLM')

#print(prueba)
print(prueba1)