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
    cruce_emas = df_results.groupby('symbol').agg({'EMA50': 'last', 'EMA21': 'last', 'price': 'last'}).reset_index()
    
    return cruce_emas


def entry_alert(currencie):
    cruce_emas = emas_indicator()
    df_filterd = cruce_emas[cruce_emas['symbol'] ==  currencie]
    ema50 = df_filterd['EMA50'].values
    ema21 = df_filterd['EMA21'].values
    price = df_filterd['price'].values
    
    if ema50 < ema21:
        return True, price
    else:
        return False, price
  

 
# prueba = emas_indicator()
# prueba1 = entry_alert('BNB')


# print(prueba)
# print(prueba1)