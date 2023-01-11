import pandas as pd
import numpy as np



# Importar datos de precios
df = pd.read_csv('./archivos/cripto_price.csv')

grouped = df.groupby("symbol")


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
df_results.to_csv('./archivos/indicadores.csv', index=False)

cruce_emas = df_results.groupby('symbol').agg({'EMA50': 'last', 'EMA21': 'last'})

for symbol, row in cruce_emas.iterrows():
    if row['EMA50'] < row['EMA21']:
        print(f"Alerta de entrada para {symbol} EMA21 acaba de pasar a EMA 50 {row['EMA50']} > {row['EMA21']}")
