import json
import pkg
import requests
import pandas as pd
import time
from datetime import datetime


def currencies_list():
  currencies = ['MASK-USDT','MATIC-USDT','DYDX-USDT','ETH-USDT','BTC-USDT','BNB-USDT','DOGE-USDT',
                'YFI-USDT','SOL-USDT','TRX-USDT','CFX-USDT','OP-USDT','XRP-USDT','LDO-USDT','FIL-USDT']
  return currencies

def price_bingx():
    try:
        currencies = currencies_list()
        data_list = []
        for currencie in currencies:
            json_str = pkg.bingx.last_price_trading_par(currencie)

            # Analiza la cadena JSON
            data = json.loads(json_str)

            # Accede a los valores del objeto JSON
            symbol = data["data"]["symbol"]
            price = data["data"]["price"]
            date = datetime.now()

            # Agrega los valores a una lista de datos
            data_list.append({'symbol': symbol, 'price': price, 'date': date})
        
        # Crea un DataFrame a partir de la lista de datos
        df = pd.DataFrame(data_list)

        # Guarda el DataFrame en un archivo CSV
        df_file = pd.read_csv('./archivos/cripto_price.csv')
        df_new = pd.concat([df_file, df], ignore_index=True)
        df_month = df_new.iloc[-70000:]
        df_month.to_csv('./archivos/cripto_price.csv', index=False)
        
    except requests.exceptions.ConnectionError as e:
        print("Error de conexión: ", e)
        print("Reintentando la solicitud en 1 minuto...")
        # Espera 1 minuto antes de volver a ejecutar la función
        time.sleep(60)
        price_bingx()
    except Exception as e:
        print("Ocurrió un error: ", e)
        # Realiza alguna acción adicional para manejar otros tipos de errores





