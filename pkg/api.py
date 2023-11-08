import json
import pkg
import requests
import pandas as pd
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

def currencies_list():
    return ['ADA-USDT','ETH-USDT','BTC-USDT','SOL-USDT','TRX-USDT','CFX-USDT','DOGE-USDT','XRP-USDT','BNB-USDT','FIL-USDT']
    # No usar DYDX, OP, LDO, 

def fetch_price(symbol):
    try:
        json_str = pkg.bingx.last_price_trading_par(symbol)
        data = json.loads(json_str)
        return {
            'symbol': data["data"]["symbol"],
            'price': data["data"]["price"],
            'date': datetime.now()
        }
    except Exception as e:
        print(f"Error al obtener el precio para {symbol}: {e}")
        return None

def price_bingx():
    try:
        currencies = currencies_list()
        data_list = []

        with ThreadPoolExecutor() as executor:
            results = executor.map(fetch_price, currencies)

        for result in results:
            if result is not None:
                data_list.append(result)
        
        new_df = pd.DataFrame(data_list)

        try:
            existing_df = pd.read_csv('./archivos/cripto_price.csv')
        except FileNotFoundError:
            existing_df = pd.DataFrame(columns=['symbol', 'price', 'date'])

        concatenated_df = pd.concat([existing_df, new_df], ignore_index=True)
        df_month = concatenated_df.iloc[-30000:]
        df_month.to_csv('./archivos/cripto_price.csv', index=False)
        
    except requests.exceptions.ConnectionError as e:
        print(f"Error de conexión: {e}")
        print("Reintentando la solicitud en 1 minuto...")
        time.sleep(30)
        price_bingx()
    except Exception as e:
        print(f"Ocurrió un error: {e}")





