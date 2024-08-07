import json
import pkg
import requests
import pandas as pd
import time
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def currencies_list():
    return ['ADA-USDT', 'AVAX-USDT', 'CFX-USDT', 'DOT-USDT', 'MATIC-USDT', 'SHIB-USDT', 'NEAR-USDT', 'LTC-USDT', 'APT-USDT', 'HBAR-USDT']

def round_time(dt=None, round_to=1800):
    if dt is None: 
        dt = datetime.now()
    seconds = (dt - dt.min).seconds
    rounding = (seconds + round_to / 2) // round_to * round_to
    return dt + timedelta(0, rounding - seconds, -dt.microsecond)

def fetch_candle(symbol, interval='30m'):
    try:
        json_str = pkg.bingx.get_candle(symbol, interval)
        data = json.loads(json_str)
        last_candle = data["data"][0]  # La última vela (incompleta)
        prev_candle = data["data"][1]  # La penúltima vela (completa)

        last_candle_data = {
            'symbol': symbol,
            'open': last_candle["open"],
            'high': last_candle["high"],
            'low': last_candle["low"],
            'close': last_candle["close"],
            'volume': last_candle["volume"],
            'date': round_time(datetime.fromtimestamp(last_candle["closeTime"] / 1000))
        }

        prev_candle_data = {
            'symbol': symbol,
            'open': prev_candle["open"],
            'high': prev_candle["high"],
            'low': prev_candle["low"],
            'close': prev_candle["close"],
            'volume': prev_candle["volume"],
            'date': round_time(datetime.fromtimestamp(prev_candle["closeTime"] / 1000))
        }

        return last_candle_data, prev_candle_data
    except Exception as e:
        logging.error(f"Error al obtener la vela para {symbol}: {e}")
        return None, None

def update_dataframe(existing_df, new_data):
    new_df = pd.DataFrame(new_data)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # Eliminar duplicados basados en 'symbol' y 'date', manteniendo la última entrada con datos completos
    combined_df = combined_df.drop_duplicates(subset=['symbol', 'date'], keep='last')
    
    return combined_df

def price_bingx():
    try:
        currencies = currencies_list()
        candle_data = []

        with ThreadPoolExecutor() as executor:
            results = executor.map(fetch_candle, currencies)

        for last_candle, prev_candle in results:
            if last_candle is not None:
                candle_data.append(last_candle)
            if prev_candle is not None:
                candle_data.append(prev_candle)

        try:
            existing_df = pd.read_csv('./archivos/cripto_price.csv')
        except FileNotFoundError:
            existing_df = pd.DataFrame(columns=['symbol', 'open', 'high', 'low', 'close', 'volume', 'date'])

        updated_df = update_dataframe(existing_df, candle_data)

        # Ordenar el DataFrame según currencies_list y fecha
        updated_df['date'] = pd.to_datetime(updated_df['date'])
        updated_df['symbol'] = pd.Categorical(updated_df['symbol'], categories=currencies_list(), ordered=True)
        updated_df.sort_values(['symbol', 'date'], inplace=True)
        updated_df.reset_index(drop=True, inplace=True)

        # Calcular el número máximo de registros para 30 días
        max_records = 14400
        # Guardar los últimos 14400 registros
        df_month = updated_df.iloc[-max_records:]
        df_month.to_csv('./archivos/cripto_price.csv', index=False)
        
    except requests.exceptions.ConnectionError as e:
        logging.error(f"Error de conexión: {e}")
        logging.info("Reintentando la solicitud en 1 minuto...")
        time.sleep(30)
        price_bingx()
    except Exception as e:
        logging.error(f"Ocurrió un error: {e}")
