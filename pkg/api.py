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
    return ['ADA-USDT', 'AVAX-USDT', 'CFX-USDT', 'DOT-USDT', 'BTC-USDT', 'SHIB-USDT', 'NEAR-USDT', 'LTC-USDT', 'APT-USDT', 'HBAR-USDT']

    # No usar: Matic

def round_time(dt=None, round_to=1800):
    if dt is None: 
        dt = datetime.now()
    seconds = (dt - dt.min).seconds
    rounding = (seconds + round_to / 2) // round_to * round_to
    return dt + timedelta(0, rounding - seconds, -dt.microsecond)

def fetch_candle(symbol, interval='30m', limit=2):
    try:
        json_str = pkg.bingx.get_candle(symbol, interval, limit)
        data = json.loads(json_str)
        candles_data = []

        for candle in data["data"]:
            candle_data = {
                'symbol': symbol,
                'open': candle["open"],
                'high': candle["high"],
                'low': candle["low"],
                'close': candle["close"],
                'volume': candle["volume"],
                'date': round_time(datetime.fromtimestamp(candle["time"] / 1000))
            }
            candles_data.append(candle_data)

        return candles_data
    except Exception as e:
        logging.error(f"Error al obtener la vela para {symbol}: {e}")
        return []

        prev_candle_data = {
            'symbol': symbol,
            'open': prev_candle["open"],
            'high': prev_candle["high"],
            'low': prev_candle["low"],
            'close': prev_candle["close"],
            'volume': prev_candle["volume"],
            # Usamos 'time' en lugar de 'closeTime'
            'date': round_time(datetime.fromtimestamp(prev_candle["time"] / 1000))
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

def price_bingx(limit=1):
    try:
        currencies = currencies_list()
        candle_data = []

        with ThreadPoolExecutor() as executor:
            results = executor.map(lambda symbol: fetch_candle(symbol, '30m', limit), currencies)

        for candles in results:
            if candles:
                candle_data.extend(candles)  # Añadir todas las velas obtenidas

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
        price_bingx(limit)
    except Exception as e:
        logging.error(f"Ocurrió un error: {e}")