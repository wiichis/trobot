import json
import pkg
import requests
import pandas as pd
import time
import logging
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def currencies_list():
    return ['ADA-USDT', 'AVAX-USDT', 'CFX-USDT', 'DOT-USDT', 'BTC-USDT', 'SHIB-USDT', 'NEAR-USDT', 'LTC-USDT', 'APT-USDT', 'HBAR-USDT']
    # No usar: MATIC

def round_time(dt=None, round_to=1800):
    """
    Redondea un objeto datetime al múltiplo más cercano de 'round_to' segundos.
    """
    if dt is None:
        dt = datetime.now()
    seconds = (dt - dt.min).seconds
    rounding = (seconds + round_to / 2) // round_to * round_to
    return dt + timedelta(0, rounding - seconds, -dt.microsecond)

def fetch_candle(symbol, interval='30m', limit=2):
    """
    Obtiene datos de velas para un símbolo específico.
    """
    try:
        json_str = pkg.bingx.get_candle(symbol, interval, limit)
        data = json.loads(json_str)
        candles_data = []

        if "data" in data:
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
        else:
            logging.error(f"No se encontraron datos de velas para {symbol}")

        return candles_data
    except Exception as e:
        logging.error(f"Error al obtener la vela para {symbol}: {e}")
        return []

def update_dataframe(existing_df, new_data):
    """
    Actualiza el DataFrame existente con los nuevos datos y elimina duplicados.
    """
    new_df = pd.DataFrame(new_data)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)

    # Eliminar duplicados basados en 'symbol' y 'date', manteniendo la última entrada
    combined_df.drop_duplicates(subset=['symbol', 'date'], keep='last', inplace=True)

    return combined_df

def price_bingx(limit=1):
    """
    Función principal para obtener y actualizar datos de velas de criptomonedas.
    """
    try:
        currencies = currencies_list()
        candle_data = []

        with ThreadPoolExecutor(max_workers=5) as executor:
            results = executor.map(lambda symbol: fetch_candle(symbol, '30m', limit), currencies)

        for candles in results:
            if candles:
                candle_data.extend(candles)

        # Verificar y crear el directorio si no existe
        os.makedirs('./archivos', exist_ok=True)

        try:
            existing_df = pd.read_csv('./archivos/cripto_price.csv')
        except FileNotFoundError:
            existing_df = pd.DataFrame(columns=['symbol', 'open', 'high', 'low', 'close', 'volume', 'date'])

        updated_df = update_dataframe(existing_df, candle_data)

        # Convertir 'date' a datetime si no lo está
        updated_df['date'] = pd.to_datetime(updated_df['date'])

        # Definir el número de registros a mantener por símbolo
        N = 2800  # Por ejemplo, para 30 días de datos con intervalos de 30 minutos

        # Mantener los últimos N registros por símbolo
        df_month = updated_df.groupby('symbol', group_keys=False).apply(lambda x: x.sort_values('date').tail(N))

        # Ordenar y reiniciar el índice
        df_month.sort_values(['symbol', 'date'], inplace=True)
        df_month.reset_index(drop=True, inplace=True)

        # Guardar el DataFrame resultante en el archivo CSV
        df_month.to_csv('./archivos/cripto_price.csv', index=False)

        #logging.info("Datos actualizados y guardados correctamente.")

    except requests.exceptions.ConnectionError as e:
        logging.error(f"Error de conexión: {e}")
        logging.info("Reintentando la solicitud en 30 segundos...")
        time.sleep(30)
        price_bingx(limit)
    except Exception as e:
        logging.error(f"Ocurrió un error: {e}")

