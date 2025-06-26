import json
import pkg
import requests
import pandas as pd
import time
import logging
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import pkg.monkey_bx

# Sesión con reintentos para requests
session = requests.Session()
retry_strategy = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def currencies_list():
    return ['XRP-USDT', 'AVAX-USDT', 'CFX-USDT', 'DOT-USDT', 'BTC-USDT', 'NEAR-USDT', 'LTC-USDT', 'APT-USDT', 'HBAR-USDT', 'BNB-USDT']
    # No usar: MATIC, ADA, SHIB

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
    Obtiene datos de velas para un símbolo específico con reintentos.
    """
    attempts = 3
    for attempt in range(1, attempts + 1):
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
        except (requests.exceptions.ConnectionError, json.JSONDecodeError) as e:
            logging.error(f"Attempt {attempt} para {symbol} falló: {e}")
            if attempt < attempts:
                time.sleep(2 ** attempt)
                continue
            logging.error(f"Max retries alcanzados para {symbol}")
            return []
        except Exception as e:
            logging.error(f"Error inesperado al obtener la vela para {symbol}: {e}")
            return []

def update_dataframe(existing_df, new_data):
    """
    Actualiza el DataFrame existente con los nuevos datos y elimina duplicados.
    """
    new_df = pd.DataFrame(new_data)
    # Asegurar que ambas fechas sean datetime para evitar duplicados por tipo
    existing_df['date'] = pd.to_datetime(existing_df['date'])
    new_df['date'] = pd.to_datetime(new_df['date'])
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)

    # Eliminar duplicados basados en 'symbol' y 'date', manteniendo la última entrada
    combined_df.drop_duplicates(subset=['symbol', 'date'], keep='last', inplace=True)

    return combined_df

def price_bingx(limit=1, max_retries=3, retry_delay=30):
    """
    Función principal para obtener y actualizar datos de velas con control de reintentos.
    """
    # Lista para registrar errores con severidad
    error_records: List[Tuple[int, str]] = []

    attempt = 0
    while attempt < max_retries:
        try:
            currencies = currencies_list()
            candle_data = []

            try:
                existing_df = pd.read_csv('./archivos/cripto_price.csv')
            except FileNotFoundError:
                existing_df = pd.DataFrame(columns=['symbol', 'open', 'high', 'low', 'close', 'volume', 'date'])

            # Calcular dinámicamente cuántas velas pedir para cada símbolo
            now = datetime.utcnow()
            threshold = now - timedelta(hours=2)
            symbol_limits = {}
            for symbol in currencies:
                if not existing_df.empty and symbol in existing_df['symbol'].values:
                    last_date = pd.to_datetime(existing_df.loc[existing_df['symbol'] == symbol, 'date']).max()
                else:
                    last_date = None
                # Límite mínimo: 4 velas si faltan o están desactualizadas, si no 1
                limit_for_symbol = 4 if (last_date is None or last_date < threshold) else 1
                if last_date is not None:
                    # Ajustar según hueco real de 30m
                    delta = now - last_date
                    gap_intervals = int(delta.total_seconds() // (30 * 60))
                    if gap_intervals > limit_for_symbol:
                        limit_for_symbol = gap_intervals
                symbol_limits[symbol] = limit_for_symbol

            with ThreadPoolExecutor(max_workers=5) as executor:
                results = executor.map(lambda symbol: fetch_candle(symbol, '30m', symbol_limits.get(symbol, 1)), currencies)

            for candles in results:
                if candles:
                    candle_data.extend(candles)

            if not candle_data:
                logging.error("No se obtuvieron datos de velas para ninguna moneda.")
                return

            try:
                existing_df = pd.read_csv('./archivos/cripto_price.csv')
            except FileNotFoundError:
                existing_df = pd.DataFrame(columns=['symbol', 'open', 'high', 'low', 'close', 'volume', 'date'])

            updated_df = update_dataframe(existing_df, candle_data)

            # Mantener solo las últimas 90 d (3 meses) de velas de 30 m → 48 velas/día × 90 d = 4 320
            N = 4320
            df_month = updated_df.groupby('symbol', group_keys=False).apply(lambda x: x.sort_values('date').tail(N))
            df_month.sort_values(['symbol', 'date'], inplace=True)
            df_month.reset_index(drop=True, inplace=True)
            df_month.to_csv('./archivos/cripto_price.csv', index=False)

            return

        except requests.exceptions.ConnectionError as e:
            attempt += 1
            # Registrar error de conexión (severidad 2)
            error_records.append((
                2,
                f"⚠️ Conexión fallida en price_bingx\n"
                f"Intento {attempt}/{max_retries}\n"
                f"❌ {e}"
            ))
            if attempt < max_retries:
                logging.info(f"Reintentando en {retry_delay} segundos...")
                time.sleep(retry_delay)
            else:
                logging.error("Max retries alcanzados. Abortando.")
        except Exception as e:
            # Registrar error inesperado (severidad 3)
            error_records.append((
                3,
                f"❗ Error inesperado en price_bingx\n"
                f"{e}"
            ))
            return
    # Enviar solo la alerta con mayor severidad registrada
    if error_records:
        _, top_msg = max(error_records, key=lambda x: x[0])
        pkg.monkey_bx.bot_send_text(top_msg)
