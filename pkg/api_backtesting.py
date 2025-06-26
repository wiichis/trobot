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
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')


def round_time(dt=None, round_to=300):
    """
    Redondea un objeto datetime al múltiplo más cercano de 'round_to' segundos.
    """
    if dt is None:
        dt = datetime.now()
    seconds = (dt - dt.min).seconds
    rounding = (seconds + round_to / 2) // round_to * round_to
    return dt + timedelta(0, rounding - seconds, -dt.microsecond)

def fetch_candle(symbol, interval='5m', limit=2):
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

def load_recent_csv(path, days=180, chunksize=100000):
    """
    Carga las filas de un CSV cuya columna 'date' sea >= ahora - days.
    Devuelve un DataFrame vacío si no existe o está vacío.
    """
    try:
        cutoff = datetime.utcnow() - timedelta(days=days)
        chunks = pd.read_csv(
            path,
            parse_dates=['date'],
            usecols=['symbol', 'open', 'high', 'low', 'close', 'volume', 'date'],
            iterator=True,
            chunksize=chunksize
        )
        df_list = []
        for chunk in chunks:
            chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
            df_list.append(chunk[chunk['date'] >= cutoff])
        df = pd.concat(df_list, ignore_index=True)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df = pd.DataFrame(columns=['symbol', 'open', 'high', 'low', 'close', 'volume', 'date'])
    return df

def price_bingx_5m(limit=1, max_retries=3, retry_delay=30):
    """
    Función principal price_bingx_5m para obtener y actualizar datos de velas de 5m con control de reintentos.
    """
    error_records: List[Tuple[int, str]] = []
    attempt = 0

    while attempt < max_retries:
        try:
            currencies = pkg.api.currencies_list()
            candle_data = []

            existing_df = load_recent_csv('./archivos/cripto_price_5m.csv', days=90)

            now = datetime.utcnow()
            threshold = now - timedelta(hours=2)
            symbol_limits = {}
            for symbol in currencies:
                if not existing_df.empty and symbol in existing_df['symbol'].values:
                    last_date = pd.to_datetime(existing_df.loc[existing_df['symbol'] == symbol, 'date']).max()
                else:
                    last_date = None
                limit_for_symbol = 4 if (last_date is None or last_date < threshold) else 1
                if last_date is not None:
                    delta = now - last_date
                    gap_intervals = int(delta.total_seconds() // (5 * 60))
                    if gap_intervals > limit_for_symbol:
                        limit_for_symbol = gap_intervals
                symbol_limits[symbol] = limit_for_symbol

            # Permite hasta 10 hilos o el total de monedas, lo que sea menor
            with ThreadPoolExecutor(max_workers=min(10, len(currencies))) as executor:
                results = executor.map(lambda symbol: fetch_candle(symbol, '5m', symbol_limits[symbol]), currencies)
            for symbol, candles in zip(currencies, results):
                if candles:
                    candle_data.extend(candles)

            # -----------------------------------------------------------------
            # Verificación y reintento para símbolos que no devolvieron velas
            obtained_symbols = {c['symbol'] for c in candle_data}
            missing_symbols = [s for s in currencies if s not in obtained_symbols]

            if missing_symbols:
                logging.warning(
                    f"Retry secuencial para símbolos faltantes: {', '.join(missing_symbols)}"
                )
                for sym in missing_symbols:
                    extra_candles = fetch_candle(sym, '5m', symbol_limits[sym])
                    if extra_candles:
                        candle_data.extend(extra_candles)
            # -----------------------------------------------------------------

            if not candle_data:
                logging.error("No se obtuvieron datos de velas para ninguna moneda.")
                return

            existing_df = load_recent_csv('./archivos/cripto_price_5m.csv', days=90)

            updated_df = update_dataframe(existing_df, candle_data)

            # Limitar data a los últimos 3 meses
            cutoff = datetime.utcnow() - timedelta(days=90)
            df_month = updated_df[updated_df['date'] >= cutoff].copy()
            df_month.sort_values(['symbol', 'date'], inplace=True)
            df_month.reset_index(drop=True, inplace=True)

            csv_path = './archivos/cripto_price_5m.csv'
            df_month.to_csv(csv_path, index=False)

            # Log de verificación final
            logging.info(
                f"Velas obtenidas para "
                f"{len({c['symbol'] for c in candle_data})}/{len(currencies)} símbolos."
            )
            return

        except requests.exceptions.ConnectionError as e:
            attempt += 1
            error_records.append((
                2,
                f"⚠️ Conexión fallida en price_bingx\n"
                f"Intento {attempt}/{max_retries}\n"
                f"❌ {e}"
            ))
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                logging.error("Max retries alcanzados. Abortando.")
        except Exception as e:
            logging.error("Unexpected error in price_bingx", exc_info=True)
            raise

    if error_records:
        _, top_msg = max(error_records, key=lambda x: x[0])
        pkg.monkey_bx.bot_send_text(top_msg)
