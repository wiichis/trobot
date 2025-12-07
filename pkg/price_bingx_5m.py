from datetime import datetime, timezone
from pathlib import Path
import time
import logging
from typing import Any, List, Tuple, Optional

import pandas as pd
import requests

from .cfg_loader import load_best_symbols
from .settings import DEFAULT_SYMBOLS

BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "archivos" / "cripto_price_5m.csv"

# Máximo de días que mantiene el archivo corto usado para señales.
SIGNAL_HISTORY_DAYS = 30

log = logging.getLogger("trobot")


# --- Helper robusto para leer velas de BingX (una sola vez y reutilizable) ---

def _fetch_bingx_candles(symbol: str, limit: int, end_time_ms: Optional[int] = None):
    """
    Obtiene velas de 5m para un símbolo desde la API de BingX.
    - Maneja respuestas en dict/list con claves variables (data/list/klines/lines/...).
    - Soporta reintentos con backoff cuando la API devuelve códigos temporales (p. ej., 109500).
    - Permite fijar un endTime (ms) para retroceder en el histórico.
    Devuelve una lista de dicts normalizados con: symbol, open, high, low, close, volume, date (UTC).
    """
    url = (
        "https://open-api.bingx.com/openApi/swap/v2/quote/klines"
        f"?symbol={symbol}&interval=5m&limit={limit}"
    )
    if end_time_ms is not None:
        url = f"{url}&endTime={int(end_time_ms)}"

    retries = 3
    sleep_base = 1.5
    last_err = None

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=8)
            response.raise_for_status()
            payload = response.json()

            # Validación de código de API
            if isinstance(payload, dict) and 'code' in payload and str(payload.get('code')) not in ('0', '200'):
                code = str(payload.get('code'))
                msg = payload.get('msg') or payload.get('message') or 'Unknown error'
                # 109500 es típico temporal; reintentar
                if code == '109500' and attempt < retries:
                    time.sleep(sleep_base * attempt)
                    continue
                raise RuntimeError(f"BingX API error code={code}: {msg}")

            data = payload.get('data', payload)
            # Si 'data' es un dict, intenta extraer la lista real de velas desde varias claves comunes
            if isinstance(data, dict):
                for k in ('lines', 'klines', 'candlesticks', 'list', 'rows', 'records', 'kline'):
                    if k in data:
                        data = data[k]
                        break

            if not isinstance(data, (list, tuple)):
                raise TypeError(f"Unexpected data format from BingX: type={type(data).__name__}")

            candles = []
            for item in data:
                if isinstance(item, (list, tuple)):
                    # Formato típico: [timestamp_ms, open, high, low, close, volume, ...]
                    timestamp_ms, open_p, high_p, low_p, close_p, volume, *_ = item
                elif isinstance(item, dict):
                    # Formato alterno con claves
                    timestamp_ms = item.get('time') or item.get('timestamp') or item.get('t')
                    open_p = item.get('open') or item.get('o')
                    high_p = item.get('high') or item.get('h')
                    low_p = item.get('low') or item.get('l')
                    close_p = item.get('close') or item.get('c')
                    volume = item.get('volume') or item.get('v')
                else:
                    # Entrada desconocida: saltar
                    continue

                # Normalización de tipos
                ts = int(str(timestamp_ms))
                # Si viniera en segundos, pásalo a ms
                if ts < 10**12:
                    ts *= 1000

                candle = {
                    'symbol': symbol,
                    'open': float(open_p),
                    'high': float(high_p),
                    'low': float(low_p),
                    'close': float(close_p),
                    'volume': float(volume),
                    'date': datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
                }
                candles.append(candle)

            return candles

        except Exception as e:
            last_err = str(e)
            if attempt < retries:
                time.sleep(sleep_base * attempt)
                continue
            # Después de agotar reintentos, propaga error
            raise RuntimeError(f"Fallo al obtener velas para {symbol}: {last_err}")

def currencies_list():
    try:
        symbols = load_best_symbols()
        cleaned = [str(sym).upper() for sym in symbols if sym]
        if cleaned:
            return cleaned
    except Exception as exc:
        log.warning("No se pudo derivar currencies_list desde best_prod.json: %s", exc)
    return [str(sym).upper() for sym in DEFAULT_SYMBOLS]
    # No usar: MATIC, ADA, BTC, LTC, SOL


def price_bingx_5m() -> None:
    """
    Descarga precios de criptomonedas en intervalos de 5 minutos desde BingX,
    actualiza el archivo CSV con datos nuevos y genera un archivo agregado a 30 minutos.
    """

    # --- Descarga y actualización de datos 5m ---
    symbols = currencies_list()
    if CSV_PATH.exists():
        df_existing = pd.read_csv(CSV_PATH)
        df_existing["date"] = pd.to_datetime(df_existing["date"], utc=True, errors="coerce")
    else:
        df_existing = pd.DataFrame()

    for symbol in symbols:
        df_symbol = df_existing[df_existing['symbol'] == symbol] if not df_existing.empty else pd.DataFrame()
        last_date = df_symbol['date'].max() if not df_symbol.empty else None

        try:
            fetch_limit = 2 if last_date is not None else 1000
            new_candles = _fetch_bingx_candles(symbol, fetch_limit)
            df_new = pd.DataFrame(new_candles)
            if not df_new.empty and last_date is not None:
                df_new = df_new[df_new['date'] > last_date]
        except Exception as error:
            print(f"Error actualizando {symbol}: {error}")
            df_new = pd.DataFrame()

        if not df_new.empty:
            df_existing = pd.concat([df_existing, df_new], ignore_index=True)

    # --- Guardar datos 5m actualizados ---
    df_existing = df_existing.drop_duplicates(subset=['symbol', 'date'], keep='last').sort_values(['symbol', 'date'])

    if not df_existing.empty:
        now_utc = datetime.now(timezone.utc)
        cutoff_time = now_utc - pd.Timedelta(days=SIGNAL_HISTORY_DAYS)
        df_existing = df_existing[df_existing['date'] >= cutoff_time]

    df_existing.to_csv(CSV_PATH, index=False)

    # --- Agregación a intervalos de 30 minutos ---
    df_5m = pd.read_csv(CSV_PATH)
    df_5m["date"] = pd.to_datetime(df_5m["date"], utc=True)
    df_5m["date_30m"] = df_5m["date"].dt.floor("30T")

    aggregation = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }
    df_30m = (
        df_5m.groupby(["symbol", "date_30m"])
        .agg(aggregation)
        .reset_index()
        .rename(columns={"date_30m": "date"})
        .sort_values(["symbol", "date"])
    )
    # Reordenar columnas para que coincidan con df_5m y eliminar date_30m
    df_30m = df_30m[["symbol", "open", "high", "low", "close", "volume", "date"]]
    df_30m["date"] = pd.to_datetime(df_30m["date"], utc=True)

    df_30m.to_csv(BASE_DIR / "archivos" / "cripto_price_30m.csv", index=False)


def completar_huecos_5m() -> None:
    """
    Revisa y completa huecos mayores a 5 minutos en las últimas 24 horas de datos 5m por símbolo.
    Solo completa el primer hueco encontrado por símbolo en cada ejecución.
    """
    if not CSV_PATH.exists():
        print("Archivo cripto_price_5m.csv no encontrado. No hay datos para completar huecos.")
        return

    df = pd.read_csv(CSV_PATH)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

    now_utc = datetime.now(timezone.utc)
    cutoff_time = now_utc - pd.Timedelta(days=1)

    updated = False

    symbols = df["symbol"].unique()

    min_date_recoverable = now_utc - pd.Timedelta(minutes=5000)


    for symbol in symbols:
        df_symbol = df[(df["symbol"] == symbol) & (df["date"] >= cutoff_time)].sort_values("date")
        if df_symbol.empty or len(df_symbol) == 1:
            continue

        # Revisar huecos
        dates = df_symbol["date"].to_list()
        gap_found = False
        for i in range(len(dates) - 1):
            diff = (dates[i + 1] - dates[i]).total_seconds()
            if diff > 300:  # más de 5 minutos
                gap_start = dates[i]
                gap_end = dates[i + 1]
                # Calcular cuántas velas faltan
                missing_intervals = int(diff // 300) - 1
                if missing_intervals <= 0:
                    continue

                # Descargar velas faltantes desde gap_start + 5m, limit=missing_intervals
                # BingX API fetches most recent candles, so we need to fetch enough candles and filter
                # We'll fetch missing_intervals + 10 candles to be safe, then filter by date
                try:
                    candles = _fetch_bingx_candles(symbol, missing_intervals + 10)
                    df_candles = pd.DataFrame(candles)
                    df_candles = df_candles[
                        (df_candles["date"] > gap_start) & (df_candles["date"] < gap_end)
                    ]
                    if not df_candles.empty:
                        # Añadir velas faltantes
                        df = pd.concat([df, df_candles], ignore_index=True)
                        updated = True
                        print(f"Hueco completado para {symbol} entre {gap_start} y {gap_end} con {len(df_candles)} velas.")
                    else:
                        print(f"No se encontraron velas para completar hueco en {symbol} entre {gap_start} y {gap_end}.")
                except Exception as e:
                    print(f"Error al descargar velas para completar hueco en {symbol}: {e}")

                gap_found = True
                break  # Solo completar el primer hueco por símbolo

        if not gap_found:
            print(f"No se encontraron huecos para completar en {symbol}.")

    if updated:
        df = df.drop_duplicates(subset=["symbol", "date"], keep="last").sort_values(["symbol", "date"])
        df.to_csv(CSV_PATH, index=False)
        print("Archivo cripto_price_5m.csv actualizado con velas completadas.")
    else:
        print("No se realizaron cambios. No se encontraron huecos para completar.")



# === completar_ultimos_3dias ===

def completar_ultimos_3dias(sleep_seconds: int = 5) -> None:
    """
    Completa huecos de datos de los últimos 3 días (sin incluir hoy) para cada símbolo.
    Por cada día, verifica y rellena todos los huecos antes de avanzar al siguiente día.
    """
    if not CSV_PATH.exists():
        print("Archivo cripto_price_5m.csv no encontrado. No hay datos para completar huecos.")
        return

    df = pd.read_csv(CSV_PATH)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

    now_utc = datetime.now(timezone.utc)
    min_date_recoverable = now_utc - pd.Timedelta(minutes=5000)
    today = now_utc.date()
    # Calcula los últimos 3 días anteriores a hoy (no incluye el día actual)
    dias_a_completar = [(today - pd.Timedelta(days=i)) for i in range(1, 4)]
    dias_a_completar = sorted(dias_a_completar)  # De más antiguo a más reciente

    symbols = df["symbol"].unique()


    max_intentos = 5

    for dia in dias_a_completar:
        print(f"\n=== Completando huecos para el día: {dia} ===")
        dia_inicio = datetime.combine(dia, datetime.min.time(), tzinfo=timezone.utc)
        dia_fin = datetime.combine(dia, datetime.max.time(), tzinfo=timezone.utc)
        for symbol in symbols:
            # --- Obtener la fecha mínima disponible por API para este símbolo ---
            try:
                api_candles = _fetch_bingx_candles(symbol, 1000)
                if api_candles:
                    fecha_min_api = min(candle["date"] for candle in api_candles)
                else:
                    fecha_min_api = None
            except Exception as e:
                print(f"{symbol}: Error al obtener fecha mínima por API: {e}")
                fecha_min_api = None

            intentos = 0
            while True:
                df = pd.read_csv(CSV_PATH)
                df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
                df_symbol = df[(df["symbol"] == symbol) & (df["date"] >= dia_inicio) & (df["date"] <= dia_fin)].sort_values("date")
                if df_symbol.empty or len(df_symbol) == 1:
                    print(f"{symbol}: Solo 0 o 1 vela para el {dia}, nada que completar.")
                    break

                dates = df_symbol["date"].to_list()
                gap_found = False
                for i in range(len(dates) - 1):
                    diff = (dates[i + 1] - dates[i]).total_seconds()
                    if diff > 300:  # más de 5 minutos
                        gap_start = dates[i]
                        gap_end = dates[i + 1]

                        # Nueva lógica: si el inicio del hueco es anterior a la fecha mínima de la API, avisar y saltar
                        if fecha_min_api is not None and gap_start < fecha_min_api:
                            print(f"{symbol}: Hueco entre {gap_start} y {gap_end} es anterior a la fecha mínima disponible por API ({fecha_min_api}). Saltando este hueco.")
                            continue

                        if gap_start < min_date_recoverable:
                            print(f"{symbol}: Hueco entre {gap_start} y {gap_end} es demasiado antiguo para recuperar por API. Saltando.")
                            break

                        missing_intervals = int(diff // 300) - 1
                        if missing_intervals <= 0:
                            continue

                        try:
                            # Siempre pedir el máximo disponible (1000) para aumentar la probabilidad de encontrar velas faltantes
                            candles = _fetch_bingx_candles(symbol, 1000)
                            df_candles = pd.DataFrame(candles)
                            df_candles = df_candles[
                                (df_candles["date"] > gap_start) & (df_candles["date"] < gap_end)
                            ]
                            if not df_candles.empty:
                                df = pd.concat([df, df_candles], ignore_index=True)
                                df = df.drop_duplicates(subset=["symbol", "date"], keep="last").sort_values(["symbol", "date"])
                                df.to_csv(CSV_PATH, index=False)
                                print(f"{symbol}: {len(df_candles)} velas completadas entre {gap_start} y {gap_end}")
                            else:
                                print(f"{symbol}: No se encontraron velas para el hueco en {dia}")
                        except Exception as e:
                            print(f"{symbol}: Error al descargar velas para hueco en {dia}: {e}")

                        gap_found = True
                        intentos += 1
                        if intentos >= max_intentos:
                            print(f"{symbol}: Máximo de {max_intentos} intentos alcanzado para el hueco entre {gap_start} y {gap_end} en {dia}. Saltando al siguiente.")
                            break
                        time.sleep(sleep_seconds)
                        break  # Sale del for para volver a chequear huecos en este símbolo y día

                if not gap_found:
                    print(f"{symbol}: Día {dia} sin huecos.")
                    break  # Sale del while para ir al siguiente símbolo
                if intentos >= max_intentos:
                    break

    print("\nProceso de completado de los últimos 3 días finalizado.")


if __name__ == "__main__":
    price_bingx_5m()


# === NUEVA FUNCIÓN: actualizar_long_ultimas_12h ===
LONG_HISTORY_DAYS = 180  # ~6 meses
LONG_HISTORY_BUFFER_DAYS = 10
MAX_BACKFILL_BATCHES = 18  # límite de solicitudes retro para no saturar la API
RETIRED_SYMBOL_GRACE_DAYS = 14


def _ensure_long_history(df_long: pd.DataFrame, now_utc: pd.Timestamp) -> pd.DataFrame:
    """Garantiza que el archivo long conserve al menos 6 meses, realizando backfill incremental."""

    if df_long.empty:
        return df_long

    now_ts = pd.Timestamp(now_utc)
    now_ts = now_ts.tz_localize('UTC') if now_ts.tzinfo is None else now_ts.tz_convert('UTC')

    target_start = now_ts - pd.Timedelta(days=LONG_HISTORY_DAYS)
    target_start_utc = target_start

    # Backfill hacia atrás por símbolo hasta alcanzar el target o agotar presupuesto de peticiones
    batches = 0
    symbols = df_long['symbol'].dropna().unique().tolist()

    def _to_utc(ts: pd.Timestamp) -> Optional[pd.Timestamp]:
        if ts is None or pd.isna(ts):  # type: ignore[arg-type]
            return None
        return ts.tz_localize('UTC') if ts.tzinfo is None else ts.tz_convert('UTC')

    for symbol in symbols:
        if batches >= MAX_BACKFILL_BATCHES:
            break
        sdf = df_long[df_long['symbol'] == symbol]
        earliest = sdf['date'].min()

        earliest_utc = _to_utc(earliest)

        while earliest_utc is not None and earliest_utc > target_start_utc and batches < MAX_BACKFILL_BATCHES:
            end_time = earliest_utc - pd.Timedelta(minutes=5)
            try:
                candles = _fetch_bingx_candles(symbol, limit=1000, end_time_ms=int(end_time.timestamp() * 1000))
            except Exception as exc:
                print(f"{symbol}: Error en backfill long → {exc}")
                break

            if not candles:
                break

            df_chunk = pd.DataFrame(candles)
            if df_chunk.empty:
                break

            df_long = pd.concat([df_long, df_chunk], ignore_index=True)
            earliest = min(earliest, df_chunk['date'].min())
            earliest_utc = _to_utc(earliest)
            batches += 1
            time.sleep(0.25)

    # Purga suave: mantener target + buffer para evitar crecimiento infinito
    cutoff = target_start - pd.Timedelta(days=LONG_HISTORY_BUFFER_DAYS)
    cutoff = cutoff.tz_localize('UTC') if cutoff.tzinfo is None else cutoff
    df_long = df_long[df_long['date'] >= cutoff]

    # Depurar símbolos retirados dejando un colchón corto para cierres pendientes
    try:
        activos = set(currencies_list())
    except Exception:
        activos = set()
    if activos:
        grace_cut = now_ts - pd.Timedelta(days=RETIRED_SYMBOL_GRACE_DAYS)
        df_long = df_long[
            (df_long['symbol'].isin(activos)) |
            (df_long['date'] >= grace_cut)
        ]

    return df_long


def actualizar_long_ultimas_12h():
    """
    Sincroniza todo 'cripto_price_5m.csv' hacia 'cripto_price_5m_long.csv'.
    Antes solo se copiaban las últimas 12h, lo que perdía velas si el job se detenía
    por más de ese lapso. Ahora detecta cualquier vela nueva en el CSV corto y la
    agrega (sin duplicados) al CSV largo, garantizando que el histórico quede al día.
    """
    import pandas as pd
    from datetime import datetime, timezone
    from pathlib import Path

    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / "archivos" / "cripto_price_5m.csv"
    long_path = base_dir / "archivos" / "cripto_price_5m_long.csv"

    if not csv_path.exists():
        print("Archivo cripto_price_5m.csv no encontrado.")
        return

    # Leer todo el CSV corto. Mantiene ~30 días pero es suficiente para cualquier
    # catch-up tras reinicios largos.
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["symbol", "date"])
    df = df.drop_duplicates(subset=["symbol", "date"], keep="last")

    if df.empty:
        print("No hay velas en cripto_price_5m.csv para sincronizar.")
        return

    now_utc = datetime.now(timezone.utc)

    added_rows = 0

    if long_path.exists():
        df_long = pd.read_csv(long_path)
        if df_long.empty:
            df_concat = df.copy()
            added_rows = len(df_concat)
        else:
            df_long["date"] = pd.to_datetime(df_long["date"], utc=True, errors="coerce")
            df_long = df_long.dropna(subset=["symbol", "date"])
            df_long = df_long.drop_duplicates(subset=["symbol", "date"], keep="last")

            short_index = pd.MultiIndex.from_frame(df[["symbol", "date"]])
            long_index = pd.MultiIndex.from_frame(df_long[["symbol", "date"]])
            new_mask = ~short_index.isin(long_index)
            df_nuevas = df.loc[new_mask].copy()
            added_rows = len(df_nuevas)

            if added_rows == 0:
                print(f"{long_path.name} ya está sincronizado con cripto_price_5m.csv.")
                return

            df_concat = pd.concat([df_long, df_nuevas], ignore_index=True)
    else:
        # Primera vez: tomar todo el CSV corto como base
        df_concat = df.copy()
        added_rows = len(df_concat)
        print(f"Archivo {long_path.name} no existía. Se crea desde cripto_price_5m.csv.")

    df_concat["date"] = pd.to_datetime(df_concat["date"], utc=True, errors="coerce")
    df_concat = _ensure_long_history(df_concat, now_utc)
    df_concat = df_concat.dropna(subset=['symbol', 'date'])
    df_concat = df_concat.drop_duplicates(subset=['symbol', 'date'], keep='last').sort_values(['symbol', 'date'])
    df_concat.to_csv(long_path, index=False)

    print(f"Archivo {long_path.name} actualizado con {added_rows} velas nuevas.")
