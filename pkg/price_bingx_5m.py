from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "archivos" / "cripto_price_5m.csv"

def currencies_list():
    return ['XRP-USDT', 'AVAX-USDT', 'CFX-USDT', 'DOT-USDT', 'NEAR-USDT', 'APT-USDT', 'HBAR-USDT', 'BNB-USDT', 'SHIB-USDT', 'SOL-USDT', 'DOGE-USDT']
    # No usar: MATIC, ADA, BTC, LTC


def price_bingx_5m() -> None:
    """
    Descarga precios de criptomonedas en intervalos de 5 minutos desde BingX,
    actualiza el archivo CSV con datos nuevos y genera un archivo agregado a 30 minutos.
    """
    def _bingx_candles(symbol: str, limit: int):
        """
        Obtiene velas (candles) de 5 minutos para un símbolo dado desde la API de BingX.
        """
        url = (
            "https://open-api.bingx.com/openApi/swap/v2/quote/klines"
            f"?symbol={symbol}&interval=5m&limit={limit}"
        )
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        data = response.json()["data"]

        candles = []
        for item in data:
            if isinstance(item, (list, tuple)):
                timestamp_ms, open_p, high_p, low_p, close_p, volume, *_ = item
            else:
                timestamp_ms = item.get("time") or item.get("timestamp")
                open_p = item["open"]
                high_p = item["high"]
                low_p = item["low"]
                close_p = item["close"]
                volume = item["volume"]

            candle = {
                "symbol": symbol,
                "open": float(open_p),
                "high": float(high_p),
                "low": float(low_p),
                "close": float(close_p),
                "volume": float(volume),
                "date": datetime.fromtimestamp(int(timestamp_ms) / 1000, tz=timezone.utc),
            }
            candles.append(candle)
        return candles

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
            new_candles = _bingx_candles(symbol, fetch_limit)
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

    def _bingx_candles(symbol: str, limit: int):
        url = (
            "https://open-api.bingx.com/openApi/swap/v2/quote/klines"
            f"?symbol={symbol}&interval=5m&limit={limit}"
        )
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        data = response.json()["data"]

        candles = []
        for item in data:
            if isinstance(item, (list, tuple)):
                timestamp_ms, open_p, high_p, low_p, close_p, volume, *_ = item
            else:
                timestamp_ms = item.get("time") or item.get("timestamp")
                open_p = item["open"]
                high_p = item["high"]
                low_p = item["low"]
                close_p = item["close"]
                volume = item["volume"]

            candle = {
                "symbol": symbol,
                "open": float(open_p),
                "high": float(high_p),
                "low": float(low_p),
                "close": float(close_p),
                "volume": float(volume),
                "date": datetime.fromtimestamp(int(timestamp_ms) / 1000, tz=timezone.utc),
            }
            candles.append(candle)
        return candles

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
                    candles = _bingx_candles(symbol, missing_intervals + 10)
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



# === NUEVA FUNCIÓN: completar_ultimos_10dias ===
import time

def completar_ultimos_3dias(sleep_seconds: int = 5) -> None:
    """
    Completa huecos de datos de los 10 días anteriores (sin incluir el actual) para cada símbolo.
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

    def _bingx_candles(symbol: str, limit: int):
        url = (
            "https://open-api.bingx.com/openApi/swap/v2/quote/klines"
            f"?symbol={symbol}&interval=5m&limit={limit}"
        )
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        data = response.json()["data"]

        candles = []
        for item in data:
            if isinstance(item, (list, tuple)):
                timestamp_ms, open_p, high_p, low_p, close_p, volume, *_ = item
            else:
                timestamp_ms = item.get("time") or item.get("timestamp")
                open_p = item["open"]
                high_p = item["high"]
                low_p = item["low"]
                close_p = item["close"]
                volume = item["volume"]

            candle = {
                "symbol": symbol,
                "open": float(open_p),
                "high": float(high_p),
                "low": float(low_p),
                "close": float(close_p),
                "volume": float(volume),
                "date": datetime.fromtimestamp(int(timestamp_ms) / 1000, tz=timezone.utc),
            }
            candles.append(candle)
        return candles

    max_intentos = 5

    for dia in dias_a_completar:
        print(f"\n=== Completando huecos para el día: {dia} ===")
        dia_inicio = datetime.combine(dia, datetime.min.time(), tzinfo=timezone.utc)
        dia_fin = datetime.combine(dia, datetime.max.time(), tzinfo=timezone.utc)
        for symbol in symbols:
            # --- Obtener la fecha mínima disponible por API para este símbolo ---
            try:
                api_candles = _bingx_candles(symbol, 1000)
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
                            candles = _bingx_candles(symbol, 1000)
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

    print("\nProceso de completado de los últimos 10 días finalizado.")


if __name__ == "__main__":
    price_bingx_5m()


# === NUEVA FUNCIÓN: actualizar_long_ultimas_12h ===
def actualizar_long_ultimas_12h():
    """
    Lee el archivo 'cripto_price_5m.csv', filtra solo las velas de las últimas 12 horas
    y agrega esas velas (sin duplicados) al archivo 'archivos/cripto_price_5m_long.csv'.
    Si el archivo long no existe, lo crea con esas velas recientes.
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

    # Leer las velas de los últimos 12 horas
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    now_utc = datetime.now(timezone.utc)
    cutoff_time = now_utc - pd.Timedelta(hours=12)
    df_ultimas_12h = df[df["date"] >= cutoff_time].copy()

    if df_ultimas_12h.empty:
        print("No hay velas de las últimas 12 horas para agregar.")
        return

    # Si el archivo long existe, leerlo y agregar solo velas nuevas (sin duplicados)
    if long_path.exists():
        df_long = pd.read_csv(long_path)
        df_long["date"] = pd.to_datetime(df_long["date"], utc=True, errors="coerce")
        df_concat = pd.concat([df_long, df_ultimas_12h], ignore_index=True)
        # Eliminar duplicados por symbol y date, dejando la última ocurrencia
        df_concat = df_concat.drop_duplicates(subset=['symbol', 'date'], keep='last').sort_values(['symbol', 'date'])
        df_concat.to_csv(long_path, index=False)
        print(f"Archivo {long_path.name} actualizado con velas de las últimas 12 horas.")
    else:
        # Crear el archivo con las velas recientes
        df_ultimas_12h = df_ultimas_12h.drop_duplicates(subset=['symbol', 'date'], keep='last').sort_values(['symbol', 'date'])
        df_ultimas_12h.to_csv(long_path, index=False)
        print(f"Archivo {long_path.name} creado con velas de las últimas 12 horas.")