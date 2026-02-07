import pkg
import pandas as pd
from datetime import datetime, timedelta
import requests
import json
import time
import os
from .settings import BEST_PROD_PATH

import math
from decimal import Decimal, ROUND_DOWN
# --- Cooldown por SL: lectura de estado compartido con indicadores ---

COOLDOWN_CSV = './archivos/cooldown.csv'

# Registro local de SL colocados para detectar fills
SL_WATCH_CSV = './archivos/sl_watch.csv'

def _normalize_orders_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura columnas coherentes: ['symbol','orderId','type','stopPrice','time'] cuando existan.
    Mapea 'orderType'→'type', 'symbol' en mayúsculas y 'stopPrice' numérico.
    """
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return pd.DataFrame(columns=['symbol','orderId','type','stopPrice','time'])
    df = df.copy()
    if 'type' not in df.columns and 'orderType' in df.columns:
        df = df.rename(columns={'orderType': 'type'})
    if 'type' not in df.columns:
        df['type'] = ''
    if 'symbol' in df.columns:
        df['symbol'] = df['symbol'].astype(str).str.upper()
    if 'stopPrice' in df.columns:
        df['stopPrice'] = pd.to_numeric(df['stopPrice'], errors='coerce')
    return df

def _cooldown_minutes_for_symbol(params_by_symbol: dict, symbol: str, default: int = 10) -> int:
    try:
        p = params_by_symbol.get(str(symbol).upper(), {}) if isinstance(params_by_symbol, dict) else {}
        cd = p.get('cooldown', p.get('cooldown_min', default))
        return int(cd)
    except Exception:
        return default

def _write_cooldown(symbol: str, ts, minutes: int = 10) -> None:
    try:
        if ts is None:
            ts = datetime.utcnow()
        expiry = ts + timedelta(minutes=int(minutes))
        df_exist = pd.read_csv(COOLDOWN_CSV, parse_dates=['expires_at']) if os.path.exists(COOLDOWN_CSV) else pd.DataFrame(columns=['symbol','expires_at'])
        df_exist['symbol'] = df_exist['symbol'].astype(str).str.upper()
        symbol = str(symbol).upper()
        if symbol in df_exist['symbol'].values:
            df_exist.loc[df_exist['symbol'] == symbol, 'expires_at'] = expiry
        else:
            df_exist = pd.concat([df_exist, pd.DataFrame({'symbol':[symbol], 'expires_at':[expiry]})], ignore_index=True)
        df_exist.to_csv(COOLDOWN_CSV, index=False)
    except Exception as e:
        print(f"Error registrando cooldown para {symbol}: {e}")


def _load_active_cooldowns(now=None) -> dict:
    now = now or datetime.utcnow()
    try:
        df = pd.read_csv(COOLDOWN_CSV, parse_dates=['expires_at'])
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"Error leyendo cooldowns: {e}")
        return {}
    df['symbol'] = df['symbol'].astype(str).str.upper()
    df = df[df['expires_at'] > now]
    try:
        df.to_csv(COOLDOWN_CSV, index=False)
    except Exception:
        pass
    return {row['symbol']: row['expires_at'].to_pydatetime() for _, row in df.iterrows()}


def _append_sl_watch(symbol: str, stop_price: float, position_side: str, order_id) -> None:
    try:
        df = pd.read_csv(SL_WATCH_CSV) if os.path.exists(SL_WATCH_CSV) else pd.DataFrame(columns=['symbol','stop_price','position_side','orderId','ts'])
        df['symbol'] = df['symbol'].astype(str).str.upper()
        symbol = str(symbol).upper()
        row = {
            'symbol': symbol,
            'stop_price': float(stop_price) if stop_price is not None else None,
            'position_side': str(position_side).upper(),
            'orderId': str(order_id) if order_id not in (None, '', float('nan')) else '',
            'ts': datetime.utcnow().isoformat()
        }
        df = df[~((df['symbol'] == symbol) & (df['position_side'] == row['position_side']))]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(SL_WATCH_CSV, index=False)
    except Exception as e:
        print(f"Error registrando SL watch para {symbol}: {e}")
    
 # Paso mínimo global de lote (0.001).  Si en el futuro necesitas pasos específicos,
# vuelve a añadir un diccionario STEP_SIZES y usa .get().

STEP_SIZE_DEFAULT = 0.001
# Overrides por símbolo para step/tick conocidos (completar según reglas del exchange).
SYMBOL_TRADING_RULES = {
    'BNB-USDT': {'qty_step': 0.01, 'price_tick': 0.1},
    'DOT-USDT': {'qty_step': 0.1, 'price_tick': 0.001},
    'HBAR-USDT': {'qty_step': 1.0, 'price_tick': 0.0001},
    # TRX y DOGE requieren ticks finos; con 0.01 el TP quedaba por debajo del precio de entrada
    'TRX-USDT': {'qty_step': 1.0, 'price_tick': 0.0001},
    'DOGE-USDT': {'qty_step': 1.0, 'price_tick': 0.0001},
}
# Splits por defecto para TP escalonado (40%, 40%, 20%)
TP_SPLITS = (0.40, 0.40, 0.20)

def _round_step(qty: float, step: float = STEP_SIZE_DEFAULT) -> float:
    """Redondea cantidad al múltiplo permitido por el contrato."""
    if step <= 0:
        return float(qty)
    raw = math.floor(float(qty) / step) * step
    decs = max(0, -int(round(math.log10(step))))
    return round(raw, decs)


def _split_position_qtys(total_qty: float, splits, step: float = STEP_SIZE_DEFAULT):
    """Divide la posición entre splits, garantizando que la suma use todo el tamaño en múltiplos de step."""
    if total_qty is None:
        return [0.0 for _ in splits]
    try:
        qty_dec = Decimal(str(abs(float(total_qty))))
    except Exception:
        return [0.0 for _ in splits]
    if qty_dec <= 0:
        return [0.0 for _ in splits]

    step = STEP_SIZE_DEFAULT if step is None else step
    if step <= 0:
        step = STEP_SIZE_DEFAULT
    step_dec = Decimal(str(step))

    # Ajustar cantidad total al múltiplo válido más cercano hacia abajo
    qty_dec = (qty_dec / step_dec).to_integral_value(rounding=ROUND_DOWN) * step_dec
    units_total = int((qty_dec / step_dec))
    if units_total <= 0:
        return [0.0 for _ in splits]

    result = []
    remaining_units = units_total
    splits = tuple(splits) if splits else (1.0,)

    for idx, split in enumerate(splits):
        if idx == len(splits) - 1:
            units = remaining_units
        else:
            try:
                fraction = Decimal(str(split))
            except Exception:
                fraction = Decimal('0')
            if fraction <= 0:
                units = 0
            else:
                units = int((fraction * Decimal(units_total)).to_integral_value(rounding=ROUND_DOWN))
            if units > remaining_units:
                units = remaining_units
        result.append(float(step_dec * units))
        remaining_units -= units
    # Si por redondeos quedó remanente, acumularlo al último TP
    if remaining_units > 0:
        result[-1] += float(step_dec * remaining_units)
    return result


def _step_size_for(symbol: str) -> float:
    return SYMBOL_TRADING_RULES.get(str(symbol).upper(), {}).get('qty_step', STEP_SIZE_DEFAULT)


def _tick_size_for(symbol: str) -> float:
    return SYMBOL_TRADING_RULES.get(str(symbol).upper(), {}).get('price_tick', 0.01)


def _round_to_tick(value: float, tick: float) -> float:
    if tick is None or tick <= 0:
        return float(value)
    try:
        tick_dec = Decimal(str(tick))
        val_dec = Decimal(str(value))
        rounded = (val_dec / tick_dec).to_integral_value(rounding=ROUND_DOWN) * tick_dec
        return float(rounded)
    except Exception:
        return float(value)

# --- Retry helper for robust order posting ---
def _post_with_retry(symbol, qty, price, stop, position_side, order_type, side, delays=(0.5, 1.0, 2.0)):
    """Intenta colocar una orden con reintentos escalonados. Valida la respuesta JSON de BingX."""
    last_err = None
    for d in delays:
        try:
            resp = pkg.bingx.post_order(symbol, qty, price, stop, position_side, order_type, side)
            ok, err_msg = _validate_order_response(resp)
            if ok:
                return True
            last_err = err_msg
            print(f"post_order({order_type}) rechazado para {symbol}: {err_msg}")
        except Exception as e:
            last_err = e
        time.sleep(d)
    print(f"Fallo post_order({order_type}) para {symbol}: {last_err}")
    return False


def _validate_order_response(resp_text):
    try:
        data = json.loads(resp_text) if isinstance(resp_text, str) else resp_text
    except Exception as e:
        return False, f"Respuesta no JSON ({e}): {resp_text}"

    if not isinstance(data, dict):
        return False, f"Formato inesperado: {data}"

    code = data.get('code')
    if code in (0, '0'):
        return True, None

    msg = data.get('msg') or data.get('message') or 'sin detalle'
    return False, f"code={code}, msg={msg}"

import pkg.price_bingx_5m

#Obtener los valores de SL TP


def get_last_take_profit_stop_loss(symbol):
    # Leer el archivo CSV con los indicadores
    df = pd.read_csv('./archivos/indicadores.csv', low_memory=False)
    df['symbol'] = df['symbol'].str.upper()

    filtered_df = df[df['symbol'] == symbol.upper()]
    if filtered_df.empty:
        return None, None

    # Preferimos columnas LONG; si no existen, usamos SHORT.
    if {'Take_Profit_Long', 'Stop_Loss_Long'}.issubset(filtered_df.columns):
        return (
            filtered_df['Take_Profit_Long'].iloc[-1],
            filtered_df['Stop_Loss_Long'].iloc[-1],
        )
    elif {'Take_Profit_Short', 'Stop_Loss_Short'}.issubset(filtered_df.columns):
        return (
            filtered_df['Take_Profit_Short'].iloc[-1],
            filtered_df['Stop_Loss_Short'].iloc[-1],
        )

    # Si no existen columnas estándar, retorna None
    return None, None


# Extraer TP ladder y SL de indicadores para un símbolo y lado
def extract_tp_sl_from_latest(latest_values: pd.DataFrame, symbol: str, side: str):
    """
    Devuelve (tp_levels:list, sl_level:float) para el símbolo y lado dados usando columnas de indicadores.
    Si no hay columnas escalonadas, intenta usar el TP clásico como único nivel.
    side ∈ {"LONG","SHORT"}
    """
    side = str(side).upper()
    sym = str(symbol).upper()
    row = latest_values[latest_values['symbol'] == sym]
    if row.empty:
        return [], None
    r = row.iloc[0]
    tps = []
    sl = None
    if side == 'LONG':
        # Preferir escalonados
        if all(col in row.columns for col in ['TP1_L','TP2_L','TP3_L']):
            tps = [float(r['TP1_L']), float(r['TP2_L']), float(r['TP3_L'])]
        elif 'Take_Profit_Long' in row.columns:
            tps = [float(r['Take_Profit_Long'])]
        # SL
        if 'Stop_Loss_Long' in row.columns:
            sl = float(r['Stop_Loss_Long'])
    else:  # SHORT
        if all(col in row.columns for col in ['TP1_S','TP2_S','TP3_S']):
            tps = [float(r['TP1_S']), float(r['TP2_S']), float(r['TP3_S'])]
        elif 'Take_Profit_Short' in row.columns:
            tps = [float(r['Take_Profit_Short'])]
        if 'Stop_Loss_Short' in row.columns:
            sl = float(r['Stop_Loss_Short'])
    # Limpieza
    tps = [float(x) for x in tps if pd.notna(x)]
    sl = float(sl) if sl is not None and pd.notna(sl) else None
    return tps, sl


#Funcion Enviar Mensajes
def bot_send_text(bot_message):

    bot_token = pkg.credentials.token
    bot_chatID = pkg.credentials.chatID
    send_text = pkg.credentials.send + bot_message
    response = requests.get(send_text)
 
    return response

def total_monkey():
    """
    Devuelve el balance de la cuenta.
    Maneja respuestas vacías o JSON mal formado para evitar que el bot se caiga.
    """
    raw = pkg.bingx.get_balance()

    # ── Validaciones defensivas ──────────────────────────────────────────
    if not raw or not raw.strip():
        print("Error: get_balance() devolvió cadena vacía.")
        return 0.0

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"Error decodificando JSON: {e} → {raw[:120]!r}")
        return 0.0
    # ─────────────────────────────────────────────────────────────────────

    balance = 0.0
    try:
        balance = float(data['data']['balance']['balance'])
    except (KeyError, ValueError, TypeError) as e:
        print(f"Error al obtener el balance: {e}")
        balance = 0.0  # Valor predeterminado en caso de error

    # Registrar histórico de balances
    datenow = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df_new = pd.DataFrame({'date': [datenow], 'balance': [balance]})

    file_path = './archivos/ganancias.csv'
    try:
        df_old = pd.read_csv(file_path)
        df_total = pd.concat([df_old, df_new]).tail(10000)
    except FileNotFoundError:
        df_total = df_new

    df_total.to_csv(file_path, index=False)

    return balance

def resultado_PnL():
    df_data = pd.read_csv('./archivos/PnL.csv')
    npl = pkg.bingx.hystory_PnL()
    npl = json.loads(npl)

    # Verificar si 'data' tiene datos
    if 'data' in npl and npl['data']:
        df = pd.DataFrame(npl['data'])

        # Verificar si la columna 'time' ya está en formato datetime
        if 'time' in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df['time']):
                print("La columna 'time' ya está en formato datetime")
            else:
                df['time'] = pd.to_datetime(df['time'], unit='ms')
        else:
            raise KeyError("La columna 'time' no está presente en los datos obtenidos.")
        
        df_concat = pd.concat([df_data, df])
        df_unique = df_concat.drop_duplicates()
        df_limited = df_unique.tail(10000)
        df_limited.to_csv('./archivos/PnL.csv', index=False)
    else:
        print("No hay datos nuevos para procesar en 'npl'.")



def monkey_result():
    # Obteniendo último resultado
    balance_actual = float(total_monkey())
    
    # Cargar los datos desde el archivo CSV en un DataFrame de pandas
    df = pd.read_csv('./archivos/ganancias.csv')
    
    # Convertir la columna 'date' en formato datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Filtrar los datos para obtener solo el día actual y la hora actual
    fecha_actual = datetime.now().date()
    hora_actual = datetime.now().hour
    df_dia_actual = df[df['date'].dt.date == fecha_actual]
    df_hora_actual = df_dia_actual[df_dia_actual['date'].dt.hour == hora_actual]
    
    # Obtener el balance al inicio y al final del día actual
    balance_inicial_dia = df_dia_actual['balance'].iloc[0]
    balance_final_dia = df_dia_actual['balance'].iloc[-1]
    
    # Obtener el balance al inicio y al final de la hora actual
    balance_inicial_hora = df_hora_actual['balance'].iloc[0]
    balance_final_hora = df_hora_actual['balance'].iloc[-1]
    
    # Calcular la diferencia del día actual y la diferencia de la hora actual
    diferencia_dia = balance_final_dia - balance_inicial_dia
    diferencia_hora = balance_final_hora - balance_inicial_hora
    
    # Calcular la fecha de una semana atrás
    fecha_semana_pasada = datetime.now().date() - timedelta(days=7)
    
    # Filtrar los datos para obtener la semana pasada
    df_semana_pasada = df[df['date'].dt.date >= fecha_semana_pasada]
    
    # Obtener el balance al inicio y al final de la semana pasada
    balance_inicial_semana = df_semana_pasada['balance'].iloc[0]
    balance_final_semana = df_semana_pasada['balance'].iloc[-1]
    
    # Calcular la diferencia de la semana pasada
    diferencia_semana = balance_final_semana - balance_inicial_semana
    
    return balance_actual, diferencia_hora, diferencia_dia, diferencia_semana


# Obteniendo las Posiciones
def total_positions(symbol):
    # Obteniendo las posiciones desde la API o fuente de datos
    positions = pkg.bingx.perpetual_swap_positions(symbol)
    
    try:
        # Intenta decodificar el JSON
        positions = json.loads(positions)
    except json.JSONDecodeError:
        # Si hay un error en la decodificación, retorna None para todos los campos
        print("Error decodificando JSON. Posible respuesta vacía o mal formada.")
        return None, None, None, None, None

    # Verifica si 'data' está en la respuesta y que no está vacía
    if 'data' in positions and positions['data']:
        # Extrae los datos necesarios
        symbol = positions['data'][0]['symbol']
        positionSide = positions['data'][0]['positionSide']
        price = float(positions['data'][0]['avgPrice'])
        positionAmt = positions['data'][0]['positionAmt']
        unrealizedProfit = positions['data'][0]['unrealizedProfit']
        return symbol, positionSide, price, positionAmt, unrealizedProfit
    else:
        # Retorna None si 'data' no está presente o está vacía
        return None, None, None, None, None


#Obteniendo Ordenes Pendientes
def obteniendo_ordenes_pendientes():
    try:
        ordenes_raw = pkg.bingx.query_pending_orders()
        ordenes = json.loads(ordenes_raw)
    except json.JSONDecodeError as e:
        print("Error decodificando JSON:", e)
        return []

    data = ordenes.get('data', {})
    orders = data.get('orders', [])

    if not orders:
        df = pd.DataFrame([{'symbol': 'zero', 'orderId': 0, 'side': 'both'}])
    else:
        df = pd.DataFrame(orders)

    try:
        df = _normalize_orders_df(df)
        csv_file = './archivos/order_id_register.csv'
        df.to_csv(csv_file, index=False)
    except Exception as e:
        print("Error al guardar en CSV:", e)

    return orders


def colocando_ordenes():
    pkg.monkey_bx.obteniendo_ordenes_pendientes()
    currencies = pkg.price_bingx_5m.currencies_list()
    # Cargar últimas señales de indicadores para mostrar TP escalonados en alertas
    latest_values = None
    try:
        _ind_path = './archivos/indicadores.csv'
        if os.path.exists(_ind_path):
            _dfi = pd.read_csv(_ind_path, low_memory=False)
            if 'date' in _dfi.columns:
                _dfi['date'] = pd.to_datetime(_dfi['date'])
            latest_values = _dfi.sort_values(['symbol','date']).groupby('symbol').last().reset_index()
    except Exception as _e:
        latest_values = None
    # --- Whitelist desde best_prod.json (si existe) ---
    best_prod_path = str(BEST_PROD_PATH)
    whitelist = None
    params_by_symbol = {}
    try:
        if os.path.exists(best_prod_path):
            with open(best_prod_path, 'r') as f:
                _prod = json.load(f)
            whitelist = set(str(x.get('symbol', '')).upper() for x in _prod if isinstance(x, dict))
            params_by_symbol = {str(x.get('symbol', '')).upper(): (x.get('params') or {}) for x in _prod if isinstance(x, dict)}
    except Exception as _e:
        whitelist = None
        params_by_symbol = {}
    if whitelist:
        currencies = [c for c in currencies if str(c).upper() in whitelist]
    # ---------------------------------------------------
    try:
        df_orders = pd.read_csv('./archivos/order_id_register.csv')
    except FileNotFoundError:
        df_orders = pd.DataFrame(columns=['symbol','orderId','type','stopPrice','time'])
    df_orders = _normalize_orders_df(df_orders)

    # Actualizar cooldowns a partir de SL ejecutados
    sync_cooldowns_from_sl_fills()
    active_cooldowns = _load_active_cooldowns()
    now = datetime.utcnow()

    try:
        df_positions = pd.read_csv('./archivos/position_id_register.csv')
    except FileNotFoundError:
        df_positions = pd.DataFrame(columns=['symbol','tipo','counter'])

    # Cargar los pesos actualizados
    PESOS_ACTUALIZADOS_PATH = './archivos/pesos_actualizados.csv'
    if os.path.exists(PESOS_ACTUALIZADOS_PATH):
        df_pesos = pd.read_csv(PESOS_ACTUALIZADOS_PATH)
    else:
        # Si no hay pesos actualizados, asignar pesos iguales
        df_pesos = pd.DataFrame({
            'symbol': currencies,
            'peso_actualizado': [1.0 / len(currencies)] * len(currencies)
        })

    # Aplicar límite máximo al peso individual (40% en normalización al 100%)
    LIMITE_PESO_INDIVIDUAL = 0.40

    # Aplicar límite máximo al peso en el DataFrame de pesos
    df_pesos['peso_ajustado'] = df_pesos['peso_actualizado'].clip(
        upper=LIMITE_PESO_INDIVIDUAL
    )

    # Normalizar los pesos ajustados al 100%
    suma_pesos_ajustados = df_pesos['peso_ajustado'].sum()
    df_pesos['peso_normalizado'] = df_pesos['peso_ajustado'] / suma_pesos_ajustados

    # Normalizar los pesos al 200%
    df_pesos['peso_normalizado'] *= 2

    # Aplicar límite máximo al peso individual al 200% (80%)
    LIMITE_PESO_INDIVIDUAL_200 = LIMITE_PESO_INDIVIDUAL * 2
    df_pesos['peso_normalizado'] = df_pesos['peso_normalizado'].clip(
        upper=LIMITE_PESO_INDIVIDUAL_200
    )

    # Obtener el total de fondos disponibles
    total_money = float(pkg.monkey_bx.total_monkey())
    capital_disponible = total_money

    # Lista para almacenar las monedas con señales activas
    active_currencies = []

    for currency in currencies:
        # Verificar si la moneda ya está en el DataFrame de órdenes o de posiciones
        if currency in df_orders['symbol'].values or currency in df_positions['symbol'].values:
            continue

        expiry = active_cooldowns.get(currency.upper())
        if expiry and now < expiry:
            continue

        try:
            # Obtener el peso normalizado de la moneda
            peso = df_pesos.loc[df_pesos['symbol'] == currency, 'peso_normalizado'].values
            if len(peso) == 0:
                peso = (1.0 / len(currencies)) * 2  # Peso por defecto normalizado al 200%
                peso = min(peso, LIMITE_PESO_INDIVIDUAL_200)
            else:
                peso = peso[0]

            # Obtener precio y tipo de alerta
            price_last, tipo = pkg.indicadores.ema_alert(currency)

            # Verificar si se obtuvo una alerta válida (nueva condición)
            if "Alerta de LONG" not in str(tipo) and "Alerta de SHORT" not in str(tipo):
                continue  # No hay alerta para esta moneda

            # Asegurar que price_last es numérico
            try:
                price_last = float(price_last)
            except (TypeError, ValueError):
                continue

            # Añadir a la lista de monedas activas
            active_currencies.append({
                'symbol': currency,
                'tipo': tipo,
                'price_last': price_last,
                'peso': peso
            })

        except Exception as e:
            pass  # Manejo de excepciones

    # Si no hay monedas activas, terminar la función
    if not active_currencies:
        # Silenciado: print("No hay señales activas en este momento.")
        return

    # Calcular la suma de los pesos de las monedas activas
    suma_pesos_activos = sum(item['peso'] for item in active_currencies)
    if suma_pesos_activos <= 0:
        return

    # Reajustar los pesos para que la suma de los pesos activos no exceda 200%
    factor_ajuste = min(2.0, suma_pesos_activos) / suma_pesos_activos
    for item in active_currencies:
        item['peso_ajustado'] = item['peso'] * factor_ajuste

    # Calcular el capital asignado a cada moneda y verificar que no exceda el capital disponible
    total_capital_asignado = 0
    for item in active_currencies:
        peso = item['peso_ajustado']
        trade = total_money * peso
        # Verificar si el capital asignado excede el capital disponible
        if total_capital_asignado + trade > capital_disponible:
            # Ajustar el trade para no exceder el capital disponible
            trade = capital_disponible - total_capital_asignado
            # Si no queda capital disponible, no colocar la orden
            if trade <= 0:
                print(f"No hay capital disponible para {item['symbol']}.")
                continue
        total_capital_asignado += trade
        item['trade'] = trade

    # Colocar las órdenes
    for item in active_currencies:
        if 'trade' not in item:
            continue  # Si no se asignó capital, pasar a la siguiente moneda

        currency = item['symbol']
        tipo = item['tipo']
        price_last = item['price_last']
        trade = item['trade']
        peso = item['peso_ajustado']

        # Ajustar cantidad al step‑size permitido por el contrato
        step_size = _step_size_for(currency)
        raw_qty = trade / price_last
        currency_amount = math.floor(raw_qty / step_size) * step_size
        # Redondeo para evitar floats interminables
        decs = max(0, -int(round(math.log10(step_size))))
        currency_amount = round(currency_amount, decs)
        if currency_amount <= 0:
            continue

        # Determinar lado de la orden
        if "LONG" in str(tipo):
            order_side = "BUY"
            position_side = "LONG"
        elif "SHORT" in str(tipo):
            order_side = "SELL"
            position_side = "SHORT"
        else:
            continue  # Si no es ninguno, no colocar la orden

        # ---- Niveles TP/SL: usar indicadores.csv; fallback a best_prod.json ----
        tp_price, sl_price = get_last_take_profit_stop_loss(currency)
        if tp_price is None or sl_price is None:
            # Fallback: usa 'tp' de best_prod para el TP del mensaje; SL conservador ±0.5%
            p = params_by_symbol.get(str(currency).upper(), {})
            try:
                tp_pct = float(p.get('tp', 0.015))
            except Exception:
                tp_pct = 0.015
            if position_side == 'LONG':
                tp_price = price_last * (1.0 + tp_pct)
                sl_price = price_last * 0.995  # -0.5% como fallback textual
            else:  # SHORT
                tp_price = price_last * (1.0 - tp_pct)
                sl_price = price_last * 1.005  # +0.5% como fallback textual
        # -----------------------------------------------------------------------

        # Colocando la orden
        entry_ok = _post_with_retry(
            currency, currency_amount, 0, 0, position_side, "MARKET", order_side
        )
        if not entry_ok:
            print(f"No se pudo abrir posición para {currency}; se omite configuración de TP/SL.")
            continue

        # Guardando las posiciones
        nueva_fila = pd.DataFrame({
            'symbol': [currency],
            'tipo': [position_side],
            'counter': [0]
        })
        df_positions = pd.concat([df_positions, nueva_fila], ignore_index=True)

        # ---------- ALERTA MARKDOWN con TP escalonados ----------
        # Intentar extraer TP ladder y SL desde indicadores; fallback a best_prod
        tps = []
        sl_level = None
        if latest_values is not None:
            try:
                tps, sl_level = extract_tp_sl_from_latest(latest_values, currency, position_side)
            except Exception:
                tps, sl_level = [], None
        if not tps:
            # Fallback a un solo TP usando best_prod.json (tp_pct) si existe
            p = params_by_symbol.get(str(currency).upper(), {})
            try:
                tp_pct = float(p.get('tp', 0.015))
            except Exception:
                tp_pct = 0.015
            if position_side == 'LONG':
                tps = [price_last * (1.0 + tp_pct)]
            else:
                tps = [price_last * (1.0 - tp_pct)]
        if sl_level is None:
            # Fallback conservador para SL sólo para el texto
            sl_level = price_last * (0.995 if position_side == 'LONG' else 1.005)

        # --- Ajuste opcional de TP1 (más cerca) desde best_prod.json ---
        try:
            p = params_by_symbol.get(str(currency).upper(), {})
            tp1_factor = p.get('tp1_factor', None)
            tp1_pct_override = p.get('tp1_pct_override', None)
            # Normalizar a float si vienen como string
            tp1_factor = float(tp1_factor) if tp1_factor is not None else None
            tp1_pct_override = float(tp1_pct_override) if tp1_pct_override is not None else None
        except Exception:
            tp1_factor = None
            tp1_pct_override = None

        if tps:
            if tp1_pct_override is not None and tp1_pct_override > 0:
                # Override absoluto en % desde precio de entrada de mercado
                if position_side == 'LONG':
                    tps[0] = float(price_last) * (1.0 + tp1_pct_override)
                else:  # SHORT
                    tps[0] = float(price_last) * (1.0 - tp1_pct_override)
            elif tp1_factor is not None and 0.0 < tp1_factor < 1.0:
                # Comprimir distancia del TP1 hacia la entrada
                base = float(tps[0])
                if position_side == 'LONG':
                    tps[0] = float(price_last) + (base - float(price_last)) * tp1_factor
                else:
                    tps[0] = float(price_last) - (float(price_last) - base) * tp1_factor
            else:
                # Fallback por defecto: TP1 más alcanzable (70% del camino)
                base = float(tps[0])
                if position_side == 'LONG':
                    tps[0] = float(price_last) + (base - float(price_last)) * 0.70
                else:
                    tps[0] = float(price_last) - (float(price_last) - base) * 0.70
        # ---------------------------------------------------------------

        # Pct firmados respecto a la entrada (positivos si a favor)
        sign = 1.0 if position_side == 'LONG' else -1.0
        def pct_to_str(px):
            try:
                return f"{sign * (px/price_last - 1.0) * 100:+.2f}%"
            except Exception:
                return "+0.00%"

        lines_tp = []
        for i, tp_px in enumerate(tps[:3], start=1):
            lines_tp.append(f"*TP{i}:* `{round(float(tp_px), 4)}` ({pct_to_str(float(tp_px))})")
        sl_pct_str = pct_to_str(float(sl_level))
        splits_line = "TP splits: 40% / 40% / 20%" if len(tps) >= 3 else "TP split: 100%"

        alert_lines = [
            "💎 *TRADE ALERT* 💎",
            f"`{currency}` | {'🟢 LONG' if position_side=='LONG' else '🔴 SHORT'}",
            "",
            f"*Entrada:* `{round(price_last, 4)}`",
            *lines_tp,
            f"*SL:* `{round(float(sl_level), 4)}` ({sl_pct_str})",
            "",
            splits_line,
            f"💰 *Capital:* `${round(trade, 2)}` — *{peso*100:.2f}%*"
        ]
        alert = "\n".join(alert_lines)
        # -------------------------------------------------------------------------------
        pkg.monkey_bx.bot_send_text(alert)

    # Guardando Posiciones fuera del bucle
    df_positions.to_csv('./archivos/position_id_register.csv', index=False)

def sync_cooldowns_from_sl_fills():
    try:
        if not os.path.exists(SL_WATCH_CSV):
            return
        df_watch = pd.read_csv(SL_WATCH_CSV)
    except Exception as e:
        print(f"No se pudo leer SL watch: {e}")
        return

    if df_watch.empty:
        return

    df_watch['symbol'] = df_watch['symbol'].astype(str).str.upper()
    df_watch['position_side'] = df_watch['position_side'].astype(str).str.upper()

    try:
        df_orders = pd.read_csv('./archivos/order_id_register.csv')
        df_orders = _normalize_orders_df(df_orders)
    except Exception:
        df_orders = pd.DataFrame(columns=['symbol','orderId','type','stopPrice'])

    pending = df_orders[df_orders['type'] == 'STOP_MARKET'].copy()
    pending['symbol'] = pending['symbol'].astype(str).str.upper()
    pending_ids = set(str(x) for x in pending.get('orderId', []))

    # Pre-cargar parámetros por símbolo para determinar duración del cooldown
    params_by_symbol = {}
    try:
        _best_path = str(BEST_PROD_PATH)
        if os.path.exists(_best_path):
            with open(_best_path, 'r') as _f:
                _prod = json.load(_f) or []
            params_by_symbol = {str(x.get('symbol', '')).upper(): (x.get('params') or {}) for x in _prod if isinstance(x, dict)}
    except Exception:
        params_by_symbol = {}

    remaining_rows = []
    now = datetime.utcnow()

    for _, row in df_watch.iterrows():
        symbol = row.get('symbol')
        order_id = str(row.get('orderId', '')).strip()
        stop_price = None
        try:
            stop_price = float(row.get('stop_price'))
        except Exception:
            stop_price = None

        still_pending = False
        if order_id:
            still_pending = order_id in pending_ids
        else:
            try:
                subset = pending[pending['symbol'] == symbol]
                if not subset.empty and stop_price is not None and 'stopPrice' in subset.columns:
                    subset['stopPrice'] = pd.to_numeric(subset['stopPrice'], errors='coerce')
                    still_pending = any(abs((float(sp) - stop_price) / stop_price) < 5e-4 for sp in subset['stopPrice'].dropna())
            except Exception:
                still_pending = False

        if still_pending:
            remaining_rows.append(row)
            continue

        minutes = _cooldown_minutes_for_symbol(params_by_symbol, symbol, default=10)
        _write_cooldown(symbol, now, minutes)

    if remaining_rows:
        pd.DataFrame(remaining_rows).to_csv(SL_WATCH_CSV, index=False)
    else:
        try:
            os.remove(SL_WATCH_CSV)
        except Exception:
            pass


def colocando_TK_SL():
    # Obteniendo posiciones sin SL o TP (arranque en frío tolerante)
    try:
        df_posiciones = pd.read_csv('./archivos/position_id_register.csv')
    except FileNotFoundError:
        df_posiciones = pd.DataFrame(columns=['symbol','tipo','counter'])
    # Asegurarse de que exista la columna 'counter'
    if 'counter' not in df_posiciones.columns:
        df_posiciones['counter'] = 0
    if not df_posiciones.empty:
        df_posiciones['counter'] += 1

    try:
        df_ordenes = pd.read_csv('./archivos/order_id_register.csv')
    except FileNotFoundError:
        df_ordenes = pd.DataFrame(columns=['symbol','orderId','type','stopPrice','time'])
    df_ordenes = _normalize_orders_df(df_ordenes)

    # Leer los últimos valores de indicadores
    df_indicadores = pd.read_csv('./archivos/indicadores.csv', low_memory=False)
    # Limpiar espacios en los nombres de columnas y en los símbolos
    df_indicadores.columns = df_indicadores.columns.str.strip()
    if 'symbol' in df_indicadores.columns:
        df_indicadores['symbol'] = df_indicadores['symbol'].str.strip().str.upper()
    latest_values = df_indicadores.groupby('symbol').last().reset_index()

    # Cargar parametros por símbolo (para ajustar TP1) desde pkg/best_prod.json
    params_by_symbol = {}
    try:
        _best_path = str(BEST_PROD_PATH)
        if os.path.exists(_best_path):
            with open(_best_path, 'r') as _f:
                _prod = json.load(_f) or []
            params_by_symbol = {str(x.get('symbol', '')).upper(): (x.get('params') or {}) for x in _prod if isinstance(x, dict)}
    except Exception as _e:
        params_by_symbol = {}

    # Agrega un pequeño delay antes de buscar la posición en BingX para dar tiempo a que la posición MARKET aparezca
    time.sleep(2)
    for index, row in df_posiciones.iterrows():
        symbol = row['symbol']
        counter = row['counter']

        # Verificar si ya existen órdenes SL y TP pendientes para este símbolo
        symbol_orders = df_ordenes[df_ordenes['symbol'] == symbol]
        sl_exists = not symbol_orders[symbol_orders['type'] == 'STOP_MARKET'].empty
        tp_exists = not symbol_orders[symbol_orders['type'] == 'TAKE_PROFIT_MARKET'].empty

        # Si ambos existen, ya está protegido: eliminamos la entrada y seguimos
        if sl_exists and tp_exists:
            df_posiciones.drop(index, inplace=True)
            continue

        # Verificar si se debe cancelar la orden después de cierto tiempo
        if counter >= 20:
            try:
                mask = (df_ordenes['symbol'] == symbol) if 'symbol' in df_ordenes.columns else pd.Series([], dtype=bool)
                msg = ""
                if mask.any():
                    df_sym = df_ordenes.loc[mask]
                    if 'orderId' in df_sym.columns and not df_sym['orderId'].isna().all():
                        orderId = df_sym['orderId'].iloc[0]
                        try:
                            pkg.bingx.cancel_order(symbol, orderId)
                            msg = f"❌ Orden cancelada por timeout para {symbol}. No se ejecutó en el tiempo límite."
                        except Exception as ce:
                            print(f"Error al cancelar la orden para {symbol}: {ce}")
                            msg = f"⏱️ Timeout para {symbol}, no se pudo cancelar la orden (ya no existe o API rechazó)."
                    else:
                        msg = f"⏱️ Timeout para {symbol}, pero no había STOP/TP pendientes para cancelar."
                else:
                    msg = f"⏱️ Timeout para {symbol}, sin órdenes pendientes registradas."
                # Retirar la posición de la cola de protección en cualquier caso
                df_posiciones.drop(index, inplace=True)
                df_posiciones.to_csv('./archivos/position_id_register.csv', index=False)
                print(msg)
                try:
                    pkg.monkey_bx.bot_send_text(msg)
                except Exception:
                    pass
                continue  # Evita múltiples mensajes y cancelaciones para la misma orden
            except Exception as e:
                print(f"Error al manejar timeout para {symbol}: {e}")
                # Aun con error, sacar de la cola para no ciclar
                try:
                    df_posiciones.drop(index, inplace=True)
                    df_posiciones.to_csv('./archivos/position_id_register.csv', index=False)
                except Exception:
                    pass
                continue

        # Obteniendo el valor de las posiciones reales
        try:
            # Obtener los detalles de la posición actual
            result = total_positions(symbol)
            if result[0] is None:
                print(f"Posición para {symbol} aún no aparece en el exchange. Reintentando en el próximo ciclo.")
                continue

            symbol_result, positionSide, price, positionAmt, unrealizedProfit = result

            try:
                position_qty = abs(float(positionAmt))
            except Exception:
                position_qty = 0.0

            if positionSide == 'LONG':
                # Niveles TP (escalonados si existen) y SL desde indicadores
                desired_tps, sl_level = extract_tp_sl_from_latest(latest_values, symbol, 'LONG')
                # Fallbacks de emergencia si no hay datos
                if not desired_tps:
                    desired_tps = [price * 1.01]
                if sl_level is None:
                    sl_level = price * 0.995

                # Ajuste opcional de TP1 desde best_prod.json (más cerca del precio de entrada)
                try:
                    p = params_by_symbol.get(str(symbol).upper(), {})
                    tp1_factor = p.get('tp1_factor', None)
                    tp1_pct_override = p.get('tp1_pct_override', None)
                    tp1_factor = float(tp1_factor) if tp1_factor is not None else None
                    tp1_pct_override = float(tp1_pct_override) if tp1_pct_override is not None else None
                except Exception:
                    tp1_factor = None
                    tp1_pct_override = None

                if desired_tps:
                    if tp1_pct_override is not None and tp1_pct_override > 0:
                        desired_tps[0] = float(price) * (1.0 + tp1_pct_override)
                    elif tp1_factor is not None and 0.0 < tp1_factor < 1.0:
                        base = float(desired_tps[0])
                        desired_tps[0] = float(price) + (base - float(price)) * tp1_factor
                    else:
                        # Fallback: TP1 más alcanzable por defecto (70% del camino)
                        base = float(desired_tps[0])
                        desired_tps[0] = float(price) + (base - float(price)) * 0.70

                # ¿Qué órdenes existen ya?
                symbol_orders = df_ordenes[df_ordenes['symbol'] == symbol]
                existing_tp = symbol_orders[symbol_orders['type'] == 'TAKE_PROFIT_MARKET']
                existing_sl = symbol_orders[symbol_orders['type'] == 'STOP_MARKET']
                tick = _tick_size_for(symbol)
                step_sz = _step_size_for(symbol)
                existing_tp_prices = set()
                if 'stopPrice' in existing_tp.columns:
                    try:
                        existing_tp_prices = set(
                            _round_to_tick(float(x), tick) for x in existing_tp['stopPrice'].tolist() if pd.notna(x)
                        )
                    except Exception:
                        existing_tp_prices = set()

                # Colocar SL si falta
                exito_sl = not existing_sl.empty
                if not exito_sl:
                    sl_px = _round_to_tick(float(sl_level), tick)
                    exito_sl = _post_with_retry(symbol, position_qty, 0, sl_px, "LONG", "STOP_MARKET", "SELL")
                    if exito_sl:
                        time.sleep(1)
                        try:
                            _ = obteniendo_ordenes_pendientes()
                            df_tmp = pd.read_csv('./archivos/order_id_register.csv')
                            df_tmp = _normalize_orders_df(df_tmp)
                            m = df_tmp[(df_tmp['symbol'] == symbol) & (df_tmp['type'] == 'STOP_MARKET')].copy()
                            order_id = None
                            if not m.empty and 'stopPrice' in m.columns:
                                m['stopPrice'] = pd.to_numeric(m['stopPrice'], errors='coerce')
                                m['diff'] = (m['stopPrice'] - float(sl_px)).abs() / max(abs(float(sl_px)), 1e-9)
                                m = m.sort_values('diff')
                                if not m.empty and m['diff'].iloc[0] < 1e-3:
                                    order_id = m['orderId'].iloc[0] if 'orderId' in m.columns else None
                            _append_sl_watch(symbol, float(sl_px), 'LONG', order_id)
                        except Exception:
                            _append_sl_watch(symbol, float(sl_px), 'LONG', None)

                # Cantidades parciales para TPs
                try:
                    pos_qty = abs(float(positionAmt))
                except Exception:
                    pos_qty = 0.0
                splits = TP_SPLITS if len(desired_tps) >= 3 else (1.0,)
                split_qtys = _split_position_qtys(pos_qty, splits, step_sz)

                # Colocar TPs faltantes (comparando por precio exacto)
                placed_all_tps = True
                for idx, tp_px in enumerate(desired_tps[:len(split_qtys)]):
                    tp_px = _round_to_tick(float(tp_px), tick)
                    # ¿Existe un TP muy parecido ya? (tolerancia relativa ~1e-4)
                    exists = any(abs((tp_px - ex_px) / ex_px) < 1e-4 for ex_px in existing_tp_prices) if existing_tp_prices else False
                    if exists:
                        continue
                    tp_qty = split_qtys[idx]
                    if tp_qty <= 0:
                        continue
                    ok = _post_with_retry(symbol, tp_qty, 0, tp_px, "LONG", "TAKE_PROFIT_MARKET", "SELL")
                    if ok:
                        time.sleep(0.3)
                        existing_tp_prices.add(tp_px)
                    else:
                        placed_all_tps = False

                # Recalcular si ya existen todos los TP deseados
                # (vuelve a leer órdenes pendientes para asegurarse)
                try:
                    _orders = obteniendo_ordenes_pendientes()
                    df_ordenes = pd.read_csv('./archivos/order_id_register.csv')
                    symbol_orders = df_ordenes[df_ordenes['symbol'] == symbol]
                    existing_tp = symbol_orders[symbol_orders['type'] == 'TAKE_PROFIT_MARKET']
                    existing_tp_prices = set(
                        _round_to_tick(float(x), tick) for x in existing_tp['stopPrice'].tolist() if pd.notna(x)
                    ) if 'stopPrice' in existing_tp.columns else set()
                    target_count = len(split_qtys)
                    have_count = 0
                    for tp_px in desired_tps[:target_count]:
                        tp_px_round = _round_to_tick(float(tp_px), tick)
                        if any(abs((tp_px_round - ex_px) / ex_px) < 1e-4 for ex_px in existing_tp_prices):
                            have_count += 1
                    placed_all_tps = (have_count >= target_count)
                except Exception:
                    pass

                # Si SL existe y todos los TP están listos, retirar de la cola
                if exito_sl and placed_all_tps:
                    df_posiciones.drop(index, inplace=True)

                # Fallbacks adicionales: garantizar al menos una protección
                symbol_orders = df_ordenes[df_ordenes['symbol'] == symbol]
                existing_tp = symbol_orders[symbol_orders['type'] == 'TAKE_PROFIT_MARKET']
                existing_sl = symbol_orders[symbol_orders['type'] == 'STOP_MARKET']
                if existing_sl.empty:
                    sl_px = _round_to_tick(float(sl_level), tick)
                    ok_sl = _post_with_retry(symbol, position_qty, 0, sl_px, "LONG", "STOP_MARKET", "SELL")
                    time.sleep(0.3)
                    if ok_sl:
                        try:
                            _ = obteniendo_ordenes_pendientes()
                            df_tmp = pd.read_csv('./archivos/order_id_register.csv')
                            df_tmp = _normalize_orders_df(df_tmp)
                            m = df_tmp[(df_tmp['symbol'] == symbol) & (df_tmp['type'] == 'STOP_MARKET')].copy()
                            order_id = None
                            if not m.empty and 'stopPrice' in m.columns:
                                m['stopPrice'] = pd.to_numeric(m['stopPrice'], errors='coerce')
                                m['diff'] = (m['stopPrice'] - float(sl_px)).abs() / max(abs(float(sl_px)), 1e-9)
                                m = m.sort_values('diff')
                                if not m.empty and m['diff'].iloc[0] < 1e-3:
                                    order_id = m['orderId'].iloc[0] if 'orderId' in m.columns else None
                            _append_sl_watch(symbol, float(sl_px), 'LONG', order_id)
                        except Exception:
                            _append_sl_watch(symbol, float(sl_px), 'LONG', None)
                if existing_tp.empty and desired_tps:
                    tp1_px = _round_to_tick(float(desired_tps[0]), tick)
                    try:
                        pos_qty = abs(float(positionAmt))
                    except Exception:
                        pos_qty = 0.0
                    tp_qty = _round_step(pos_qty, step_sz)
                    if tp_qty > 0:
                        _post_with_retry(symbol, tp_qty, 0, tp1_px, "LONG", "TAKE_PROFIT_MARKET", "SELL")
                        time.sleep(0.3)

            elif positionSide == 'SHORT':
                # Niveles TP (escalonados si existen) y SL desde indicadores
                desired_tps, sl_level = extract_tp_sl_from_latest(latest_values, symbol, 'SHORT')
                if not desired_tps:
                    desired_tps = [price * 0.99]
                if sl_level is None:
                    sl_level = price * 1.005

                # Ajuste opcional de TP1 desde best_prod.json para SHORT
                try:
                    p = params_by_symbol.get(str(symbol).upper(), {})
                    tp1_factor = p.get('tp1_factor', None)
                    tp1_pct_override = p.get('tp1_pct_override', None)
                    tp1_factor = float(tp1_factor) if tp1_factor is not None else None
                    tp1_pct_override = float(tp1_pct_override) if tp1_pct_override is not None else None
                except Exception:
                    tp1_factor = None
                    tp1_pct_override = None

                if desired_tps:
                    if tp1_pct_override is not None and tp1_pct_override > 0:
                        desired_tps[0] = float(price) * (1.0 - tp1_pct_override)
                    elif tp1_factor is not None and 0.0 < tp1_factor < 1.0:
                        base = float(desired_tps[0])
                        desired_tps[0] = float(price) - (float(price) - base) * tp1_factor
                    else:
                        # Fallback: TP1 más alcanzable por defecto (70% del camino)
                        base = float(desired_tps[0])
                        desired_tps[0] = float(price) - (float(price) - base) * 0.70

                # ¿Qué órdenes existen ya?
                symbol_orders = df_ordenes[df_ordenes['symbol'] == symbol]
                existing_tp = symbol_orders[symbol_orders['type'] == 'TAKE_PROFIT_MARKET']
                existing_sl = symbol_orders[symbol_orders['type'] == 'STOP_MARKET']
                tick = _tick_size_for(symbol)
                step_sz = _step_size_for(symbol)
                existing_tp_prices = set()
                if 'stopPrice' in existing_tp.columns:
                    try:
                        existing_tp_prices = set(
                            _round_to_tick(float(x), tick) for x in existing_tp['stopPrice'].tolist() if pd.notna(x)
                        )
                    except Exception:
                        existing_tp_prices = set()

                # SL si falta
                exito_sl = not existing_sl.empty
                if not exito_sl:
                    sl_px = _round_to_tick(float(sl_level), tick)
                    exito_sl = _post_with_retry(symbol, position_qty, 0, sl_px, "SHORT", "STOP_MARKET", "BUY")
                    if exito_sl:
                        time.sleep(1)
                # Registrar SL en watch para detectar fill y disparar cooldown
                try:
                    _ = obteniendo_ordenes_pendientes()
                    df_ordenes = pd.read_csv('./archivos/order_id_register.csv')
                    m = df_ordenes[(df_ordenes['symbol']==symbol) & (df_ordenes['type']=='STOP_MARKET')].copy()
                    order_id = None
                    if not m.empty:
                        if 'stopPrice' in m.columns:
                            m['stopPrice'] = pd.to_numeric(m['stopPrice'], errors='coerce')
                            m['diff'] = (m['stopPrice'] - float(sl_level)).abs() / float(sl_level)
                            m = m.sort_values('diff')
                            if not m.empty and m['diff'].iloc[0] < 1e-3:
                                order_id = m['orderId'].iloc[0] if 'orderId' in m.columns else None
                    _append_sl_watch(symbol, float(sl_level), 'SHORT', order_id)
                except Exception:
                    _append_sl_watch(symbol, float(sl_level), 'SHORT', None)

                # Cantidades parciales
                try:
                    pos_qty = abs(float(positionAmt))
                except Exception:
                    pos_qty = 0.0
                splits = TP_SPLITS if len(desired_tps) >= 3 else (1.0,)
                split_qtys = _split_position_qtys(pos_qty, splits, step_sz)

                # Colocar TPs faltantes
                placed_all_tps = True
                for idx, tp_px in enumerate(desired_tps[:len(split_qtys)]):
                    tp_px = _round_to_tick(float(tp_px), tick)
                    exists = any(abs((tp_px - ex_px) / ex_px) < 1e-4 for ex_px in existing_tp_prices) if existing_tp_prices else False
                    if exists:
                        continue
                    tp_qty = split_qtys[idx]
                    if tp_qty <= 0:
                        continue
                    ok = _post_with_retry(symbol, tp_qty, 0, tp_px, "SHORT", "TAKE_PROFIT_MARKET", "BUY")
                    if ok:
                        time.sleep(0.3)
                        existing_tp_prices.add(tp_px)
                    else:
                        placed_all_tps = False

                # Revalidar existencia de todos los TP
                try:
                    _orders = obteniendo_ordenes_pendientes()
                    df_ordenes = pd.read_csv('./archivos/order_id_register.csv')
                    symbol_orders = df_ordenes[df_ordenes['symbol'] == symbol]
                    existing_tp = symbol_orders[symbol_orders['type'] == 'TAKE_PROFIT_MARKET']
                    existing_tp_prices = set(
                        _round_to_tick(float(x), tick) for x in existing_tp['stopPrice'].tolist() if pd.notna(x)
                    ) if 'stopPrice' in existing_tp.columns else set()
                    target_count = len(split_qtys)
                    have_count = 0
                    for tp_px in desired_tps[:target_count]:
                        tp_px_round = _round_to_tick(float(tp_px), tick)
                        if any(abs((tp_px_round - ex_px) / ex_px) < 1e-4 for ex_px in existing_tp_prices):
                            have_count += 1
                    placed_all_tps = (have_count >= target_count)
                except Exception:
                    pass

                if exito_sl and placed_all_tps:
                    df_posiciones.drop(index, inplace=True)

                # Fallbacks adicionales: garantizar al menos una protección
                symbol_orders = df_ordenes[df_ordenes['symbol'] == symbol]
                existing_tp = symbol_orders[symbol_orders['type'] == 'TAKE_PROFIT_MARKET']
                existing_sl = symbol_orders[symbol_orders['type'] == 'STOP_MARKET']
                if existing_sl.empty:
                    sl_px = _round_to_tick(float(sl_level), tick)
                    ok_sl = _post_with_retry(symbol, position_qty, 0, sl_px, "SHORT", "STOP_MARKET", "BUY")
                    time.sleep(0.3)
                    if ok_sl:
                        try:
                            _ = obteniendo_ordenes_pendientes()
                            df_tmp = pd.read_csv('./archivos/order_id_register.csv')
                            df_tmp = _normalize_orders_df(df_tmp)
                            m = df_tmp[(df_tmp['symbol'] == symbol) & (df_tmp['type'] == 'STOP_MARKET')].copy()
                            order_id = None
                            if not m.empty and 'stopPrice' in m.columns:
                                m['stopPrice'] = pd.to_numeric(m['stopPrice'], errors='coerce')
                                m['diff'] = (m['stopPrice'] - float(sl_px)).abs() / max(abs(float(sl_px)), 1e-9)
                                m = m.sort_values('diff')
                                if not m.empty and m['diff'].iloc[0] < 1e-3:
                                    order_id = m['orderId'].iloc[0] if 'orderId' in m.columns else None
                            _append_sl_watch(symbol, float(sl_px), 'SHORT', order_id)
                        except Exception:
                            _append_sl_watch(symbol, float(sl_px), 'SHORT', None)
                if existing_tp.empty and desired_tps:
                    tp1_px = _round_to_tick(float(desired_tps[0]), tick)
                    try:
                        pos_qty = abs(float(positionAmt))
                    except Exception:
                        pos_qty = 0.0
                    tp_qty = _round_step(pos_qty, step_sz)
                    if tp_qty > 0:
                        _post_with_retry(symbol, tp_qty, 0, tp1_px, "SHORT", "TAKE_PROFIT_MARKET", "BUY")
                        time.sleep(0.3)

        except Exception as e:
            print(f"Error al configurar SL/TP para {symbol}: {e}")
            pass

    # Guardando Posiciones
    df_posiciones.to_csv('./archivos/position_id_register.csv', index=False)


#Cerrando Posiciones antiguas
def filtrando_posiciones_antiguas() -> pd.DataFrame:
    try:
        # Cargar los datos
        data = pd.read_csv('./archivos/order_id_register.csv')
        
        # Ajustar por zona horaria sumando 5 horas al tiempo actual
        # Tiempo Server AWS
        current_time = pd.Timestamp.now() - timedelta(hours=9)
        # Tiempo Mac
        # current_time = pd.Timestamp.now() + timedelta(hours=5)
        
        # Comprobar si la columna 'symbol' está en el DataFrame
        if 'symbol' not in data.columns:
            raise KeyError("La columna 'symbol' no se encuentra en el DataFrame.")
        
        # Filtro de columnas
        data_filtered = data[['symbol', 'orderId', 'type', 'time', 'stopPrice']].copy()
        data_filtered['time'] = pd.to_datetime(data_filtered['time'], unit='ms')
        
        # Calcular la diferencia de tiempo
        data_filtered['time_difference'] = (current_time - data_filtered['time']).dt.total_seconds() / 60
        
        # Filtrar entradas con más de 1 minuto de diferencia y de tipo 'STOP_MARKET'
        data_filtered = data_filtered[(data_filtered['time_difference'] > 1) & (data_filtered['type'] == 'STOP_MARKET')]
        
        # Remover duplicados basado en 'symbol'
        data_filtered = data_filtered.drop_duplicates(subset='symbol')
        
        # Resetear el índice
        data_filtered.reset_index(drop=True, inplace=True)
        
        return data_filtered

    except FileNotFoundError:
        return pd.DataFrame(columns=['symbol', 'orderId', 'type', 'time', 'stopPrice'])  # Retorna un DataFrame con las columnas esperadas pero vacío

    except KeyError as e:
        return pd.DataFrame(columns=['symbol', 'orderId', 'type', 'time', 'stopPrice'])  # Retorna un DataFrame con las columnas esperadas pero vacío
    


def unrealized_profit_positions():
    # Cargar los indicadores
    df_indicadores = pd.read_csv('./archivos/indicadores.csv', low_memory=False)
    # Limpiar nombres de columnas y símbolo
    df_indicadores.columns = df_indicadores.columns.str.strip()
    df_indicadores['symbol'] = df_indicadores['symbol'].str.strip().str.upper()
    
    # Verificar que las columnas necesarias existen
    required_columns = ['symbol', 'close', 'Stop_Loss_Long', 'Stop_Loss_Short']
    missing_columns = [col for col in required_columns if col not in df_indicadores.columns]
    if missing_columns:
        print(f"Las siguientes columnas faltan en 'indicadores.csv': {missing_columns}")
        return
    
    # Convertir símbolos a mayúsculas para asegurar coincidencia
    df_indicadores['symbol'] = df_indicadores['symbol'].str.upper()
    
    # Agrupar por 'symbol' y obtener la última fila de cada grupo
    latest_values = df_indicadores.groupby('symbol').last().reset_index()

    # Cargar parametros por símbolo (para be_trigger) desde pkg/best_prod.json
    params_by_symbol = {}
    try:
        _best_path = str(BEST_PROD_PATH)
        if os.path.exists(_best_path):
            with open(_best_path, 'r') as _f:
                _prod = json.load(_f) or []
            params_by_symbol = {str(x.get('symbol', '')).upper(): (x.get('params') or {}) for x in _prod if isinstance(x, dict)}
    except Exception as _e:
        params_by_symbol = {}

    # Obtener datos filtrados de la función anterior
    data_filtered = filtrando_posiciones_antiguas()
    
    # Verificar si 'data_filtered' no está vacío
    if data_filtered.empty:
        # Silenciado: print("No hay posiciones antiguas para procesar.")
        return
    
    # Convertir símbolos a mayúsculas
    data_filtered['symbol'] = data_filtered['symbol'].str.upper()
    
    # Extraer la lista de símbolos
    symbols = data_filtered['symbol'].tolist()
    
    for symbol in symbols:
        # Silenciado: print(f"Procesando símbolo: {symbol}")

        # Obtener datos del símbolo en 'latest_values'
        symbol_data = latest_values[latest_values['symbol'] == symbol]

        # Verificar si 'symbol_data' está vacío
        if symbol_data.empty:
            # Silenciado: print(f"No hay datos de indicadores para el símbolo: {symbol}")
            continue

        # Obtener el precio actual
        precio_actual = symbol_data['close'].iloc[0]

        # Obtener datos de posición utilizando 'total_positions'
        result = total_positions(symbol)

        # Verificar si 'result' es None o no tiene suficientes datos
        if not result or result[0] is None:
            # Silenciado: print(f"No hay datos de posición para el símbolo: {symbol}")
            continue  # Saltar a la siguiente iteración del bucle

        # Desempaquetar el resultado
        symbol_result, positionSide, price, positionAmt, unrealizedProfit = result

        # Parámetro de break-even por símbolo (default 0.0 desactivado)
        p = params_by_symbol.get(str(symbol).upper(), {})
        try:
            be_trigger = float(p.get('be_trigger', 0.0))
        except Exception:
            be_trigger = 0.0
        TINY_BE = 0.0002  # 2 bps para cubrir fees/ticks

        # Obtener el último valor de 'stopPrice' y 'orderId' para el símbolo
        filtered_data = data_filtered[data_filtered['symbol'] == symbol]

        # Verificar si 'filtered_data' está vacío
        if filtered_data.empty:
            # Silenciado: print(f"No se encontraron datos de 'order_id_register.csv' para el símbolo: {symbol}")
            continue

        # Acceder de forma segura a 'stopPrice' y 'orderId'
        last_stop_price = filtered_data['stopPrice'].iloc[-1]
        orderId = filtered_data['orderId'].iloc[-1]

        # Asegurar que 'positionAmt' es numérico
        try:
            positionAmt = float(positionAmt)
        except ValueError:
            # Silenciado: print(f"Cantidad de posición inválida para {symbol}: {positionAmt}")
            continue
        position_qty = abs(positionAmt)

        if positionSide == 'LONG':
            stop_loss = symbol_data['Stop_Loss_Long'].iloc[0]
            potencial_nuevo_sl = stop_loss
            # Break-Even: si el avance supera be_trigger, subir SL a entrada + tiny
            if be_trigger > 0.0:
                try:
                    be_price = float(price) * (1.0 + be_trigger)
                    if float(precio_actual) >= be_price:
                        be_stop = float(price) * (1.0 + TINY_BE)
                        potencial_nuevo_sl = max(potencial_nuevo_sl, be_stop)
                except Exception:
                    pass
            if potencial_nuevo_sl > last_stop_price and potencial_nuevo_sl != last_stop_price:
                try:
                    pkg.bingx.cancel_order(symbol, orderId)
                    time.sleep(1)
                    _post_with_retry(symbol, position_qty, 0, potencial_nuevo_sl, "LONG", "STOP_MARKET", "SELL")
                    # Actualizar watch con el nuevo SL
                    try:
                        _ = obteniendo_ordenes_pendientes()
                        df_ordenes = pd.read_csv('./archivos/order_id_register.csv')
                        df_ordenes = _normalize_orders_df(df_ordenes)
                        m = df_ordenes[(df_ordenes['symbol']==symbol) & (df_ordenes['type']=='STOP_MARKET')].copy()
                        order_id = None
                        if not m.empty and 'stopPrice' in m.columns:
                            m['diff'] = (m['stopPrice'] - float(potencial_nuevo_sl)).abs() / float(potencial_nuevo_sl)
                            m = m.sort_values('diff')
                            if not m.empty and m['diff'].iloc[0] < 1e-3:
                                order_id = m['orderId'].iloc[0] if 'orderId' in m.columns else None
                        _append_sl_watch(symbol, float(potencial_nuevo_sl), 'LONG', order_id)
                    except Exception:
                        _append_sl_watch(symbol, float(potencial_nuevo_sl), 'LONG', None)
                    print(f"Stop Loss actualizado para {symbol} (LONG) a {potencial_nuevo_sl}")
                except Exception as e:
                    print(f"Error al actualizar el Stop Loss para {symbol}: {e}")
            else:
                print(f"SL actual para {symbol} (LONG) es suficientemente bueno, no se modifica.")

        elif positionSide == 'SHORT':
            stop_loss = symbol_data['Stop_Loss_Short'].iloc[0]
            potencial_nuevo_sl = stop_loss
            # Break-Even: si la ganancia supera be_trigger, bajar SL a entrada - tiny
            if be_trigger > 0.0:
                try:
                    be_price = float(price) * (1.0 - be_trigger)
                    if float(precio_actual) <= be_price:
                        be_stop = float(price) * (1.0 - TINY_BE)
                        potencial_nuevo_sl = min(potencial_nuevo_sl, be_stop)
                except Exception:
                    pass
            if potencial_nuevo_sl < last_stop_price and potencial_nuevo_sl != last_stop_price:
                try:
                    pkg.bingx.cancel_order(symbol, orderId)
                    time.sleep(1)
                    _post_with_retry(symbol, position_qty, 0, potencial_nuevo_sl, "SHORT", "STOP_MARKET", "BUY")
                    # Actualizar watch con el nuevo SL
                    try:
                        _ = obteniendo_ordenes_pendientes()
                        df_ordenes = pd.read_csv('./archivos/order_id_register.csv')
                        df_ordenes = _normalize_orders_df(df_ordenes)
                        m = df_ordenes[(df_ordenes['symbol']==symbol) & (df_ordenes['type']=='STOP_MARKET')].copy()
                        order_id = None
                        if not m.empty and 'stopPrice' in m.columns:
                            m['diff'] = (m['stopPrice'] - float(potencial_nuevo_sl)).abs() / float(potencial_nuevo_sl)
                            m = m.sort_values('diff')
                            if not m.empty and m['diff'].iloc[0] < 1e-3:
                                order_id = m['orderId'].iloc[0] if 'orderId' in m.columns else None
                        _append_sl_watch(symbol, float(potencial_nuevo_sl), 'SHORT', order_id)
                    except Exception:
                        _append_sl_watch(symbol, float(potencial_nuevo_sl), 'SHORT', None)

                    print(f"Stop Loss actualizado para {symbol} (SHORT) a {potencial_nuevo_sl}")
                except Exception as e:
                    print(f"Error al actualizar el Stop Loss para {symbol}: {e}")
            else:
                print(f"SL actual para {symbol} (SHORT) es suficientemente bueno, no se modifica.")
        else:
            # Silenciado: print(f"positionSide desconocido para {symbol}: {positionSide}")
            continue
  
