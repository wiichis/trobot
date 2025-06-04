import pandas as pd
import matplotlib.pyplot as plt
import talib
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# =============================
# SECCIÓN DE VARIABLES
# =============================
INITIAL_BALANCE = 300  # Balance inicial en dólares
POSITION_SIZE_PERCENTAGE = 0.1  # Porcentaje del balance para cada posición

# Parámetros de indicadores (valores iniciales, se actualizarán con la optimización)
RSI_PERIOD = 14  
ATR_PERIOD = 14  
EMA_SHORT_PERIOD = 8  
EMA_LONG_PERIOD = 20  
ADX_PERIOD = 7  

TP_MULTIPLIER = 5  
SL_MULTIPLIER = 2.5  

VOLUME_THRESHOLD = 0.68  
VOLATILITY_THRESHOLD = 1.07  

RSI_OVERSOLD = 30  
RSI_OVERBOUGHT = 65

# Comisión en futuros de BingX (suponiendo que se opera como taker: 0.04% en entrada y salida)
COMMISSION_RATE = 0.0004

# Nuevo parámetro: solo operar monedas con rendimiento acumulado >= PERFORMANCE_THRESHOLD
PERFORMANCE_THRESHOLD = 0  

# Lista de monedas deshabilitadas para no operar en la simulación (monedas con bajo rendimiento)
DISABLED_COINS = ["ADA-USDT", "SHIB-USDT", "AVAX-USDT", "BTC-USDT","XRP-USDT","LTC-USDT"]
# =============================
# FIN DE LA SECCIÓN DE VARIABLES
# =============================

def load_data(filepath='./archivos/cripto_price.csv'):
    try:
        crypto_data = pd.read_csv(filepath, parse_dates=['date'])
        crypto_data.sort_values(by='date', inplace=True)
        return crypto_data
    except Exception as e:
        print(f"Error leyendo el archivo: {e}")
        return None

def filter_duplicates(crypto_data):
    crypto_data = crypto_data.drop_duplicates()
    crypto_data = crypto_data[crypto_data['volume'] > 0]
    crypto_data = crypto_data.reset_index(drop=True)
    return crypto_data

# =============================
# Confirmación de señales usando velas de 5 minutos
# =============================
def add_5m_confirmation(data_30m, data_5m, symbol_col='symbol', ema_fast_col='EMA_Short', ema_slow_col='EMA_Long', min_confirm=4):
    # Creamos un campo de intervalo 30m para cada fila
    data_5m = data_5m.copy()
    data_5m['interval_start'] = data_5m['date'].dt.floor('30T')
    
    # Juntamos las de 30m con las de 5m en intervalos
    merged = data_5m.merge(
        data_30m[[symbol_col, 'date']],
        left_on=[symbol_col, 'interval_start'],
        right_on=[symbol_col, 'date'],
        how='inner',
        suffixes=('_5m', '_30m')
    )
    # Confirmaciones por intervalo
    confirm = merged.groupby(['date_30m', symbol_col]).apply(
        lambda x: (x[ema_fast_col] > x[ema_slow_col]).sum() >= min_confirm
    ).reset_index(name='Confirmacion_5m')
    
    # Mezclar con data_30m
    data_30m = data_30m.merge(confirm, left_on=['date', symbol_col], right_on=['date_30m', symbol_col], how='left')
    if 'date_30m' in data_30m.columns:
        data_30m = data_30m.drop(columns=['date_30m'])
    data_30m['Confirmacion_5m'] = data_30m['Confirmacion_5m'].fillna(False)
    return data_30m

def backtest_strategy(data, use_dynamic_sl=True, data_5m=None):
    balance = INITIAL_BALANCE
    equity_curve = []
    trade_log = []
    
    # Nuevo diccionario para acumular rendimiento por moneda
    coin_performance = {}

    # Preprocesamiento y cálculo de indicadores
    data = filter_duplicates(data)
    data = calculate_indicators(
        data,
        rsi_period=RSI_PERIOD,
        atr_period=ATR_PERIOD,
        ema_short_period=EMA_SHORT_PERIOD,
        ema_long_period=EMA_LONG_PERIOD,
        adx_period=ADX_PERIOD,
        tp_multiplier=TP_MULTIPLIER,
        sl_multiplier=SL_MULTIPLIER,
        volume_threshold=VOLUME_THRESHOLD,
        volatility_threshold=VOLATILITY_THRESHOLD,
        rsi_oversold=RSI_OVERSOLD,
        rsi_overbought=RSI_OVERBOUGHT
    )
    # Asegurar orden cronológico de los datos tras indicadores
    data.sort_values(by='date', inplace=True)
    data.reset_index(drop=True, inplace=True)

    open_positions = {}

    for i in range(len(data) - 1):
        current_row = data.iloc[i]
        next_row = data.iloc[i + 1]
        date = current_row['date']
        symbol = current_row['symbol']
        close_price = current_row['close']
        high_price = current_row['high']
        low_price = current_row['low']

        # Actualizar posiciones abiertas
        for sym in list(open_positions.keys()):
            position = open_positions[sym]
            if position['is_open']:
                exit_price = None
                exit_date = None
                # --- Si hay data_5m, usar precios intra-vela para gestión de SL/TP/Trailing ---
                if data_5m is not None:
                    # Determinar el rango de tiempo de la vela 30m actual
                    entry_30m_date = current_row['date']
                    next_30m_date = next_row['date']
                    # Extraer las 6 velas de 5m para el símbolo
                    mask_5m = (
                        (data_5m['symbol'] == sym) &
                        (data_5m['date'] >= entry_30m_date) &
                        (data_5m['date'] < next_30m_date)
                    )
                    sub_5m = data_5m[mask_5m].copy()
                    # Si no hay velas de 5m, fallback al comportamiento de 30m
                    if len(sub_5m) == 0:
                        sub_5m = None
                else:
                    sub_5m = None

                if sub_5m is not None and len(sub_5m) > 0:
                    # Recorrer las velas de 5m en orden cronológico
                    for idx_5m, row_5m in sub_5m.iterrows():
                        high_5m = row_5m['high']
                        low_5m = row_5m['low']
                        close_5m = row_5m['close']
                        date_5m = row_5m['date']
                        # Comprobar ejecución de SL/TP intra-vela
                        if position['type'] == 'LONG':
                            if low_5m <= position['stop_loss']:
                                exit_price = position['stop_loss']
                                exit_date = date_5m
                                break
                            elif high_5m >= position['take_profit']:
                                exit_price = position['take_profit']
                                exit_date = date_5m
                                break
                        elif position['type'] == 'SHORT':
                            if high_5m >= position['stop_loss']:
                                exit_price = position['stop_loss']
                                exit_date = date_5m
                                break
                            elif low_5m <= position['take_profit']:
                                exit_price = position['take_profit']
                                exit_date = date_5m
                                break
                        # Trailing stop dinámico intra-vela
                        if exit_price is None and use_dynamic_sl:
                            if 'trailing_distance' not in position:
                                if position['type'] == 'LONG':
                                    position['trailing_distance'] = position['entry_price'] - position['stop_loss']
                                else:
                                    position['trailing_distance'] = position['stop_loss'] - position['entry_price']
                            if position['type'] == 'LONG':
                                new_sl = close_5m - position['trailing_distance']
                                if low_5m <= new_sl:
                                    exit_price = new_sl
                                    exit_date = date_5m
                                    break
                                elif new_sl > position['stop_loss']:
                                    position['stop_loss'] = new_sl
                            elif position['type'] == 'SHORT':
                                new_sl = close_5m + position['trailing_distance']
                                if high_5m >= new_sl:
                                    exit_price = new_sl
                                    exit_date = date_5m
                                    break
                                elif new_sl < position['stop_loss']:
                                    position['stop_loss'] = new_sl
                else:
                    # --- Comportamiento original solo con vela 30m ---
                    if position['type'] == 'LONG':
                        if low_price <= position['stop_loss']:
                            exit_price = position['stop_loss']
                        elif high_price >= position['take_profit']:
                            exit_price = position['take_profit']
                    elif position['type'] == 'SHORT':
                        if high_price >= position['stop_loss']:
                            exit_price = position['stop_loss']
                        elif low_price <= position['take_profit']:
                            exit_price = position['take_profit']
                    # Ajustar trailing stop solo con vela siguiente del mismo símbolo
                    if exit_price is None and use_dynamic_sl and next_row['symbol'] == sym:
                        if 'trailing_distance' not in position:
                            if position['type'] == 'LONG':
                                position['trailing_distance'] = position['entry_price'] - position['stop_loss']
                            else:
                                position['trailing_distance'] = position['stop_loss'] - position['entry_price']
                        if position['type'] == 'LONG':
                            new_sl = next_row['close'] - position['trailing_distance']
                            if next_row['low'] <= new_sl:
                                exit_price = new_sl
                            elif new_sl > position['stop_loss']:
                                position['stop_loss'] = new_sl
                        elif position['type'] == 'SHORT':
                            new_sl = next_row['close'] + position['trailing_distance']
                            if next_row['high'] >= new_sl:
                                exit_price = new_sl
                            elif new_sl < position['stop_loss']:
                                position['stop_loss'] = new_sl
                    if exit_price is not None:
                        exit_date = next_row['date']

                if exit_price is not None:
                    # Calcular ganancia bruta y luego restar comisiones en entrada y salida
                    if position['type'] == 'LONG':
                        raw_profit = (exit_price - position['entry_price']) * position['size']
                    else:
                        raw_profit = (position['entry_price'] - exit_price) * position['size']
                    commission_cost = (position['entry_price'] + exit_price) * position['size'] * COMMISSION_RATE
                    profit = raw_profit - commission_cost
                    position['commission'] = commission_cost

                    balance += profit
                    position['is_open'] = False
                    position['exit_date'] = exit_date if exit_date is not None else next_row['date']
                    position['exit_price'] = exit_price
                    position['profit'] = profit
                    trade_log.append(position)
                    
                    # Actualizar rendimiento acumulado para la moneda
                    if sym in coin_performance:
                        coin_performance[sym] += profit
                    else:
                        coin_performance[sym] = profit

                    del open_positions[sym]

        # Filtrar períodos con bajo volumen o alta volatilidad
        if current_row.get('Low_Volume', False) or current_row.get('High_Volatility', False):
            continue

        # Nuevo: solo abrir posición si la moneda cumple el rendimiento mínimo
        if symbol in coin_performance and coin_performance[symbol] < PERFORMANCE_THRESHOLD:
            continue

        # Nuevo: omitir apertura de nuevas posiciones para monedas deshabilitadas
        if symbol in DISABLED_COINS and symbol not in open_positions:
            continue

        # Abrir nuevas posiciones si no hay posición abierta para el símbolo
        if symbol not in open_positions and balance > 0:
            position_size = balance * POSITION_SIZE_PERCENTAGE
            size = position_size / close_price

            if current_row.get('Long_Signal', False):
                take_profit = current_row['Take_Profit_Long']
                stop_loss = current_row['Stop_Loss_Long']
                position = {
                    'symbol': symbol,
                    'type': 'LONG',
                    'entry_date': date,
                    'entry_price': close_price,
                    'take_profit': take_profit,
                    'stop_loss': stop_loss,
                    'size': size,
                    'is_open': True
                }
                open_positions[symbol] = position
            elif current_row.get('Short_Signal', False):
                take_profit = current_row['Take_Profit_Short']
                stop_loss = current_row['Stop_Loss_Short']
                position = {
                    'symbol': symbol,
                    'type': 'SHORT',
                    'entry_date': date,
                    'entry_price': close_price,
                    'take_profit': take_profit,
                    'stop_loss': stop_loss,
                    'size': size,
                    'is_open': True
                }
                open_positions[symbol] = position

        equity_curve.append({'date': date, 'balance': balance})

    # Procesar cierre de posiciones abiertas al final de los datos por símbolo
    for sym, position in list(open_positions.items()):
        if position['is_open']:
            # Obtener última fila de datos de ese símbolo
            sym_data = data[data['symbol'] == sym]
            last_sym_row = sym_data.iloc[-1]
            exit_price = last_sym_row['close']
            exit_date = last_sym_row['date']
            # Calcular ganancia bruta y restar comisiones en entrada y salida
            if position['type'] == 'LONG':
                raw_profit = (exit_price - position['entry_price']) * position['size']
            else:
                raw_profit = (position['entry_price'] - exit_price) * position['size']
            commission_cost = (position['entry_price'] + exit_price) * position['size'] * COMMISSION_RATE
            profit = raw_profit - commission_cost
            position['commission'] = commission_cost
            balance += profit
            position['is_open'] = False
            position['exit_date'] = exit_date
            position['exit_price'] = exit_price
            position['profit'] = profit
            trade_log.append(position)
            # Actualizar rendimiento acumulado para la moneda
            if sym in coin_performance:
                coin_performance[sym] += profit
            else:
                coin_performance[sym] = profit

    return balance, trade_log, equity_curve

def calculate_indicators(
    data,
    rsi_period=14,
    atr_period=14,
    ema_short_period=12,
    ema_long_period=26,
    adx_period=14,
    tp_multiplier=4.5,
    sl_multiplier=1.5,
    volume_threshold=0.8,
    volatility_threshold=1.2,
    rsi_oversold=30,
    rsi_overbought=70,
    output_filepath='./archivos/indicadores.csv'
):
    data = data.sort_values(by=['symbol', 'date'])
    data.ffill(inplace=True)
    data['volume'] = data['volume'].fillna(0)
    symbols = data['symbol'].unique()

    # Usar groupby para procesar todos los símbolos de manera eficiente
    for symbol, df_symbol in data.groupby('symbol'):
        df_symbol = df_symbol[df_symbol['close'] > 0]

        required_periods = max(rsi_period, atr_period, ema_long_period, adx_period, 26)
        if len(df_symbol) < required_periods:
            continue

        try:
            # Indicadores vectorizados
            data.loc[df_symbol.index, 'RSI'] = talib.RSI(df_symbol['close'], timeperiod=rsi_period)
            data.loc[df_symbol.index, 'ATR'] = talib.ATR(df_symbol['high'], df_symbol['low'], df_symbol['close'], timeperiod=atr_period)
            data.loc[df_symbol.index, 'OBV'] = talib.OBV(df_symbol['close'], df_symbol['volume'])
            data.loc[df_symbol.index, 'OBV_Slope'] = data.loc[df_symbol.index, 'OBV'].diff()
            data.loc[df_symbol.index, 'EMA_Short'] = talib.EMA(df_symbol['close'], timeperiod=ema_short_period)
            data.loc[df_symbol.index, 'EMA_Long'] = talib.EMA(df_symbol['close'], timeperiod=ema_long_period)
            macd, macd_signal, macd_hist = talib.MACD(df_symbol['close'], fastperiod=12, slowperiod=26, signalperiod=9)
            data.loc[df_symbol.index, 'MACD'] = macd
            data.loc[df_symbol.index, 'MACD_Signal'] = macd_signal
            data.loc[df_symbol.index, 'MACD_Hist'] = macd_hist
            data.loc[df_symbol.index, 'MACD_Bullish'] = (
                (macd > macd_signal) & (macd.shift(1) <= macd_signal.shift(1))
            ).astype('float')
            data.loc[df_symbol.index, 'MACD_Bearish'] = (
                (macd < macd_signal) & (macd.shift(1) >= macd_signal.shift(1))
            ).astype('float')
            data.loc[df_symbol.index, 'Hammer'] = talib.CDLHAMMER(
                df_symbol['open'], df_symbol['high'], df_symbol['low'], df_symbol['close'])
            data.loc[df_symbol.index, 'ShootingStar'] = talib.CDLSHOOTINGSTAR(
                df_symbol['open'], df_symbol['high'], df_symbol['low'], df_symbol['close'])
            data.loc[df_symbol.index, 'Avg_Volume'] = df_symbol['volume'].rolling(window=20).mean()
            data.loc[df_symbol.index, 'Rel_Volume'] = df_symbol['volume'] / data.loc[df_symbol.index, 'Avg_Volume']
            data.loc[df_symbol.index, 'Volatility'] = data.loc[df_symbol.index, 'ATR']
            data.loc[df_symbol.index, 'Avg_Volatility'] = data.loc[df_symbol.index, 'Volatility'].rolling(window=20).mean()
            data.loc[df_symbol.index, 'Rel_Volatility'] = data.loc[df_symbol.index, 'Volatility'] / data.loc[df_symbol.index, 'Avg_Volatility']
            data.loc[df_symbol.index, 'ADX'] = talib.ADX(
                df_symbol['high'], df_symbol['low'], df_symbol['close'], timeperiod=adx_period)
            data.loc[df_symbol.index, 'Take_Profit_Long'] = df_symbol['close'] + (data.loc[df_symbol.index, 'ATR'] * tp_multiplier)
            data.loc[df_symbol.index, 'Stop_Loss_Long'] = (df_symbol['close'] - (data.loc[df_symbol.index, 'ATR'] * sl_multiplier)).clip(lower=1e-8)
            data.loc[df_symbol.index, 'Take_Profit_Short'] = (df_symbol['close'] - (data.loc[df_symbol.index, 'ATR'] * tp_multiplier)).clip(lower=1e-8)
            data.loc[df_symbol.index, 'Stop_Loss_Short'] = df_symbol['close'] + (data.loc[df_symbol.index, 'ATR'] * sl_multiplier)
        except Exception as e:
            continue

    # Lógica para las columnas, señales, etc. igual que antes
    data['Trend_Up'] = data['EMA_Short'] > data['EMA_Long']
    data['Trend_Down'] = data['EMA_Short'] < data['EMA_Long']
    data['Trend_Up_Long_Term'] = data['EMA_Short'] > talib.EMA(data['close'], timeperiod=50)
    data['Trend_Down_Long_Term'] = data['EMA_Short'] < talib.EMA(data['close'], timeperiod=50)
    data['Hammer'] = data['Hammer'].fillna(0)
    data['ShootingStar'] = data['ShootingStar'].fillna(0)
    data['MACD_Bullish'] = data['MACD_Bullish'].fillna(0)
    data['MACD_Bearish'] = data['MACD_Bearish'].fillna(0)
    data['ADX'] = data['ADX'].fillna(0)
    data['Rel_Volume'] = data['Rel_Volume'].fillna(1)
    data['Rel_Volatility'] = data['Rel_Volatility'].fillna(1)
    data['Low_Volume'] = data['Rel_Volume'] < volume_threshold
    data['High_Volatility'] = data['Rel_Volatility'] > volatility_threshold

    # Señales, igual que antes (puedes mantener esa parte)
    # ...

    data.reset_index(drop=True, inplace=True)

    # Señales básicas Long/Short (sobrescribe si existen)
    data['Long_Signal'] = (
        (data['Trend_Up']) &
        (data['RSI'] < rsi_overbought) &
        (data['MACD_Bullish'] > 0) &
        (data['ADX'] > 15)
    )

    data['Short_Signal'] = (
        (data['Trend_Down']) &
        (data['RSI'] > rsi_oversold) &
        (data['MACD_Bearish'] > 0) &
        (data['ADX'] > 15)
    )

    return data

def plot_simulation_results(csv_file='./archivos/backtesting_results.csv'):
    df = pd.read_csv(csv_file)
    # Agrupar sumando profit y commission
    resultados = df.groupby('symbol').agg({'profit': 'sum', 'commission': 'sum'}).reset_index()
    resultados = resultados.sort_values('profit', ascending=False)

    x = range(len(resultados))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar([p - width/2 for p in x], resultados['profit'], width, label='Profit')
    ax.bar([p + width/2 for p in x], resultados['commission'], width, label='Comisión')
    ax.set_xticks(x)
    ax.set_xticklabels(resultados['symbol'])
    ax.set_xlabel('Moneda')
    ax.set_ylabel('Monto ($)')
    ax.set_title('Resultado Global y Comisiones por Moneda')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def optimize_parameters(data, max_evals=50):
    import numpy as np
    def objective(params):
        global RSI_PERIOD, ATR_PERIOD, EMA_SHORT_PERIOD, EMA_LONG_PERIOD, ADX_PERIOD
        global TP_MULTIPLIER, SL_MULTIPLIER, RSI_OVERSOLD, RSI_OVERBOUGHT
        RSI_PERIOD = params['rsi']
        ATR_PERIOD = params['atr']
        EMA_SHORT_PERIOD = params['ema_short']
        EMA_LONG_PERIOD = params['ema_long']
        ADX_PERIOD = params['adx']
        TP_MULTIPLIER = params['tp_mult']
        SL_MULTIPLIER = params['sl_mult']
        RSI_OVERSOLD = params['rsi_os']
        RSI_OVERBOUGHT = params['rsi_ob']
        # Cargar datos de 5m para cada evaluación
        data_5m = load_data('./archivos/cripto_price_5m.csv')
        if data_5m is None:
            return {'loss': 1e10, 'status': STATUS_OK}
        # Calcular indicadores para 5m usando los parámetros optimizados
        data_5m_tmp = calculate_indicators(
            data_5m.copy(),
            rsi_period=params['rsi_5m'],
            ema_short_period=params['ema_short_5m'],
            ema_long_period=params['ema_long_5m'],
            volume_threshold=params['volume_threshold'],
            volatility_threshold=params['volatility_threshold'],
        )
        # Calcular indicadores para 30m
        data_tmp = calculate_indicators(
            data.copy(),
            rsi_period=params['rsi'],
            atr_period=params['atr'],
            ema_short_period=params['ema_short'],
            ema_long_period=params['ema_long'],
            adx_period=params['adx'],
            tp_multiplier=params['tp_mult'],
            sl_multiplier=params['sl_mult'],
            volume_threshold=params['volume_threshold'],
            volatility_threshold=params['volatility_threshold'],
            rsi_oversold=params['rsi_os'],
            rsi_overbought=params['rsi_ob']
        )
        # Confirmación usando los parámetros optimizados de min_confirm_5m
        data_tmp = add_5m_confirmation(
            data_tmp,
            data_5m_tmp,
            ema_fast_col='EMA_Short',
            ema_slow_col='EMA_Long',
            min_confirm=params['min_confirm_5m']
        )
        data_tmp['Long_Signal'] = data_tmp['Long_Signal'] & data_tmp['Confirmacion_5m']
        data_tmp['Short_Signal'] = data_tmp['Short_Signal'] & data_tmp['Confirmacion_5m']
        final_balance, _, _ = backtest_strategy(data_tmp, use_dynamic_sl=True, data_5m=data_5m_tmp)
        profit = final_balance - INITIAL_BALANCE
        return {'loss': -profit, 'status': STATUS_OK}

    space = {
        'rsi': hp.choice('rsi', [8, 12, 14]),
        'atr': hp.choice('atr', [10, 12, 14]),
        'ema_short': hp.choice('ema_short', [5, 10, 16]),
        'ema_long': hp.choice('ema_long', [10, 25, 30]),
        'adx': hp.choice('adx', [5, 8, 10]),
        'tp_mult': hp.choice('tp_mult', [1, 4, 15]),
        'sl_mult': hp.choice('sl_mult', [0.5, 2, 7]),
        'rsi_os': hp.choice('rsi_os', [25, 34, 38]),
        'rsi_ob': hp.choice('rsi_ob', [65, 69, 76]),
        # Parámetros para 5m
        'ema_short_5m': hp.choice('ema_short_5m', [3, 5, 7]),
        'ema_long_5m': hp.choice('ema_long_5m', [7, 10, 14]),
        'rsi_5m': hp.choice('rsi_5m', [7, 10, 14]),
        'min_confirm_5m': hp.choice('min_confirm_5m', [3, 4, 5]),
        # Nuevos parámetros
        'volume_threshold': hp.uniform('volume_threshold', 0.5, 1.2),
        'volatility_threshold': hp.uniform('volatility_threshold', 0.9, 1.5)
    }

    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials,
                rstate=np.random.default_rng(42))

    # Opciones originales usadas en el espacio de búsqueda
    rsi_options = [8, 12, 14]
    atr_options = [10, 12, 14]
    ema_short_options = [5, 10, 16]
    ema_long_options = [10, 25, 30]
    adx_options = [5, 8, 10]
    tp_mult_options = [1, 4, 15]
    sl_mult_options = [0.5, 2, 7]
    rsi_os_options = [25, 34, 38]
    rsi_ob_options = [65, 69, 76]
    ema_short_5m_options = [3, 5, 7]
    ema_long_5m_options = [7, 10, 14]
    rsi_5m_options = [7, 10, 14]
    min_confirm_5m_options = [3, 4, 5]
    best_params = {
        'rsi': rsi_options[best['rsi']],
        'atr': atr_options[best['atr']],
        'ema_short': ema_short_options[best['ema_short']],
        'ema_long': ema_long_options[best['ema_long']],
        'adx': adx_options[best['adx']],
        'tp_mult': tp_mult_options[best['tp_mult']],
        'sl_mult': sl_mult_options[best['sl_mult']],
        'rsi_os': rsi_os_options[best['rsi_os']],
        'rsi_ob': rsi_ob_options[best['rsi_ob']],
        'ema_short_5m': ema_short_5m_options[best['ema_short_5m']],
        'ema_long_5m': ema_long_5m_options[best['ema_long_5m']],
        'rsi_5m': rsi_5m_options[best['rsi_5m']],
        'min_confirm_5m': min_confirm_5m_options[best['min_confirm_5m']],
        'volume_threshold': best['volume_threshold'],
        'volatility_threshold': best['volatility_threshold']
    }

    print("\n=== Parámetros Óptimos Encontrados ===")
    for param, value in best_params.items():
        print(f"{param:18}: {value}")
    print("====================================\n")
    return best_params

def main():
    # Cargar datos de 30m
    data = load_data('./archivos/cripto_price.csv')
    if data is None:
        print("No se pudo cargar data, revisa tu CSV.")
        return

    # Cargar datos de 5m
    data_5m = load_data('./archivos/cripto_price_5m.csv')
    if data_5m is None:
        print("No se pudo cargar data de 5m, revisa tu CSV.")
        return

    # Optimizar parámetros usando hyperopt
    best_params = optimize_parameters(data, max_evals=50)
    global RSI_PERIOD, ATR_PERIOD, EMA_SHORT_PERIOD, EMA_LONG_PERIOD, ADX_PERIOD
    global TP_MULTIPLIER, SL_MULTIPLIER, RSI_OVERSOLD, RSI_OVERBOUGHT
    RSI_PERIOD       = best_params['rsi']
    ATR_PERIOD       = best_params['atr']
    EMA_SHORT_PERIOD = best_params['ema_short']
    EMA_LONG_PERIOD  = best_params['ema_long']
    ADX_PERIOD       = best_params['adx']
    TP_MULTIPLIER    = best_params['tp_mult']
    SL_MULTIPLIER    = best_params['sl_mult']
    RSI_OVERSOLD     = best_params['rsi_os']
    RSI_OVERBOUGHT   = best_params['rsi_ob']

    # Calcular indicadores para 5 minutos con valores óptimos de 5m y nuevos parámetros
    data_5m = calculate_indicators(
        data_5m,
        rsi_period=best_params['rsi_5m'],
        ema_short_period=best_params['ema_short_5m'],
        ema_long_period=best_params['ema_long_5m'],
        volume_threshold=best_params['volume_threshold'],
        volatility_threshold=best_params['volatility_threshold'],
    )

    # Impresión de datos de simulación final
    print("\n=== DATOS PARA SIMULACIÓN FINAL ===")
    print(f"Filas cargadas      : {len(data):,}")
    print(f"Símbolos únicos     : {data['symbol'].nunique()}")
    print("============================\n")

    # Calcular indicadores para 30m (por si no están calculados aún)
    data = calculate_indicators(
        data,
        rsi_period=RSI_PERIOD,
        atr_period=ATR_PERIOD,
        ema_short_period=EMA_SHORT_PERIOD,
        ema_long_period=EMA_LONG_PERIOD,
        adx_period=ADX_PERIOD,
        tp_multiplier=TP_MULTIPLIER,
        sl_multiplier=SL_MULTIPLIER,
        volume_threshold=best_params['volume_threshold'],
        volatility_threshold=best_params['volatility_threshold'],
        rsi_oversold=RSI_OVERSOLD,
        rsi_overbought=RSI_OVERBOUGHT
    )

    # Confirmación por 5m antes del backtest usando los valores óptimos de min_confirm_5m
    data = add_5m_confirmation(
        data,
        data_5m,
        ema_fast_col='EMA_Short',
        ema_slow_col='EMA_Long',
        min_confirm=best_params['min_confirm_5m']
    )
    data['Long_Signal'] = data['Long_Signal'] & data['Confirmacion_5m']
    data['Short_Signal'] = data['Short_Signal'] & data['Confirmacion_5m']

    # Ejecutar backtest con SL dinámico, pasando data_5m
    final_balance_dynamic, trades_dynamic, equity_curve_dynamic = backtest_strategy(data, use_dynamic_sl=True, data_5m=data_5m)
    # Impresión formateada de resultados finales
    print("\n=== RESULTADOS FINALES ===")
    print(f"Balance final con SL dinámico: USD {final_balance_dynamic:,.2f}\n")
    df_trades_dynamic = pd.DataFrame(trades_dynamic)
    df_trades_dynamic.to_csv('./archivos/backtesting_results_dynamic.csv', index=False)

    # Resumen de trades por moneda
    summary = df_trades_dynamic.groupby('symbol')['profit'].agg(['count', 'sum', 'mean'])
    summary = summary.sort_values(by='sum', ascending=False)
    print("----- Resumen de trades por moneda (ordenado por profit desc) -----")
    print(summary.to_string(
        float_format=lambda x: f"{x:,.4f}",
        header=True,
        index_names=False,
        justify='center'
    ))
    print("-------------------------------------------------------------\n")

    # Mostrar detalles para las 3 monedas con mayor ganancia
    top_n = 3
    top_symbols = summary.head(top_n).index.tolist()
    for symbol in top_symbols:
        stats = summary.loc[symbol]
        print(f"\n{symbol} -> Trades: {int(stats['count'])}, Ganancia total: {stats['sum']:.2f}, Ganancia promedio: {stats['mean']:.4f}")
        print(f"\nDetalles de {symbol} trades:")
        trades = df_trades_dynamic[df_trades_dynamic['symbol'] == symbol]
        print(trades[['entry_date', 'entry_price', 'exit_price', 'size', 'profit']])

    df_equity_dynamic = pd.DataFrame(equity_curve_dynamic)
    df_equity_dynamic.to_csv('./archivos/equity_curve_dynamic.csv', index=False)
    
    # Mostrar gráfico de resultados para la versión dinámica (opcional)
    plot_simulation_results('./archivos/backtesting_results_dynamic.csv')

    print("\n=== BLOQUE PARA PEGAR EN VARIABLES INICIALES ===\n")
    print(f"RSI_PERIOD = {best_params['rsi']}")
    print(f"ATR_PERIOD = {best_params['atr']}")
    print(f"EMA_SHORT_PERIOD = {best_params['ema_short']}")
    print(f"EMA_LONG_PERIOD = {best_params['ema_long']}")
    print(f"ADX_PERIOD = {best_params['adx']}")
    print(f"TP_MULTIPLIER = {best_params['tp_mult']}")
    print(f"SL_MULTIPLIER = {best_params['sl_mult']}")
    print(f"VOLUME_THRESHOLD = {best_params['volume_threshold']:.4f}")
    print(f"VOLATILITY_THRESHOLD = {best_params['volatility_threshold']:.4f}")
    print(f"RSI_OVERSOLD = {best_params['rsi_os']}")
    print(f"RSI_OVERBOUGHT = {best_params['rsi_ob']}")

    # Top 5 monedas con peor rendimiento
    worst_symbols = summary.tail(5).index.tolist()
    coins_repr = "[" + ", ".join([f'\"{c}\"' for c in worst_symbols]) + "]"
    print(f"DISABLED_COINS = {coins_repr}")
    print("\n=== FIN DEL BLOQUE ===\n")

if __name__ == "__main__":
    main()