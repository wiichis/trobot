import pandas as pd
import matplotlib.pyplot as plt
import talib
import logging
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import random
import numpy as np

random.seed(42)
np.random.seed(42)

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

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
DISABLED_COINS = ["HBAR-USDT", "DOT-USDT", "LTC-USDT", "ADA-USDT"]
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

def backtest_strategy(data, use_dynamic_sl=True):
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

                # Ajustar trailing stop con la siguiente vela solo si se usa SL dinámico
                if exit_price is None and use_dynamic_sl:
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
                    position['exit_date'] = next_row['date']
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

    # Procesar la última vela
    last_row = data.iloc[-1]
    date = last_row['date']
    for sym in list(open_positions.keys()):
        position = open_positions[sym]
        if position['is_open']:
            current_price = last_row['close']
            if position['type'] == 'LONG':
                raw_profit = (current_price - position['entry_price']) * position['size']
            else:
                raw_profit = (position['entry_price'] - current_price) * position['size']
            commission_cost = (position['entry_price'] + current_price) * position['size'] * COMMISSION_RATE
            profit = raw_profit - commission_cost
            position['commission'] = commission_cost

            balance += profit
            position['is_open'] = False
            position['exit_date'] = date
            position['exit_price'] = current_price
            position['profit'] = profit
            trade_log.append(position)
            
            # Actualizar rendimiento acumulado para la moneda en la última vela
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
    data.sort_values(by=['symbol', 'date'], inplace=True)
    data = data.copy()
    symbols = data['symbol'].unique()

    for symbol in symbols:
        df_symbol = data[data['symbol'] == symbol].copy()
        df_symbol = df_symbol[df_symbol['close'] > 0]
        df_symbol['close'] = df_symbol['close'].ffill()
        df_symbol['open'] = df_symbol['open'].ffill()
        df_symbol['high'] = df_symbol['high'].ffill()
        df_symbol['low'] = df_symbol['low'].ffill()
        df_symbol['volume'] = df_symbol['volume'].fillna(0)

        required_periods = max(rsi_period, atr_period, ema_long_period, adx_period, 26)
        if len(df_symbol) < required_periods:
            continue

        try:
            df_symbol['RSI'] = talib.RSI(df_symbol['close'], timeperiod=rsi_period)
            df_symbol['ATR'] = talib.ATR(df_symbol['high'], df_symbol['low'], df_symbol['close'], timeperiod=atr_period)
            df_symbol['OBV'] = talib.OBV(df_symbol['close'], df_symbol['volume'])
            df_symbol['OBV_Slope'] = df_symbol['OBV'].diff()
            df_symbol['EMA_Short'] = talib.EMA(df_symbol['close'], timeperiod=ema_short_period)
            df_symbol['EMA_Long'] = talib.EMA(df_symbol['close'], timeperiod=ema_long_period)
            df_symbol['MACD'], df_symbol['MACD_Signal'], df_symbol['MACD_Hist'] = talib.MACD(
                df_symbol['close'], fastperiod=12, slowperiod=26, signalperiod=9)
            df_symbol['MACD_Bullish'] = (
                (df_symbol['MACD'] > df_symbol['MACD_Signal']) &
                (df_symbol['MACD'].shift(1) <= df_symbol['MACD_Signal'].shift(1))
            ).astype('boolean')
            df_symbol['MACD_Bearish'] = (
                (df_symbol['MACD'] < df_symbol['MACD_Signal']) &
                (df_symbol['MACD'].shift(1) >= df_symbol['MACD_Signal'].shift(1))
            ).astype('boolean')
            df_symbol['Hammer'] = talib.CDLHAMMER(
                df_symbol['open'], df_symbol['high'], df_symbol['low'], df_symbol['close'])
            df_symbol['ShootingStar'] = talib.CDLSHOOTINGSTAR(
                df_symbol['open'], df_symbol['high'], df_symbol['low'], df_symbol['close'])
            df_symbol['Avg_Volume'] = df_symbol['volume'].rolling(window=20).mean()
            df_symbol['Rel_Volume'] = df_symbol['volume'] / df_symbol['Avg_Volume']
            df_symbol['Volatility'] = df_symbol['ATR']
            df_symbol['Avg_Volatility'] = df_symbol['Volatility'].rolling(window=20).mean()
            df_symbol['Rel_Volatility'] = df_symbol['Volatility'] / df_symbol['Avg_Volatility']
            df_symbol['ADX'] = talib.ADX(
                df_symbol['high'], df_symbol['low'], df_symbol['close'], timeperiod=adx_period)
            df_symbol['Take_Profit_Long'] = df_symbol['close'] + (df_symbol['ATR'] * tp_multiplier)
            df_symbol['Stop_Loss_Long'] = (df_symbol['close'] - (df_symbol['ATR'] * sl_multiplier)).clip(lower=1e-8)
            df_symbol['Take_Profit_Short'] = df_symbol['close'] - (df_symbol['ATR'] * tp_multiplier)
            df_symbol['Stop_Loss_Short'] = df_symbol['close'] + (df_symbol['ATR'] * sl_multiplier)
            df_symbol['Take_Profit_Short'] = df_symbol['Take_Profit_Short'].clip(lower=1e-8)
            df_symbol = df_symbol.ffill().fillna(0)
            bool_cols = df_symbol.select_dtypes(include=['boolean']).columns
            df_symbol[bool_cols] = df_symbol[bool_cols].astype(float)
            data.loc[df_symbol.index, df_symbol.columns] = df_symbol
        except Exception as e:
            continue

    data['Trend_Up'] = data['EMA_Short'] > data['EMA_Long']
    data['Trend_Down'] = data['EMA_Short'] < data['EMA_Long']
    data['Hammer'] = data['Hammer'].fillna(0)
    data['ShootingStar'] = data['ShootingStar'].fillna(0)
    data['MACD_Bullish'] = data['MACD_Bullish'].fillna(False).astype('boolean')
    data['MACD_Bearish'] = data['MACD_Bearish'].fillna(False).astype('boolean')
    data['ADX'] = data['ADX'].fillna(0)
    data['Trend_Up'] = data['Trend_Up'].astype('boolean')
    data['Trend_Down'] = data['Trend_Down'].astype('boolean')
    data['Low_Volume'] = data['Rel_Volume'] < volume_threshold
    data['High_Volatility'] = data['Rel_Volatility'] > volatility_threshold
    data['EMA_Long_Term'] = talib.EMA(data['close'], timeperiod=50)
    data['Trend_Up_Long_Term'] = data['EMA_Short'] > data['EMA_Long_Term']
    data['Trend_Down_Long_Term'] = data['EMA_Short'] < data['EMA_Long_Term']
    
    data['Long_Signal'] = (
        ((data['Hammer'] != 0) & data['Trend_Up'] & data['Trend_Up_Long_Term'] & (data['RSI'] < rsi_oversold)) |
        (data['MACD_Bullish'] & (data['ADX'] > 25) & data['Trend_Up_Long_Term'] & (data['RSI'] < rsi_overbought))
    ) & (~data['Low_Volume']) & (~data['High_Volatility'])
    data['Long_Signal'] = data['Long_Signal'].astype('boolean')

    data['Short_Signal'] = (
        ((data['ShootingStar'] != 0) & data['Trend_Down'] & data['Trend_Down_Long_Term'] & (data['RSI'] > rsi_overbought)) |
        (data['MACD_Bearish'] & (data['ADX'] > 25) & data['Trend_Down_Long_Term'] & (data['RSI'] > rsi_oversold))
    ) & (~data['Low_Volume']) & (~data['High_Volatility'])
    data['Short_Signal'] = data['Short_Signal'].astype('boolean')

    data.reset_index(drop=True, inplace=True)
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
        final_balance, _, _ = backtest_strategy(data)
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
        'rsi_ob': hp.choice('rsi_ob', [65, 69, 76])
    }
    
    trials = Trials()
    best = fmin(fn=objective, 
                space=space, 
                algo=tpe.suggest, 
                max_evals=max_evals, 
                trials=trials, 
                rstate=np.random.default_rng(42))
    
    rsi_options = [8, 12, 14]
    atr_options = [10, 12, 14]
    ema_short_options = [8, 10, 12]
    ema_long_options = [18, 25, 30]
    adx_options = [7, 8, 10]
    tp_mult_options = [1, 4, 15]
    sl_mult_options = [0.5, 2, 5]
    rsi_os_options = [25, 34, 38]
    rsi_ob_options = [65, 69, 73]
    
    best_params = {
        'rsi': rsi_options[best['rsi']],
        'atr': atr_options[best['atr']],
        'ema_short': ema_short_options[best['ema_short']],
        'ema_long': ema_long_options[best['ema_long']],
        'adx': adx_options[best['adx']],
        'tp_mult': tp_mult_options[best['tp_mult']],
        'sl_mult': sl_mult_options[best['sl_mult']],
        'rsi_os': rsi_os_options[best['rsi_os']],
        'rsi_ob': rsi_ob_options[best['rsi_ob']]
    }
    
    print('Best parameters found:', best_params)
    return best_params

def main():
    data = load_data('./archivos/cripto_price.csv')
    if data is None:
        print("No se pudo cargar data, revisa tu CSV.")
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

    # Ejecutar backtest con SL dinámico
    final_balance_dynamic, trades_dynamic, equity_curve_dynamic = backtest_strategy(data, use_dynamic_sl=True)
    print("Balance final con SL dinámico:", final_balance_dynamic)
    df_trades_dynamic = pd.DataFrame(trades_dynamic)
    df_trades_dynamic.to_csv('./archivos/backtesting_results_dynamic.csv', index=False)
    df_equity_dynamic = pd.DataFrame(equity_curve_dynamic)
    df_equity_dynamic.to_csv('./archivos/equity_curve_dynamic.csv', index=False)
    
    # Mostrar gráfico de resultados para la versión dinámica (opcional)
    plot_simulation_results('./archivos/backtesting_results_dynamic.csv')

if __name__ == "__main__":
    main()