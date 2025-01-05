import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import talib
import logging

# Si las funciones están en otro archivo llamado 'indicadores.py', asegúrate de importarlas correctamente.
# En caso contrario, puedes definirlas en este mismo archivo.

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# =============================
# SECCIÓN DE VARIABLES
# =============================

# Parámetros generales
INITIAL_BALANCE = 300  # Balance inicial en dólares
POSITION_SIZE_PERCENTAGE = 0.2  # Porcentaje del balance para cada posición

# Parámetros de indicadores
RSI_PERIOD = 12  # Período del RSI
ATR_PERIOD = 12  # Período del ATR
EMA_SHORT_PERIOD = 10  # Período de la EMA corta
EMA_LONG_PERIOD = 26  # Período de la EMA larga
ADX_PERIOD = 16  # Período del ADX

# Multiplicadores para TP y SL basados en ATR
TP_MULTIPLIER = 5.2  # Multiplicador para el Take Profit
SL_MULTIPLIER = 1.1 # Multiplicador para el Stop Loss

# Umbrales para filtrar ruido del mercado
VOLUME_THRESHOLD = 0.68 # Umbral para volumen bajo (80% del volumen promedio)
VOLATILITY_THRESHOLD = 1.07  # Umbral para volatilidad alta (120% de la volatilidad promedio)

# Niveles de RSI para señales
RSI_OVERSOLD = 34  # Nivel de sobreventa para RSI
RSI_OVERBOUGHT = 69  # Nivel de sobrecompra para RSI

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
    # Eliminar duplicados exactos y aquellos con volumen igual a 0
    crypto_data = crypto_data.drop_duplicates()
    crypto_data = crypto_data[crypto_data['volume'] > 0]
    crypto_data = crypto_data.reset_index(drop=True)
    return crypto_data

def backtest_strategy(data):
    balance = INITIAL_BALANCE
    equity_curve = []
    trade_log = []

    # Preprocesamiento de datos
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

    for index, current_row in data.iterrows():
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

                # Verificar stop loss y take profit
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

                if exit_price is not None:
                    # Calcular ganancia/pérdida
                    if position['type'] == 'LONG':
                        profit = (exit_price - position['entry_price']) * position['size']
                    elif position['type'] == 'SHORT':
                        profit = (position['entry_price'] - exit_price) * position['size']

                    balance += profit  # Actualizar el balance con la ganancia/pérdida
                    position['is_open'] = False
                    position['exit_date'] = date
                    position['exit_price'] = exit_price
                    position['profit'] = profit
                    trade_log.append(position)
                    del open_positions[sym]

        # Filtrar períodos de bajo volumen y alta volatilidad
        if current_row.get('Low_Volume', False) or current_row.get('High_Volatility', False):
            continue  # Saltar este ciclo si hay bajo volumen o alta volatilidad

        # Abrir nuevas posiciones si no hay una abierta para el símbolo
        if symbol not in open_positions and balance > 0:
            position_size = balance * POSITION_SIZE_PERCENTAGE
            size = position_size / close_price  # Cantidad de unidades

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

        # Registrar el balance en cada paso
        equity_curve.append({'date': date, 'balance': balance})

    # Cerrar posiciones abiertas al final del período
    for position in open_positions.values():
        if position['is_open']:
            current_price = data[data['symbol'] == position['symbol']]['close'].iloc[-1]
            if position['type'] == 'LONG':
                profit = (current_price - position['entry_price']) * position['size']
            elif position['type'] == 'SHORT':
                profit = (position['entry_price'] - current_price) * position['size']
            balance += profit
            position['is_open'] = False
            position['exit_date'] = data['date'].iloc[-1]
            position['exit_price'] = current_price
            position['profit'] = profit
            trade_log.append(position)

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
    # Ordenar el DataFrame por símbolo y fecha
    data.sort_values(by=['symbol', 'date'], inplace=True)
    data = data.copy()

    symbols = data['symbol'].unique()

    for symbol in symbols:
        df_symbol = data[data['symbol'] == symbol].copy()

        # Validación y limpieza de datos
        df_symbol = df_symbol[df_symbol['close'] > 0]
        df_symbol['close'] = df_symbol['close'].ffill()
        df_symbol['open'] = df_symbol['open'].ffill()
        df_symbol['high'] = df_symbol['high'].ffill()
        df_symbol['low'] = df_symbol['low'].ffill()
        df_symbol['volume'] = df_symbol['volume'].fillna(0)

        # Verificar si hay suficientes datos
        required_periods = max(rsi_period, atr_period, ema_long_period, adx_period, 26)  # 26 es el período más largo usado en MACD
        if len(df_symbol) < required_periods:
            continue  # Saltar al siguiente símbolo si no hay suficientes datos

        try:
            # Calcular indicadores técnicos
            df_symbol['RSI'] = talib.RSI(df_symbol['close'], timeperiod=rsi_period)
            df_symbol['ATR'] = talib.ATR(
                df_symbol['high'], df_symbol['low'], df_symbol['close'], timeperiod=atr_period)
            df_symbol['OBV'] = talib.OBV(df_symbol['close'], df_symbol['volume'])
            df_symbol['OBV_Slope'] = df_symbol['OBV'].diff()

            # Medias Móviles Exponenciales
            df_symbol['EMA_Short'] = talib.EMA(df_symbol['close'], timeperiod=ema_short_period)
            df_symbol['EMA_Long'] = talib.EMA(df_symbol['close'], timeperiod=ema_long_period)

            # MACD
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

            # Patrones de velas
            df_symbol['Hammer'] = talib.CDLHAMMER(
                df_symbol['open'], df_symbol['high'], df_symbol['low'], df_symbol['close'])
            df_symbol['ShootingStar'] = talib.CDLSHOOTINGSTAR(
                df_symbol['open'], df_symbol['high'], df_symbol['low'], df_symbol['close'])

            # Calcular Volumen Promedio y Volumen Relativo
            df_symbol['Avg_Volume'] = df_symbol['volume'].rolling(window=20).mean()
            df_symbol['Rel_Volume'] = df_symbol['volume'] / df_symbol['Avg_Volume']

            # Calcular Volatilidad (usando el ATR)
            df_symbol['Volatility'] = df_symbol['ATR']
            df_symbol['Avg_Volatility'] = df_symbol['Volatility'].rolling(window=20).mean()
            df_symbol['Rel_Volatility'] = df_symbol['Volatility'] / df_symbol['Avg_Volatility']

            # Calcular ADX
            df_symbol['ADX'] = talib.ADX(
                df_symbol['high'], df_symbol['low'], df_symbol['close'], timeperiod=adx_period)

            # Calcular TP y SL basados en ATR para posiciones LARGAS
            df_symbol['Take_Profit_Long'] = df_symbol['close'] + (df_symbol['ATR'] * tp_multiplier)
            df_symbol['Stop_Loss_Long'] = df_symbol['close'] - (df_symbol['ATR'] * sl_multiplier)
            df_symbol['Stop_Loss_Long'] = df_symbol['Stop_Loss_Long'].clip(lower=1e-8)

            # Calcular TP y SL basados en ATR para posiciones CORTAS
            df_symbol['Take_Profit_Short'] = df_symbol['close'] - (df_symbol['ATR'] * tp_multiplier)
            df_symbol['Stop_Loss_Short'] = df_symbol['close'] + (df_symbol['ATR'] * sl_multiplier)
            df_symbol['Take_Profit_Short'] = df_symbol['Take_Profit_Short'].clip(lower=1e-8)

            # Reemplazar valores NaN
            df_symbol = df_symbol.ffill().fillna(0)

            # Actualizar el DataFrame principal
            data.loc[df_symbol.index, df_symbol.columns] = df_symbol

        except Exception as e:
            continue  # Si ocurre un error, saltar al siguiente símbolo

    # Definir señales de tendencia
    data['Trend_Up'] = data['EMA_Short'] > data['EMA_Long']
    data['Trend_Down'] = data['EMA_Short'] < data['EMA_Long']

    # Asegurarse de que no haya valores NaN en los indicadores
    data['Hammer'] = data['Hammer'].fillna(0)
    data['ShootingStar'] = data['ShootingStar'].fillna(0)
    data['MACD_Bullish'] = data['MACD_Bullish'].fillna(False).astype('boolean')
    data['MACD_Bearish'] = data['MACD_Bearish'].fillna(False).astype('boolean')
    data['ADX'] = data['ADX'].fillna(0)

    # Convertir columnas booleanas al tipo correcto
    data['Trend_Up'] = data['Trend_Up'].astype('boolean')
    data['Trend_Down'] = data['Trend_Down'].astype('boolean')

    # Filtrar períodos de bajo volumen y alta volatilidad
    data['Low_Volume'] = data['Rel_Volume'] < volume_threshold
    data['High_Volatility'] = data['Rel_Volatility'] > volatility_threshold

    # Definir tendencia a largo plazo
    data['EMA_Long_Term'] = talib.EMA(data['close'], timeperiod=50)
    data['Trend_Up_Long_Term'] = data['EMA_Short'] > data['EMA_Long_Term']
    data['Trend_Down_Long_Term'] = data['EMA_Short'] < data['EMA_Long_Term']

    # Señales combinadas con filtrado de ruido y análisis de RSI
    data['Long_Signal'] = (
        (
            ((data['Hammer'] != 0) & data['Trend_Up'] & data['Trend_Up_Long_Term'] & (data['RSI'] < rsi_oversold)) |
            (data['MACD_Bullish'] & (data['ADX'] > 25) & data['Trend_Up_Long_Term'] & (data['RSI'] < rsi_overbought))
        ) &
        (~data['Low_Volume']) &
        (~data['High_Volatility'])
    ).astype('boolean')

    data['Short_Signal'] = (
        (
            ((data['ShootingStar'] != 0) & data['Trend_Down'] & data['Trend_Down_Long_Term'] & (data['RSI'] > rsi_overbought)) |
            (data['MACD_Bearish'] & (data['ADX'] > 25) & data['Trend_Down_Long_Term'] & (data['RSI'] > rsi_oversold))
        ) &
        (~data['Low_Volume']) &
        (~data['High_Volatility'])
    ).astype('boolean')

    # Restablecer índices si es necesario y retornar el DataFrame actualizado
    data.reset_index(drop=True, inplace=True)

    # Guardar el DataFrame resultante en 'indicadores.csv'
    data.to_csv(output_filepath, index=False)

    return data

# Ejecución del backtesting
if __name__ == "__main__":
    data = load_data('./archivos/cripto_price.csv')

    if data is not None:
        final_balance, trades, equity_curve = backtest_strategy(data)

        df_trades = pd.DataFrame(trades)
        df_trades.to_csv('./archivos/backtesting_results.csv', index=False)

        df_equity = pd.DataFrame(equity_curve)
        df_equity.to_csv('./archivos/equity_curve.csv', index=False)

        # Cálculo de métricas adicionales
        total_trades = len(df_trades)
        winning_trades = df_trades[df_trades['profit'] > 0]
        losing_trades = df_trades[df_trades['profit'] <= 0]
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0

        # Cálculo del drawdown máximo
        df_equity['balance'] = df_equity['balance'].astype(float)
        df_equity['cummax'] = df_equity['balance'].cummax()
        df_equity['drawdown'] = df_equity['balance'] - df_equity['cummax']
        max_drawdown = df_equity['drawdown'].min()

        # Cálculo del rendimiento total y anualizado
        total_return = (final_balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100
        days = (pd.to_datetime(df_equity['date'].iloc[-1]) - pd.to_datetime(df_equity['date'].iloc[0])).days
        annualized_return = (1 + total_return/100) ** (365/days) - 1 if days > 0 else 0

        # Mensajes de resultados detallados
        logging.info("\n--- Resultados del Backtesting ---")
        logging.info(f"Balance inicial: ${INITIAL_BALANCE:.2f}")
        logging.info(f"Balance final: ${final_balance:.2f}")
        logging.info(f"Ganancia/Pérdida total: ${final_balance - INITIAL_BALANCE:.2f} ({total_return:.2f}%)")
        logging.info(f"Retorno anualizado: {annualized_return * 100:.2f}%")
        logging.info(f"Máximo drawdown: ${max_drawdown:.2f}")
        logging.info(f"Número total de trades: {total_trades}")
        logging.info(f"Trades ganadores: {len(winning_trades)}")
        logging.info(f"Trades perdedores: {len(losing_trades)}")
        logging.info(f"Porcentaje de aciertos: {win_rate:.2f}%")
        logging.info("----------------------------------\n")

        # Generar y personalizar el gráfico de la curva de equity
        df_equity['date'] = pd.to_datetime(df_equity['date'])
        plt.figure(figsize=(14, 7))
        plt.plot(df_equity['date'], df_equity['balance'], label='Balance')
        plt.title('Curva de Equity')
        plt.xlabel('Fecha')
        plt.ylabel('Balance ($)')
        plt.grid(True)

        # Anotaciones de máximos y mínimos
        max_balance = df_equity['balance'].max()
        min_balance = df_equity['balance'].min()
        max_balance_date = df_equity[df_equity['balance'] == max_balance]['date'].iloc[0]
        min_balance_date = df_equity[df_equity['balance'] == min_balance]['date'].iloc[0]

        plt.annotate(f'Balance Máximo: ${max_balance:.2f}', xy=(max_balance_date, max_balance),
                     xytext=(max_balance_date, max_balance + INITIAL_BALANCE * 0.1),
                     arrowprops=dict(facecolor='green', shrink=0.05), fontsize=10, ha='center')

        plt.annotate(f'Balance Mínimo: ${min_balance:.2f}', xy=(min_balance_date, min_balance),
                     xytext=(min_balance_date, min_balance - INITIAL_BALANCE * 0.1),
                     arrowprops=dict(facecolor='red', shrink=0.05), fontsize=10, ha='center')

        plt.legend()
        plt.tight_layout()
        plt.savefig('./archivos/equity_curve.png')  # Guarda el gráfico como PNG
        plt.show()