import pkg
import pandas as pd
from datetime import datetime, timedelta
import os

import pkg.monkey_bx
from pkg.cfg_loader import load_best_symbols

# =============================
# SECCIÓN DE VARIABLES
# =============================

INCREMENTO_PESO = 0.05  # Porcentaje de incremento/decremento para ajustar los pesos (1% por defecto)
TRANSACTIONS_FILE_PATH = './archivos/PnL.csv'  # Ruta al archivo de transacciones
PESOS_ACTUALIZADOS_PATH = './archivos/pesos_actualizados.csv'  # Ruta para guardar los pesos actualizados

# =============================
# FIN DE LA SECCIÓN DE VARIABLES
# =============================

def _current_symbols():
    syms = load_best_symbols()
    return [str(s).upper() for s in syms] if syms else []

def load_transaction_data(filepath=TRANSACTIONS_FILE_PATH):
    """
    Carga los datos de transacciones desde un archivo CSV.
    """
    try:
        df_transactions = pd.read_csv(filepath, parse_dates=['time'])
        return df_transactions
    except Exception as e:
        print(f"Error al leer el archivo de transacciones: {e}")
        return None

def filter_last_30_days(df_transactions):
    """
    Filtra los datos para obtener las transacciones de los últimos 30 días.
    """
    fecha_final = df_transactions['time'].max()
    fecha_inicial = fecha_final - timedelta(days=30)
    df_last_30_days = df_transactions[
        (df_transactions['time'] >= fecha_inicial) & (df_transactions['time'] <= fecha_final)
    ]
    return df_last_30_days

def calculate_net_profit(df_transactions):
    """
    Calcula el rendimiento neto por cada criptomoneda.
    """
    # Filtrar solo las transacciones de ganancias y pérdidas realizadas
    df_pnl = df_transactions[df_transactions['incomeType'] == 'REALIZED_PNL'].copy()

    # Convertir la columna 'income' a numérica
    df_pnl['income'] = pd.to_numeric(df_pnl['income'], errors='coerce')

    # Agrupar por símbolo y sumar las ganancias/pérdidas
    df_net_profit = df_pnl.groupby('symbol')['income'].sum().reset_index()
    df_net_profit.rename(columns={'income': 'net_profit'}, inplace=True)

    return df_net_profit

def calculate_weights(df_net_profit, incremento=INCREMENTO_PESO):
    """
    Calcula los pesos basados en el rendimiento neto y los pesos anteriores.
    Retorna un DataFrame con columnas ['symbol', 'peso_actualizado', 'cambio']
    """
    # Intentar cargar los pesos anteriores
    if os.path.exists(PESOS_ACTUALIZADOS_PATH):
        df_pesos_previos = pd.read_csv(PESOS_ACTUALIZADOS_PATH)
    else:
        df_pesos_previos = pd.DataFrame(columns=['symbol', 'peso_actualizado'])

    # Crear DataFrame con la lista de criptomonedas actuales
    symbols = _current_symbols()
    if not symbols:
        return pd.DataFrame(columns=['symbol','peso_actualizado','cambio'])

    df_symbols = pd.DataFrame({'symbol': symbols})

    # Merge con pesos previos
    df_pesos = df_symbols.merge(df_pesos_previos, on='symbol', how='left')

    # Asignar peso_actual: si hay peso previo, usarlo; si no, 10%
    df_pesos['peso_actual'] = df_pesos['peso_actualizado'].fillna(0.10)
    df_pesos.drop(columns=['peso_actualizado'], inplace=True)

    # Combinar con net_profit
    df = df_pesos.merge(df_net_profit, on='symbol', how='left')
    df['net_profit'] = df['net_profit'].fillna(0)

    # Aplicar incremento o decremento basado en el rendimiento neto
    df['peso_actualizado'] = df['peso_actual']
    df.loc[df['net_profit'] > 0, 'peso_actualizado'] += df['peso_actual'] * incremento
    df.loc[df['net_profit'] < 0, 'peso_actualizado'] -= df['peso_actual'] * incremento

    # Asegurarse de que los pesos no sean negativos
    df['peso_actualizado'] = df['peso_actualizado'].clip(lower=0, upper=0.80)

    # Normalizar los pesos para que sumen 1
    #suma_pesos = df['peso_actualizado'].sum()
    #df['peso_actualizado'] = df['peso_actualizado'] / suma_pesos

    # Determinar cambio respecto al peso anterior
    df_prev_pesos = df_pesos_previos.set_index('symbol')['peso_actualizado'].to_dict()
    cambios = []
    for index, row in df.iterrows():
        symbol = row['symbol']
        peso_anterior = df_prev_pesos.get(symbol, 0.10)  # Si no existe, asumimos 10%
        peso_actualizado = row['peso_actualizado']
        if peso_actualizado > peso_anterior:
            cambio = 'Subió'
        elif peso_actualizado < peso_anterior:
            cambio = 'Bajó'
        else:
            cambio = 'Sin cambio'
        cambios.append(cambio)
    df['cambio'] = cambios

    # Ordenar por peso_actualizado descendente
    df.sort_values(by='peso_actualizado', ascending=False, inplace=True)

    return df[['symbol', 'peso_actualizado', 'cambio']]

def pesos_ok():
    # Verificar si el archivo de transacciones existe
    if not os.path.exists(TRANSACTIONS_FILE_PATH):
        print(f"El archivo {TRANSACTIONS_FILE_PATH} no existe.")
        return

    # Cargar los datos de transacciones
    df_transactions = load_transaction_data()
    if df_transactions is None:
        return

    symbols = _current_symbols()
    if not symbols:
        print("No hay símbolos configurados para calcular pesos.")
        return

    # Filtrar las transacciones para incluir solo las criptomonedas de la lista
    df_transactions = df_transactions[df_transactions['symbol'].str.upper().isin(symbols)]

    # Filtrar los datos de los últimos 30 días
    if df_transactions.empty:
        print("Sin transacciones recientes para los símbolos actuales; se mantienen pesos previos.")
        return

    df_last_30_days = filter_last_30_days(df_transactions)
    if df_last_30_days.empty:
        print("No hay PnL de los últimos 30 días para los símbolos actuales; se mantienen pesos previos.")
        return

    # Calcular el rendimiento neto por criptomoneda
    df_net_profit = calculate_net_profit(df_last_30_days)

    # Calcular los pesos basados en el rendimiento neto y pesos previos
    df_pesos_actualizados = calculate_weights(df_net_profit)


    # Mostrar los pesos actualizados ordenados de mayor a menor
    print("Pesos actualizados basados en el rendimiento de los últimos 30 días:")
    for index, row in df_pesos_actualizados.iterrows():
        symbol = row['symbol']
        peso = row['peso_actualizado']
        cambio = row['cambio']
        print(f"{symbol}: Peso = {peso*100:.2f}%, {cambio}")

    # Preparar mensaje de pesos actualizado
    lineas = [
        f"{'━' * 20}",
        "⚖️ *REBALANCEO SEMANAL*",
        f"{'━' * 20}",
        "Pesos recalculados según rendimiento de los últimos 30 días:",
        "",
    ]
    for _, row in df_pesos_actualizados.iterrows():
        symbol_clean = row['symbol'].replace('-USDT', '')
        peso = row['peso_actualizado']
        cambio = row['cambio']
        if cambio == 'Subió':
            emoji = '📈'
        elif cambio == 'Bajó':
            emoji = '📉'
        else:
            emoji = '➡️'
        lineas.append(f"{emoji} *{symbol_clean}* — `{peso*100:.1f}%` ({cambio})")

    pkg.monkey_bx.bot_send_text("\n".join(lineas))

    # Guardar los pesos actualizados para referencia futura
    df_pesos_actualizados[['symbol', 'peso_actualizado']].to_csv(PESOS_ACTUALIZADOS_PATH, index=False)
    print(f"\nPesos actualizados guardados en {PESOS_ACTUALIZADOS_PATH}")
