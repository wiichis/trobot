import pkg
import pandas as pd
from datetime import datetime, timedelta
import os

# =============================
# SECCIÓN DE VARIABLES
# =============================

INCREMENTO_PESO = 0.01  # Porcentaje de incremento/decremento para ajustar los pesos (1% por defecto)
TRANSACTIONS_FILE_PATH = './archivos/PnL.csv'  # Ruta al archivo de transacciones
PESOS_ACTUALIZADOS_PATH = './archivos/pesos_actualizados.csv'  # Ruta para guardar los pesos actualizados

# =============================
# FIN DE LA SECCIÓN DE VARIABLES
# =============================

# Obtener la lista de criptomonedas desde otro archivo
currencies = pkg.api.currencies_list()

def load_transaction_data(filepath=TRANSACTIONS_FILE_PATH):
    """
    Carga los datos de transacciones desde un archivo CSV.

    Parámetros:
    - filepath: Ruta al archivo CSV.

    Retorna:
    - DataFrame con los datos de transacciones.
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

    Parámetros:
    - df_transactions: DataFrame con los datos de transacciones.

    Retorna:
    - DataFrame filtrado con los datos de los últimos 30 días.
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

    Parámetros:
    - df_transactions: DataFrame con los datos de transacciones.

    Retorna:
    - DataFrame con columnas ['symbol', 'net_profit']
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
    Calcula los pesos basados en el rendimiento neto.

    Parámetros:
    - df_net_profit: DataFrame con columnas ['symbol', 'net_profit']
    - incremento: Porcentaje de incremento/decremento para ajustar los pesos.

    Retorna:
    - DataFrame con columnas ['symbol', 'peso_actualizado']
    """
    # Asignar peso inicial (1/n para cada criptomoneda)
    num_symbols = df_net_profit['symbol'].nunique()
    peso_inicial = 1.0 / num_symbols
    df_net_profit['peso_actual'] = peso_inicial

    # Aplicar incremento o decremento basado en el rendimiento neto
    df_net_profit['peso_actualizado'] = df_net_profit['peso_actual']
    df_net_profit.loc[df_net_profit['net_profit'] > 0, 'peso_actualizado'] += df_net_profit['peso_actual'] * incremento
    df_net_profit.loc[df_net_profit['net_profit'] < 0, 'peso_actualizado'] -= df_net_profit['peso_actual'] * incremento

    # Asegurarse de que los pesos no sean negativos
    df_net_profit['peso_actualizado'] = df_net_profit['peso_actualizado'].clip(lower=0)

    # Normalizar los pesos para que sumen 1
    suma_pesos = df_net_profit['peso_actualizado'].sum()
    df_net_profit['peso_actualizado'] = df_net_profit['peso_actualizado'] / suma_pesos

    return df_net_profit[['symbol', 'peso_actualizado']]

def pesos_ok():
    # Verificar si el archivo de transacciones existe
    if not os.path.exists(TRANSACTIONS_FILE_PATH):
        print(f"El archivo {TRANSACTIONS_FILE_PATH} no existe.")
        return

    # Cargar los datos de transacciones
    df_transactions = load_transaction_data()
    if df_transactions is None:
        return

    # Filtrar las transacciones para incluir solo las criptomonedas de la lista
    df_transactions = df_transactions[df_transactions['symbol'].isin(currencies)]

    # Filtrar los datos de los últimos 30 días
    df_last_30_days = filter_last_30_days(df_transactions)

    # Calcular el rendimiento neto por criptomoneda
    df_net_profit = calculate_net_profit(df_last_30_days)

    # Crear DataFrame con la lista de criptomonedas
    df_all_symbols = pd.DataFrame({'symbol': currencies})

    # Combinar con los rendimientos netos
    df_net_profit = df_all_symbols.merge(df_net_profit, on='symbol', how='left')
    df_net_profit['net_profit'] = df_net_profit['net_profit'].fillna(0)

    # Calcular los pesos basados en el rendimiento neto
    df_pesos_actualizados = calculate_weights(df_net_profit)

    # Mostrar los pesos actualizados
    print("Pesos actualizados basados en el rendimiento de los últimos 30 días:")
    print(df_pesos_actualizados)

    # Guardar los pesos actualizados para referencia futura
    df_pesos_actualizados.to_csv(PESOS_ACTUALIZADOS_PATH, index=False)
    print(f"\nPesos actualizados guardados en {PESOS_ACTUALIZADOS_PATH}")

