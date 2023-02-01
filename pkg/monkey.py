import pkg
import pandas as pd
from datetime import datetime

def saving_operations():
    currencies = pkg.api.currencies_list()
    date = datetime.now()
    df = pd.read_csv('./archivos/monkey_register.csv')
    #Limpieza y orden
    df.drop_duplicates(subset=['date', 'symbol', 'price'], keep='last', inplace=True)
    df.sort_values('date', inplace=True)
    df.to_csv('./archivos/monkey_register.csv', index=False)

    #Calculado el total de dinero
    total_monkey = df['USD_Total'].sum() + df['result_USD'].sum() - df['USD_Trade'].sum()
    trade = total_monkey / 10

    for currencie in currencies:
        try:
            price_last, stop_lose, profit, tipo = pkg.indicadores.ema_alert(currencie)
            
            currency_amount = trade / price_last
                        
            #Guardando las operaciones  
            if tipo == '=== Alerta de LONG ===':
                #Consulta para evitar que se abra otra orden cuando ya exite una abierta
                query = f"symbol == '{currencie}' and status == 'open'"
                if df.query(query).empty:
                    print('entrada long')
                    status = 'open'
                    df.loc[len(df)] = [date, currencie, price_last, 'LONG', currency_amount, stop_lose, profit, status,'espera',0, trade, 0]
                    df.to_csv('./archivos/monkey_register.csv', index=False)

            elif tipo == '=== Alerta de SHORT ===':
                #Consulta para evitar que se abra otra orden cuando ya exite una abierta
                query = f"symbol == '{currencie}' and status == 'open'"
                if df.query(query).empty:
                    print('entrada short')
                    status = 'open'
                    df.loc[len(df)] = [date, currencie, price_last, 'SHORT', currency_amount, stop_lose, profit, status,'espera',0, trade, 0]
                    df.to_csv('./archivos/monkey_register.csv', index=False)
        except:
            continue



def trading_result():
    #Obteniendo el ultimo precio actualizado
    currencies = pkg.api.currencies_list()
    df_price = pd.read_csv('./archivos/cripto_price.csv')
    df = pd.read_csv('./archivos/monkey_register.csv')
    
    for currencie in currencies: 
        #Obteniendo el ultimo precio de la moneda a anlizar
        price_now = df_price[(df_price['symbol'] == currencie)].tail(1)
        price_last = price_now['price'].item()
        df_open = df[(df['symbol'] == currencie) & (df['status'] == 'open')].head(1) 
        #Comparando Ganacias o perdidas
        try:
            if df_open['tipo'].iloc[0] == 'LONG':
                    print('regla 1')
                    #Ganancia en Long
                    if price_last > df_open['profit'].item():
                        print('esto es long')
                        df_open['status'] = 'close'
                        df_open['result'] = 'ganancia'
                        df_open['result_USD'] = (df_open['profit'] * df_open['currency_amount']) - df_open['USD_Trade']
                        df_open['USD_Total'] += df_open['result_USD'] + df_open['USD_Trade']
                        df_open['USD_Trade'] = 0
                        df = pd.concat([df, df_open])
                        df.to_csv('./archivos/monkey_register.csv', index=False)
                        #Perdida en Long
                    elif price_last < df_open['stop_lose'].item():
                        print('esto es long perdida')
                        df_open['status'] = 'close'
                        df_open['result'] = 'perdida'
                        df_open['result_USD'] = df_open['USD_Trade'] - (df_open['stop_lose'] * df_open['currency_amount'])
                        df_open['USD_Total'] += df_open['result_USD'] + df_open['USD_Trade']
                        df_open['USD_Trade'] = 0
                        df = pd.concat([df, df_open])
                        df.to_csv('./archivos/monkey_register.csv', index=False)
            elif df_open['tipo'].iloc[0] == 'SHORT':
                    print('regla 2')
                    #Ganancia en Short
                    if price_last < df_open['profit'].item():
                        print('esto es short')
                        df_open['status'] = 'close'
                        df_open['result'] = 'ganancia'
                        df_open['result_USD'] = (df_open['profit'] * df_open['currency_amount']) - df_open['USD_Trade']
                        df_open['USD_Total'] += df_open['result_USD'] + df_open['USD_Trade']
                        df_open['USD_Trade'] = 0
                        df = pd.concat([df, df_open])
                        df.to_csv('./archivos/monkey_register.csv', index=False)
                        #Perdida en Short
                    elif price_last > df_open['stop_lose'].item():
                        print('esto es short perdida')
                        df_open['status'] = 'close'
                        df_open['result'] = 'perdida'
                        df_open['result_USD'] = df_open['USD_Trade'] - (df_open['stop_lose'] * df_open['currency_amount'])
                        df_open['USD_Total'] += df_open['result_USD'] + df_open['USD_Trade']
                        df_open['USD_Trade'] = 0
                        df = pd.concat([df, df_open])
                        df.to_csv('./archivos/monkey_register.csv', index=False)
        except:
            continue
    

def monkey_result():
    df = pd.read_csv('./archivos/monkey_register.csv')
    # Sumando la columna resultados
    total_result = df["result_USD"].sum().astype(float)
    print(total_result)
    # get the last value of the "USD_Total" column
    total_USD_trade = df["USD_Trade"].sum().astype(float)
    final_usd_total = (df['USD_Total'].sum() + df['result_USD'].sum() - df['USD_Trade'].sum()).astype(float)
        
    # return the results
    return total_result, total_USD_trade, final_usd_total

