import pkg
import pandas as pd
from datetime import datetime
import requests

#Funcion Enviar Mensajes
def bot_send_text(bot_message):

    bot_token = pkg.credentials.token
    bot_chatID = pkg.credentials.chatID
    send_text = pkg.credentials.send + bot_message
    response = requests.get(send_text)

    return response

def saving_operations():
    currencies = pkg.api.currencies_list()
    date = datetime.now()
    df = pd.read_csv('./archivos/monkey_register.csv')
    #Limpieza y orden
    df.sort_values('date', inplace=True)
    df.to_csv('./archivos/monkey_register.csv', index=False)

    #Calculado el total de dinero
    for currencie in currencies:
        df = pd.read_csv('./archivos/monkey_register.csv')

        #Contantdo la cantidad de ordenes abiertas
        df_open = df[df['status'] == 'open']
        df_close = df[df['status'] == 'close']
        contador = len(df_open) - len(df_close)
        max_contador = 20

        if contador <=10:
            max_contador = 10
        elif contador >10 and contador <=15:
            max_contador = 15
        elif contador > 15:
            max_contador = 21
        
        print(f'Contador: {contador}, Max Contador:  {max_contador}')

        #Comprobar si hay dinero en caja.
        total_monkey = df['USD_Total'].sum()
        trade = total_monkey / (max_contador - contador)

        if total_monkey >= trade:
            total_usd = trade * -1

            try:
                price_last, stop_lose, profit, tipo, envelope_superior, envelope_inferior = pkg.indicadores.ema_alert(currencie)
                
                currency_amount = trade / price_last
                            
                #Guardando las operaciones  
                if tipo == '=== Alerta de LONG ===':
                    #Consulta para evitar que se abra otra orden cuando ya exite una abierta
                    count = df[df['symbol']== currencie]['symbol'].count()
                    if count %2 == 0:

                        status = 'open'
                        df.loc[len(df)] = [date, currencie, price_last, 'LONG', currency_amount, stop_lose, profit, status,'espera',0, trade, total_usd]
                        df.to_csv('./archivos/monkey_register.csv', index=False)

                        #Enviando Mensajes
                        alert = f' 🚨 🤖 🚨 \n *{tipo}* \n 🚧 *{currencie}* \n *Precio Actual:* {round(price_last,3)} \n *Stop Loss* en: {round(stop_lose,3)} \n *Profit* en: {round(profit,3)}\n *Trade: * {round(trade,2)}'
                        bot_send_text(alert)
 
                elif tipo == '=== Alerta de SHORT ===':
                    #Consulta para evitar que se abra otra orden cuando ya exite una abierta
                    count = df[df['symbol']== currencie]['symbol'].count()
                    if count %2 == 0:
                        status = 'open'
                        df.loc[len(df)] = [date, currencie, price_last, 'SHORT', currency_amount, stop_lose, profit, status,'espera',0, trade, total_usd]
                        df.to_csv('./archivos/monkey_register.csv', index=False)

                        #Enviando Mensajes
                        alert = f' 🚨 🤖 🚨 \n *{tipo}* \n 🚧 *{currencie}* \n *Precio Actual:* {round(price_last,3)} \n *Stop Loss* en: {round(stop_lose,3)} \n *Profit* en: {round(profit,3)}\n *Trade: * {round(trade,2)}'
                        bot_send_text(alert)

            except:
                continue
        else:
            continue


def trading_result():
    #Obteniendo el ultimo precio actualizado
    currencies = pkg.api.currencies_list()
    df = pd.read_csv('./archivos/monkey_register.csv')
    date = datetime.now()
        
    for currencie in currencies: 
        #Obteniendo el ultimo precio de la moneda a anlizar
        df_price = pd.read_csv('./archivos/cripto_price.csv')
        df_open = df[(df['symbol'] == currencie) & (df['status'] == 'open')].tail(1)
        df_price = df_price[(df_price['symbol'] == currencie)].tail(1)

        price_last = df_price.iloc[-1]['price']
                  
        try:
            #Comparando Tiempo de la orden en minutos.
            date1 = pd.to_datetime(df_open['date'])
            date = pd.to_datetime(date)
            diference = date - date1
            diference = diference.iloc[0]
            minutes = diference.total_seconds() / 60
        except IndexError:
            continue

        #Comparando Ganacias o perdidas

        if df_open['tipo'].iloc[0] == 'LONG':
            count = df[df['symbol']== currencie]['symbol'].count()
            if count %2 != 0:
                #Ganancia en Long
                if minutes < 16:
                    if price_last > df_open['profit'].item():
                        df_open['date'] = date
                        df_open['status'] = 'close'
                        df_open['result'] = 'ganancia'
                        df_open['result_USD'] = (df_open['profit'] * df_open['currency_amount']) - df_open['USD_Trade']
                        df_open['USD_Total'] = df_open['result_USD'] + df_open['USD_Trade']
                        df_open['USD_Trade'] = 0
                        df = pd.concat([df, df_open])
                        df.to_csv('./archivos/monkey_register.csv', index=False)
                    #Enviando Mensajes
                        alert = f' 💵 🤖 💵 \n *Ganancia LONG* \n 🚧' + currencie
                        bot_send_text(alert)

                #Perdida en Long
                    if price_last < df_open['stop_lose'].item():
                        df_open['date'] = date
                        df_open['status'] = 'close'
                        df_open['result'] = 'perdida'
                        df_open['result_USD'] = (df_open['stop_lose'] * df_open['currency_amount']) - df_open['USD_Trade'] 
                        df_open['USD_Total'] = df_open['result_USD'] + df_open['USD_Trade']
                        df_open['USD_Trade'] = 0
                        df = pd.concat([df, df_open])
                        df.to_csv('./archivos/monkey_register.csv', index=False)

                        #Enviando Mensajes
                        alert = f' 💸 🤖 💸 \n *Perdida LONG* \n 🚧' + currencie
                        bot_send_text(alert)
                elif price_last > df_open['price'].item()*1.001 or price_last < df_open['stop_lose'].item():
                    df_open['date'] = date
                    df_open['status'] = 'close'
                    df_open['result'] = 'tiempo excede'
                    df_open['result_USD'] = (price_last * df_open['currency_amount']) - df_open['USD_Trade'] 
                    df_open['USD_Total'] = df_open['result_USD'] + df_open['USD_Trade']
                    df_open['USD_Trade'] = 0
                    df = pd.concat([df, df_open])
                    df.to_csv('./archivos/monkey_register.csv', index=False)


        elif df_open['tipo'].iloc[0] == 'SHORT':
            count = df[df['symbol']== currencie]['symbol'].count()
            if count %2 != 0:
                #Ganancia en Short
                if minutes < 13:
                    if price_last < df_open['profit'].item():
                        df_open['date'] = date
                        df_open['status'] = 'close'
                        df_open['result'] = 'ganancia'
                        df_open['result_USD'] = df_open['USD_Trade'] - (df_open['profit'] * df_open['currency_amount']) 
                        df_open['USD_Total'] = df_open['result_USD'] + df_open['USD_Trade']
                        df_open['USD_Trade'] = 0
                        df = pd.concat([df, df_open])
                        df.to_csv('./archivos/monkey_register.csv', index=False)

                        #Enviando Mensajes
                        alert = f' 💵 🤖 💵 \n *Ganancia SHORT* \n 🚧' + currencie
                        bot_send_text(alert)

                    #Perdida en Short
                    if price_last > df_open['stop_lose'].item():
                        df_open['date'] = date
                        df_open['status'] = 'close'
                        df_open['result'] = 'perdida'
                        df_open['result_USD'] = df_open['USD_Trade'] - (df_open['stop_lose'] * df_open['currency_amount'])
                        df_open['USD_Total'] = df_open['result_USD'] + df_open['USD_Trade']
                        df_open['USD_Trade'] = 0
                        df = pd.concat([df, df_open])
                        df.to_csv('./archivos/monkey_register.csv', index=False)

                        #Enviando Mensajes
                        alert = f' 💸 🤖 💸 \n *Perdida SHORT* \n 🚧' + currencie
                        bot_send_text(alert)
                elif price_last < df_open['price'].item()*0.999 or price_last > df_open['stop_lose'].item():
                    df_open['date'] = date
                    df_open['status'] = 'close'
                    df_open['result'] = 'tiempo excede'
                    df_open['result_USD'] = df_open['USD_Trade'] - (price_last * df_open['currency_amount'])
                    df_open['USD_Total'] = df_open['result_USD'] + df_open['USD_Trade']
                    df_open['USD_Trade'] = 0
                    df = pd.concat([df, df_open])
                    df.to_csv('./archivos/monkey_register.csv', index=False)

def monkey_result():
    df = pd.read_csv('./archivos/monkey_register.csv')
    # Sumando la columna resultados
    total_result = df["result_USD"].sum().astype(float)
    print(total_result)
    # get the last value of the "USD_Total" column
    total_USD_trade = df["USD_Trade"].sum().astype(float)
    final_usd_total = df['USD_Total'].sum().astype(float)
        
    # return the results
    return total_result, total_USD_trade, final_usd_total

