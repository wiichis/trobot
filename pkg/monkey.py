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

#Enviando Tuits
def send_tuits(cripto,text, user, likes):
    send_tuits_liks = f'📡 *Noticias Tuits* 📇 *{cripto}:* \n 🐦*Tuit*: {text} \n \n 🥸 *User:* {user} \n 💚 *Likes:* {likes}'
    bot_send_text(send_tuits_liks)


def saving_operations():
    currencies = pkg.api.currencies_list()
    date = datetime.now()
    df = pd.read_csv('./archivos/monkey_register.csv')
    #Limpieza y orden
    df.sort_values('date', inplace=True)
    df.to_csv('./archivos/monkey_register.csv', index=False)

    #Calculado el total de dinero
    total_monkey = df['USD_Total'].sum()
    if total_monkey >= 20:
        trade = 20
        total_usd = -20
    else:
        trade = 0.01
        total_usd = 0.01

    for currencie in currencies:
        try:
            price_last, stop_lose, profit, tipo = pkg.indicadores.ema_alert(currencie)
            
            currency_amount = trade / price_last
                        
            #Guardando las operaciones  
            if tipo == '=== Alerta de LONG ===':
                #Consulta para evitar que se abra otra orden cuando ya exite una abierta
                count = df[df['symbol']== currencie]['symbol'].count()
                if count %2 == 0:
                    print('entrada long')
                    status = 'open'
                    df.loc[len(df)] = [date, currencie, price_last, 'LONG', currency_amount, stop_lose, profit, status,'espera',0, trade, total_usd]
                    df.to_csv('./archivos/monkey_register.csv', index=False)

                    #Enviando Mensajes
                    alert = f' 🚨 🤖 🚨 \n *{tipo}* \n 🚧 *{currencie}* \n *Precio Actual:* {round(price_last,3)} \n *Stop Loss* en: {round(stop_lose,3)} \n *Profit* en: {round(profit,3)}'
                    bot_send_text(alert)

                    #Enviando Tuits
                    text, user, likes = pkg.tweets.get_tweets(currencie)
                    send_tuits(currencie, text, user, likes)  

            elif tipo == '=== Alerta de SHORT ===':
                #Consulta para evitar que se abra otra orden cuando ya exite una abierta
                count = df[df['symbol']== currencie]['symbol'].count()
                if count %2 == 0:
                    print('entrada short')
                    status = 'open'
                    df.loc[len(df)] = [date, currencie, price_last, 'SHORT', currency_amount, stop_lose, profit, status,'espera',0, trade, total_usd]
                    df.to_csv('./archivos/monkey_register.csv', index=False)

                    #Enviando Mensajes
                    alert = f' 🚨 🤖 🚨 \n *{tipo}* \n 🚧 *{currencie}* \n *Precio Actual:* {round(price_last,3)} \n *Stop Loss* en: {round(stop_lose,3)} \n *Profit* en: {round(profit,3)}'
                    bot_send_text(alert)

                    #Enviando Tuits
                    text, user, likes = pkg.tweets.get_tweets(currencie)
                    send_tuits(currencie, text, user, likes)  
        except:
            continue



def trading_result():
    #Obteniendo el ultimo precio actualizado
    currencies = pkg.api.currencies_list()
    df_price = pd.read_csv('./archivos/cripto_price.csv')
    df = pd.read_csv('./archivos/monkey_register.csv')
    date = datetime.now()
    
    for currencie in currencies: 
        #Obteniendo el ultimo precio de la moneda a anlizar
        price_now = df_price[(df_price['symbol'] == currencie)].tail(1)
        price_last = price_now['price'].item()
        df_open = df[(df['symbol'] == currencie) & (df['status'] == 'open')].tail(1) 
        #Comparando Ganacias o perdidas
        try:
            if df_open['tipo'].iloc[0] == 'LONG':
                count = df[df['symbol']== currencie]['symbol'].count()
                if count %2 != 0:
                    print('regla 1')
                    #Ganancia en Long
                    if price_last > df_open['profit'].item():
                        print('esto es long')
                        df_open['date'] = date
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
                        df_open['date'] = date
                        df_open['status'] = 'close'
                        df_open['result'] = 'perdida'
                        df_open['result_USD'] = df_open['USD_Trade'] - (df_open['stop_lose'] * df_open['currency_amount'])
                        df_open['USD_Total'] += df_open['result_USD'] + df_open['USD_Trade']
                        df_open['USD_Trade'] = 0
                        df = pd.concat([df, df_open])
                        df.to_csv('./archivos/monkey_register.csv', index=False)
            elif df_open['tipo'].iloc[0] == 'SHORT':
                count = df[df['symbol']== currencie]['symbol'].count()
                if count %2 != 0:
                    print('regla 2')
                    #Ganancia en Short
                    if price_last < df_open['profit'].item():
                        print('esto es short')
                        df_open['date'] = date
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
                        df_open['date'] = date
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
    final_usd_total = (df['USD_Total'].sum() + df['result_USD'].sum()).astype(float)
        
    # return the results
    return total_result, total_USD_trade, final_usd_total

