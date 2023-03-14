import pkg
import pandas as pd
from datetime import datetime
import requests
import json

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


# Calucando El Valor de las inversiones
def total_monkey():
    monkey = pkg.bingx.getBalance()
    monkey = json.loads(monkey)
    balance = monkey['data']['account']['balance']
    return balance

money = total_monkey()
