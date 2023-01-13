from calendar import month
import requests
import schedule
import pkg

#Funcion Enviar Mensajes
def bot_send_text(bot_message):

    bot_token = pkg.credentials.token
    bot_chatID = pkg.credentials.chatID
    send_text = pkg.credentials.send + bot_message
    response = requests.get(send_text)

    return response

#Enviando Tuits
def send_tuits(cripto,text, user, likes):
    send_tuits_liks = f'Noticias Tuits  📡  📇 {cripto} \n 🐦Tuit: {text} \n \n 🥸 User: {user} \n 💚 Likes: {likes}'
    bot_send_text(send_tuits_liks)

#Indicador EMA
def ema():
    currencies = pkg.api.currencies_list()
    for currencie in currencies:
        status, price = pkg.indicadores.entry_alert(currencie)
        if status == True:
            stop_lose = price * 0.99
            profit = price * 1.03
            enter_alert = f'🤖🚨 Alerta de Entrada \n 🚧 {currencie} setear Stop Loss en {stop_lose} y Recogida de Ganancia en {profit}'
            bot_send_text(enter_alert)

            #Tuits
            text, user, likes = pkg.tweets.get_tweets(currencie)
            send_tuits(currencie, text, user, likes)                

    
def run_5min():
    pkg.api.get_data()
    ema()

if __name__ == '__main__':
    schedule.every(0.2).minutes.do(run_5min) #4.6
   
    while True:
        schedule.run_pending()