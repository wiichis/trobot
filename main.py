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
        try:
            stop_lose, profit, tipo = pkg.indicadores.ema_alert(currencie)  
                            
            alert = f'🤖🚨 {tipo} \n 🚧 {currencie} setear: \n Stop Loss en {stop_lose} \n Recogida de Ganancia en {profit}'
            bot_send_text(alert)

            #Tuits
            text, user, likes = pkg.tweets.get_tweets(currencie)
            send_tuits(currencie, text, user, likes)                
        except:
            continue
    
def run_5min():
    pkg.api.get_data()
    ema()

if __name__ == '__main__':
    schedule.every(5).minutes.do(run_5min) #4.6
   
    while True:
        schedule.run_pending()