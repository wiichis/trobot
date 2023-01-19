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
    send_tuits_liks = f'📡 *Noticias Tuits* 📇 *{cripto}:* \n 🐦*Tuit*: {text} \n \n 🥸 *User:* {user} \n 💚 *Likes:* {likes}'
    bot_send_text(send_tuits_liks)

#Indicador EMA
def ema():
    currencies = pkg.api.currencies_list()
    for currencie in currencies:
        try:
            price_last, stop_lose, profit, tipo = pkg.indicadores.ema_alert(currencie)  
                            
            alert = f' 🚨 🤖 🚨 \n *{tipo}* \n 🚧 El precio de *{currencie}* es *{price_last}* \n Setear: \n *Stop Loss* en: {round(stop_lose,3)} \n *Profit* en: {round(profit,3)}'
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
    schedule.every(4.6).minutes.do(run_5min) #4.6
   
    while True:
        schedule.run_pending()

