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
                            
            alert = f' 🚨 🤖 🚨 \n *{tipo}* \n 🚧 *{currencie}* \n *Precio Actual:* {round(price_last,3)} \n *Stop Loss* en: {round(stop_lose,3)} \n *Profit* en: {round(profit,3)}'
            bot_send_text(alert)

            #Tuits
            # text, user, likes = pkg.tweets.get_tweets(currencie)
            # send_tuits(currencie, text, user, likes)                
        except:
            continue

def monkey_result():
    total_result, total_USD_trade, final_usd_total = pkg.monkey.monkey_result()
    monkey_USD = f'*==RESULTADOS==* \n Resultados Trade: *{total_result}* \n Total Trade: *{total_USD_trade}* \n Total Dinero: *{final_usd_total}*'
    bot_send_text(monkey_USD)
    
def run():
    pkg.api.get_data()
    pkg.monkey.saving_operations()
    pkg.monkey.trading_result()
    ema()

if __name__ == '__main__':
    schedule.every(4).minutes.do(run) 

    hours = list(map(lambda x: x if x > 9 else "0"+str(x), range(6,24)))
    for hour in hours:
        schedule.every().day.at(f"{hour}:00").do(monkey_result)
   
    while True:
        schedule.run_pending()

