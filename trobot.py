from bs4 import BeautifulSoup  
import requests
import schedule
import numpy as np
import locale
import credentials
import api

#Configurando Dolares
locale.setlocale(locale.LC_ALL,'en_US.UTF-8')

#Fucncion Enviar Mensajes
def bot_send_text(bot_message):

    bot_token = credentials.token
    bot_chatID = credentials.chatID
    send_text = credentials.send + bot_message
    response = requests.get(send_text)

    return response


def btc_scraping():
    url = requests.get('https://awebanalysis.com/es/coin-details/bitcoin/')
    soup = BeautifulSoup(url.content, 'html.parser')
    result = soup.find('td', {'class': 'wbreak_word align-middle coin_price'})
    format_result = result.text

    return format_result



#Alertando cuando se rompe el techo o el piso
def report_max_min(value_max_min):
    btc_price_max_min = f'Alerta el valor {value_max_min} se acaba de romper el nuevo {value_max_min} es {locale.currency(price_list_filter[0])}'
    bot_send_text(btc_price_max_min)

#Alerta para comprar o vender FIAT
def report_buy_sell(value_buy_sell, value_mean, dif_percent):
    btc_price_buy_sell = f'El precio del BTC es {locale.currency(price_list_filter[0])} {value_mean} {dif_percent}% al valor promedio {locale.currency(mean_number)} se recomienda {value_buy_sell}'
    bot_send_text(btc_price_buy_sell)

#Poner la orden de compra
def order_value(order_value_text, order_value_money):
    order_value_act = f'Se recomienda actualizar el valor de la orden de {order_value_text} a {locale.currency(order_value_money)} ahora mismo'
    bot_send_text(order_value_act)


#Leyendo el archivo
def read():
    global price_list_filter
    price_list = [] 
    with open("./archivos/btc_price.csv", "r", encoding="utf-8") as f:
        for i in f:
            price_list.append(i)
    #Quitando los "\n"
    price_list_filter = list(filter(lambda x: x != "\n", price_list))
    price_list_filter = list(np.float_(price_list_filter))

def btc_price_list():
    actual_value = btc_scraping()
    actual_value_int = float(actual_value.replace(",","").replace("$",""))
    price_list_filter.insert(0, actual_value_int)
    if len(price_list_filter) > 1440:
        price_list_filter.pop()

    #Alerta de cambio de Techo o piso
    if actual_value_int == max(price_list_filter):
        report_max_min('maximo')
    elif actual_value_int == min(price_list_filter):
        report_max_min('minimo') 

 
    #Calulo promedio y mean
    global mean_number
    mean_number = int(np.mean(price_list_filter))

    #Alerta actualizar orden de compra
    price_list_number_day = price_list_filter[0:720]
    mean_number_day = int(np.mean(price_list_number_day))

    if actual_value_int < min(price_list_number_day[1:480]):
        if actual_value_int + 2000 < max(price_list_number_day[1:144]):
            order_value('compra', actual_value_int + 2000)
    elif actual_value_int > max(price_list_number_day[1:480]):
        if actual_value_int -2000 > min(price_list_number_day[1:144]):
            order_value('venta', actual_value_int - 2000)


    #Alerta compra o venta
    if actual_value_int < mean_number * 0.94:
        report_buy_sell('aumentar el capital de trabajo','inferior en un',dif_percent = int(100-((actual_value_int/mean_number)*100)))
    elif actual_value_int > mean_number * 1.06:
        report_buy_sell('vender BTC por dinero FIAT','superior en un',dif_percent = int(((actual_value_int/mean_number)*100)-100))
    return 
    
#Guardando el archivo
def write():
    with open("./archivos/btc_price.csv", "w", encoding="utf-8") as f:
        for i in price_list_filter:
            f.write(str(i) + '\n')
                        
def run():
    read()
    btc_price_list()
    write()

def report():
    btc_price = f'Reporte Mensual BTC\nEl precio del BTC es {locale.currency(price_list_filter[0])}\nEl max es {locale.currency(max(price_list_filter))}\nEl min es {locale.currency(min(price_list_filter))}\nEl promedio es {locale.currency(mean_number)}'
    bot_send_text(btc_price)

#Reporte Diario
def btc_price_day():
    run()
    price_list_day = price_list_filter[0:30]
    btc_report_day = f'Reporte Diario BTC\nAbrió con {locale.currency(price_list_day[20])}\nCerró con {locale.currency(price_list_day[0])}\nEl max fue {locale.currency(max(price_list_day))}\nEl min fue {locale.currency(min(price_list_day))}'
    bot_send_text(btc_report_day)

if __name__ == '__main__':
    schedule.every().day.at("18:00").do(btc_price_day)
    schedule.every(280).minutes.do(report)
    schedule.every(30).minutes.do(run)
    schedule.every(5).minutes.do(api.df)

    while True:
        schedule.run_pending()