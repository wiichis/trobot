from bs4 import BeautifulSoup  
import requests
import schedule
import numpy as np
import credentials

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
    btc_price_max_min = f'Alerta el valor {value_max_min} se acaba de romper el nuevo {value_max_min} es {price_list_filter[0]}'
    bot_send_text(btc_price_max_min)

#Alerta para comprar o vender
def report_buy_sell(value_buy_sell, value_mean):
    btc_price_buy_sell = f'El precio del BTC es {price_list_filter[0]} {value_mean} al promedio ${mean_number} se recomienda {value_buy_sell} lo antes posible'
    bot_send_text(btc_price_buy_sell)


#Leyendo el archivo
def read():
    global price_list_filter
    price_list = [] 
    with open("./archivos/btc_price.csv", "r", encoding="utf-8") as f:
        for i in f:
            price_list.append(i)
    #Quitando los "\n"
    price_list_filter = list(filter(lambda x: x != "\n", price_list))

def btc_price_list():
    actual_value = btc_scraping()
    price_list_filter.insert(0, actual_value)
    if len(price_list_filter) > 1400:
        price_list_filter.pop()

    #Alerta de cambio de Techo o piso
    if actual_value == max(price_list_filter):
        report_max_min('maximo')
    elif actual_value == min(price_list_filter):
        report_max_min('minimo')    

    #Quitando $ de la lista original y convirtiendo en float
    global price_list_number
    global mean_number
    price_list_number = {i.replace(",","").replace("$","") for i in price_list_filter}
    price_list_number = {float(i) for i in price_list_number}
    
    # Alerta compra o venta
    mean_number = int(sum(price_list_number)/len(price_list_number))
    actual_value_int = float(actual_value.replace(",","").replace("$",""))
    if actual_value_int < mean_number*1.01:
        report_buy_sell('comprar','inferior')
    elif actual_value_int > mean_number*1.05:
        report_buy_sell('vender','superior en mas del 5%')
    return 
#Guardando el archivo
def write():
    with open("./archivos/btc_price.csv", "w", encoding="utf-8") as f:
        for i in price_list_filter:
            f.write("\n" + i)
                        
def run():
    read()
    btc_price_list()
    write()

def report():
    btc_price = f'Reporte Mensual BTC\nEl precio del BTC es {price_list_filter[0]}\nEl max es {max(price_list_filter)}El min es {min(price_list_filter)}El promedio es ${mean_number}'
    bot_send_text(btc_price)

#Reporte Diario
def btc_price_day():
    run()
    price_list_day = price_list_filter[0:30]
    btc_report_day = f'Reporte Diario BTC\nAbrió con {price_list_day[20]}Cerró con {price_list_day[0]}\nEl max fue {max(price_list_day)}El min fue {min(price_list_day)}'
    bot_send_text(btc_report_day)

if __name__ == '__main__':
    schedule.every().day.at("18:00").do(btc_price_day)
    schedule.every(280).minutes.do(report)
    schedule.every(30).minutes.do(run)


    while True:
        schedule.run_pending()