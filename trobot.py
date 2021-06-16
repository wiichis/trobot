from bs4 import BeautifulSoup  
import requests
import schedule
import numpy as np

def bot_send_text(bot_message):

    bot_token = '1794033765:AAHRZZFudN6zqDo2HJ6OeteGzRGV9vplpdo'
    bot_chatID = '566709397'
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message

    response = requests.get(send_text)

    return response


def btc_scraping():
    url = requests.get('https://awebanalysis.com/es/coin-details/bitcoin/')
    soup = BeautifulSoup(url.content, 'html.parser')
    result = soup.find('td', {'class': 'wbreak_word align-middle coin_price'})
    format_result = result.text

    return format_result

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
    price_list_filter.insert(0, btc_scraping())
    if len(price_list_filter) > 800:
        price_list_filter.pop()
    #Quitando $ de la lista original y convirtiendo en float
    global price_list_number
    price_list_number = {i.lstrip("$") for i in price_list_filter}
    price_list_number = {i.replace(",","") for i in price_list_number}
    price_list_number = {float(i) for i in price_list_number}
    return 

#Guardando el archivo
def write():
    with open("./archivos/btc_price.csv", "w", encoding="utf-8") as f:
        for i in price_list_filter:
            f.write(i + "\n")
            
def run():
    read()
    btc_price_list()
    write()

def report():
    btc_price = f'El precio del BTC es {price_list_filter[0]}\nEl max es {max(price_list_filter)}El min es {min(price_list_filter)}El promedio es ${int(sum(price_list_number)/len(price_list_number))}'
    bot_send_text(btc_price)

#Reporte Diario
def btc_price_day():
    run()
    price_list_day = price_list_filter[0:30]
    btc_report_day = f'Reporte Diario BTC\nArranco con {price_list_day[14]}Cerro con {price_list_day[0]}\nEl max fue {max(price_list_day)}El min fue {min(price_list_day)}'
    bot_send_text(btc_report_day)


if __name__ == '__main__':
    schedule.every().day.at("18:00").do(btc_price_day)
    schedule.every(250).minutes.do(report)
    schedule.every(35).minutes.do(run)


    while True:
        schedule.run_pending()