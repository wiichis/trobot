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

price_list = [] 

def btc_price_list():
    price_list.append(btc_scraping())
    #Quitando $ de la lista original y convirtiendo en float
    global price_list_number
    price_list_number = {i.lstrip("$") for i in price_list}
    price_list_number = {i.replace(",","") for i in price_list_number}
    price_list_number = {float(i) for i in price_list_number}

    #print(f'El precio del BTC es {price_list[-1]}, el max es {max(price_list)}, el min es {min(price_list)} y el promedio es {btc_price_mean}')
    #print(price_list_number)
    return 


def report():
    btc_price = f'El precio del BTC es {price_list[-1]}, el max es {max(price_list)}, el min es {min(price_list)} y el promedio es ${int(sum(price_list_number)/len(price_list_number))}'
    bot_send_text(btc_price)


if __name__ == '__main__':
    
    #schedule.every().day.at("08:00").do(report)
    schedule.every(300).minutes.do(report)
    schedule.every(45).minutes.do(btc_price_list)
    

    while True:
        schedule.run_pending()