from bs4 import BeautifulSoup  
import requests
import schedule

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
    #max_btc = max(price_list)
    #min_btc = min(price_list)
    #print(f'El precio del BTC es {price_list[-1]}, el max es {max_btc} y el min es {min_btc}')

    return 


def report():
    btc_price = f'El precio del BTC es {price_list[-1]}, el max es {max(price_list)} y el min es {min(price_list)}'
    bot_send_text(btc_price)


if __name__ == '__main__':
    
    #schedule.every().day.at("08:00").do(report)
    schedule.every(60).minutes.do(report)
    schedule.every(10).minutes.do(btc_price_list)
    

    while True:
        schedule.run_pending()