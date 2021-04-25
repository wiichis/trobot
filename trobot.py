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
    print(price_list)

    return 


def report():
    btc_price = f'Buenos días Caballes el precio de Bitcoin es de {btc_price_list()}'
    bot_send_text(btc_price)

if __name__ == '__main__':
    
    #schedule.every().day.at("08:00").do(report)
    #schedule.every(10).minutes.do(report)
    schedule.every(2).minutes.do(btc_price_list)
    

    while True:
        schedule.run_pending()