from importlib.metadata import files
from operator import index
from urllib import request
from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json
import credentials
import pandas as pd


def currencies_list():
  currencies = ['OP','WAVES','XRP','XLM','EWT','DOGE','LDO','MASK','MATIC','DYDX','ETH','BTC',
                'BNB','ADA','SOL','DOT','AVAX']
  return currencies

 
def get_data():
  url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
  parameters = {
    'start':'1',
    'limit':'200',
    'convert':'USD'
  }
  headers = {
    'Accepts': 'application/json',
    'X-CMC_PRO_API_KEY': credentials.key,
  }

  session = Session()
  session.headers.update(headers)

  #Importando listado de currencies
  currencies = currencies_list()

  try:
    response = session.get(url, params=parameters)
    data = json.loads(response.text)

  #Obteniendo los campos buscados y guardando en un diccionario, tambien filtramos las currencies que necesitamos
    price_now = {}
    for entry in data['data']:
        symbol =entry['symbol']
        if symbol in currencies:
          price = entry['quote']['USD']['price']
          volume = entry['quote']['USD']['volume_24h']
          date = entry['quote']['USD']['last_updated']
          price_now[symbol] = {'symbol':symbol,'price': price,'volume':volume, 'date':date}
        
    
    #Pasando el diccionario a un dataframe y guardadno en un archivo
    df = pd.DataFrame(price_now)
    df = df.transpose()
    df_file = pd.read_csv('./archivos/cripto_price.csv')
    df_new = pd.concat([df_file,df],ignore_index=True)
    df_month = df_new.iloc[-570000:] #Tres Meses
    df_month.to_csv('./archivos/cripto_price.csv',index = False)

  except (ConnectionError, Timeout, TooManyRedirects) as e:
    print(e)
    



