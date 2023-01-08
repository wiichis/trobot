from importlib.metadata import files
from operator import index
from urllib import request
from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json
import credentials
import pandas as pd


# Optimism op 103, waves 133, xrp 6, xlm 26, ewt 173, doge 8, ldo 37, mask 129, matic 10, dydx 188,
# eth 2, btc 1
 


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

  try:
    response = session.get(url, params=parameters)
    data = json.loads(response.text)

  #Obteniendo los campos buscados y guardando en un diccionario
    price_now = {}
    for entry in data['data']:
        symbol =entry['symbol']
        price = entry['quote']['USD']['price']
        volume = entry['quote']['USD']['volume_24h']
        price_now[symbol] = {'price': price,'volume':volume}
        
    
    #Pasando el diccionario a un dataframe y guardadno en un archivo
    df = pd.DataFrame([price_now])
    df_file = pd.read_csv('./archivos/cripto_price.csv')
    df_new = pd.concat([df_file,df],ignore_index=True)
    df_month = df_new.iloc[-105120:]
    df_month.to_csv('./archivos/cripto_price.csv',index = False)

  except (ConnectionError, Timeout, TooManyRedirects) as e:
    print(e)
    



