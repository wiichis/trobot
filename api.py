from importlib.metadata import files
from operator import index
from urllib import request
from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json
import credentials
import pandas as pd


def get_data():
  url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
  parameters = {
    'start':'1',
    'limit':'20',
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
        quote = entry['quote']['USD']['price']
        price_now.setdefault(symbol, quote)
    
    #Pasando el diccionario a un dataframe y guardadno en un archivo
    df = pd.DataFrame([price_now])
    df_file = pd.read_csv('./archivos/cripto_price.csv')
    df_new = pd.concat([df_file,df],ignore_index=True)
    df_month = df_new.iloc[-105120:]
    df_month.to_csv('./archivos/cripto_price.csv',index = False)

  except (ConnectionError, Timeout, TooManyRedirects) as e:
    print(e)
    


