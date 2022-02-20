import requests
import schedule
import pandas as pd
import numpy as np
import locale #Formato Monedas
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

#Leyendo el archivo
def read():
    pd.options.display.float_format = '${:,.2f}'.format
    global df
    df = pd.read_csv('./archivos/cripto_price.csv')
    df = df[['BTC','ETH','XRP','SOL','LUNA','DOGE']]

#Reporte Mensual
def report():
    list = df.iloc[-1]['BTC']
    maximos = df.max()
    minimos = df.min()
    promedio = df.mean()
    cripto_price = f'--REPORTE MENSUAL--\n  MAX:\n{maximos}\n\n    MIN:\n{minimos}\n\n    MEAN:\n{promedio}'


    # btc_price = f'Reporte Mensual BTC\nEl precio del BTC es {locale.currency(price_list_filter[0])}\n
    #               El max es {locale.currency(max(price_list_filter))}\n
    #               El min es {locale.currency(min(price_list_filter))}\n
    #               El promedio es {locale.currency(mean_number)}'
    bot_send_text(cripto_price)

# #Alertando cuando se rompe el techo o el piso
# def report_max_min(value_max_min):
#     btc_price_max_min = f'Alerta el valor {value_max_min} se acaba de romper el nuevo {value_max_min} es {locale.currency(price_list_filter[0])}'
#     bot_send_text(btc_price_max_min)

# #Alerta para comprar o vender FIAT
# def report_buy_sell(value_buy_sell, value_mean, dif_percent):
#     btc_price_buy_sell = f'El precio del BTC es {locale.currency(price_list_filter[0])} {value_mean} {dif_percent}% al valor promedio {locale.currency(mean_number)} se recomienda {value_buy_sell}'
#     bot_send_text(btc_price_buy_sell)

# #Poner la orden de compra
# def order_value(order_value_text, order_value_money):
#     order_value_act = f'Se recomienda actualizar el valor de la orden de {order_value_text} a {locale.currency(order_value_money)} ahora mismo'
#     bot_send_text(order_value_act)

    
#     #Alerta de cambio de Techo o piso
#     if actual_value_int == max(price_list_filter):
#         report_max_min('maximo')
#     elif actual_value_int == min(price_list_filter):
#         report_max_min('minimo') 

 
#     #Calulo promedio y mean
#     global mean_number
#     mean_number = int(np.mean(price_list_filter))

#     #Alerta actualizar orden de compra
#     price_list_number_day = price_list_filter[0:720]
#     mean_number_day = int(np.mean(price_list_number_day))

    # if actual_value_int < min(price_list_number_day[1:480]):
    #     if actual_value_int + 2000 < max(price_list_number_day[1:144]):
    #         order_value('compra', actual_value_int + 2000)
    # elif actual_value_int > max(price_list_number_day[1:480]):
    #     if actual_value_int -2000 > min(price_list_number_day[1:144]):
    #         order_value('venta', actual_value_int - 2000)


    # #Alerta compra o venta
    # if actual_value_int < mean_number * 0.94:
    #     report_buy_sell('aumentar el capital de trabajo','inferior en un',dif_percent = int(100-((actual_value_int/mean_number)*100)))
    # elif actual_value_int > mean_number * 1.06:
    #     report_buy_sell('vender BTC por dinero FIAT','superior en un',dif_percent = int(((actual_value_int/mean_number)*100)-100))
    # return 
    
                        
def run():
    read()
    report()

   



#Reporte Diario
# def btc_price_day():
#     run()
#     price_list_day = price_list_filter[0:30]
#     btc_report_day = f'Reporte Diario BTC\nAbrió con {locale.currency(price_list_day[20])}\nCerró con {locale.currency(price_list_day[0])}\nEl max fue {locale.currency(max(price_list_day))}\nEl min fue {locale.currency(min(price_list_day))}'
#     bot_send_text(btc_report_day)

if __name__ == '__main__':
    #schedule.every().day.at("18:00").do(btc_price_day)
    #schedule.every(280).minutes.do(report)
    schedule.every(1).minutes.do(run)
    schedule.every(5).minutes.do(api.get_data)

    while True:
        schedule.run_pending()