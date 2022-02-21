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
    cripto_price = f'--REPORTE MENSUAL--\n  MAX:\n{df.max()}\n\n    MIN:\n{df.min()}\n\n    MEAN:\n{df.mean()}'
    bot_send_text(cripto_price)

#Alertando cuando se rompe el techo o el piso
def report_max_min(value_max_min,cripto_name,cripto_value):
    btc_price_max_min = f'Alerta el valor {value_max_min} del {cripto_name} se acaba de romper el nuevo {value_max_min} es {cripto_value}'
    bot_send_text(btc_price_max_min)

#Obteniendo Datos de cambio de Techo o piso
def floor_ceiling():
    cripto_list = list(df)
    for cripto in cripto_list:
        if df.iloc[-1][cripto] == df[cripto].max():
            report_max_min('maximo',cripto,df.iloc[-1][cripto])
        elif df.iloc[-1][cripto] == df[cripto].min():
            report_max_min('minimo', cripto,df.iloc[-1][cripto]) 


#Alertando para comprar o vender FIAT
def report_buy_sell(up_down,cripto_percent,cripto_name, cripto_value):
    btc_price_buy_sell = f'Se reporta un movimiento importante {up_down} del {cripto_percent}% del valor del {cripto_name} el precio es {cripto_value}'
    bot_send_text(btc_price_buy_sell)

 #Obteniendo cambios o movimientos 
def change_alert():
    pd.options.display.float_format = '${:,.2f}'.format #ver si se arregla el formato
    cripto_list = list(df)
    for cripto in cripto_list:
        cripto_percent = ((df.iloc[-1][cripto] / df.iloc[-3][cripto])*100) - 100
        if cripto_percent < -0.1:
            report_buy_sell('a la baja',cripto_percent,cripto, df.iloc[-1][cripto])
        elif cripto_percent > 0.1:
            report_buy_sell('al alza',cripto_percent,cripto, df.iloc[-1][cripto])
        return 
       

#Alerta poner la orden de compra
def report_order_value(order_value_text, order_value_money,cripto_name):
    order_value_act = f'Se recomienda actualizar el valor de la orden de {order_value_text} del {cripto_name} a {locale.currency(order_value_money)} ahora mismo'
    bot_send_text(order_value_act)
 

#Obteniendo valor actualizar orden de compra
def order_value():
    cripto_list = list(df)
    for cripto in cripto_list:
        actual_value = df.iloc[-1][cripto]
        if actual_value < df.iloc[-1390][cripto].min():  #Actualizar a 2880
            if actual_value + (actual_value * 0.05) < df.iloc[-1000][cripto].max(): #Actualizar a 1728
                report_order_value('compra', actual_value + (actual_value * 0.05),cripto)
        elif actual_value > df.iloc[-1390][cripto].max(): #Actualizar a 2880
            if actual_value - (actual_value * 0.05) > df.iloc[-1000][cripto].min(): #Actualizar a 1728
                report_order_value('venta', actual_value - (actual_value * 0.05),cripto)

    
def run_5min():
    read()
    floor_ceiling()
    change_alert()
    order_value()

def run():
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
    schedule.every(5).minutes.do(run_5min)
    schedule.every(60).minutes.do(run)
    schedule.every(5).minutes.do(api.get_data)
    

    while True:
        schedule.run_pending()