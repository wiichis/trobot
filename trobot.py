from calendar import month
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
    df = df[['BTC','ETH','XRP','SOL','LUNA','AVAX']]

#Reporte Mensual
def report_month():
    cripto_price_month = f'--🗓 REPORTE MENSUAL--\n\n    MAX:\n{df.max()}\n\n    MIN:\n{df.min()}\n\n    MEAN:\n{df.mean()}'
    bot_send_text(cripto_price_month)

#Reporte 10 Dias
def report_10_days():
    cripto_price_month = f'--📮 REPORTE 🔟 Días--\n\n    MAX:\n{df.iloc[-2880:].max()}\n\n    MIN:\n{df.iloc[-2880:].min()}\n\n    MEAN:\n{df.iloc[-2880:].mean()}'
    bot_send_text(cripto_price_month)

#Alerta Reporte Diario
def report_price_day():
    report_day = f'--⏰ REPORTE DIARIO --\n\n    OPEN:\n{df.iloc[-144]}\n\n    CLOSE:\n{df.iloc[-1]}\n\n    MAX:\n{df.iloc[-144:].max()}\n\n    MIN:\n{df.iloc[-144:].min()}        '
    bot_send_text(report_day)

#Alertando cuando se rompe el techo o el piso
def report_max_min(value_max_min,cripto_name,cripto_value):
    btc_price_max_min = f'🚨Alerta🚨 el valor {value_max_min} de {cripto_name} se acaba de romper el nuevo {value_max_min} es ${round(cripto_value,3)}'
    bot_send_text(btc_price_max_min)

#Obteniendo Datos de cambio de Techo o piso
def floor_ceiling():
    cripto_list = list(df)
    for cripto in cripto_list:
        if df.iloc[-1][cripto] == df[cripto].max():
            report_max_min('máximo 📈',cripto,df.iloc[-1][cripto])
        elif df.iloc[-1][cripto] == df[cripto].min():
            report_max_min('minimo 📉', cripto,df.iloc[-1][cripto]) 


#Alertando para comprar o vender 
def report_buy_sell(up_down,cripto_percent,cripto_name, cripto_value,c_mean_4h):
    btc_price_buy_sell = f'Se reporta 📫 un movimiento importante  {up_down} de {cripto_percent}% del valor de {cripto_name} el precio es ${round(cripto_value,3)} vs el promedio de las últimas 4 horas ${round(c_mean_4h,3)}'
    bot_send_text(btc_price_buy_sell)


 #Obteniendo cambios o movimientos 
def change_alert():
    cripto_list = list(df)
    for cripto in cripto_list:
        c_mean_4h = df.iloc[-48:][cripto].mean()
        cripto_dif = df.iloc[-1][cripto] - c_mean_4h
        cripto_abs = abs(cripto_dif)
        cripto_percent = round(cripto_abs / c_mean_4h,3)
        if cripto_dif < 0 :
            if cripto_abs > c_mean_4h * 0.02:
                report_buy_sell('a la baja ⬇️ 🔴',cripto_percent,cripto, df.iloc[-1][cripto],c_mean_4h)
        elif cripto_percent > 0:
            if cripto_abs > c_mean_4h * 0.02:
                report_buy_sell('al alza ⬆️ 🟢',cripto_percent,cripto, df.iloc[-1][cripto],c_mean_4h)
        

#Alerta poner la orden de compra
def report_order_value(order_value_text, order_value_money,cripto_name):
    order_value_act = f'Se recomienda actualizar el valor de la orden de {order_value_text} de {cripto_name} a {locale.currency(order_value_money)} ahora mismo'
    bot_send_text(order_value_act)
 

#Obteniendo valor actualizar orden de compra
def order_value():
    cripto_list = list(df)
    for cripto in cripto_list:
        actual_value = df.iloc[-1][cripto]
        min_value = df.iloc[-2880:-1][cripto].min() 
        min_value_20 = df.iloc[-5760:-1][cripto].min()
        min_value_30 = df.iloc[-8640:-1][cripto].min()

        max_value = df.iloc[-2880:-1][cripto].max()
        max_value_20 = df.iloc[-5760:-1][cripto].max()
        max_value_30 = df.iloc[-8640:-1][cripto].max()

        #Calculando el porcentaje en base a la data del mes.
        per = 0.05

        if actual_value < min_value_20:
            per = 0.04
            if actual_value < min_value_30:
                per = 0.03
        elif actual_value > max_value_20:
            per = 0.04
            if actual_value > max_value_30:
                per = 0.03

        if actual_value < min_value:  
            if actual_value + (actual_value * 0.05) < max_value:
                report_order_value('compra 💵', actual_value + (actual_value * per),cripto)
        elif actual_value > max_value: 
            if actual_value - (actual_value * 0.05) > min_value: 
                report_order_value('venta 💸', actual_value - (actual_value * per),cripto)

    
def run_5min():
    api.get_data()
    read()
    floor_ceiling()
    order_value()

def run_10min():
    change_alert()
    

if __name__ == '__main__':
    schedule.every().day.at("06:00").do(report_price_day)
    schedule.every().day.at("18:00").do(report_price_day)
    schedule.every().day.at("15:00").do(report_10_days)
    schedule.every().day.at("12:00").do(report_month)
    schedule.every(5).minutes.do(run_5min)
    schedule.every(10).minutes.do(run_10min)

    

    while True:
        schedule.run_pending()