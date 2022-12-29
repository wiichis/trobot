from calendar import month
import requests
import schedule
import pandas as pd
import numpy as np
import credentials
import api
import give_me_tweets


#Funcion Enviar Mensajes
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
    df = df[['BTC','ETH','BNB','ADA','XRP','SOL','DOT','AVAX']]


#Reporte Mensual
def report_month():
    cripto_price_month = f'--🗓 REPORTE MENSUAL--\n\n    MAX:\n{df.iloc[-8640:].max()}\n\n    MIN:\n{df.iloc[-8640:].min()}\n\n    MEAN:\n{df.iloc[-8640:].mean()}'
    bot_send_text(cripto_price_month)

#Reporte 15 Dias
def report_15_days():
    cripto_price_month = f'--📮 REPORTE 1️⃣5️⃣ Días--\n\n    MAX:\n{df.iloc[-2880:].max()}\n\n    MIN:\n{df.iloc[-2880:].min()}\n\n    MEAN:\n{df.iloc[-2880:].mean()}'
    bot_send_text(cripto_price_month)

#Reporte 10 Dias
def report_10_days():
    cripto_price_month = f'--📮 REPORTE 🔟 Días--\n\n    MAX:\n{df.iloc[-2880:].max()}\n\n    MIN:\n{df.iloc[-2880:].min()}\n\n    MEAN:\n{df.iloc[-2880:].mean()}'
    bot_send_text(cripto_price_month)

#Reporte 5 Dias
def report_5_days():
    cripto_price_month = f'--📮 REPORTE 5️⃣ Días--\n\n    MAX:\n{df.iloc[-1440:].max()}\n\n    MIN:\n{df.iloc[-1440:].min()}\n\n    MEAN:\n{df.iloc[-1440:].mean()}'
    bot_send_text(cripto_price_month)


#Alerta Reporte Diario
def report_price_day():
    report_day = f'--⏰ REPORTE DIARIO --\n\n    OPEN:\n{df.iloc[-144]}\n\n    CLOSE:\n{df.iloc[-1]}\n\n    MAX:\n{df.iloc[-144:].max()}\n\n    MIN:\n{df.iloc[-144:].min()}        '
    bot_send_text(report_day)


#Alertando cuando se rompe el techo o el piso por mes
def report_max_min_month(value_max_min,cripto_name,cripto_value):
    btc_price_max_min = f'🚨Alerta🚨 el valor {value_max_min} mensual de {cripto_name} se acaba de romper el nuevo {value_max_min} es ${round(cripto_value,3)}'
    bot_send_text(btc_price_max_min)

#Obteniendo Datos de cambio de Techo o piso por mes
def floor_ceiling_month():
    cripto_list = list(df)
    for cripto in cripto_list:
        if df.iloc[-1][cripto] == df.iloc[-8640:][cripto].max():
            report_max_min_month('máximo 📈',cripto,df.iloc[-1][cripto])
        elif df.iloc[-1][cripto] == df.iloc[-8640:][cripto].min():
            report_max_min_month('minimo 📉', cripto,df.iloc[-1][cripto]) 

#Alertando cuando se rompe el techo o el piso anul
def report_max_min_year(value_max_min,cripto_name,cripto_value):
    btc_price_max_min = f'🚨🚧Alerta🚧🚨 el valor {value_max_min} anual de {cripto_name} se acaba de romper el nuevo {value_max_min} es ${round(cripto_value,3)}'
    bot_send_text(btc_price_max_min)

#Obteniendo Datos de cambio de Techo o piso anual
def floor_ceiling_year():
    cripto_list = list(df)
    for cripto in cripto_list:
        if df.iloc[-1][cripto] == df[cripto].max():
            report_max_min_year('máximo 📈🚀',cripto,df.iloc[-1][cripto])
        elif df.iloc[-1][cripto] == df[cripto].min():
            report_max_min_year('minimo 📉🫡', cripto,df.iloc[-1][cripto]) 


#Alertando para comprar o vender 
def report_buy_sell(up_down,cripto_percent,cripto_name, cripto_value,c_mean_4h):
    btc_price_buy_sell = f'Se reporta 📫 un movimiento importante  {up_down} de {cripto_percent * 100}% del valor de {cripto_name} el precio es ${round(cripto_value,3)} vs el promedio de las últimas 4 horas ${round(c_mean_4h,3)}'
    bot_send_text(btc_price_buy_sell)

#Enviando Tuits
def send_tuits(cripto,text, user, likes):
    send_tuits_liks_rts = f'Noticias Tuits  📡  📇 {cripto} \n 🐦Tuit: {text} \n \n 🥸 User: {user} \n 💚 Likes: {likes}'
    bot_send_text(send_tuits_liks_rts)

 #Obteniendo cambios o movimientos 
def change_alert():
    cripto_list = list(df)
    for cripto in cripto_list:
        c_mean_4h = df.iloc[-48:][cripto].mean()
        cripto_dif = df.iloc[-1][cripto] - c_mean_4h
        cripto_abs = abs(cripto_dif)
        cripto_percent = round(cripto_abs / c_mean_4h,3)
        if cripto_dif < 0 :
            if cripto_abs > c_mean_4h * 0.027:
                report_buy_sell('a la baja ⬇️ 🔴',cripto_percent,cripto, df.iloc[-1][cripto],c_mean_4h)
                text, user, likes = give_me_tweets.get_tweets(cripto)
                send_tuits(cripto, text, user, likes)
                
        elif cripto_percent > 0:
            if cripto_abs > c_mean_4h * 0.027:
                report_buy_sell('al alza ⬆️ 🟢',cripto_percent,cripto, df.iloc[-1][cripto],c_mean_4h)
                text, user, likes = give_me_tweets.get_tweets(cripto)
                send_tuits(cripto, text, user, likes)                

#Alerta poner la orden de compra 5 Dias
def report_order_value_5_days(order_value_text, order_value_money,cripto_name,per):
    order_value_act = f'🤖 5️⃣ Días recomienda actualizar el valor de la orden de {order_value_text} con {per * 100}% de {cripto_name} a ${round(order_value_money,3)} ahora mismo'
    bot_send_text(order_value_act)

#Obteniendo valor actualizar orden de compra 5 Dias
def order_value_5_days():
    cripto_list = list(df)
    for cripto in cripto_list:
        actual_value = df.iloc[-1][cripto]
        min_value = df.iloc[-1440:-1][cripto].min() 
        min_value_8 = df.iloc[-2304:-1][cripto].min()
        min_value_11 = df.iloc[-3168:-1][cripto].min()
        min_value_14 = df.iloc[-4032:-1][cripto].min()

        max_value = df.iloc[-1440:-1][cripto].max()
        max_value_8 = df.iloc[-2304:-1][cripto].max()
        max_value_11 = df.iloc[-3168:-1][cripto].max()
        max_value_14 = df.iloc[-4032:-1][cripto].max()

        #Calculando el porcentaje en base a la data de 15 días
        per = 0.05

        if actual_value < min_value_8:
            per = 0.04
            if actual_value < min_value_11:
                per = 0.03
                if actual_value < min_value_14:
                    per = 0.02
        elif actual_value > max_value_8:
            per = 0.04
            if actual_value > max_value_11:
                per = 0.03
                if actual_value > max_value_14:
                    per = 0.02

        if actual_value < min_value:  
            if actual_value + (actual_value * 0.05) < max_value:
                report_order_value_5_days('compra 💵', actual_value + (actual_value * per),cripto,per)
        elif actual_value > max_value: 
            if actual_value - (actual_value * 0.05) > min_value: 
                report_order_value_5_days('venta 💸', actual_value - (actual_value * per),cripto,per)


    
def run_5min():
    api.get_data()
    read()
    floor_ceiling_month()
    floor_ceiling_year()
    order_value_5_days()

def run_10min():
    change_alert()
    

if __name__ == '__main__':
    schedule.every().day.at("11:00").do(report_price_day)
    schedule.every().day.at("23:00").do(report_price_day)
    schedule.every().day.at("14:00").do(report_month)
    schedule.every().day.at("20:00").do(report_10_days)
    schedule.every().day.at("17:00").do(report_5_days)
    schedule.every().day.at("02:00").do(report_15_days)
    schedule.every(5).minutes.do(run_5min)
    schedule.every(10).minutes.do(run_10min)

    

    while True:
        schedule.run_pending()