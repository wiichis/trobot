from calendar import month
import schedule
import pkg


def monkey_result():
    balance_actual, diferencia_hora, diferencia_dia, diferencia_semana = pkg.monkey_bx.monkey_result()
    monkey_USD = f'*📊===RESULTADOS===* \n Balance Actual: *{round(balance_actual,2)}* \n Resultado Última Hora: *{round(diferencia_hora,2)}* \n Resultado Día: *{round(diferencia_dia,2)}* \n Resultado Semana: *{round(diferencia_semana,2)}*'
    pkg.monkey.bot_send_text(monkey_USD)
    
def run_bingx():
    pkg.api.price_bingx()
    pkg.monkey_bx.obteniendo_ordenes_pendientes()
    pkg.monkey_bx.colocando_ordenes()


def run_fast():
    pkg.monkey_bx.colocando_TK_SL()


if __name__ == '__main__':
    schedule.every(1).minutes.do(run_bingx)
    schedule.every(0.9).minutes.do(run_fast)  
  
    hours = list(map(lambda x: str(x).zfill(2), range(0, 24)))
    for hour in hours:
        schedule.every().day.at(f"{hour}:59").do(monkey_result)
        schedule.every().day.at(f"{hour}:30").do(pkg.monkey_bx.resultado_PnL)

    while True:
        schedule.run_pending()

