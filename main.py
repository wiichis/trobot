from calendar import month
import os
import pkg.api_backtesting
import pkg.calcular_pesos
import pkg.monkey_bx
import schedule
import time
import pkg

def monkey_result():
    balance_actual, diferencia_hora, diferencia_dia, diferencia_semana = pkg.monkey_bx.monkey_result()
    monkey_USD = f'*📊===RESULTADOS===* \n Balance Actual: *{round(balance_actual,2)}* \n Resultado Última Hora: *{round(diferencia_hora,2)}* \n Resultado Día: *{round(diferencia_dia,2)}* \n Resultado Semana: *{round(diferencia_semana,2)}*'
    pkg.monkey_bx.bot_send_text(monkey_USD)
    
def run_bingx():
    pkg.monkey_bx.obteniendo_ordenes_pendientes()
    pkg.monkey_bx.colocando_ordenes()

    
def pesos():
    pkg.calcular_pesos.pesos_ok()
    
    
def run_fast():
    path = './archivos/indicadores.csv'
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print("❌ indicadores.csv no existe o está vacío. Se omite run_fast.")
        return
    pkg.monkey_bx.colocando_TK_SL()
    
def posiciones_antiguas():
    pkg.monkey_bx.unrealized_profit_positions()



if __name__ == '__main__':

    # Velas de 5 minutos, siempre completas (ejecuta unos segundos después del cierre)
    schedule.every(5).minutes.at(":01").do(pkg.api_backtesting.price_bingx_5m)

    # Velas de 30 minutos, siempre completas (ejecuta unos segundos después del cierre)
    schedule.every(5).minutes.at(":02").do(pkg.api.price_bingx)

    schedule.every(10).minutes.at(":03").do(run_bingx)
    schedule.every(25).seconds.do(run_fast)
    schedule.every(6).hours.do(pkg.monkey_bx.resultado_PnL)
    schedule.every(5).minutes.do(posiciones_antiguas)
    schedule.every().saturday.at("01:00").do(pesos)    

  
    hours = list(map(lambda x: str(x).zfill(2), range(0, 24)))
    for hour in hours:
        schedule.every().day.at(f"{hour}:59").do(monkey_result)

    while True:
        schedule.run_pending()
        time.sleep(1)
