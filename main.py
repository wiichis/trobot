import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import os
import pkg.price_bingx_5m
import schedule
import time
import threading
import pkg

def monkey_result():
    balance_actual, diferencia_hora, diferencia_dia, diferencia_semana = pkg.monkey_bx.monkey_result()
    monkey_USD = f'*📊===RESULTADOS===* \n Balance Actual: *{round(balance_actual,2)}* \n Resultado Última Hora: *{round(diferencia_hora,2)}* \n Resultado Día: *{round(diferencia_dia,2)}* \n Resultado Semana: *{round(diferencia_semana,2)}*'
    pkg.monkey_bx.bot_send_text(monkey_USD)
    
def run_bingx():
    pkg.indicadores.update_indicators() 
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

    for minute in range(1, 60, 5):
        schedule.every().hour.at(f":{minute:02d}").do(pkg.price_bingx_5m.price_bingx_5m)

    schedule.every(12).hours.at(":01").do(pkg.price_bingx_5m.actualizar_long_ultimas_12h)
    schedule.every().hour.at(":02").do(pkg.price_bingx_5m.completar_huecos_5m)
    schedule.every(12).hours.at(":02").do(pkg.price_bingx_5m.completar_ultimos_3dias)

    # Colocar órdenes 2 minutos después de cada cierre de vela 5‑min (ahora en minutos 03, 08, ..., 58)
    for minute in range(3, 60, 5):
        schedule.every().hour.at(f":{minute:02d}").do(run_bingx)

    schedule.every(50).seconds.do(run_fast)
    schedule.every(6).hours.do(pkg.monkey_bx.resultado_PnL)
    schedule.every(5).minutes.do(posiciones_antiguas)
    schedule.every().saturday.at("01:00").do(pesos)    

  
    # Reporte de resultados cada hora al minuto 59, ejecutado en un hilo independiente
    schedule.every().hour.at(":59").do(lambda: threading.Thread(target=monkey_result).start())

    while True:
        schedule.run_pending()
        time.sleep(1)