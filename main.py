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

    # --- ─── Velas 5‑min y 30‑min perfectamente alineadas ─── ---
    # 5‑min candles: HH:00, HH:05, HH:10, … HH:55
    for minute in range(1, 60, 5):
        schedule.every().hour.at(f":{minute:02d}").do(pkg.api_backtesting.price_bingx_5m)

    # 30‑min candles: HH:00 y HH:30
    schedule.every().hour.at(":00").do(pkg.api.price_bingx)
    schedule.every().hour.at(":30").do(pkg.api.price_bingx)

    # Colocar órdenes 1 minuto después de cada cierre de vela 5‑min
    for minute in range(2, 60, 5):   # 01, 06, 11, … 56
        schedule.every().hour.at(f":{minute:02d}").do(run_bingx)

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
