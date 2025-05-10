from calendar import month
import pkg.api_backtesting
import pkg.calcular_pesos
import pkg.monkey_bx
import schedule
import pkg

def monkey_result():
    balance_actual, diferencia_hora, diferencia_dia, diferencia_semana = pkg.monkey_bx.monkey_result()
    monkey_USD = f'*📊===RESULTADOS===* \n Balance Actual: *{round(balance_actual,2)}* \n Resultado Última Hora: *{round(diferencia_hora,2)}* \n Resultado Día: *{round(diferencia_dia,2)}* \n Resultado Semana: *{round(diferencia_semana,2)}*'
    pkg.monkey_bx.bot_send_text(monkey_USD)
    
def run_bingx():
    pkg.api.price_bingx()
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

def run_candles_5m():
    # Ejecuta la actualización de velas de 5m una vez al día
    pkg.api_backtesting.price_bingx_5m()


if __name__ == '__main__':
    schedule.every(1).minutes.do(run_bingx)
    schedule.every(0.4).minutes.do(run_fast)
    schedule.every(6).hours.do(pkg.monkey_bx.resultado_PnL)
    schedule.every(25).minutes.do(posiciones_antiguas)
    schedule.every().saturday.at("01:00").do(pesos)    
    
    # Programar actualización diaria de velas de 5m a la medianoche
    schedule.every().day.at("21:30").do(run_candles_5m)
  
    hours = list(map(lambda x: str(x).zfill(2), range(0, 24)))
    for hour in hours:
        schedule.every().day.at(f"{hour}:59").do(monkey_result)

    while True:
        schedule.run_pending()
