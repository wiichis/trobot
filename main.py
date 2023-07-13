from calendar import month
import schedule
import pkg


def monkey_result():
    balance_actual, diferencia_hora, diferencia_dia = pkg.monkey_bx.monkey_result()
    monkey_USD = f'*📊===RESULTADOS===* \n Balance Actual: *{round(balance_actual,2)}* \n Resultado Última Hora: *{round(diferencia_hora,2)}* \n Resultado Día: *{round(diferencia_dia,2)}*'
    pkg.monkey.bot_send_text(monkey_USD)
    
def run_bingx():
    print('Cerrar Todas las Ordenes Pendientes: ',pkg.bingx.cancel_all_orders('BTC-USDT'))
    #print(pkg.bingx.one_clickLclose_all_positions())

    print('Posiciones Pendientes: ',pkg.bingx.perpetual_swap_positions('BTC-USDT'))
    pkg.api.price_bingx()
    pkg.monkey_bx.colocando_ordenes()
    pkg.monkey_bx.obteniendo_ordenes_pendientes()
    #pkg.monkey_bx.prueba_short()
    pkg.monkey_bx.cerrando_ordenes()

def run():
    pkg.api.price_bingx()
    pkg.monkey.saving_operations()
    pkg.indicadores.emas_indicator()
    pkg.monkey.trading_result()


if __name__ == '__main__':
    schedule.every(1).minutes.do(run_bingx) 
  
    hours = list(map(lambda x: str(x).zfill(2), range(0, 24)))
    for hour in hours:
        schedule.every().day.at(f"{hour}:59").do(monkey_result)

    while True:
        schedule.run_pending()

