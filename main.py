from calendar import month
import schedule
import pkg


def monkey_result():
    total_result, total_USD_trade, final_usd_total = pkg.monkey.monkey_result()
    monkey_USD = f'*==RESULTADOS==* \n Resultados Trade: *{round(total_result,2)}* \n Total Trade: *{round(total_USD_trade,2)}* \n Total Dinero: *{round(final_usd_total,2)}*'
    pkg.monkey.bot_send_text(monkey_USD)
    
def run_bingx():
    
    print(pkg.monkey_bx.total_positions('BTC-USDT'))
    #print('Cerrar Todas las Ordenes Pendientes: ',pkg.bingx.cancel_all_orders('BTC-USDT'))
    #print('Cerrar una orden: ',pkg.bingx.one_clickLclose_all_positions())
    #print('Post Order: ',pkg.bingx.post_order())
    print(pkg.monkey_bx.total_monkey())
    pkg.monkey_bx.obteniendo_ordenes_pendientes()
    pkg.api.price_bingx()

def run():
    pkg.api.price_bingx()
    pkg.monkey.saving_operations()
    pkg.indicadores.emas_indicator()
    pkg.monkey.trading_result()


if __name__ == '__main__':
    schedule.every(0.9).minutes.do(run) 

    hours = list(map(lambda x: x if x > 9 else "0"+str(x), range(1,24)))
    for hour in hours:
        schedule.every().day.at(f"{hour}:00").do(monkey_result)
   
    while True:
        schedule.run_pending()

