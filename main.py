from calendar import month
import schedule
import pkg


def monkey_result():
    total_result, total_USD_trade, final_usd_total = pkg.monkey.monkey_result()
    monkey_USD = f'*==RESULTADOS==* \n Resultados Trade: *{total_result}* \n Total Trade: *{total_USD_trade}* \n Total Dinero: *{final_usd_total}*'
    pkg.monkey.bot_send_text(monkey_USD)
    
def run():
    pkg.api.get_data()
    pkg.monkey.saving_operations()
    pkg.monkey.trading_result()

if __name__ == '__main__':
    schedule.every(2).minutes.do(run) 

    hours = list(map(lambda x: x if x > 9 else "0"+str(x), range(6,24)))
    for hour in hours:
        schedule.every().day.at(f"{hour}:00").do(monkey_result)
   
    while True:
        schedule.run_pending()

