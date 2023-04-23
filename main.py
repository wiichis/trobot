from calendar import month
import schedule
import pkg


def monkey_result():
    total_result, total_USD_trade, final_usd_total = pkg.monkey.monkey_result()
    monkey_USD = f'*==RESULTADOS==* \n Resultados Trade: *{round(total_result,2)}* \n Total Trade: *{round(total_USD_trade,2)}* \n Total Dinero: *{round(final_usd_total,2)}*'
    pkg.monkey.bot_send_text(monkey_USD)
    
def run():
    
    #print(pkg.monkey_bx.total_positions('LINK-USDT'))
    #print('Close Order: ',pkg.monkey_bx.close_orders('LINK-USDT'))
    #print('Close Order: ',pkg.bingx.one_clickLclose_all_positions())
    print('Post Order: ',pkg.bingx.post_order())
    print(pkg.monkey_bx.total_monkey())
    pkg.api.price_bingx()

if __name__ == '__main__':
    schedule.every(0.5).minutes.do(run) 

    hours = list(map(lambda x: x if x > 9 else "0"+str(x), range(1,24)))
    for hour in hours:
        schedule.every().day.at(f"{hour}:00").do(monkey_result)
   
    while True:
        schedule.run_pending()

