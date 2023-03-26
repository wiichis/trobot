from calendar import month
import schedule
import pkg


def monkey_result():
    total_result, total_USD_trade, final_usd_total = pkg.monkey.monkey_result()
    monkey_USD = f'*==RESULTADOS==* \n Resultados Trade: *{round(total_result,2)}* \n Total Trade: *{round(total_USD_trade,2)}* \n Total Dinero: *{round(final_usd_total,2)}*'
    pkg.monkey.bot_send_text(monkey_USD)
    
def run():
    pkg.api.get_data()
    #pkg.monkey.saving_operations()
    pkg.indicadores.emas_indicator()
    #pkg.monkey.trading_result()
    print(pkg.monkey_bx.total_monkey())
    #print(pkg.monkey_bx.total_positions("BTC-USDT"))
    #print(pkg.monkey_bx.close_positions("BTC-USDT"))
           
    #print(pkg.monkey_bx.saving_operations())
    print("placeOpenOrder:", pkg.bingx.placeOrder("BTC-USDT", "Bid", 0, 0.00068, "Market", "Open",27800,27000))
    #print("placeOpenOrder:", pkg.bingx.placeOrder("BTC-USDT", "Bid", 0, 0.0004, "Market", "Open"))
    #print("getPositions:", pkg.bingx.getPositions("BTC-USDT"))
        #print("ClosePositions:", pkg.bingx.oneClickClosePosition("BTC-USDT",1638345765501796352))
    

if __name__ == '__main__':
    schedule.every(0.1).minutes.do(run) 

    hours = list(map(lambda x: x if x > 9 else "0"+str(x), range(1,24)))
    for hour in hours:
        schedule.every().day.at(f"{hour}:00").do(monkey_result)
   
    while True:
        schedule.run_pending()

