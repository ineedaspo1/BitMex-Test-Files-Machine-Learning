import requests
import sqlite3
import time

def make_table(table_name):
    conn = sqlite3.connect('BitMex_hist.db')
    cur = conn.cursor()
    cur.execute('DROP TABLE IF EXISTS ' +table_name)
    str1 = 'CREATE TABLE '
    str2 = '(Time TEXT, VWAP INTEGER, Open INTEGER, High INTEGER, Close INTEGER, Low INTEGER, Volume INTEGER)'
    final_str = str(str1 + table_name + str2)
    print(final_str)
    cur.execute(str(final_str))
    

def get_data(year, m):
    conn = sqlite3.connect('BitMex_hist.db')
    cur = conn.cursor()
    month_map = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
    str1 = 'https://www.bitmex.com/api/v1/trade/bucketed?binSize=1d&partial=false&symbol=ETHUSD&count=1&reverse=false&startTime='
    #str1 = 'https://www.bitmex.com/api/v1/trade/bucketed?binSize=5m&partial=false&symbol=XBTUSD&count=288&reverse=false&startTime='
    day = 1
    month = 1
    for k in range(m):
        for i in range(month_map[month]):
            if day < 10:
                day_2 = str('0'+ str(day))
            elif day > 9:
                day_2 = str(day)
            if month < 10:
                month_2 = str('0'+ str(month))
            elif month > 9:
                month_2 = str(month)
            str2 = '%2000%3A00'
            final_str = str(str1 + year + '-' + month_2 + '-' + str(day_2) + str2)
            r = requests.get(final_str)
            data = r.json()
            print(month, day, year)
            for i in data:
                Time = i['timestamp']
                Vwap = i['vwap']
                Open = i['open']
                High = i['high']
                Close = i['close']
                Low = i['low']
                Volume = i['volume']
                cur.execute('INSERT INTO eth (Time, VWAP, Open, High, Close, Low, Volume) VALUES ( ?, ?, ?, ?, ?, ?, ? )', (Time, Vwap, Open, High, Low, Close, Volume))
                conn.commit()
            day += 1
            if day > month_map[month]:
                day = 1
                month +=1
                time.sleep(45)
            print(month, day, year)

def get_data_month(month, year):
    conn = sqlite3.connect('BitMex_hist.db')
    cur = conn.cursor()
    month_map = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
    #str1 = 'https://www.bitmex.com/api/v1/trade/bucketed?binSize=1h&partial=false&symbol=XBTUSD&count=24&reverse=false&startTime='
    str1 = 'https://www.bitmex.com/api/v1/trade/bucketed?binSize=1h&partial=false&symbol=ETHUSD&count=24&reverse=false&startTime='
    day = 1
    for i in range(month_map[month]):
        if day < 10:
            day_2 = str('0'+ str(day))
        elif day > 9:
            day_2 = str(day)
        if month < 10:
            month_2 = str('0'+ str(month))
        elif month > 9:
            month_2 = str(month)
        str2 = '%2000%3A00'
        final_str = str(str1 + year + '-' + month_2 + '-' + str(day_2) + str2)
        r = requests.get(final_str)
        data = r.json()
        print(data)
        for i in data:
            Time = i['timestamp']
            Vwap = i['vwap']
            Open = i['open']
            High = i['high']
            Low = i['low']
            Close = i['close']
            Volume = i['volume']
            cur.execute('INSERT INTO OHCL (Time, VWAP, Open, High, Low, Close, Volume) VALUES ( ?, ?, ?, ?, ?, ?, ? )', (Time, Vwap, Open, High, Low, Close, Volume))
            conn.commit()
        day += 1
        print(day)

if __name__ == "__main__":
    
    make_table('ETH')
    #get_data_month(4,'2017')
    get_data('2019',4)


