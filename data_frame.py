import pandas as pd
import sqlite3
import numpy as np
import os

####################################### Pull DATA from SQL ######################################
conn = sqlite3.connect('BitMex_hist.db')
cur = conn.cursor()
cur.execute('SELECT * FROM OHCL')
price_data = []
sql_data = []
Open = []
High = []
Close = []
Low = []
Volume = []

for row in cur:
    sql_data.append(row)

for row in sql_data:                                     # for Vwap
    if type(row[1]) ==  float:
        price_data.append(row[1]) 
        
    elif type(row[1]) == int:
        price_data.append(row[1]*1.0)
        
    else:
        price_data.append(price_data[-1])
        
for row in sql_data:                                     #For Open Prices
    if type(row[2]) ==  float:
        Open.append(row[2])
    elif type(row[2]) == int:
        Open.append(row[2]*1.0)
    else:
        Open.append(Open[-1])

for row in sql_data:                                     #For High Prices
    if type(row[3]) ==  float:
        High.append(row[3])
    elif type(row[3]) == int:
        High.append(row[3]*1.0)
    else:
        High.append(High[-1])

for row in sql_data:                                     #For Close Prices
    if type(row[4]) ==  float:
        Close.append(row[4])
    elif type(row[4]) == int:
        Close.append(row[4]*1.0)
    else:
        Close.append(Close[-1])
        
for row in sql_data:                                     #For Low Prices
    if type(row[5]) ==  float:
        Low.append(row[5])
    elif type(row[5]) == int:
        Low.append(row[5]*1.0)
    else:
        Low.append(Low[-1])
        
for row in sql_data:                                     #For Low Prices
    if type(row[6]) ==  float:
        Volume.append(row[6])
    elif type(row[6]) == int:
        Volume.append(row[6]*1.0)
    else:
        Volume.append(Volume[-1])

frame = list(zip(Open,High,Close,Low))


change = []
last_val = 0
for i in price_data:
    if last_val != 0:
        difference = i - last_val
        if difference >= 0:
            change.append(1)
        else:
            change.append(0)
    last_val = i    
change.append(5)
    



###################################### Feature Creation #############################################

def calculateRSI(prices, n = 13):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)                               #### RSI Function

    for i in range(n, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0
        else:
            upval = 0
            downval = -delta
        up = (up*(n-1)+upval)/n
        down = (down*(n-1)+downval)/n
        rs = up/down
        rsi[i] = 100. - 100./(1+rs)

    return rsi

def TR(frame = frame):
    TR = []
    Last_period = []
    for i in frame:
        if len(TR) == 0:
            TR.append(int(i[1]-i[3]))
        else:
            method_1 = i[1]-i[3]
            method_2 = abs(i[1]-Last_period[2])
            method_3 = abs(i[3]-Last_period[2])
            TR.append(max([method_1,method_2,method_3]))
        Last_period = i 
    return TR

def ATR(TR, n = 14):
    ATR = []
    ATR_slope = []
    first_val = max(TR[0:n-1]) - min(TR[0:n-1])
    ATR.append(first_val)
    for i in range(n, len(TR)):
        ATR.append((((ATR[-1] * (n-1)) + TR[i])/n))
        ATR_slope.append((TR[i] - ATR[-1])/(2*(TR[i]+ATR[-1])))
    for i in range(n-1):
        ATR.insert(0,first_val)   
        ATR_slope.insert(0,first_val)    
    ATR_slope.append(5)
    
    return ATR_slope



def BB_low(prices, n = 20, f = 1.2):
    High_BB = []
    Low_BB = []
    for i in range(n, len(prices)):
        sma = np.mean(prices[i-n:i])
        stdv = np.std(prices[i-n:i])
        hBB = sma + (f * stdv)
        lBB = sma - (f * stdv)
        High_BB.append(hBB)
        Low_BB.append(lBB)
    first_high = High_BB[0]
    first_low = Low_BB[0]
    for i in range(n):
        High_BB.insert(0,first_high)
        Low_BB.insert(0,first_low)
    
    #tup = list(zip(prices,High_BB, Low_BB))
    return Low_BB   

'''BBL_5_1 = BB_low(price_data, n=5, f = 1)
BBL_10_1 = BB_low(price_data, n=10, f = 1)
BBL_15_1 = BB_low(price_data, n=15, f = 1)
BBL_24_1 = BB_low(price_data, n=24, f = 1)
BBL_48_1 = BB_low(price_data, n=48, f = 1)'''



def BB_high(prices, n = 20, f = 1.2):
    High_BB = []
    Low_BB = []
    for i in range(n, len(prices)):
        sma = np.mean(prices[i-n:i])
        stdv = np.std(prices[i-n:i])
        hBB = sma + (f * stdv)
        lBB = sma - (f * stdv)
        High_BB.append(hBB)
        Low_BB.append(lBB)
    first_high = High_BB[0]
    first_low = Low_BB[0]
    for i in range(n):
        High_BB.insert(0,first_high)
        Low_BB.insert(0,first_low)
    
    #tup = list(zip(prices,High_BB, Low_BB))
    return High_BB   
'''
BBH_5_1 = BB_high(price_data, n=5, f = 1)
BBH_10_1 = BB_high(price_data, n=10, f = 1)
BBH_15_1 = BB_high(price_data, n=15, f = 1)
BBH_24_1 = BB_high(price_data, n=24, f = 1)
BBH_48_1 = BB_high(price_data, n=48, f = 1) '''


def EMA(prices, n = 10):
    first_Avg = np.mean(prices[:n-1])
    multiplier = (2/(n+1))
    ema = [first_Avg]
    for i in range(n, len(prices)):
        hold = ((prices[i]-ema[-1])*multiplier)+ema[-1]
        ema.append(hold)
    for i in range(n-1):
        ema.insert(0,first_Avg)
 
    return ema    
    
'''ema_5 = EMA(price_data, 5)
ema_10 = EMA(price_data, 10)'''

    
def MACD(prices,x = 12, y = 26, z = 9):
    low = EMA(prices, x)
    high = EMA(prices, y)
    hold = list(zip(low, high))
    line = []
    histogram = []
    for i in hold:
        line.append(i[0]-i[1])
    signal = EMA(line, z)    
    hold_2 = list(zip(line, signal))
    for i in hold_2:
        histogram.append(i[0]-i[1])
    new_hist = histogram[x:]
    first_hist = new_hist[0]
    for i in range(x):
        new_hist.insert(0,first_hist)
    
    return new_hist




###################################### Data Frame Construction ########################################
def export_db():
    df = pd.DataFrame(change)
    df['Volume'] = Volume
    df['Volume_rsi'] = calculateRSI(Volume, n = 14)
    df['RSi_5'] = calculateRSI(price_data, n = 5)
    df['RSi_7'] = calculateRSI(price_data, n = 7)
    df['RSi_10'] = calculateRSI(price_data, n = 10)
    df['RSi_14'] = calculateRSI(price_data, n = 14)
    df['RSi_24'] = calculateRSI(price_data, n = 24)
    df['MACD_standard'] = MACD(price_data)
    df['MACD_short'] = MACD(price_data, 6,13,4)
    df['MACD_LONG'] = MACD(price_data, 15, 34, 12)
    df['ATR_5'] = ATR(TR(),5)
    df['ATR_7'] = ATR(TR(),7)
    df['ATR_14'] = ATR(TR(),14)
    df['ATR_20'] = ATR(TR(),20)
    df['ATR_48'] = ATR(TR(),48)
    df['BBL_5_1'] = BB_low(price_data, n=5, f = 1)
    df['BBL_10_1'] = BB_low(price_data, n=10, f = 1)
    df['BBL_15_1'] = BB_low(price_data, n=15, f = 1)
    df['BBL_24_1'] = BB_low(price_data, n=24, f = 1)
    df['BBL_48_1'] = BB_low(price_data, n=48, f = 1)
    df['BBH_5_1'] = BB_high(price_data, n=5, f = 1)
    df['BBH_10_1'] = BB_high(price_data, n=10, f = 1)
    df['BBH_15_1'] = BB_high(price_data, n=15, f = 1)
    df['BBH_24_1'] = BB_high(price_data, n=24, f = 1)
    df['BBH_48_1'] = BB_high(price_data, n=48, f = 1)
    
    df.drop(df.tail(1).index,inplace=True)
    
    print(df.head())
    print(df.tail())
    
    if os.path.exists('data_frame.csv'):
        os.remove('data_frame.csv')
    df.to_csv('data_frame.csv',mode = 'w', index=False, header = None)
    
    ################################### Prices #############################3
    df2 = pd.DataFrame(price_data)
    if os.path.exists('list_price.csv'):
        os.remove('list_price.csv')
    df2.to_csv('list_price.csv', mode = 'w', index=False, header = None)

if __name__ == "__main__":
    export_db()
