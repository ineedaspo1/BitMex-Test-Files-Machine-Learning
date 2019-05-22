import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.svm import SVC  
from sklearn.metrics import explained_variance_score
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression

file_path = '/Users/kylekoshiyama/Desktop/BitMex_Bot/data_frame.csv'
dataset = np.loadtxt(file_path, delimiter=",")

run = False
#run = True


x = dataset[:,1:20]  #20,000 hours and 244,799 on 5 min 
y = dataset[:,0]
#price = dataset[:,1]
#print(len(x))
#x = x.reshape((11639,4))
#y = y.reshape((11639,))
cut = 16000
offset = 1000
fin = 1
x_train = x[offset:cut]
y_train = y[offset:cut]
x_test = x[cut:-fin]
y_test = y[cut:-fin]
x_fin = x[fin:]



def NN(x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, shape = 28):
    model = keras.Sequential()
    model.add(keras.layers.Dense(50,input_shape=(shape,), activation='relu'))
    model.add(keras.layers.Dense(25, activation='sigmoid'))
    #model.add(keras.layers.Dense(6, activation='relu'))
    model.add(keras.layers.Dense(25, activation='tanh'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    #model.add(keras.layers.Dense(1))
    #model.compile(loss = "binary_crossentropy", optimizer = SGD(lr=0.005), metrics = ['mse', 'acc'])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['mae', 'acc'])
    model.fit(x_train, y_train, epochs=5)
    #scores = model.evaluate(x_train, y_train)
    
    predictions = model.predict_proba(x_test)
    print("Explained Variance:", explained_variance_score(y_test, predictions))
    
    return predictions


def random_forrest(x_train = x_train, y_train = y_train, x_test = x_test, y_test=y_test):
    sc = StandardScaler()  
    x_train = sc.fit_transform(x_train)  
    x_test = sc.transform(x_test)  
    regressor = RandomForestRegressor(n_estimators=125, random_state=0, n_jobs = -1)  
    #regressor = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=None, random_state=0)
    regressor.fit(x_train, y_train) 
    predictions = regressor.predict(x_test)
    print(regressor.feature_importances_)
    print("Explained Variance:", explained_variance_score(y_test, predictions))
    print("R^2: {}".format(regressor.score(x_test, y_test)))
    print(np.mean(predictions))
    return predictions


def Logistic(x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test):
    #log = LogisticRegression(penalty='l1', solver='liblinear')
    log = LogisticRegression()
    log.fit(x_train, y_train)
    predictions = log.predict(x_test)
    print("Explained Variance:", explained_variance_score(y_test, predictions))
    print(np.mean(predictions))
    
    return predictions



################################### BackTester ##############################3


price = np.loadtxt('/Users/kylekoshiyama/Desktop/BitMex_Bot//list_price.csv', delimiter=",")

'''
rf = random_forrest()
length = len(rf)
sample = price[:-length]
if run == True:
    test = list(zip(rf,sample))
    #testb = list(zip(random_forrest()[len(x_fin):], price))'''

    
#price = np.loadtxt('/Users/kylekoshiyama/Desktop/BitMex_Bot//list_price.csv', delimiter=",")
#test = list(zip(random_forrest(),price))
#test = list(zip(NN(shape=13),price))
#test = list(zip(SVM(),price))

def Backtester(test, lower_bound, higher_bound):
    net_profit = 0
    current_value = 0
    max_drawdown = 0
    last_price = 0
    current_diff = 0
    trade_count = 0
    good_trade_count = 0
    bad_trade_count = 0
    first_price = 0
    price_change_hist = []
    run_returns = []
    price_hist = []
    return_difference = []
    print('Trade Log') #signal start of backtester
    for i in test:
        if first_price == 0:
            first_price = i[1]
        if current_value == 0: #to start the backetester
            if i[0] >= lower_bound: #starts backtester with a buy position
                current_value += i[1] #sets the current value for the backtester
        elif current_value > 0: #if we are long
            if i[0] <= lower_bound: #if the prediction is that it will go down
                net_profit += (i[1] - current_value) #current Value - what we bought at
                #print('Profit from Long =', (i[1] - current_value))
                if (i[1] - current_value) > 0:
                    good_trade_count += 1
                else:
                    bad_trade_count += 1
                current_value = -i[1] #now are are short
                trade_count += 1
        elif current_value < 0: #if we are currently short
            if i[0] >= higher_bound:
                net_profit += ((-current_value) - i[1])
                #print('Profit from Short =', ((-current_value) - i[1]))
                if ((-current_value) - i[1]) > 0:
                    good_trade_count += 1
                else:
                    bad_trade_count += 1
                current_value = i[1] #now we are long
                trade_count += 1
        if net_profit < max_drawdown:
            max_drawdown = net_profit
        if last_price == 0:
            last_price = i[1]
        elif last_price != 0:
            current_diff = last_price - i[1]
            price_change_hist.append(current_diff)
            last_price = i[1]
        if bad_trade_count == 0:
            win_ratio = 100
        elif bad_trade_count != 0:
            win_ratio = good_trade_count/bad_trade_count
        
        run_returns.append(net_profit)
        price_hist.append(i[1] - first_price)
        
        
        
    l_run_diff = list(zip(run_returns, price_hist))
    for k in l_run_diff:
        return_difference.append(k[0]-k[1])    
    
    
    '''
    plt.plot(run_returns, label = 'Model')
    #plt.plot(price_change_hist)
    plt.plot(price_hist, label = 'Market Returns')
    plt.title('Price Over Time')
    plt.ylabel('Price')
    plt.xlabel('Time (Hours)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print()
    print('Net Profit =', net_profit)
    print('Alpha =', net_profit - (price_hist[-1] - price_hist[0]))
    print('Market =', price_hist[-1] - price_hist[0])
    print('Trading Days =', len(test)/24)
    print('Number of trades =', trade_count)     
    print('Win Ratio =', win_ratio)
    print('Max Drawdown =', max_drawdown)
    print('Current Position =', current_value)'''
    
    return net_profit, trade_count, win_ratio 


def Final_Backtester(test, lower_bound, higher_bound):
    net_profit = 0
    current_value = 0
    max_drawdown = 0
    last_price = 0
    current_diff = 0
    trade_count = 0
    good_trade_count = 0
    bad_trade_count = 0
    first_price = 0
    price_change_hist = []
    run_returns = []
    price_hist = []
    return_difference = []
    print('Trade Log') #signal start of backtester
    for i in test:
        if first_price == 0:
            first_price = i[1]
        if current_value == 0: #to start the backetester
            if i[0] >= lower_bound: #starts backtester with a buy position
                current_value += i[1] #sets the current value for the backtester
        elif current_value > 0: #if we are long
            if i[0] <= lower_bound: #if the prediction is that it will go down
                net_profit += (i[1] - current_value) #current Value - what we bought at
                #print('Profit from Long =', (i[1] - current_value))
                if (i[1] - current_value) > 0:
                    good_trade_count += 1
                else:
                    bad_trade_count += 1
                current_value = -i[1] #now are are short
                trade_count += 1
        elif current_value < 0: #if we are currently short
            if i[0] >= higher_bound:
                net_profit += ((-current_value) - i[1])
                #print('Profit from Short =', ((-current_value) - i[1]))
                if ((-current_value) - i[1]) > 0:
                    good_trade_count += 1
                else:
                    bad_trade_count += 1
                current_value = i[1] #now we are long
                trade_count += 1
        if net_profit < max_drawdown:
            max_drawdown = net_profit
        if last_price == 0:
            last_price = i[1]
        elif last_price != 0:
            current_diff = last_price - i[1]
            price_change_hist.append(current_diff)
            last_price = i[1]
        if bad_trade_count == 0:
            win_ratio = 100
        elif bad_trade_count != 0:
            win_ratio = good_trade_count/bad_trade_count
        
        run_returns.append(net_profit)
        price_hist.append(i[1] - first_price)
        
        
        
    l_run_diff = list(zip(run_returns, price_hist))
    for k in l_run_diff:
        return_difference.append(k[0]-k[1])    
    
    
    
    plt.plot(run_returns, label = 'Model')
    #plt.plot(price_change_hist)
    plt.plot(price_hist, label = 'Market Returns')
    plt.title('Price Over Time')
    plt.ylabel('Price')
    plt.xlabel('Time (Hours)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print()
    print('Net Profit =', net_profit)
    print('Alpha =', net_profit - (price_hist[-1] - price_hist[0]))
    print('Market =', price_hist[-1] - price_hist[0])
    print('Trading Days =', len(test)/24)
    print('Number of trades =', trade_count)     
    print('Win Ratio =', win_ratio)
    print('Max Drawdown =', max_drawdown)
    print('Current Position =', current_value)
    
    return net_profit, trade_count, win_ratio 


######################### Optimizer ##########################

def threshold_optimizer(test, lb_limit = 0.2, ub_limit = 0.8, 
                        trade_min = 0, win_thresh = 0, max_drawdown_thresh = -10000, diff_thresh = 0.1):
    max_profit = 0
    optimal_low = 0
    optimal_high = 0      
    #length = len(the_range)                                         
    for k in np.arange(lb_limit, ub_limit, 0.01):
        for i in np.arange(lb_limit, ub_limit, 0.01):
            info = Backtester(test,k, i)
            run_profit = info[0]
            trade_count = info[1]
            win_ratio = info[2]
            diff = abs(i - k)
            diff_2 = abs(k -i)
            print(max_profit)
            print(run_profit)
            print(k)
            print(i)
            print(win_ratio)
            print()
            if run_profit > max_profit and trade_count >= trade_min and win_ratio >= win_thresh and diff_thresh < diff and diff_thresh < diff_2:
                max_profit = run_profit
                optimal_low = k
                optimal_high = i

   
    print('Max Profit =', max_profit)
    print('Optimal Low =', optimal_low)
    print('Optimal High =', optimal_high)

    return [test, optimal_low, optimal_high]
        

if __name__ == "__main__":
    #NN()
    random_forrest()
    #threshold_optimizer(test, 0.2, 0.8, trade_min = 100, win_thresh = 1.2, diff_thresh = 0.1)
    #Final_Backtester(test, 0.2, 0.33)
    print('you got this')
    