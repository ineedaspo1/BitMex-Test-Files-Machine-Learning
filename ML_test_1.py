import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC  
from sklearn.metrics import explained_variance_score


file_path = '/Users/kylekoshiyama/Desktop/Bot_2.0/data_frame.csv'
dataset = np.loadtxt(file_path, delimiter=",")


x = dataset[:,1:15]
y = dataset[:,0]
#price = dataset[:,1]
#print(len(x))
#x = x.reshape((11639,4))
#y = y.reshape((11639,))
cut = 15000
offset = 2000
x_train = x[offset:cut]
y_train = y[offset:cut]
x_test = x[cut:]
y_test = y[cut:]


def NN(x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, shape = 13):
    model = keras.Sequential()
    model.add(keras.layers.Dense(12,input_shape=(13,), activation='relu'))
    model.add(keras.layers.Dense(9, activation='sigmoid'))
    #model.add(keras.layers.Dense(6, activation='relu'))
    model.add(keras.layers.Dense(6, activation='tanh'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    #model.add(keras.layers.Dense(1))
    model.compile(loss = "binary_crossentropy", optimizer = SGD(lr=0.01), metrics = ['mse', 'acc'])
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics'mae', 'acc'
    model.fit(x_train, y_train, epochs=5)
    #scores = model.evaluate(x_train, y_train)
    
    ynew = model.predict_proba(x_test)
    predictions = []
    for i in range(len(x_test)):
        predictions.append(ynew[i])
        print(ynew[i])
    print("Explained Variance:", explained_variance_score(y_test, predictions))
    
    return predictions


def random_forrest(x_train = x_train, y_train = y_train, x_test = x_test):
    sc = StandardScaler()  
    x_train = sc.fit_transform(x_train)  
    x_test = sc.transform(x_test)  
    regressor = RandomForestRegressor(n_estimators=25, random_state=0)  
    #regressor = GradientBoostingClassifier(n_estimators=25, learning_rate=0.05, max_depth=None, random_state=0).fit(x_train, y_train)
    regressor.fit(x_train, y_train) 
    predictions = regressor.predict(x_test) 
    #print(predictions)
    return predictions


def SVM(x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test):
    svclassifier = SVC(kernel='linear')  
    svclassifier.fit(x_train, y_train)
    predictions = svclassifier.predict(x_test) 
    #print("Explained Variance:", explained_variance_score(y_test, predictions))
    
    return predictions



################################### BackTester ##############################3

price = np.loadtxt('/Users/kylekoshiyama/Desktop/Bot_2.0/list_price.csv', delimiter=",")
#test = list(zip(random_forrest(),price))
test = list(zip(NN(shape=13),price))
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
    print('Trade Log')
    for i in test:
        if first_price == 0:
            first_price = i[1]
        if current_value == 0: #to start the backetester
            if i[0] <= lower_bound:
                current_value += i[1]
        elif current_value > 0: #if we are long
            if i[0] <= higher_bound:
                net_profit += (i[1] - current_value) #current Value - what we bought at
                print('Profit from Long =', (i[1] - current_value))
                if (i[1] - current_value) > 0:
                    good_trade_count += 1
                else:
                    bad_trade_count += 1
                current_value = -i[1] #now are are short
                trade_count += 1
        elif current_value < 0: #if we are currently short
            if i[0] >= lower_bound:
                net_profit += ((-current_value) - i[1])
                print('Profit from Short =', ((-current_value) - i[1]))
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
    
    return net_profit 

Backtester(test, 0.48, 0.52)


