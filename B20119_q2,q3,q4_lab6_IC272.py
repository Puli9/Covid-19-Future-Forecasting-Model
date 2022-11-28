# Sreesha B20119  Mobile no: 8639196385 


import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from statsmodels.tsa.ar_model import AutoReg as AR
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_percentage_error as MAPE
# Train test split
series = pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')
test_size = 0.35 # 35% for testing
X = series.values
tst_sz = math.ceil(len(X)*test_size)
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]

#q2
pd.DataFrame(train).plot(legend=None,ylabel='New confirmed cases',title='test set')
plt.show()
pd.DataFrame(test).plot(legend=None,ylabel='New confirmed cases',title='test set')
plt.show()

window = 5 # The lag=5
model = AR(train, lags=window,old_names=False)
model_fit = model.fit() # fit/train the model
coef = model_fit.params # Get the coefficients of AR model
print(np.ndarray.round(coef,3))
#using these coefficients walk forward over time steps in test, one step each time
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list() # List to hold the predictions.
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-window,length)]    
    yhat = coef[0] # Initialize to w0
    for d in range(window):
        yhat += coef[d+1] * lag[window-d-1] # Add other values
        
    obs = test[t]
    predictions.append(yhat) #Append predictions to compute RMSE later
    history.append(obs) # Append actual test value to history, to be used in next step.

# q2(i)
plt.scatter(test,predictions)
plt.xlabel('actual values')
plt.ylabel("predicted values")
plt.show()
# q2(ii)
plt.plot(test)
plt.plot(predictions,color='red')
plt.legend(['Actual values','predicted values'])
plt.show()
# q2(iii)
rmse_per= (mse(test,predictions,squared=False)*100)/(sum(test)/len(test))
print("RMSE(%)="+'%.3f'%rmse_per)
mape=MAPE(test,predictions)*100
print('MAPE =',round(mape,3))

#q3
lags=[1,5,10,15,25]
rmse=[]
mape=[]
for i in lags:
    
    model = AR(train, lags=i,old_names=False)
    model_fit = model.fit() # fit/train the model
    coef = model_fit.params # Get the coefficients of AR model
    #using these coefficients walk forward over time steps in test, one step each time
    history = train[len(train)-i:]

    history = [history[i] for i in range(len(history))]

    predictions = list() # List to hold the predictions.
    for t in range(len(test)):
        length = len(history)
        lag = [history[i] for i in range(length-i,length)]
        yhat = coef[0] # Initialize to w0
        for d in range(i):
            yhat += coef[d+1] * lag[i-d-1] # Add other values
        
        obs = test[t]
        predictions.append(yhat) #Append predictions to compute RMSE later
        history.append(obs) # Append actual test value to history, to be used in next step.
    rmse_per= (mse(test,predictions,squared=False)*100)/(sum(test)[0]/len(test))
    print("RMSE(%) for lag=",i,'is '+'%.3f'%rmse_per)
    rmse.append(rmse_per)
    mape_res=MAPE(test,predictions)*100
    print('MAPE for lag=',i,'is',round(mape_res,3))
    mape.append(mape_res)


plt.bar(lags,mape)
plt.xticks(lags,lags)
plt.xlabel("Lag values")
plt.ylabel("MAPE")
plt.show()
plt.bar(lags,rmse)
plt.xticks(lags,lags)
plt.xlabel("Lag values")
plt.ylabel("RMSE(%)")
plt.show()


# q4
b=[]
for i in range(len(train)):
    b.append(train[i][0])
b=pd.Series(b)
i=1
while (i>0):
    a=b.autocorr(lag=i)
    if abs(a) <= 2/(len(train)**0.5):
        break
    i=i+1

print("The heuiristic value for the optical number of lags is",i-1)

window = i-1 # The lag=5
model = AR(train, lags=window,old_names=False)
model_fit = model.fit() # fit/train the model
coef = model_fit.params # Get the coefficients of AR model

#using these coefficients walk forward over time steps in test, one step each time
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list() # List to hold the predictions.
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-window,length)]
    yhat = coef[0] # Initialize to w0
    for d in range(window):
        yhat += coef[d+1] * lag[window-d-1] # Add other values
        
    obs = test[t]
    predictions.append(yhat) #Append predictions to compute RMSE later
    history.append(obs) # Append actual test value to history, to be used in next step.


rmse_per= (mse(test,predictions,squared=False)*100)/(sum(test)/len(test))
print("RMSE(%)="+'%.3f'%rmse_per)
mape=MAPE(test,predictions)*100
print('MAPE =',round(mape,3))

