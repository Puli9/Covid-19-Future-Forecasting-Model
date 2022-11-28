# Sreesha B20119  Mobile no: 8639196385 

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

df= pd.read_csv("daily_covid_cases.csv")
y=df[df.columns[1]]   # 2nd Column values i.e Number of covid-19 cases
x=pd.to_datetime(df[df.columns[0]],infer_datetime_format=True)

# Q1
#a
plt.plot(x,y)  # line plot
plt.xlabel("Month-Year")
plt.ylabel("New confirmed cases")
plt.title("Lineplot-Q1a")
plt.xticks(rotation=45)
plt.show()

#b
print("Autocorrelation coefficient is",round(y.autocorr(),3)) # autocorrelation

#c
plt.scatter(y[:-1],y[1:])  #Scatter plot between given time sequence and one day lagged generated sequence
plt.ylabel("Original values")
plt.xlabel("One-day lagged values")
plt.show()

#d
l= range(1,7)  
auto_corr=[]
for i in l:
    auto_corr.append(y.autocorr(lag=i))  # correlation between given sequence and i lagged generated sequence
    
plt.plot(l,auto_corr)
plt.xlabel("lag")
plt.ylabel("Autocorrelation Coefficient")
plt.show()

#e
plot_acf(y,lags=6)
plt.xlabel("Lag")
plt.show()



    
    
    