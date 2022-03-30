#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('pip install chart_studio')


# In[35]:


#get_ipython().system('pip install pmdarima')


# # Stock Market Price Trend Prediction Using Time Series Forecasting

# In[46]:


# Importing libraries
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight') 
# Above is a special style template for matplotlib, highly useful for visualizing time series data
#get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import rcParams
from plotly import tools
import chart_studio.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
import statsmodels.api as sm
from numpy.random import normal, seed
from scipy.stats import norm
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima_model import ARIMA
import math
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ### Importing time series data

# ### How to import data?
# First, we import all the datasets needed for this kernel. The required time series column is imported as a datetime column using **parse_dates** parameter and is also selected as index of the dataframe using **index_col** parameter. 
# #### Data being used:-
# 1. Maruti Suzuki Stock Data

# In[17]:


maruti = pd.read_csv('MARUTI.NS.csv', index_col='Date', parse_dates=['Date'])
maruti.head()


# In[20]:


maruti.tail()


# In[48]:


maruti.info()


# In[49]:


maruti.describe()


# ### Visualizing the dataset

# In[23]:


maruti['2010':'2021'].plot(subplots=True, figsize=(10,12))
plt.title('Maruti Suzuki India Limited stock attributes from 2010 to 2021')
plt.savefig('stocks.png')
plt.show()


# ### Visualize the per day closing price of the stock.

# In[19]:


#plot close price
plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Close Prices')
plt.plot(maruti['Close'])
plt.title('Maruti Suzuki India Limited closing price')
plt.show()


# ### Lets us plot the scatterplot:

# In[21]:


df_close = maruti['Close']
df_close.plot(style='k.')
plt.title('Scatter plot of closing price')
plt.show()


# ### Histogram of Closing price

# In[24]:


plt.figure(figsize=(10,6))
df_close = maruti['Close']
df_close.plot(style='k.',kind='hist')
plt.title('Hisogram of closing price')
plt.show()


# First, we need to check if a series is stationary or not because time series analysis only works with stationary data.
# 
# Testing For Stationarity:
# 
# To identify the nature of the data, we will be using the null hypothesis.
# 
# H0: The null hypothesis: It is a statement about the population that either is believed to be true or is used to put forth an argument unless it can be shown to be incorrect beyond a reasonable doubt.
# 
# H1: The alternative hypothesis: It is a claim about the population that is contradictory to H0 and what we conclude when we reject H0.
# 
# #Ho: It is non-stationary
# #H1: It is stationary
# If we fail to reject the null hypothesis, we can say that the series is non-stationary. This means that the series can be linear.
# 
# If both mean and standard deviation are flat lines(constant mean and constant variance), the series becomes stationary.

# In[50]:


# Cleaning and Preparing Time Series Data
maruti = maruti.fillna(method='ffill')


# In[27]:


def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    #Plot rolling statistics:
    plt.plot(timeseries, color='yellow',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=False)
    print("Results of dickey fuller test")
    adft = adfuller(timeseries,autolag='AIC')
    # output for dft will give us without defining what the values are.
    #hence we manually write what values does it explains using a for loop
    output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key,values in adft[4].items():
        output['critical value (%s)'%key] =  values
    print(output)

test_stationarity(maruti['Close'])


# We see that the p-value is greater than 0.05 so we cannot reject the Null hypothesis. Also, the test statistics is greater than the critical values. so the data is non-stationary.
# 
# For time series analysis we separate Trend and Seasonality from the time series.

# In[30]:


result = seasonal_decompose(maruti['Close'], model='multiplicative', freq = 30)
fig = plt.figure()  
fig = result.plot()  
fig.set_size_inches(16, 9)


# In[31]:


rcParams['figure.figsize'] = 10, 6
df_log = np.log(maruti['Close'])
moving_avg = df_log.rolling(12).mean()
std_dev = df_log.rolling(12).std()
plt.legend(loc='best')
plt.title('Moving Average')
plt.plot(std_dev, color ="black", label = "Standard Deviation")
plt.plot(moving_avg, color="red", label = "Mean")
plt.legend()
plt.show()


# Now we are going to create an ARIMA model and will train it with the closing price of the stock on the train data. So let us split the data into training and test set and visualize it.

# In[32]:


train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Closing Prices')
plt.plot(df_log, 'green', label='Train data')
plt.plot(test_data, 'blue', label='Test data')
plt.legend()


# Its time to choose parameters p,q,d for ARIMA model. Last time we chose the value of p,d, and q by observing the plots of ACF and PACF but now we are going to use Auto ARIMA to get the best parameters without even plotting ACF and PACF graphs.

# Auto ARIMA: Automatically discover the optimal order for an ARIMA model.
# The auto_arima function seeks to identify the most optimal parameters for an ARIMA model, and returns a fitted ARIMA model. This function is based on the commonly-used R function, forecast::auto.arima.
# The auro_arima function works by conducting differencing tests (i.e., Kwiatkowski–Phillips–Schmidt–Shin, Augmented Dickey-Fuller or Phillips–Perron) to determine the order of differencing, d, and then fitting models within ranges of defined start_p, max_p, start_q, max_q ranges. If the seasonal optional is enabled, auto_arima also seeks to identify the optimal P and Q hyper- parameters after conducting the Canova-Hansen to determine the optimal order of seasonal differencing, D.

# In[37]:


model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
test='adf',       # use adftest to find optimal 'd'
max_p=3, max_q=3, # maximum p and q
m=1,              # frequency of series
d=None,           # let model determine 'd'
seasonal=False,   # No Seasonality
start_P=0, 
D=0, 
trace=True,
error_action='ignore',  
suppress_warnings=True, 
stepwise=True)
print(model_autoARIMA.summary())


# In[38]:


model_autoARIMA.plot_diagnostics(figsize=(15,8))
plt.show()


# So how to interpret the plot diagnostics?
# 
# Top left: The residual errors seem to fluctuate around a mean of zero and have a uniform variance.
# 
# Top Right: The density plot suggest normal distribution with mean zero.
# 
# Bottom left: All the dots should fall perfectly in line with the red line. Any significant deviations would imply the distribution is skewed.
# 
# Bottom Right: The Correlogram, aka, ACF plot shows the residual errors are not autocorrelated. Any autocorrelation would imply that there is some pattern in the residual errors which are not explained in the model. So you will need to look for more X’s (predictors) to the model.
# 
# Overall, it seems to be a good fit. Let’s start forecasting the stock prices.
# 
# Next, create an ARIMA model with provided optimal parameters p, d and q.

# In[39]:


model = ARIMA(train_data, order=(3, 1, 2))
fitted = model.fit(disp=-1)
print(fitted.summary())


# Now let's start forecast the stock prices on the test dataset keeping 95% confidence level.

# In[44]:


# Forecast
fc, se, conf = fitted.forecast(297, alpha=0.05)  # 95% confidence
fc_series = pd.Series(fc, index=test_data.index)
lower_series = pd.Series(conf[:, 0], index=test_data.index)
upper_series = pd.Series(conf[:, 1], index=test_data.index)
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train_data, label='training')
plt.plot(test_data, color = 'blue', label='Actual Stock Price')
plt.plot(fc_series, color = 'orange',label='Predicted Stock Price')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.10)
plt.title('Maruti Suzuki India Limited Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Actual Stock Price')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# As you can see our model did quite handsomely. Let us also check the commonly used accuracy metrics to judge forecast results:

# In[47]:


# report performance
mse = mean_squared_error(test_data, fc)
print('MSE: '+str(mse))
mae = mean_absolute_error(test_data, fc)
print('MAE: '+str(mae))
rmse = math.sqrt(mean_squared_error(test_data, fc))
print('RMSE: '+str(rmse))
mape = np.mean(np.abs(fc - test_data)/np.abs(test_data))
print('MAPE: '+str(mape))