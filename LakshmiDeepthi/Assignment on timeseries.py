
# In[1]:


import pandas_datareader.data as pdr
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np 
plt.style.use('fivethirtyeight') 
# Above is a special style template for matplotlib, highly useful for visualizing time series data
# get_ipython().run_line_magic('matplotlib', 'inline')
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
from statsmodels.tsa.arima.model import ARIMA
import math
from sklearn.metrics import mean_squared_error


# #### Column  Meaning
# - High-   that day high price
# - Low-    low Price on that day
# - Open-   opening price on that day
# - close-  Closing price on that day
# - Volume- volume would refer to the number of shares of a security traded between its daily open and close.
# 

# - <a href='#1'>1. Importing the dataset</a>
#     - <a href='#1.1'>1.1 preparing the dataset</a>
#     - <a href='#1.2'>1.2 Visualizing the datasets</a>
#     - <a href='#1.3'>1.3 Resampling</a>
#     - <a href='#1.4'>1.4 Shifting and lagging</a>
# - <a href='#2'>2. Finance and Statistics</a>
#     - <a href='#2.1'>2.1 Percent change</a>
#     - <a href='#2.2'>2.2 Stock returns</a>
#     - <a href='#2.3'>2.3 Absolute change in successive rows</a>
#     - <a href='#2.4'>2.4 Comparing two or more time series</a>
#     - <a href='#2.5'>2.5 Window functions</a>
#     - <a href='#2.6'>2.6 OHLC charts</a>
#     - <a href='#2.7'>2.7 Candlestick charts</a>
#     - <a href='#2.8'>2.8 Autocorrelation and Partial Autocorrelation</a>
# - <a href='#3'>3. Time series decomposition and Random Walks</a>
#     - <a href='#3.1'>3.1 Trends, Seasonality and Noise</a>
#     - <a href='#3.2'>3.2 White Noise</a>
#     - <a href='#3.3'>3.3 Random Walk</a>
#     - <a href='#3.4'>3.4 Stationarity</a>
# - <a href='#4'>4. Modelling using statsmodels</a>
#     - <a href='#4.1'>4.1 Predicting the models</a>
#         - <a href="#4.1.1">4.1.1 ARIMA Model</a>
#     - <a href='#4.2'>4.2 State space methods</a>
#         - <a href='#4.2.1'>4.2.1 SARIMAX MODEL</a>
#         - <a href='#4.2.2'>4.2.2 DYNAMIC FACTORS</a>

# # <a id='1'>1.Importing the dataset</a>

# In[43]:


LG_data=pdr.DataReader('066570.KS',"yahoo",start="2010") # importing the LG Electronics data
LG_data.tail(10)


# ### <a id='1.1'>1.1.preparing the dataset</a>

# In[3]:


LG_data.index


# In[4]:


LG_data.isnull().sum()


# In[5]:


LG_data.info()


# ### <a id='1.2'>1.2 Visualizing the datasets</a>

# In[6]:


LG_data.plot(subplots=True,figsize=(10,15))
plt.title('LG stock attributes from 2010 to 2022')
plt.savefig('stocks.png')
plt.show()


# In[7]:


plt.style.available


# ### <a id='1.3'> 1.3 Resampling </a>

# In[8]:


LG_data.resample(rule="A").max().plot(figsize=(12,8))


# ## 
# - from the above fig i observed that from 2019 to 2020 volume is increased and then it starts fall down.

# In[9]:


LG_data.resample(rule="QS").max().plot(subplots=True,figsize=(12,16),xlim=["2019-01-04","2022-04-01"])
plt.show()


# In[10]:


LG_data.resample(rule="QS").mean().plot(kind="bar",subplots=True,figsize=(12,16),xlim=["2019-01-04","2022-04-01"])
plt.show()


# In[11]:


LG_data.resample(rule="M").mean().plot(subplots=True,figsize=(12,16),xlim=["2019-01-04","2022-04-01"])
plt.show()


# In[12]:


LG_data.High['2019':'2020'].resample("M").plot(figsize=(12,8))
plt.show()


# #### Assumptions:
# - from the above figure i understand that "HIGH" sale price was decreased in apr 2020 due to covid -19 effect.

# In[13]:


LG_data.High['2020':'2021'].resample("M").plot(figsize=(12,8))
plt.show()


# 
# - From the above fig we can observe that in 2021- jan there is tremondous incraease in high price.after that there is fluctuations but no fall down. 

# In[14]:


# here i want to observe how the past i.e 2010-2015
LG_data.High['2010':'2019'].resample("M").plot(figsize=(12,8))
plt.show()


# -  from above figure end of 2015 the sales was down because of foreign rivals i.e samsung and apple , sales more falls in South korea.
# -source: https://au.finance.yahoo.com/news/lg-electronics-sees-50-fall-221627707.html

# In[15]:


LG_data.columns.duplicated().any()


# In[16]:


LG_data = LG_data.reset_index()
LG_data['Date'] = pd.to_datetime(LG_data['Date'])
# 'ts' is now datetime of 'Timestamp', you just need to set it to index
LG_data = LG_data.set_index('Date')


# ### <a id= '1.4'>1.4 Shifting and lagging</a>

# In[42]:


LG_data.High['2019':'2022'].asfreq(freq='D').plot(legend=True)
shifted=LG_data.High['2019':'2022'].asfreq(freq='D').shift(100).plot(legend=True)
shifted.legend(["high",'high-lagged'])
plt.show()


# # <a id='2'>2. Finance and statistics</a>

# ## <a id='2.1'>2.1 Percent change</a>

# In[18]:


LG_data['Change'] = LG_data.High.div(LG_data.High.shift())
LG_data['Change'].plot(figsize=(20,8))


# ## <a id="2.2"> 2.2 stock returns </a>

# In[19]:


LG_data['Return'] = LG_data.Change.sub(1).mul(100)
LG_data['Return'].plot(figsize=(20,8))


# ## <a id='2.3'>2.3 Absolute change in successive rows</a>

# In[20]:


LG_data.High.diff().plot(figsize=(20,6))


# ## <a id='2.4'>2.4 Comparing two or more time series</a> 

# In[44]:


sams_data=pdr.DataReader('005930.KS',"yahoo",start="2010") # importing the samsung Electronics data
sams_data.tail(10)


# In[46]:


sams_data.info()


# In[52]:


# Before Normalization
plt.figure(figsize=(12,8))
LG_data.High['2019':'2022'].plot()
sams_data.High['2019':'2022'].plot()
plt.legend(['LG','SAMSUNG'])
plt.show()


# 
# - From the above figure we observed that sales of samsung is lower than LG electronics.
# - In jan,2021 LG sales are increased.

# In[65]:


# After Normalization
normalized_LG = LG_data.High['2019':'2022'].div(LG_data.High.iloc[0]).mul(100)
normalized_samsung = sams_data.High['2019':'2022'].div(sams_data.High.iloc[0]).mul(100)
normalized_LG.plot()
normalized_samsung.plot()
plt.legend(['LG','samsung'])
plt.show()


# - After normalization sales of samsung electronics is high .

# ## <a id='2.5'>2.5 Window functions</a>

# In[68]:


rolling_LG_data = LG_data.High.rolling('90D').mean()
LG_data.High.plot()
rolling_LG_data.plot()
plt.legend(['High','Rolling Mean'])
# Plotting a rolling mean of 90 day window with original High attribute of LG stocks
plt.show()


# In[70]:


samsung_mean = sams_data.High.expanding().mean()
samsung_std = sams_data.High.expanding().std()
sams_data.High.plot()
samsung_mean.plot()
samsung_std.plot()
plt.legend(['High','Expanding Mean','Expanding Standard Deviation'])
plt.show()


# ## <a id='2.6'>2.6 OHLC charts</a>

# In[73]:


# OHLC chart of april 2021 to april 2022
trace = go.Ohlc(x=LG_data['04-2021':'04-2022'].index,
                open=LG_data['04-2021':'04-2022'].Open,
                high=LG_data['04-2021':'04-2022'].High,
                low=LG_data['04-2021':'04-2022'].Low,
                close=LG_data['04-2021':'04-2022'].Close)
data = [trace]
iplot(data, filename='simple_ohlc')


# ## <a id='2.7'> 2.7 Candlestick chart

# In[76]:


# Candlestick chart from jan 2021 to april 2021
trace = go.Candlestick(x=LG_data['01-2021':'04-2021'].index,
                open=LG_data['01-2021':'04-2021'].Open,
                high=LG_data['01-2021':'04-2021'].High,
                low=LG_data['01-2021':'04-2021'].Low,
                close=LG_data['01-2021':'04-2021'].Close)
data = [trace]
iplot(data, filename='simple_candlestick')


# ## <a id='2.8'>2.8 Autocorrelation and Partial Autocorrelation</a>

# In[78]:


# Autocorrelation of 
plot_acf(LG_data["High"],lags=20,title="HIGH")
plt.show()


# In[81]:


plot_pacf(LG_data["High"],lags=20,title="HIGH")
plt.show()


# # <a id='3'>3. Time series decomposition and Random walks</a>

# ## <a id='3.1'>3.1. Trends, seasonality and noise</a>

# In[82]:


LG_data["High"].plot(figsize=(16,8))


# In[84]:


rcParams['figure.figsize'] = 11, 9
decomposed_LG_volume = sm.tsa.seasonal_decompose(LG_data["High"],period=360) # The frequncy is annual
figure = decomposed_LG_volume.plot()
plt.show()


# ## <a id='3.2'>3.2. White noise</a>

# In[85]:


rcParams['figure.figsize'] = 16, 6
white_noise = np.random.normal(loc=0, scale=1, size=1000)
# loc is mean, scale is variance
plt.plot(white_noise)


# ## <a id='3.3'>3.3. Random Walk</a>

# ### Augmented Dickey-fuller test on LG  and Samsung data

# In[86]:


adf = adfuller(sams_data["Volume"])
print("p-value of samsung: {}".format(float(adf[1])))
adf = adfuller(LG_data["Volume"])
print("p-value of LG: {}".format(float(adf[1])))


# ##### As SAMSUNG has p-value 0.00000000465964 which is less than 0.05, null hypothesis is rejected and this is not a random walk.
# ##### Now LG has p-value 0.0000000063154 which is more than 0.05, null hypothesis is rejected and this is not a  random walk.

# ## <a id='3.4'>3.4 Stationarity</a>

# In[88]:


decomposed_LG_volume.trend.plot()


# In[89]:


decomposed_LG_volume.trend.diff().plot()


# # <a id='4'>4. Modelling using statstools</a>

# ## <a id='4.1'>4.1 Predicting the models</a>

# ### <a id='4.1.1'>4.1.1 ARIMA Model</a>

# In[141]:


# ARMA is a depreciated model so it showing NotImplementedError
from statsmodels.tsa.arima.model import ARIMA
train_sample = LG_data["Volume"].diff().iloc[1:].values
LG = ARIMA(np.asarray(train_sample), order=(4,0,4),trend="c")
res = LG.fit()
print(res.summary())
predicted_result = result.predict(start=0, end=500)
result.plot_diagnostics()
# calculating error
rmse = math.sqrt(mean_squared_error(train_sample[1:502], predicted_result))
print("The root mean squared error is {}.".format(rmse))


# In[144]:


plt.plot(train_sample[1:500],color='red')
plt.plot(predicted_result,color='blue')
plt.legend(['Actual','Predicted'])
plt.title('LG Volume prices')
plt.show()


# # <a id='4.2'>4.2 State space methods</a>

# ### <a id="4.2.1">4.2.1 SARIMAX MODEL</a>

# In[142]:


train_sample = LG_data["Volume"].diff().iloc[1:].values
model = sm.tsa.SARIMAX(train_sample,order=(4,0,4),trend='c')
result = model.fit(maxiter=1000,disp=False)
print(result.summary())
predicted_result1 = result.predict(start=0, end=500)
result.plot_diagnostics()
# calculating error
rmse = math.sqrt(mean_squared_error(train_sample[1:502], predicted_result))
print("The root mean squared error is {}.".format(rmse))


# In[143]:


plt.plot(train_sample[1:500],color='red')
plt.plot(predicted_result1,color='blue')
plt.legend(['Actual','Predicted'])
plt.title('LG Volume prices')
plt.show()


# ### <a id="4.2.2">4.2.2 DYNAMIC FACTORS</a>

# In[148]:


train_sample1 = pd.concat([LG_data["Close"].diff().iloc[1:],sams_data["Close"].diff().iloc[1:]],axis=1)
model = sm.tsa.DynamicFactor(train_sample1, k_factors=1, factor_order=2)
result = model.fit(maxiter=1000,disp=False)
print(result.summary())
predicted_result3 = result.predict(start=0, end=1000)
result.plot_diagnostics()
# calculating error
rmse = math.sqrt(mean_squared_error(train_sample.iloc[1:1002].values, predicted_result.values))
print("The root mean squared error is {}.".format(rmse))


# In[155]:


plt.plot(train_sample1[1:502],color='red')
plt.plot(predicted_result3,color='green')
plt.legend(['Actual','predicted'])
plt.title(' Closing prices')
plt.show()
