#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')

df1=pd.read_csv('Crop_recommendation.csv')
from sklearn.model_selection import train_test_split
train,test=train_test_split(df1,test_size=0.3,random_state=1)
train.to_csv("train_data.csv")
test.to_csv('test_data.csv')
df1['label'].value_counts()


# In[2]:


df=pd.read_csv("train_data.csv").drop('Unnamed: 0',axis=1)
df.head()


# In[3]:



y=df['label']
x=df.drop('label',axis=1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
x_train.shape


# In[4]:


from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=5,
                                       n_estimators=100, oob_score=True)


# In[5]:


classifier_rf.fit(x_train, y_train)


# In[6]:


### hyperparameter Tunung

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
params = {
    'max_depth': [2,3,5,10,20],
    'min_samples_leaf': [5,10,20,50,100,200],
    'n_estimators': [10,25,30,50,100,200]
}


# In[7]:


from sklearn.model_selection import GridSearchCV
# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf,
                           param_grid=params,
                           cv = 4,
                           n_jobs=-1, verbose=1, scoring="accuracy")


# In[8]:


grid_search.fit(x_train, y_train)


# In[9]:


grid_search.predict([[13,73,20,30.504209,35.48886,5.391560,162.592772]])[0]


# In[12]:


import joblib
model=joblib.dump(grid_search, 'rf.pkl')
rf_model = joblib.load('rf.pkl')


# In[11]:


rf_model.predict([[13,73,20,30.504209,35.48886,5.391560,162.592772]])[0]


# In[ ]:




if request.method=='POST'
        n1=request.form["N"]
        n2=request.form["P"]
        n3=request.form["K"]
        n4=request.form["temp"]
        n5=request.form["Humidity"]
        n6=request.form["ph"]
        n7=request.form["rainfall"]

        data=[[float("N"),float("P"),float("K"),float("temp"),float("Humidity"),float("ph"),float("rainfall")]]
        