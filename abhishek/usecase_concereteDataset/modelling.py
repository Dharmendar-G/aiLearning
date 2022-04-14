import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
from numpy import mean
from numpy import std
from numpy import absolute
from numpy import arange
from sklearn.model_selection import GridSearchCV
import pickle
import joblib

class Model:
    def __init__(self):
        pass

    def linear_regression(self,data):
        X = data.iloc[:, :8]
        y = data['strength']
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.2)
        model = LinearRegression()
        model.fit(X_train, y_train)
        predict = model.predict(X_test)
        score = r2_score(y_test, predict)
        return score

    def random_forest(self, data):
        X = data.iloc[:, :8]
        y = data['strength']
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.2)
        reg=RandomForestRegressor()
        reg.fit(X_train,y_train.values.ravel())
        pred=reg.predict(X_test)
        #r2_score(y_test,pred)
        return "Success"


    def save_model(self, trained_model):
        #saved_model = pickle.dumps(trained_model) 
        joblib.dump(trained_model, 'filename.pkl')



