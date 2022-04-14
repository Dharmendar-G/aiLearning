import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from get_data import get_Data
from univariate_analysis import Univariate_Analysis

class Preprocessing:
    def __init__(self):
        pass

    def preprocess(self, data):
        data.dropnull()
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        data = data.drop_duplicates()
        data_cleaned = data[~((data < (Q1 - 1.5 *(Univariate_Analysis.iqr(get_Data())))) |(data > (Q3 + 1.5 *(Univariate_Analysis.iqr(get_Data()))))).any(axis=1)]
        
        #trating skewness
        for i in list(data.columns):
            data_cleaned[i] = (data_cleaned[i]).transform(np.log)

        #scaling using standard scaler
        scaler = StandardScaler()
        df = scaler.fit_transform(data_cleaned)
        return df
            
