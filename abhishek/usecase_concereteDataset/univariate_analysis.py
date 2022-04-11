import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from get_data import get_Data

class Univariate_Analysis:
    def __init__(self):
        pass

    def get_data_info(self, data):
        info = data.info()
        return info

    def iqr(self, data):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1  
        return IQR

    def lower_range_iqr(self, data):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_range = Q1-(1.5 * IQR) 
        return lower_range

    def upper_range_iqr(self, data):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        upper_range = Q3 + (1.5 * IQR)  
        return upper_range

    def no_of_outliers(self, data, col):
        outlier = []
        colms = list(data.columns)
        index = colms.index(col)
        for x in data[col]:
            if (x>self.upper_range_iqr[col]) or (x<self.lower_range_iqr[col]):
                outlier.append(x)
        return len(outlier)   
        
         


