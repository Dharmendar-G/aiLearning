


import pandas as pd
import numpy as np
import datetime
import os
import re

df1 = pd.read_excel('raw-logs.xlsx')
df1.head()

class key_value:
    def __init__(self,data):
        self.df = data
    
    def find_ip(self):
        a = self.df['_source.logData'].apply(lambda x: re.findall(r'\d+\.\d+\.\d+\.\d+',x))
        self.df['IP Address'] = pd.DataFrame([i[0] if len(i) > 0 else 'NA' for i in a ])
       
    def find_host(self):  
        b = self.df["_source.logData"].apply(lambda x: re.findall(r'"host":"[A-Za-z0-9]+-*[A-Za-z0-9]+"',x))
        self.df['Host_name'] = pd.DataFrame([str(i[0]).split(':')[1].strip() if len(i) > 0 else np.nan for i in b])

b = key_value(df1)
b.find_ip()
b.find_host()
df1.head(30)
df1["Host_name"].value_counts()
df1["Host_name"].isna().sum()




