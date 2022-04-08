import pandas as pd
import numpy as np
import datetime
import os
import re

# data
df = pd.read_excel('raw-logs.xlsx')
df.head()

# Key Values Extraction 
class key_value:
    def __init__(self,data):
        self.df = data
        
    def find_ip(self):
        ip_col = 'IP_Address'
        for col in self.df.columns:
            ip = re.findall('\d{3}\.\d{3}\.\d{2}.\d+', str(df[col].tolist()))
            if len(ip)>len(ip_col):
                a = self.df[col].apply(lambda x: re.findall(r'\d+\.\d+\.\d+\.\d+',x))
                self.df[ip_col] = pd.DataFrame([i[0] if len(i) > 0 else 'NA' for i in a ])
       
    def find_host(self): 
        h_col = 'Host_Name'
        for col in self.df.columns:
            if h.split('_')[0].lower() in str(self.df[col].tolist()):
                b = self.df[col].apply(lambda x: re.findall(r'"host":"[A-Za-z0-9]+-[A-Za-z0-9]+"',x))
                self.df[h_col] = pd.DataFrame([str(i[0]).split(':')[1].strip() if len(i) > 0 else np.nan for i in b])

    def extract_timestamp(self):
        ts_col = 'timeStamp'
        ts = []
        for col in self.df.columns:
            if ts_col.lower() in str(self.df[col].tolist()):
                for i in range(len(self.df)):
                    ts.append(re.findall(r'\w{3}\s\w+\s\d+\s\d+.\d+.\d+.\d+|\d+\-\d{2}\-\d{2}\s\d+.\d+.\d+|\d{2}\-\w{3}\-\d+\s\d+.\d+.\d+.\d+|\w{3}\s+\d{2}\s\d+.\d+.\d+|\d{6}\s\d{2}\:\d+\:\d+|\d{2}\/\w{3}\/\d+.\:\d+.\d+.\d+..\d+|\d+\-\d{2}\-\d{2}.\d+.\d+.\d+.\d+',self.df.loc[i, col])[0])
                self.df[ts_col] = pd.DataFrame(ts)

b = key_value(df)
b.find_ip()
b.find_host()
b.extract_timestamp()

df.head()