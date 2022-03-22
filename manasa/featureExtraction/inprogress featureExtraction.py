#importing Modules
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import re

df1 = pd.read_csv('C:\\Users\\MorampudiManasa\\Downloads\\1.csv')
df2 = pd.read_csv('C:\\Users\\MorampudiManasa\\Downloads\\2.csv')
df3 = pd.read_csv('C:\\Users\\MorampudiManasa\\Downloads\\3.csv')
df4 = pd.read_csv('C:\\Users\\MorampudiManasa\\Downloads\\4.csv')
df5 = pd.read_csv('C:\\Users\\MorampudiManasa\\Downloads\\5.csv')
df6 = pd.read_csv('C:\\Users\\MorampudiManasa\\Downloads\\6.csv')

data = ' '.join(list(df4['text'].values))
data

pattern = "Requested Policy Effective Date: \w/[0-9]"

u=re.findall(pattern,data)

print(u)

# Writing Company:

pattern ="Writing Company: \w+ \w+ \w+"

u1=re.findall(pattern,data)

print(u1)

#p="ProducerName : \w+ \w+ \w+"