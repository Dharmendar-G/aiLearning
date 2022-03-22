print("hello world")
import os
# print(os.getcwd())
# print(os.listdir(os.getcwd()))
os.chdir('SivaShankar/featureExtraction')
print(os.getcwd())
import pandas as pd
df = pd.read_csv('../../dataset/1.csv',encoding = 'utf-8')
print(df.head())