#importing required libraries
import pandas as pd
import os
os.getcwd()
os.chdir("output")
os.remove('finaldf.csv')
dfs = [pd.read_csv(f, index_col=0)
        for f in os.listdir(os.getcwd()) if f.endswith('csv')]
finaldf = pd.concat(dfs, axis=0)
finaldf.reset_index(inplace=True)
finaldf.drop(['index'],axis=1,inplace=True)
finaldf.to_csv('finaldf.csv')
data = pd.read_csv('finaldf.csv')
print(data.head())
print(data.shape)
