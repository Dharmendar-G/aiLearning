import os
# print(os.getcwd())
# print(os.listdir(os.getcwd()))

# change the dir 
os.chdir('SivaShankar/featureExtraction')
print(os.getcwd())
# using Relative path 
# ../ => it get backs to the previous dir
import pandas as pd
df = pd.read_csv('../../dataset/1.csv',encoding = 'utf-8')
print(df.head())

# using absolute path to read files.
# D:/ailearning/aiLearning/dataset/1.csv