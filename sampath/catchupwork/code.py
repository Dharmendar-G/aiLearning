import json
import csv
import os

path='C:\\Users\\MajjiSampathVinayKum\\Desktop\\Notes Assignments\\ailearning\\aiLearning\\sampath\\catchupwork\\input.json'
#Method1

import numpy as np
import pandas as pd

df=pd.read_json(path)
#print(df)
df2=df.to_csv('C:\\Users\\MajjiSampathVinayKum\\Desktop\\Notes Assignments\\ailearning\\aiLearning\\sampath\\catchupwork\\method1.csv')
#print(df2)
#type(df2)

#Method2
with open(path) as j:
   data=json.load(j)
   #print(data)

with open('C:\\Users\\MajjiSampathVinayKum\\Desktop\\Notes Assignments\\ailearning\\aiLearning\\sampath\\catchupwork\\method2.csv','w') as k:
   d=csv.writer(k)
   d.writerow(data)