#!/usr/bin/env python
# coding: utf-8

# In[1]:
##importing required libraries
import pandas as pd
import os
import zipfile
import re
import time
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules


# In[30]:


#importing zipfile module
from zipfile import ZipFile
file_name = 'associationRuleMining.zip'
with ZipFile(file_name,'r') as zip:
    zip.printdir()
    
    print('extracting....')
    zip.extractall()
    print('done')

#with zipfile.ZipFile('associationRuleMining.zip') as thezip:
# a = zip.infolist()# list

#getting current working directory
os.getcwd()
#changing directory to output path
os.chdir('output')
#rechecking the current directory
os.getcwd()
#reading all csv files of listdir
df = [pd.read_csv(f, index_col=0)
        for f in os.listdir(os.getcwd()) if f.endswith('csv')]
#concatinating all csv files into one dataframe
finaldf = pd.concat(df, axis=0)
#reset the index
finaldf.reset_index(inplace=True)
finaldf.drop(['index'],axis=1,inplace=True)
# In[48]:
#printing top 10 records
finaldf.head(10)
#checking shape
finaldf.shape
# In[95]:
#finaldf.head(10)

#checking unique values
finaldf['vulnerabilities'].unique()

#length of unique values 
len(finaldf['vulnerabilities'].unique())

#printing all vulnerabilities elements
for i in finaldf.vulnerabilities:
    print(i)

#value counts of vulnerabilities column
finaldf['vulnerabilities'].value_counts()

#creating list of vulnerabilities elements
lists=[]
for i in range(len(finaldf.vulnerabilities)):
    a = finaldf["vulnerabilities"][i].strip("[]").replace("'","")
    lists.append(a.split(","))
lists

len(lists)

list_vul = lists.copy()

#creating list of tuple set of vulnerabilities
t5 = []
for i in list_vul:
    t5.append(tuple(i))

t5
#creating freq dict for vulnerabilities columns
freq6 = {}
for i in t5:
    if i in freq6:
        freq6[i] = freq6[i]+1
    else:
        freq6[i] = 1
print(freq6)

#creating list of values of frequency
list_vals6 = []
for i in freq6.values():
    list_vals6.append(i)

list_vals6

len(list_vals6)

t2 = t5[:14699]

#creating sample of freq_dict with 14,700 rows for printing
freq7 = {}
for i in t2:
    if i in freq7:
        freq7[i] = freq7[i]+1
    else:
        freq7[i] = 1
print(freq7)   

list_vals = []
for i in freq7.values():
    list_vals.append(i)

list_vals7

len(list_vals7)

#sorting the dict based on the values
dict_sorted_val = {k:v for k,v in sorted(freq7.items(),key = lambda v:v[1],reverse=True)}
dict_sorted_val

#sorted values appending into list
list_frq_val7=[]
for i in dict_sorted_val.values():
    list_frq_val7.append(i)

#list_frq_val7

#len(list_frq_val7)

finaldf.head()

sample_data = finaldf[:28420].copy()

sample_data

sample_data['vul_freq_val'] = list_vals6

sample_data.head(50)
