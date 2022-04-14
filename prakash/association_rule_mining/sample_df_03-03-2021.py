#!/usr/bin/env python
# coding: utf-8

# In[10]:


#importing required libraries
import pandas as pd
import os
import zipfile
import re
import time
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules


# In[11]:


#importing zipfile module
from zipfile import ZipFile
file_name = 'associationRuleMining.zip'
with ZipFile(file_name,'r') as zip:
    zip.printdir()
    
    print('extracting....')
    zip.extractall()
    print('done')


# In[12]:


#getting current working directory
os.getcwd()


# In[13]:


#changing directory to output path
os.chdir('output')


# In[14]:


#reading all csv files of listdir
df = [pd.read_csv(f, index_col=0)
        for f in os.listdir(os.getcwd()) if f.endswith('csv')]


# In[15]:


#concatinating all csv files into one dataframe
finaldf = pd.concat(df, axis=0)


# In[16]:


#reset the index
finaldf.reset_index(inplace=True)
finaldf.drop(['index'],axis=1,inplace=True)


# In[59]:


#printing top 10 records
finaldf.head(5)


# In[18]:


#checking shape
finaldf.shape


# In[19]:


#length of unique values 
len(finaldf['vulnerabilities'].unique())


# In[20]:


#value counts of vulnerabilities column
finaldf['vulnerabilities'].value_counts()


# In[60]:


#creating list of vulnerabilities elements
lists=[]
for i in range(len(finaldf.vulnerabilities)):
    a = finaldf["vulnerabilities"][i].strip("[]").replace("'","")
    lists.append(a.split(","))
#lists


# In[22]:


len(lists)


# In[23]:


list_vul = lists.copy()


# In[24]:


#creating list of tuple set of vulnerabilities
t5 = []
for i in list_vul:
    t5.append(tuple(i))


# In[44]:


dict = {}
dict1 = {}
for ele in t5:
    dict[ele] = dict.get(ele,0)+1
for key,value in dict.items():
    print(key[0],':',value)
    


# In[45]:


#dict.strip()


# In[46]:


# new_col1 = []
# for x in range(len(finaldf.vulnerabilities)):
#     sg1 = finaldf.vulnerabilities.iloc[x].strip('[]').split(',')
#     sg1 = [x.strip("'' ") for x in sg1]
#     cn = []
#     for x in sg1:
#         print(x)
#         cn.append(dict[x])
#         new_col1.append(cn)


# In[52]:


freq = {}
for x in range(len(finaldf.vulnerabilities)):
        sg = finaldf.vulnerabilities.iloc[x].strip('[]').split(',')

        for x in sg:
            y = x.strip(" ''")
            if y in freq.keys():
                freq[y] +=1
            else:
                freq[y] = 1
    


# In[53]:


#freq


# In[54]:


new_col = []
for x in range(len(finaldf.vulnerabilities)):
    sg = finaldf.vulnerabilities.iloc[x].strip('[]').split(',')
    sg = [x.strip("'' ") for x in sg]
    cn = []
    for x in sg:
        cn.append(freq[x])
    new_col.append(cn)


# In[61]:


#new_col


# In[48]:


df = finaldf.copy()


# In[49]:


df.head()


# In[56]:


df['freq_col_vul'] = new_col


# In[57]:


df.head()


# In[58]:


df.to_csv('sample_data.csv',index=False)


# In[ ]:




