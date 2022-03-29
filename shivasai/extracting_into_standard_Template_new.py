#!/usr/bin/env python
# coding: utf-8

# 1.Is the lowest floor living area off ground by means of: (Piers/Posts/Piles/Columns/Solid perimeter walls/Parallel shear walls)
# 2.Does basement or enclosed area contain machinery and equipment? (Yes/No)
# 3.Garage or enclosed area is used for: (Parking/Storage/Access)
# 4.Type of building:(Split-level//Townhouse/Rowhouse//High Rise)
# 

# In[1]:


import pandas as pd
import numpy as np
import re
import nltk


# In[2]:


df=pd.read_csv('StandardTemplate.csv',encoding='cp1252')
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_columns', 500)


# In[3]:


df.head()


# In[4]:


df1 = pd.read_csv('1.csv')
df2 = pd.read_csv('2.csv')
df3 = pd.read_csv('3.csv')
df4 = pd.read_csv('4.csv')
df5 = pd.read_csv('5.csv')
df6 = pd.read_csv('6.csv')


# In[5]:


data = ' '.join(list(df4['text'].values))
data


# In[6]:


def features(dataframe,Reg_exp):
    data = ' '.join(list(dataframe['text'].values))
    match1 = re.findall(Reg_exp, data)  

    mail=match1[0].split(':')
    a = mail[1].split(' ')
    z=[]
    for i in a:
        if i=='oa':
            b=a.index(i)
            x=a[b-1]
            z.append(x)
        
        elif i == 'a':
            b = a.index(i)
            y=a[b-1]
            z.append(y)
    s=','.join(z)
    return s


# In[7]:


feature1 = r'Is the lowest floor living area off ground by means of:\s\w+\s\w+\s\w+\s\w+\s\w+\s\w+\s\w+\s\w+\s\w+\s\w+\s\w+\s\w+\s\w+\s\w+'    
p=features(df4,feature1)
df.loc[3,'Is the lowest floor living area off ground by means of: (Piers/Posts/Piles/Columns/Solid perimeter walls/Parallel shear walls)'] = p


# In[8]:


feature2 = re.findall(r' Does basement or enclosed area contain machinery and equipment\?[\s\w+]{9}',data)
c=feature2[0].split("?")
d=c[1].split(' ')
q=[]
for i in d:
    if i=='oa':
        e=d.index(i)
        x=d[e-1]
        q.append(x)
        
        
    elif i == 'a':
        e = d.index(i)
        y=d[e-1]
        q.append(y)
        


df.loc[3,'Does basement or enclosed area contain machinery and equipment? (Yes/No)']=q


# In[9]:


feature3=r' Garage or enclosed area is used for:[\s\w+]{8}/\w+/\w+'
r=features(df4,feature3)
df.loc[3,'Garage or enclosed area is used for: (Parking/Storage/Access)']=r
    


# In[10]:


feature4=r'Type of building:[\s\w+]\w+-\w+\s\w+/\w+\s\w+\s\w+\s\w+\s\w+'
s=features(df4,feature4)
df.loc[3,'Type of building:(Split-level//Townhouse/Rowhouse//High Rise)']=s


# In[12]:


df.head()



# In[ ]:




