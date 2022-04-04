#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import nltk
import pandas as pd
import numpy as np
import re 


# In[ ]:


df = pd.read_csv('StandardTemplate.csv',encoding='cp1252')
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_columns', 500)


# In[ ]:


df.head()


# In[ ]:


df1 = pd.read_csv('1.csv')
df2 = pd.read_csv('2.csv')
df3 = pd.read_csv('3.csv')
df4 = pd.read_csv('4.csv')
df5 = pd.read_csv('5.csv')
df6 = pd.read_csv('6.csv')


# In[ ]:


df_list =[df1,df2,df3,df4,df5,df6]


# In[ ]:


data = ' '.join(list(df4['text'].values))
data


# In[ ]:


def fun1(data1):
    match1 = re.findall(r'Is building elevated .includes dwelling crawl spaces.\?\s[\w\s]+[^\d.]',data)
    return match1


# In[ ]:


def fun2(data1):
    match2 = re.findall(r'Is the area below the elevated floor enclosed\?\s[\w]+\so\s[\w]+',data)

    return match2


# In[ ]:


def fun3(data1):
    match3 = re.findall(r'Type of enclosure walls:\s[\w\s]+',data)

    return match3


# In[ ]:


def fun4(data1):
    match4 = re.findall(r'Basement enclosed area:\s[\w\s]+ ',data)

    return match4


# In[ ]:


def fun(feature,column):
    list1=[]
    for i in range(6):
        try: 
            
            data1 = ' '.join(list(df_list[i]['text'].values))
            match = feature(data1)
            

            if (len(match) >= 1) and (match != None):
                
                a = match[0].split(' ')
                
                result = ''
                
                for val in a:
                    if val == 'ao':
                        b = a.index(val)
                        result = a[b-1]
                    
                list1.append(result)
                    

            else:

                list1.append('-')
        
        except Exception as e:

                list1.append('-')


    print(list1)


    for i in range(6):
        df.loc[i,column] = list1[i]


# In[ ]:


fun(fun1,'Is building elevated (includes dwelling crawl spaces)? (Yes/No)')


# In[ ]:


fun(fun2,'Is the area below the elevated floor enclosed? (Yes/No).1')


# In[ ]:


fun(fun3,'Type of enclosure walls: (Breakaway/Lattice/Solid perimeter)')


# In[ ]:


fun(fun4,'Basement enclosed area: (None/Finished/Unfinished)')


# In[ ]:


df.head()

