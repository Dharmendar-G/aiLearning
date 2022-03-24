#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import nltk
import re

import os
print(os.getcwd())


# In[2]:


data=pd.read_csv("../mukund/StandardTemplate.csv", encoding='cp1252')
pd.set_option('display.max_colwidth',1000)
pd.set_option('display.max_columns',500)


# In[3]:


df0 = pd.read_csv('../dataset/1.csv')
df1 = pd.read_csv('../dataset/2.csv')
df2 = pd.read_csv('../dataset/3.csv')
df3 = pd.read_csv('../dataset/4.csv')
df4 = pd.read_csv('../dataset/5.csv')
df5 = pd.read_csv('../dataset/6.csv')


# In[4]:


df_list=[df0,df1,df2,df3,df4,df5]


# In[5]:


def feature1(data1):
   
    match3 = re.findall(r'Insured Name:\s[A-Za-z]+\s[A-Za-z]+\s[A-Za-z]+\s[A-Za-z]+\s[A-Za-z]+\s[A-Za-z]+\s[A-Za-z]+|Insured Name:\s[A-Za-z]+\s[A-Za-z]+|INSURED’S NAME:\s[_]+[A-Za-z]+\s[A-Za-z]+\s[A-Za-z]+|Insured’s Name:\s[A-Za-z]+\s[A-Za-z]+\s[A-Za-z]+\s[A-Za-z]+\s[A-Za-z]+|Insured’s Name:\s[A-Za-z]+\s[A-Za-z]+\s[A-Za-z]+\s&\s[A-Za-z]+\s[A-Za-z]+', data1)
    return match3


def feature2(data):
    
    match1 = re.findall(r'Mailing Address:\s*[__]*\s[\d{2}]*\s[A-Za-z]+\s[A-Za-z]+\s', data)
    
    return match1

def feature3(data1):
    
    match3 = re.findall(r'City/State/Zipcode:\s[A-Za-z\s]*[A-Za-z,]+\s[A-Za-z]+\s[0-9]+', data1)
    
    return match3


def feature4(data1):

    match4 = re.findall(r'Property Location:\s[\d]+\s[A-Za-z]+\s[A-Za-z]+\s', data1)

    return match4


# In[6]:


def fun(feature,column):
    list1=[]
    for i in range(6):
        try: 
            data1 = ' '.join(list(df_list[i]['text'].values))
            

            match = feature(data1)
            #print(match)
    

            if (len(match) >= 1) and (match != None):
            
                result = ''
            
                for k in range(len(match)):

                    mail=match[k].split(':')
                    
                    result = result +  mail[1]
                    
                        
                list1.append(result)

            else:

                list1.append(np.nan)
        
        except Exception as e:

                list1.append(np.nan)


    print(list1)


    for i in range(6):
        data.loc[i,column] = list1[i]


# In[7]:


fun(feature1,'Insured Name')

fun(feature2,'Mailing Address')

fun(feature3,'City/State/Zipcode')

fun(feature4,'Property Location')


# In[9]:


print(data.head(10))
data.to_csv('submission.csv',index=False)

# In[ ]:




