#!/usr/bin/env python
# coding: utf-8

# # find the followings keywords in csv file
# 
# 1. 'Building Occupancy(Single family/2-4 family)'
# 
# 2. 'Flood deductible',
# 
# 3. 'What is the deductible on the homeowners policy?',
# 
# 4. 'Is dwelling in course of construction?(Yes/No)'

# In[1]:


# from email import header
import pandas as pd
import numpy as np
import glob
import re


# In[2]:


def match(lst_sent, key):
    for i in lst_sent:
        if re.search(f'{key}'.lower(),i):
            print(f"found-----> {key}")
#             print(f"found in {i}")
        else:
            pass
#             print("Not found")


# In[3]:


def lst_sent(n,key):
    lst=[] #creating an empty list to store unique sentences in csv. 
    for i in range(1,n):   
        df = pd.read_csv(str(i)+'.csv')   #reading all 6 csv
        for i in df['new_col'].unique():
            lst.append((str(i).lower()))
              
    match(lst, key)  
    return 


# In[4]:


#find if keyword "Building Occupancy" in csv files"
lst_sent(7,'Building Occupancy')


# In[5]:


#find if keyword "flood deductible" in csv files"
lst_sent(7,'flood deductible')


# In[6]:


#find if keyword "What is the deductible on the homeowners policy?" in csv files"
lst_sent(7,'What is the deductible on the homeowners policy?')


# In[7]:


#find if keyword "Is dwelling in course of construction?" in csv files"
lst_sent(7,'Is dwelling in course of construction?')


# In[ ]:





# In[ ]:




