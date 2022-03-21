#!/usr/bin/env python
# coding: utf-8

# In[22]:


# Import Libraries
import numpy as np
import pandas as pd


# In[57]:


print("This is practice programm")

def process_data(data):
    print("Data is Analysis")
    shape_data = data.shape
    size_data = data.size
    print("Dataset shape : ",shape_data)
    print("Dataset size : ",size_data)
    
    return data
    
def read_data(data):
    print("data is readed")
    df=pd.read_csv(data)
    return df
def write_data(data):
    print("Data is written into the database")
    print(data)
def main():
    data= read_data()
    modified_data = process_data(data)
    write_data(modified_data)
if __name__=="__main__":
    main()


# In[ ]:




