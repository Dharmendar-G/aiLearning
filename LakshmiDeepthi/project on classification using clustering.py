#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymongo

client=pymongo.MongoClient('mongodb://localhost:27017/')
mydb_deepthi=client.Classification_Project.seeds_dataset
cursor = mydb_deepthi.find()
entries = list(cursor)


df = pd.DataFrame(entries)
df


# In[ ]:




