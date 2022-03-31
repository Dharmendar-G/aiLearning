#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import os
import zipfile
import re
import time
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules


# In[7]:


# from zipfile import ZipFile
# file_name = 'associationRuleMining.zip'
# with ZipFile(file_name,'r') as thezip:
#     a = thezip.infolist()   
# len(a)
# d = len(a)-1
# print("no of csv files : ", d)
# no_of_files_batch = 500
# no_of_batches = (d/500)
# print("total number of batches ",no_of_batches)
# s= 1
# c = 1
# list_bdfs = []
# min_range = 1
# max_range = len(a)
# for x in range(min_range,max_range,no_of_files_batch):
#     g = (s*(no_of_files_batch)+1 )
#     try:
#         list_bdfs.append([pd.read_csv(thezip.open(a[y].filename,mode='r')) for y in range(x,g)])
#     except IndexError:
#         g = max_range
#         list_bdfs.append([pd.read_csv(thezip.open(a[y].filename,mode='r')) for y in range(x,g)])
        
#     s+=1
#     c+=1
    
# count = 1
# freq = {}

# '''x = 1 => first batch => has list of 500 dfs
# pbdf for first instance => the df for first batch
# final => stores all the pbdf generated in each batch as list of pbdfs
# '''    
# final = [ ]
# for x in list_bdfs:
#     pbdf = pd.concat(x)
#     #display(pbdf)
#     for x in range(len(pbdf.vulnerabilities)):
#         sg = pbdf.vulnerabilities.iloc[x].strip('[]').split(',')

#         for x in sg:
#             y = x.strip(" ''")
#             if y in freq.keys():
#                 freq[y] +=1
#             else:
#                 freq[y] = 1
    
#     final.append(pbdf)
    
    


# In[8]:


# with zipfile.ZipFile('associationRuleMining.zip') as thezip:
#     a = thezip.infolist()# list 
#     #print(a)
#     d = len(a)-1 # 25000 files
#     print("no of csv files : ", d)
#     no_of_batches = round(d/500) # 500 files per batch => 50b * 500 files = 25000 files
#     print("total number of batches ",no_of_batches)

#     s = 1
#     c = 1
#     no_of_files_batch = 500 # no of files per batch
#     bdf = [] # list of batch dfs=> [[b1],[b2],......]
#     max_range = len(a)
#     min_range = 1 # as the  0 th index has the folder path 
#     for x in range(min_range,max_range,no_of_files_batch):

#         g = (s*(no_of_files_batch)+1 )

#         try:
#             bdf.append([pd.read_csv(thezip.open(a[y].filename,mode='r')) for y in range(x,g)])
#         except IndexError:
#             g = max_range
#             bdf.append([pd.read_csv(thezip.open(a[y].filename,mode='r')) for y in range(x,g)])
            

#         s+=1 
#         c+=1
        
# cou = 1
# freq = {}

# '''x = 1 => first batch => has list of 500 dfs
# pbdf for first instance => the df for first batch
# final => stores all the pbdf generated in each batch as list of pbdfs
# '''
# final = [ ]
# for x in bdf:
#     pbdf = pd.concat(x)
#     #display(pbdf)
#     for x in range(len(pbdf.vulnerabilities)):
#         sg = pbdf.vulnerabilities.iloc[x].strip('[]').split(',')

#         for x in sg:
#             y = x.strip(" ''")
#             if y in freq.keys():
#                 freq[y] +=1
#             else:
#                 freq[y] = 1
    
#     final.append(pbdf)
        
# # generates overall df for 25000 csvs
# fdf = pd.concat(final)

# print("columns in the dataframe\n ",fdf.columns)

# # to remove the columns that match with the Unnamed:
# rem = []
# for x in fdf.columns:
#     if re.search("Unnamed:",x):
#         rem.append(x)

# fdf.drop(rem,axis=1,inplace = True)

# print("columns in dataframe after removing the unwanted columns : \n",fdf.columns)


# # to create the new columns with list of freqs of the related vulns of cveid's
# new_col = []
# for x in range(len(fdf.vulnerabilities)):
#     sg = fdf.vulnerabilities.iloc[x].strip('[]').split(',')
#     sg = [x.strip("'' ") for x in sg]
#     cn = []
#     for x in sg:
#         cn.append(freq[x])
#     new_col.append(cn)
    
    
# # adding the new_col to the end of the dataframe fdf.
# fdf_n = fdf.assign(freq_vulns = new_col)

# print(fdf_n.head())
# compression_opts = dict(method = 'zip',archive_name = "associate_csv_update.csv")
# fdf_n.to_csv("associate_csv_update.zip",compression = compression_opts )

# print("Done Saving the update.....!!!!")


# In[9]:


#fdf


# In[10]:


#fdf_n


# In[11]:


#fdf_n.head(50)


# In[ ]:


# file_name = 'associationRuleMining.zip'
# with ZipFile(file_name,'r') as zip:
#     a = zip.infolist()# list 
# len(a)
# d = len(a)-1
# print("no of csv files : ", d)
# no_of_files_batch = 500
# no_of_batches = (d/500)
# print("total number of batches ",no_of_batches)
# s= 1
# c = 1
# list_bdfs = []
# min_range = 1
# max_range = len(a)
# for x in range(min_range,max_range,no_of_files_batch):
#     g = (s*(no_of_files_batch)+1 )
#     try:
#         list_bdfs.append([pd.read_csv(zip.open(a[y].filename,mode='r')) for y in range(x,g)])
#     except IndexError:
#         g = max_range
#         list_bdfs.append([pd.read_csv(zip.open(a[y].filename,mode='r')) for y in range(x,g)])
        
#     s+=1
#     c+=1
    
# count = 1
# freq = {}

# '''x = 1 => first batch => has list of 500 dfs
# pbdf for first instance => the df for first batch
# final => stores all the pbdf generated in each batch as list of pbdfs
# '''    

# final = [ ]
# for x in list_bdfs:
#     pbdf = pd.concat(x)
#     #display(pbdf)
#     for x in range(len(pbdf.vulnerabilities)):
#         sg = pbdf.vulnerabilities.iloc[x].strip('[]').split(',')

#         for x in sg:
#             y = x.strip(" ''")
#             if y in freq.keys():
#                 freq[y] +=1
#             else:
#                 freq[y] = 1
    
#     final.append(pbdf)


# In[ ]:





# In[ ]:


# path = f"{os.getcwd()}/dataset//associationRuleMining.zip"
# zf = zipfile.ZipFile(path)

# start = time.time()
# with zf as thezip:
#     a = thezip.infolist() # list 
#     b = [pd.read_csv(thezip.open(a[x].filename,mode='r')) for x in range(1,len(a))]
#     fdf = pd.concat(b)
# end = time.time()
# # Return the current time in seconds 
# print(f"Total number of csv files  :  {len(a)}\n")
# print(f"time taken to read the files and make them to a dataframe: {end-start} sec \n ")


# In[2]:


#importing required libraries
import pandas as pd
import os
import zipfile
import re
import time
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

#importing zipfile module
from zipfile import ZipFile
file_name = 'associationRuleMining.zip'
with ZipFile(file_name,'r') as zip:
    zip.printdir()
    
    print('extracting....')
    zip.extractall()
    print('done')

#getting current working directory
os.getcwd()
#changing directory to output path
os.chdir('output')
#reading all csv files of listdir
df = [pd.read_csv(f, index_col=0)
        for f in os.listdir(os.getcwd()) if f.endswith('csv')]
#concatinating all csv files into one dataframe
finaldf = pd.concat(df, axis=0)
#reset the index
finaldf.reset_index(inplace=True)
finaldf.drop(['index'],axis=1,inplace=True)
#printing top 5 records
print(finaldf.head())
#checking shape
print(finaldf.shape)
#checking unique values
finaldf['vulnerabilities'].unique()
#length of unique values 
print(len(finaldf['vulnerabilities'].unique()))
#value counts of vulnerabilities column
finaldf['vulnerabilities'].value_counts()
#creating list of vulnerabilities elements
lists=[]
for i in range(len(finaldf.vulnerabilities)):
    a = finaldf["vulnerabilities"][i].strip("[]").replace("'","")
    lists.append(a.split(","))
print(lists)

list_vul = lists.copy()
#creating list of tuple set of vulnerabilities
t5 = []
for i in list_vul:
    t5.append(tuple(i))
    
#creating freq dict for vulnerabilities columns
freq6 = {}
for i in t5:
    if i in freq6:
        freq6[i] = freq6[i]+1
    else:
        freq6[i] = 1
#print(freq6)        
#creating list of values of frequency
list_vals6 = []
for i in freq6.values():
    list_vals6.append(i)
    
print(list_vals6)

#print(len(list_vals6))
t2 = t5[:14699]
#creating sample of freq_dict with 14,700 rows for printing
freq7 = {}
for i in t2:
    if i in freq7:
        freq7[i] = freq7[i]+1
    else:
        freq7[i] = 1
        
print(freq7)

list_vals7 = []
for i in freq7.values():
    list_vals7.append(i)
    
len(list_vals7)
#creating sample data
sample_data = finaldf[:28420].copy()
#adding new column of vul_freq_val
sample_data['vul_freq_val'] = list_vals6

print(sample_data.head())

#converting sample_data to csv file
sample_data.to_csv('sample_df',index = False)


print(list_vals6)
# print('######################################################')
# print(freq7)
# print('######################################################')
# print(sample_data.head())
